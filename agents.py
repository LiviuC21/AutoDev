"""agents.py — agenții AI: GeminiArchitect (plan/review/conversație) și QwenWorker (cod local prin Ollama)."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from google import genai
from google.genai import types

from config import (
    GEMINI_MODEL, MAX_ATTEMPTS, OLLAMA_URL, QWEN_MODEL, QWEN_NUM_CTX,
    QWEN_NUM_PREDICT, QWEN_PROMPT_CHAR_BUDGET, Cancelled,GEMINI_MODELS
)

MAX_HISTORY_PAIRS = 8  # perechi (utilizator, Gemini) păstrate în conversație
TRANSIENT_MARKERS = ("429", "500", "503", "unavailable", "resource_exhausted", "overloaded", "deadline")


class AgentError(RuntimeError):
    """Eroare la comunicarea cu un agent (Gemini sau Qwen/Ollama)."""


# --------------------------------------------------------------------------- #
# Utilitare comune
# --------------------------------------------------------------------------- #
def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def run_cancellable(func: Callable[[], Any], cancel: Optional[threading.Event]) -> Any:
    """Rulează o funcție blocantă într-un thread și o abandonează dacă se apasă Stop."""
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - propagăm orice eroare în thread-ul apelant
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        thread.join(0.2)
    if "error" in box:
        raise box["error"]
    return box.get("value")


def normalize_rel(path: str) -> str:
    """Normalizează o cale relativă din plan; returnează '' dacă e nesigură."""
    rel = str(path).strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    parts = [p for p in rel.lstrip("/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return ""
    return "/".join(parts)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n… [tăiat]"


def _clip_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    return text[:half] + "\n… [partea din mijloc a fost omisă] …\n" + text[-half:]


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("Răspunsul JSON nu este un obiect.")
    return data


def _clean_steps(raw: Any) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        rel = normalize_rel(item.get("path", ""))
        if not rel or rel in seen:
            continue
        seen.add(rel)
        description = str(item.get("description", "")).strip()
        requirements = str(item.get("requirements", "")).strip() or description
        deps_raw = item.get("depends_on", [])
        deps = [normalize_rel(d) for d in deps_raw if isinstance(d, str)] if isinstance(deps_raw, list) else []
        steps.append({
            "path": rel,
            "description": description,
            "requirements": requirements,
            "depends_on": [d for d in deps if d],
        })
    return steps


def _format_files(contents: Optional[dict[str, str]]) -> str:
    if not contents:
        return "(niciun conținut disponibil)"
    return "\n\n".join(f"### {rel}\n```\n{code}\n```" for rel, code in contents.items())


# --------------------------------------------------------------------------- #
# Gemini — Arhitect + Reviewer
# --------------------------------------------------------------------------- #
ARCHITECT_SYSTEM = """Ești Arhitectul software și Reviewer-ul unui sistem cu doi agenți.
Un model local mic (Qwen2.5-Coder 7B) scrie codul, CÂTE UN FIȘIER PE RÂND, fără să vadă tot proiectul și
cu un context foarte mic. De aceea:
- împarți proiectul în fișiere mici, cu o singură responsabilitate (ideal sub ~200 de linii);
- pentru fiecare fișier scrii cerințe foarte concrete: nume exacte de clase/funcții/metode, semnături,
  importuri dintre fișiere, formate de date, comportament, tratarea erorilor;
- ordonezi fișierele astfel încât fiecare să importe doar fișiere scrise înaintea lui.
Răspunzi în limba română în câmpurile `message` și `reply` (pentru utilizator), dar scrii ÎNTOTDEAUNA în limba ENGLEZĂ câmpurile `description`, `requirements` și notele de review `REFA` (pentru Qwen).O
Proiecte Python: țintă Linux, Python 3.11+, rulează într-un .venv. Doar fișierul de intrare are blocul
`if __name__ == "__main__":`; celelalte module trebuie să poată fi importate fără efecte secundare
(fără input(), fără rețea, fără ferestre la import). Dacă folosești biblioteci externe, primul fișier din plan
este `requirements.txt`. Proiecte Android: Kotlin + Gradle, structură standard (app/src/main/...).Reguli stricte de UI/UX (Design 2026):
- Pentru aplicații desktop Python cu GUI, NU folosi Tkinter standard. Folosește PyQt6 (cu Fusion + stylesheet QSS modern) sau CustomTkinter (dark mode), adăugând librăria în requirements.txt.
- Design modern: temă dark curată (#18181b / #27272a), culori de accent moderne (ex. #3b82f6 sau #8b5cf6), colțuri rotunjite (border-radius 8px-10px), spațiere generoasă (padding/margins), etichete clare și placeholder-e pe fiecare câmp de input, butoane cu efect hover și layout aerisit.
- Pentru aplicații Android: folosește Material Design 3 (MaterialComponents / Jetpack Compose sau XML modern cu carduri rotunjite, temă dark/dinamică și spațiere corectă). """

PLAN_SCHEMA = """{
  "message": "scurt răspuns pentru utilizator (2-4 propoziții): ce vei construi",
  "entry_file": "calea fișierului de intrare (Python: ex. main.py; Android: activitatea principală)",
  "files": [
    {
      "path": "cale/relativă/fișier.ext",
      "description": "rolul fișierului într-o frază",
      "requirements": "cerințe detaliate și concrete pentru Qwen",
      "depends_on": ["alte/fișiere/din/proiect.py de care depinde"]
    }
  ]
}"""

CHAT_SCHEMA = """{
  "reply": "răspunsul pentru utilizator, în română",
  "entry_file": "(opțional) noul fișier de intrare, doar dacă se schimbă",
  "tasks": [
    {
      "path": "fișier de creat sau modificat",
      "description": "rolul fișierului",
      "requirements": "descrierea COMPLETĂ a ce trebuie să facă fișierul după modificare, cu detalii concrete",
      "depends_on": ["fișiere de care depinde"]
    }
  ]
}"""


class GeminiArchitect:
    """Arhitect și Reviewer bazat pe `google-genai`."""

    def __init__(self, api_key: str, model: str = GEMINI_MODEL) -> None:
        if not api_key:
            raise AgentError("Lipsește GEMINI_API_KEY. Adaug-o din butonul Setări.")
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.history: list[tuple[str, str]] = []  # (rol, text), rol = "user" | "model"

    # ----- apel de bază ---------------------------------------------------- #
    def _contents(self, prompt: str, use_history: bool) -> list[types.Content]:
        contents: list[types.Content] = []
        if use_history:
            for role, text in self.history[-MAX_HISTORY_PAIRS * 2:]:
                contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        return contents

    def _generate(self, prompt: str, json_mode: bool, use_history: bool,
                  cancel: Optional[threading.Event]) -> str:
        contents = self._contents(prompt, use_history)
        kwargs: dict[str, Any] = {"system_instruction": ARCHITECT_SYSTEM, "temperature": 0.3}
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**kwargs)
        last_error: Optional[BaseException] = None
        models_to_try = [self.model] + [m for m in GEMINI_MODELS if m != self.model]
        for model_name in models_to_try:
            _check_cancel(cancel)
            try:
                response = run_cancellable(
                    lambda m=model_name: self.client.models.generate_content(
                        model=m, contents=contents, config=config),
                    cancel,
                )
                self.model = model_name
                return (response.text or "").strip()
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc)
                if any(k in msg for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    continue
                raise AgentError(f"Eroare Gemini ({model_name}): {exc}") from exc
        raise AgentError(f"Toate modelele Gemini au atins limita: {last_error}")

    def _json_call(self, prompt: str, use_history: bool, cancel: Optional[threading.Event]) -> dict[str, Any]:
        current = prompt
        for _ in range(2):
            text = self._generate(current, json_mode=True, use_history=use_history, cancel=cancel)
            try:
                return parse_json(text)
            except ValueError:
                current = prompt + "\n\nRăspunsul tău anterior nu a fost JSON valid. Răspunde DOAR cu obiectul JSON cerut."
        raise AgentError("Gemini nu a returnat un JSON valid după două încercări.")

    def _remember(self, user_text: str, model_text: str) -> None:
        self.history.append(("user", user_text))
        self.history.append(("model", model_text))
        del self.history[:-MAX_HISTORY_PAIRS * 2]

    # ----- planificare ----------------------------------------------------- #
    def plan(self, requirements: str, project_type: str, existing_files: list[str],
             existing_contents: Optional[dict[str, str]] = None,
             cancel: Optional[threading.Event] = None) -> dict[str, Any]:
        """Returnează {"message", "entry_file", "files": [{path, description, requirements, depends_on}]}."""
        existing = "\n".join(f"- {f}" for f in existing_files) or "(folderul este gol)"
        prompt = f"""Tip proiect: {project_type}

Cerințele utilizatorului:
{requirements}

Fișiere existente în folderul proiectului:
{existing}

Conținutul fișierelor existente (pentru a ține cont de ele):
{_format_files(existing_contents)}

Sarcină: creează planul de implementare. Dacă există deja cod, nu rescrie ce funcționează; include în plan
doar fișierele noi sau cele care trebuie modificate (la cele existente descrie în `requirements` comportamentul
final complet dorit). Folosește căi relative cu `/`, fără `..`.
Răspunde DOAR cu JSON valid, exact în această schemă:
{PLAN_SCHEMA}"""
        data = self._json_call(prompt, use_history=False, cancel=cancel)
        steps = _clean_steps(data.get("files"))
        if not steps:
            raise AgentError("Planul primit de la Gemini nu conține niciun fișier.")
        paths = {s["path"] for s in steps}
        known = paths | set(existing_files)
        entry = normalize_rel(str(data.get("entry_file", "")))
        if project_type == "Python" and entry not in known:
            python_files = [s["path"] for s in steps if s["path"].endswith(".py")]
            entry = "main.py" if "main.py" in paths else (python_files[-1] if python_files else "")
        message = str(data.get("message", "")).strip() or "Planul este gata, încep implementarea."
        self._remember(requirements, message)
        return {"message": message, "entry_file": entry, "files": steps}

    # ----- review ---------------------------------------------------------- #
    def review(self, rel: str, description: str, requirements: str, code: str, run_log: str,
               project_type: str, attempt: int,
               cancel: Optional[threading.Event] = None) -> tuple[str, str]:
        """Returnează ("OK" | "REFA", corecții)."""
        prompt = f"""Fă review la un fișier scris de Qwen.

Fișier: {rel} (proiect {project_type})
Rol: {description}
Încercarea {attempt}/{MAX_ATTEMPTS}.

Cerințele date lui Qwen:
{requirements}

Codul scris de Qwen:
```
{_clip_middle(code, 30000)}
```

Rezultatul verificării automate (sintaxă / rulare / import):
{_clip(run_log, 4000) or "(fără log)"}

Reguli de răspuns:
- Prima linie trebuie să fie EXACT `OK` sau `REFA`, fără alt text pe ea.
- Dacă logul arată o excepție, o eroare de sintaxă sau un import eșuat, răspunsul este `REFA`.
- `REFA` doar pentru probleme reale: bug-uri, cerințe neîndeplinite, nume/semnături diferite de cele cerute,
  cod incomplet, placeholdere. Nu cere refacere pentru stil minor.
- După `REFA`, listează corecții numerotate și concrete (funcția/zona, ce e greșit, cum se repară), maximum 10,
  cu cel mult fragmente scurte de cod. Nu rescrie tot fișierul.
- După `OK` poți adăuga o singură propoziție."""
        text = self._generate(prompt, json_mode=False, use_history=False, cancel=cancel)
        match = re.match(r"\W*(OK|REFA)\b", text.strip(), re.I)
        if match:
            verdict = match.group(1).upper()
            notes = text.strip()[match.end():].strip()
        elif re.search(r"\bREFA\b", text):
            verdict, notes = "REFA", text.strip()
        else:
            verdict, notes = "REFA", "Verdict neclar de la reviewer; cod retrimis spre refacere.\n" + text.strip()
        return verdict, notes

    # ----- conversație / feedback ------------------------------------------ #
    def chat(self, message: str, project_type: str, files: list[str],
             contents: Optional[dict[str, str]] = None,
             cancel: Optional[threading.Event] = None) -> dict[str, Any]:
        """Feedback ulterior. Returnează {"reply", "entry_file", "tasks": [...]}; `tasks` poate fi gol."""
        listing = "\n".join(f"- {f}" for f in files) or "(niciun fișier)"
        prompt = f"""Proiectul ({project_type}) este deja generat. Utilizatorul trimite un mesaj nou.

Mesajul utilizatorului:
{message}

Fișierele proiectului:
{listing}

Conținutul fișierelor:
{_format_files(contents)}

Dacă mesajul cere modificări, bug-fix-uri sau funcții noi, pune în `tasks` fișierele de creat/modificat
(în ordinea dependențelor), fiecare cu descrierea completă a comportamentului final.
Dacă e doar o întrebare sau o discuție, răspunde în `reply` și lasă `tasks` gol.
Răspunde DOAR cu JSON valid, exact în această schemă:
{CHAT_SCHEMA}"""
        data = self._json_call(prompt, use_history=True, cancel=cancel)
        reply = str(data.get("reply", "")).strip() or "Am înțeles."
        tasks = _clean_steps(data.get("tasks"))
        entry = normalize_rel(str(data.get("entry_file", "")))
        self._remember(message, reply)
        return {"reply": reply, "entry_file": entry, "tasks": tasks}

# ----- review vizual (Vision) ------------------------------------------ #
    def review_screenshot(self, image_path: Path, app_goal: str,
                          cancel: Optional[threading.Event] = None) -> tuple[str, str]:
        """Analizează captura de ecran de pe telefon și returnează (verdict, observații)."""
        img_bytes = image_path.read_bytes()
        prompt = (
            f"You are a senior Android UI/UX reviewer. Goal of the app: {app_goal}\n"
            "Inspect this screenshot taken on a real Android phone. Check for:\n"
            "1. Crashes, blank screens, or system error dialogs.\n"
            "2. Overlapping text, clipped buttons, or broken layout.\n"
            "3. Clean modern dark/Material 3 appearance.\n"
            "Reply on line 1 with ONLY 'OK' or 'REFA', followed by short concrete notes in Romanian."
        )
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            system_instruction="You are a strict Android UI/UX reviewer.",
            temperature=0.2,
        )
        last_error: Optional[BaseException] = None
        models_to_try = [self.model] + [m for m in GEMINI_MODELS if m != self.model]
        raw = ""
        for model_name in models_to_try:
            _check_cancel(cancel)
            try:
                response = run_cancellable(
                    lambda m=model_name: self.client.models.generate_content(
                        model=m, contents=contents, config=config),
                    cancel,
                )
                self.model = model_name
                raw = (response.text or "").strip()
                break
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if any(k in str(exc) for k in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")):
                    continue
                raise AgentError(f"Eroare Gemini Vision ({model_name}): {exc}") from exc
        else:
            raise AgentError(f"Toate modelele Gemini au atins limita: {last_error}")

        lines = raw.splitlines() or ["REFA"]
        first, rest = lines[0], lines[1:]
        verdict = "OK" if "OK" in first.upper() and "REFA" not in first.upper() else "REFA"
        return verdict, "\n".join(rest).strip()

# --------------------------------------------------------------------------- #
# Qwen — Muncitorul (Ollama, streaming)
# --------------------------------------------------------------------------- #
QWEN_SYSTEM = """You are an expert software engineer writing exactly ONE file of a larger project.
Output ONLY the complete final content of that file inside a single fenced code block. No explanations
before or after the block. No placeholders, no TODO, no "..." elisions: the file must be complete and working.
Use the exact names, imports and signatures given in the requirements and in the other project files shown.
Do not import modules that are not part of the project or the standard library unless the requirements list them.
Write comments and docstrings in Romanian; keep identifiers in English.
If a previous version and review notes are provided, apply EVERY correction and return the whole corrected file."""

_FENCE_RE = re.compile(r"```[ \t]*[\w+#.\-]*[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Extrage curat codul dintr-un răspuns Qwen (tolerant la fence-uri lipsă sau netrminate)."""
    text = text.replace("\r\n", "\n").strip("\n")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")[1:]
        while lines and lines[-1].strip() == "":
            lines.pop()
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        code = "\n".join(lines)
    else:
        blocks = _FENCE_RE.findall(text)
        if blocks:
            code = max(blocks, key=len)
        elif "```" in text:
            rest = text[text.find("```") + 3:]
            newline = rest.find("\n")
            code = rest[newline + 1:] if newline != -1 else ""
        else:
            code = text
    return code.strip("\n") + "\n"

QWEN_EDIT_SYSTEM = """You are an expert software engineer modifying an EXISTING file.
DO NOT rewrite the whole file. Output ONLY one or more SEARCH/REPLACE blocks for the exact lines that must change:

<<<<<<< SEARCH
exact existing lines from the current file
=======
new replacement lines
>>>>>>> REPLACE

Rules:
- Copy the SEARCH lines EXACTLY as they appear in the current file (same indentation and spacing).
- Keep SEARCH blocks short (3-10 lines of context) so they match uniquely.
- Never output the whole file unless every single line changes. No explanations outside the blocks."""

_PATCH_RE = re.compile(
    r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE",
    re.S,
)


def apply_patches(original: str, raw: str) -> str:
    """Aplică blocurile SEARCH/REPLACE pe fișierul existent. Fallback la cod complet dacă nu există blocuri."""
    patches = _PATCH_RE.findall(raw.replace("\r\n", "\n"))
    if not patches:
        # Plasă de siguranță: dacă Qwen a rescris tot fișierul în loc de patch, îl folosim normal.
        return extract_code(raw)

    updated = original.replace("\r\n", "\n")
    for search_block, replace_block in patches:
        if search_block in updated:
            updated = updated.replace(search_block, replace_block, 1)
        elif search_block.strip() and search_block.strip() in updated:
            updated = updated.replace(search_block.strip(), replace_block.strip(), 1)
    return updated.strip("\n") + "\n"

class QwenWorker:
    """Scrie cod prin Ollama, cu streaming live și suport pentru Stop."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = QWEN_MODEL) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_truncated = False

    def check_available(self) -> Optional[str]:
        """Returnează un mesaj de eroare dacă Ollama/modelul nu sunt disponibile, altfel None."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            return f"Ollama nu răspunde la {self.base_url}: {exc}. Pornește-l cu `ollama serve`."
        if self.model not in names and f"{self.model}:latest" not in names:
            return f"Modelul {self.model} nu este descărcat. Rulează: ollama pull {self.model}"
        return None

    # ----- prompt ---------------------------------------------------------- #
    def build_prompt(self, rel: str, description: str, requirements: str, project_type: str,
                     project_files: Optional[list[str]] = None,
                     context: Optional[dict[str, str]] = None,
                     previous_code: Optional[str] = None,
                     feedback: Optional[str] = None) -> tuple[str, str]:
        """Construiește (system, prompt) încadrat în QWEN_PROMPT_CHAR_BUDGET."""
        is_edit = bool(previous_code and previous_code.strip())
        system_prompt = QWEN_EDIT_SYSTEM if is_edit else QWEN_SYSTEM

        requirements = _clip(requirements, QWEN_PROMPT_CHAR_BUDGET // 3)
        head = f"Project type: {project_type}\nFile to write: {rel}\nRole: {description}\n\nRequirements:\n{requirements}\n"
        if project_files:
            names = "\n".join(f"- {f}" for f in project_files[:60])
            head += f"\nProject files:\n{names}\n"
        footer = (
            f"\nReturn ONLY SEARCH/REPLACE blocks to update {rel}."
            if is_edit
            else f"\nReturn ONLY the complete content of {rel} in a single fenced code block."
        )

        budget = QWEN_PROMPT_CHAR_BUDGET - len(system_prompt) - len(head) - len(footer) - 200
        feedback_text = ""
        if feedback:
            feedback_text = "\nReview notes to apply (mandatory):\n" + _clip(feedback, max(500, min(3000, budget // 4))) + "\n"
            budget -= len(feedback_text)
        previous_text = ""
        if previous_code:
            previous_text = "\nCurrent version of the file:\n```\n" + _clip_middle(previous_code, max(500, int(budget * 0.5))) + "\n```\n"
            budget -= len(previous_text)
        context_text = ""
        for path, code in (context or {}).items():
            if budget < 400:
                break
            part_limit = max(200, min(3000, budget - 100))
            chunk = f"\nOther project file {path}:\n```\n{_clip(code, part_limit)}\n```\n"
            context_text += chunk
            budget -= len(chunk)
        return system_prompt, head + context_text + previous_text + feedback_text + footer

    # ----- generare -------------------------------------------------------- #
    def generate(self, system: str, prompt: str, on_token: Optional[Callable[[str], None]] = None,
                 cancel: Optional[threading.Event] = None) -> str:
        _check_cancel(cancel)
        self.last_truncated = False
        payload = {
            "model": self.model,
            "system": system,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "num_ctx": QWEN_NUM_CTX,
                "num_predict": QWEN_NUM_PREDICT,
                "temperature": 0.2,
                "top_p": 0.9,
            },
        }
        url = f"{self.base_url}/api/generate"
        try:
            response = run_cancellable(
                lambda: requests.post(url, json=payload, stream=True, timeout=(10, 600)), cancel)
        except Cancelled:
            raise
        except requests.RequestException as exc:
            raise AgentError(f"Nu mă pot conecta la Ollama ({url}): {exc}") from exc
        if response.status_code != 200:
            body = response.text[:300]
            response.close()
            raise AgentError(f"Ollama a răspuns cu HTTP {response.status_code}: {body}")

        finished = threading.Event()

        def watchdog() -> None:
            # Închide conexiunea imediat ce se apasă Stop, ca să nu așteptăm următorul token.
            while not finished.is_set():
                if cancel is not None and cancel.wait(0.2):
                    try:
                        response.close()
                    except Exception:  # noqa: BLE001
                        pass
                    return

        if cancel is not None:
            threading.Thread(target=watchdog, daemon=True).start()

        chunks: list[str] = []
        try:
            for raw in response.iter_lines():
                _check_cancel(cancel)
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    continue
                if "error" in data:
                    raise AgentError(f"Ollama: {data['error']}")
                piece = data.get("response", "")
                if piece:
                    chunks.append(piece)
                    if on_token is not None:
                        on_token(piece)
                if data.get("done"):
                    self.last_truncated = data.get("done_reason") == "length"
                    break
        except (Cancelled, AgentError):
            raise
        except Exception as exc:  # noqa: BLE001
            if cancel is not None and cancel.is_set():
                raise Cancelled() from exc
            raise AgentError(f"Conexiunea cu Ollama s-a întrerupt: {exc}") from exc
        finally:
            finished.set()
            try:
                response.close()
            except Exception:  # noqa: BLE001
                pass
        _check_cancel(cancel)
        return "".join(chunks)

    def write_file(self, rel: str, description: str, requirements: str, project_type: str,
                   project_files: Optional[list[str]] = None,
                   context: Optional[dict[str, str]] = None,
                   previous_code: Optional[str] = None,
                   feedback: Optional[str] = None,
                   on_token: Optional[Callable[[str], None]] = None,
                   cancel: Optional[threading.Event] = None) -> str:
        """Generează conținutul complet al unui fișier și îl returnează curățat de fence-uri."""
        system, prompt = self.build_prompt(
            rel, description, requirements, project_type, project_files, context, previous_code, feedback)
        raw = self.generate(system, prompt, on_token, cancel)
        if previous_code and previous_code.strip():
            code = apply_patches(previous_code, raw)
        else:
            code = extract_code(raw)
        if not code.strip():
            raise AgentError("Qwen a returnat un răspuns gol.")
        return code
