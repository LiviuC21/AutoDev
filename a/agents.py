import json
import re
import requests
from google import genai


class GeminiArchitect:
    def __init__(self, api_key: str, model: str = "gemini-2.5-pro"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.history = []

    def ask(self, prompt: str) -> str:
        self.history.append(f"USER/SYSTEM:\n{prompt}")
        full_prompt = "\n\n---\n\n".join(self.history[-8:])
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )
        text = response.text or ""
        self.history.append(f"GEMINI:\n{text}")
        return text

    def create_plan(self, user_request: str, project_type: str) -> list[dict]:
        subprompt = f"""
Ești un Arhitect Software senior. Utilizatorul vrea un proiect de tip: {project_type}.
Cererea utilizatorului: "{user_request}"

Împarte proiectul în pași cât mai mărunți (câte un fișier pe pas) pentru programatorul tău local (Qwen).
Răspunde STRICT cu un bloc JSON valid, fără text în afara lui, în formatul:
[
  {{"fisier": "cale/relativa/fisier.py", "cerinta": "descriere exactă și detaliată a ce trebuie să conțină fișierul"}}
]
"""
        raw = self.ask(subprompt)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            raise ValueError(f"Gemini nu a returnat un plan JSON valid:\n{raw}")
        return json.loads(match.group(0))

    def review_code(self, filename: str, task: str, code: str, console_error: str = "") -> tuple[bool, str]:
        err_section = f"\nEroare din consolă la rulare:\n{console_error}\n" if console_error else ""
        prompt = f"""
Verifică codul scris de Qwen pentru fișierul `{filename}`.
Cerința inițială: {task}
{err_section}
Codul lui Qwen:
---
{code}
---

Dacă codul este complet, corect și gata de salvat, răspunde pe prima linie cu: OK
Dacă are greșeli, lipsuri sau erori, răspunde pe prima linie cu: REFA
și explică scurt și concret ce trebuie corectat.
"""
        verdict = self.ask(prompt).strip()
        is_ok = verdict.upper().startswith("OK")
        return is_ok, verdict


class QwenWorker:
    def __init__(self, model: str = "qwen2.5-coder:7b", url: str = "http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def write_code_stream(self, filename: str, task: str, feedback: str = "", on_chunk=None) -> str:
        fb_section = f"\nFeedback de la Gemini (corectează neapărat):\n{feedback}\n" if feedback else ""
        prompt = f"""Ești un programator expert. Scrie codul COMPLET și funcțional pentru fișierul `{filename}`.
Cerință: {task}
{fb_section}
Returnează DOAR codul final al fișierului, într-un singur bloc de cod markdown, fără explicații lungi."""

        r = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_ctx": 8192},
            },
            stream=True,
            timeout=300,
        )
        r.raise_for_status()

        full_text = []
        for line in r.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response", "")
                if chunk:
                    full_text.append(chunk)
                    if on_chunk:
                        on_chunk(chunk)
        return self._extract_code("".join(full_text))

    @staticmethod
    def _extract_code(text: str) -> str:
        fence = "`" * 3
        match = re.search(rf"{fence}[a-zA-Z0-9_+-]*\n(.*?){fence}", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()
