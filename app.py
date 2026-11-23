"""app.py — AutoDev Studio: interfață PyQt6 (temă dark) care orchestrează Gemini (Arhitect) și Qwen (Muncitor)."""
from __future__ import annotations

import html
import json
import sys
import zipfile
import threading
import time
import traceback
import re
from runner import (
    deploy_and_capture_android, export_windows_exe, open_android_studio,
    prepare_android_project, run_gradle_build, validate_file,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from remote_api import RemoteAPIServer

from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPalette, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from agents import AgentError, GeminiArchitect, QwenWorker, list_ollama_models
from config import (
    GEMINI_MODELS, MAX_ATTEMPTS, PROJECT_TYPES, Cancelled, Settings,
    list_project_files, safe_join,
)
from github_manager import push_project

TEXT_SUFFIXES = {
    ".py", ".kt", ".java", ".xml", ".json", ".txt", ".gradle", ".kts", ".md",
    ".properties", ".toml", ".cfg", ".ini", ".yml", ".yaml", ".pro",
    ".html", ".htm", ".css", ".js", ".ts", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".c",
}
MAX_CONTEXT_FILE_BYTES = 20000

STATE_FILENAME = ".autodev_state.json"


def load_state(project_dir: str) -> Optional[dict]:
    if not project_dir:
        return None
    path = Path(project_dir) / STATE_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_state(project_dir: str, state: dict) -> None:
    path = Path(project_dir) / STATE_FILENAME
    tmp = Path(project_dir) / f"{STATE_FILENAME}.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

COLORS = {
    "info": "#9aa7b8", "plan": "#c792ea", "task": "#e5c07b", "ok": "#7fd18b",
    "warn": "#f0a35e", "err": "#f07178", "user": "#61afef", "gemini": "#7fd18b", "sys": "#9aa7b8",
}

def read_cpu_temp() -> Optional[float]:
    """Citește temperatura CPU (°C) din /sys/class/hwmon (k10temp / coretemp / acpitz)."""
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.is_dir():
        return None
    priority = ("k10temp", "coretemp", "zenpower", "acpitz")
    found: dict[str, float] = {}
    for entry in hwmon.iterdir():
        try:
            name = (entry / "name").read_text(encoding="utf-8").strip()
            if name in priority:
                raw = (entry / "temp1_input").read_text(encoding="utf-8").strip()
                found[name] = int(raw) / 1000.0
        except (OSError, ValueError):
            continue
    for key in priority:
        if key in found:
            return found[key]
    return None

_SIG_PY_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+[^\n:]+", re.M)
_SIG_KT_RE = re.compile(r"^\s*(?:(?:public|private|internal|protected|data|sealed|open|abstract|override|suspend)\s+)*(?:class|interface|object|enum\s+class|fun)\s+[^\n{=]+", re.M)
_SIG_XML_ID_RE = re.compile(r'android:id="@\+id/([^"]+)"')


def build_project_map(project_dir: str, exclude: set[str], budget: int = 2500) -> str:
    """Extrage o hartă compactă de semnături (clase, funcții, ID-uri XML) din restul fișierelor."""
    lines: list[str] = []
    used = 0
    for rel in list_project_files(project_dir):
        if rel in exclude:
            continue
        suffix = Path(rel).suffix.lower()
        if suffix not in {".py", ".kt", ".java", ".xml"}:
            continue
        try:
            path = safe_join(project_dir, rel)
            if path.stat().st_size > MAX_CONTEXT_FILE_BYTES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, ValueError):
            continue

        if suffix == ".py":
            sigs = [m.strip() for m in _SIG_PY_RE.findall(text)]
        elif suffix in {".kt", ".java"}:
            sigs = [m.strip() for m in _SIG_KT_RE.findall(text)]
        else:
            ids = _SIG_XML_ID_RE.findall(text)
            sigs = [f"ids: {', '.join(ids)}"] if ids else []

        if not sigs:
            continue
        entry = f"- {rel}: " + " | ".join(sigs[:12])
        if used + len(entry) > budget:
            break
        lines.append(entry)
        used += len(entry) + 1
    return "\n".join(lines)
# --------------------------------------------------------------------------- #
# Sesiune (rămâne între mesajele din panoul 1)
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    architect: Optional[GeminiArchitect] = None
    api_key: str = ""
    gemini_model: str = ""
    project_dir: str = ""
    project_type: str = ""
    entry_file: str = ""
    built: bool = False

# --------------------------------------------------------------------------- #
# Worker — toată logica rulează aici, comunică doar prin semnale
# --------------------------------------------------------------------------- #
class PipelineWorker(QThread):
    chat_message = pyqtSignal(str, str)   # (autor, text) -> panoul 1
    agent_log = pyqtSignal(str, str)      # (text, tip) -> panoul 2
    code_set = pyqtSignal(str, str)       # (titlu, text) -> panoul 3 sus (înlocuiește)
    code_chunk = pyqtSignal(str)          # fragment live de la Qwen
    console = pyqtSignal(str)             # panoul 3 jos
    finished_run = pyqtSignal(bool, str)  # (reușit, rezumat)

    def __init__(self, settings: Settings, session: Session, mode: str, text: str,
                 auto_push: bool, cancel: threading.Event) -> None:
        super().__init__()
        self.settings = settings
        self.session = session
        self.mode = mode
        self.text = text
        self.auto_push = auto_push
        self.cancel = cancel
        self.project_dir = session.project_dir
        self.project_type = session.project_type
        self.qwen = QwenWorker(
            model=settings.ollama_model,
            num_thread=settings.ollama_threads,
        )
        self._buffer: list[str] = []
        self._last_flush = 0.0

    # ----- ciclu de viață -------------------------------------------------- #
    def run(self) -> None:
        try:
            problem = self.qwen.check_available()
            if problem:
                raise AgentError(problem)
            if self.mode == "build":
                ok, summary = self._build()
            elif self.mode == "resume":
                ok, summary = self._resume()
            else:
                ok, summary = self._feedback()
            self.finished_run.emit(ok, summary)
        except Cancelled:
            self.agent_log.emit("⛔ Oprit de utilizator.", "warn")
            self.console.emit("⛔ Oprit de utilizator.")
            self.finished_run.emit(False, "Procesul a fost oprit.")
        except AgentError as exc:
            self.agent_log.emit(f"✘ {exc}", "err")
            self.finished_run.emit(False, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.console.emit("✘ Eroare neașteptată:\n" + traceback.format_exc())
            self.finished_run.emit(False, f"Eroare neașteptată: {exc}")

    def _check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    # ----- streaming cod --------------------------------------------------- #
    def _on_token(self, piece: str) -> None:
        self._buffer.append(piece)
        if time.monotonic() - self._last_flush >= 0.05:
            self._flush()

    def _flush(self) -> None:
        if self._buffer:
            self.code_chunk.emit("".join(self._buffer))
            self._buffer.clear()
        self._last_flush = time.monotonic()

    # ----- citire fișiere -------------------------------------------------- #
    def _read_files(self, files: list[str], budget: int) -> dict[str, str]:
        out: dict[str, str] = {}
        used = 0
        for rel in files:
            if Path(rel).name == STATE_FILENAME or Path(rel).suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                path = safe_join(self.project_dir, rel)
                if path.stat().st_size > MAX_CONTEXT_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                continue
            if used + len(text) > budget:
                break
            out[rel] = text
            used += len(text)
        return out
        
    def _context_for(self, rel: str, depends_on: list[str]) -> dict[str, str]:
        files = set(list_project_files(self.project_dir))
        strict_deps = [d for d in depends_on if d in files and d != rel]
        return self._read_files(strict_deps, 20000)

    def _read_requested_file(self, rel: str) -> Optional[str]:
        try:
            path = safe_join(self.project_dir, rel)
            if not path.is_file() or path.stat().st_size > MAX_CONTEXT_FILE_BYTES:
                self.console.emit(f"⚠ Modelul a cerut {rel}, dar nu poate fi citit.")
                return None
            self.agent_log.emit(f"📂 Modelul local a cerut fișierul: {rel}", "info")
            self.console.emit(f"📂 NEED_FILE satisfăcut: {rel}")
            return path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None

    # ----- un fișier: Qwen scrie -> salvare -> validare -> Gemini verifică -- #
    def _produce_file(self, step: dict) -> bool:
        rel = step["path"]
        target = safe_join(self.project_dir, rel)
        architect = self.session.architect
        assert architect is not None
        previous = target.read_text(encoding="utf-8", errors="replace") if target.is_file() else None
        last_code = previous
        feedback: Optional[str] = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._check()
            if attempt == 1:
                self.agent_log.emit(f"📝 Sarcină pentru Qwen → {rel}\n{step['requirements']}", "task")
            else:
                self.agent_log.emit(f"🔁 Refacere {rel} (încercarea {attempt}/{MAX_ATTEMPTS}) cu corecțiile lui Gemini.", "task")

            title = f"{rel} — încercarea {attempt}/{MAX_ATTEMPTS}"
            self.code_set.emit(title, "")
            deps = step.get("depends_on", [])
            pmap = build_project_map(self.project_dir, exclude={rel, *deps})
            code = self.qwen.write_file(
                rel, step["description"], step["requirements"], self.project_type,
                project_files=list_project_files(self.project_dir),
                context=self._context_for(rel, deps),
                project_map=pmap,
                on_need_file=self._read_requested_file,
                previous_code=last_code, feedback=feedback,
                on_token=self._on_token, cancel=self.cancel,
            )
            self._flush()
            self.code_set.emit(title, code)
            if self.qwen.last_truncated:
                self.console.emit("⚠ Răspunsul lui Qwen a atins limita QWEN_NUM_PREDICT și poate fi incomplet.")

            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            self.console.emit(f"💾 Salvat: {rel} ({len(code.splitlines())} linii)")

            result = validate_file(self.project_type, self.project_dir, rel,
                                   self.session.entry_file, self.console.emit, self.cancel)
            self.console.emit(f"{'✔' if result.ok else '✘'} {result.summary}")

            self._check()
            self.agent_log.emit(f"🔍 Gemini face review la {rel}…", "info")
            verdict, notes = architect.review(
                rel, step["description"], step["requirements"], code, result.as_log(4000),
                self.project_type, attempt, self.cancel)
            if verdict == "OK" and not result.ok:
                verdict = "REFA"
                notes = f"Verificarea automată a eșuat, deci fișierul nu poate fi acceptat.\n{result.as_log(3000)}"
            self.agent_log.emit(f"Verdict {rel}: {verdict}" + (f"\n{notes}" if notes else ""),
                                "ok" if verdict == "OK" else "warn")
            if verdict == "OK":
                return True
            feedback = notes or result.as_log(3000)
            last_code = code

        self.agent_log.emit(f"⚠ {rel}: limita de {MAX_ATTEMPTS} încercări a fost atinsă; păstrez ultima versiune.", "err")
        return False

    def _save_progress(self, steps: list[dict], completed: list[str], done: bool = False) -> None:
        try:
            save_state(self.project_dir, {
                "goal": self.text,
                "project_type": self.project_type,
                "entry_file": self.session.entry_file,
                "steps": steps,
                "completed": completed,
                "done": done,
            })
        except OSError as exc:
            self.console.emit(f"⚠ Nu am putut salva checkpoint-ul: {exc}")

    def _run_steps(self, steps: list[dict], completed: Optional[list[str]] = None,
                   save_checkpoint: bool = False) -> list[str]:
        failed: list[str] = []
        done_list = list(completed or [])
        for index, step in enumerate(steps, 1):
            self._check()
            if step["path"] in done_list:
                self.agent_log.emit(f"⏭ Pasul {index}/{len(steps)}: {step['path']} (deja terminat)", "info")
                continue
            self.agent_log.emit(f"━━ Pasul {index}/{len(steps)}: {step['path']} ━━", "plan")
            if self._produce_file(step):
                done_list.append(step["path"])
                if save_checkpoint:
                    self._save_progress(steps, done_list, done=False)
            else:
                failed.append(step["path"])
        return failed

    def _maybe_push(self, commit_message: str) -> str:
        if not (self.auto_push and self.project_type == "Python"):
            return ""
        self._check()
        self.console.emit("🚀 Upload pe GitHub…")
        result = push_project(self.project_dir, self.settings, self.console.emit, self.cancel, commit_message)
        self.console.emit(f"{'✔' if result.ok else '✘'} {result.summary}")
        return f" GitHub: {result.summary}"

    def _final_audit(self) -> list[str]:
        """Rulează auditul global Gemini și aplică automat eventualele corecții de integrare."""
        architect = self.session.architect
        assert architect is not None
        self._check()
        files = list_project_files(self.project_dir)
        self.agent_log.emit("🔎 Gemini face auditul final pe tot proiectul…", "info")
        audit = architect.audit(self.text, self.project_type, files,
                                self._read_files(files, 80000), self.cancel)
        self.chat_message.emit("Gemini (Audit)", audit["summary"])
        tasks = audit["tasks"]
        if not tasks:
            self.agent_log.emit(f"✔ Audit final OK: {audit['summary']}", "ok")
            return []
        self.agent_log.emit(
            "🛠 Auditul a găsit corecții de integrare:\n"
            + "\n".join(f"• {t['path']} — {t['description']}" for t in tasks),
            "warn",
        )
        return self._run_steps(tasks)

    # ----- moduri ---------------------------------------------------------- #
    def _build(self) -> tuple[bool, str]:
        architect = self.session.architect
        assert architect is not None

        if "android" in self.project_type.lower():
            proj_path = Path(self.project_dir)
            prepare_android_project(proj_path, proj_path.name, log=self.console.emit)
            open_android_studio(proj_path, log=self.console.emit)

        files = list_project_files(self.project_dir)
        self.agent_log.emit("📐 Gemini planifică proiectul…", "info")
        plan = architect.plan(self.text, self.project_type, files,
                              self._read_files(files, 60000), self.cancel)
        self.session.entry_file = plan["entry_file"]
        self.chat_message.emit("Gemini", plan["message"])

        lines = [f"Plan: {plan['message']}", f"Fișier de intrare: {plan['entry_file'] or '—'}"]
        lines += [f"{i}. {s['path']} — {s['description']}" for i, s in enumerate(plan["files"], 1)]
        self.agent_log.emit("\n".join(lines), "plan")

        self._save_progress(plan["files"], [], done=False)
        failed = self._run_steps(plan["files"], completed=[], save_checkpoint=True)
        if not failed:
            failed.extend(self._final_audit())
        if "android" in self.project_type.lower():
            proj_path = Path(self.project_dir)
            pkg = f"com.autodev.{re.sub(r'[^A-Za-z0-9]', '', proj_path.name).lower() or 'autodevapp'}"
            ok_build, build_out = run_gradle_build(proj_path, log=self.console.emit)
            self.console.emit("✔ Gradle build reușit!" if ok_build else f"✘ Gradle build eșuat:\n{build_out}")
            if ok_build:
                ok_adb, shot, msg = deploy_and_capture_android(proj_path, pkg, log=self.console.emit)
                if ok_adb and shot:
                    self.agent_log.emit("👁 Gemini Vision analizează captura de pe telefon…", "info")
                    verdict, notes = architect.review_screenshot(shot, self.text, self.cancel)
                    self.agent_log.emit(f"Verdict UI telefon: {verdict}\n{notes}",
                                        "ok" if verdict == "OK" else "warn")
                    self.chat_message.emit("Gemini (Vision)", f"Verdict UI: {verdict}\n{notes}")
                else:
                    self.console.emit(f"⚠ ADB: {msg}")
        self.session.built = True
        done_paths = [s["path"] for s in plan["files"] if s["path"] not in failed]
        self._save_progress(plan["files"], done_paths, done=(not failed))
        total = len(plan["files"])
        summary = f"Proiect generat: {total - len(failed)}/{total} fișiere acceptate."
        if failed:
            summary += " Neacceptate: " + ", ".join(failed) + "."
        summary += self._maybe_push("AutoDev Studio: proiect generat")
        return not failed, summary

    def _resume(self) -> tuple[bool, str]:
        architect = self.session.architect
        assert architect is not None
        state = load_state(self.project_dir)
        if not state or not state.get("steps"):
            return False, "Nu există un checkpoint valid de reluat."

        self.text = state.get("goal", "")
        self.session.entry_file = state.get("entry_file", "")
        steps = state["steps"]
        completed = list(state.get("completed", []))
        if self.text and not architect.history:
            architect._remember(self.text, "Plan restaurat din checkpoint.")

        self.agent_log.emit(
            f"▶ Reluare din checkpoint: {len(completed)}/{len(steps)} fișiere deja terminate.", "plan"
        )
        failed = self._run_steps(steps, completed=completed, save_checkpoint=True)
        if not failed:
            failed.extend(self._final_audit())

        if "android" in self.project_type.lower():
            proj_path = Path(self.project_dir)
            pkg = f"com.autodev.{re.sub(r'[^A-Za-z0-9]', '', proj_path.name).lower() or 'autodevapp'}"
            ok_build, build_out = run_gradle_build(proj_path, log=self.console.emit)
            self.console.emit("✔ Gradle build reușit!" if ok_build else f"✘ Gradle build eșuat:\n{build_out}")
            if ok_build:
                ok_adb, shot, msg = deploy_and_capture_android(proj_path, pkg, log=self.console.emit)
                if ok_adb and shot:
                    self.agent_log.emit("👁 Gemini Vision analizează captura de pe telefon…", "info")
                    verdict, notes = architect.review_screenshot(shot, self.text, self.cancel)
                    self.agent_log.emit(f"Verdict UI telefon: {verdict}\n{notes}",
                                        "ok" if verdict == "OK" else "warn")
                    self.chat_message.emit("Gemini (Vision)", f"Verdict UI: {verdict}\n{notes}")
                else:
                    self.console.emit(f"⚠ ADB: {msg}")

        self.session.built = True
        done_paths = [s["path"] for s in steps if s["path"] not in failed]
        self._save_progress(steps, done_paths, done=(not failed))
        total = len(steps)
        summary = f"Proiect reluat și finalizat: {total - len(failed)}/{total} fișiere acceptate."
        if failed:
            summary += " Neacceptate: " + ", ".join(failed) + "."
        summary += self._maybe_push("AutoDev Studio: proiect finalizat din checkpoint")
        return not failed, summary

    def _feedback(self) -> tuple[bool, str]:
        architect = self.session.architect
        assert architect is not None
        files = list_project_files(self.project_dir)
        self.agent_log.emit("💬 Gemini analizează feedback-ul…", "info")
        result = architect.chat(self.text, self.project_type, files,
                                self._read_files(files, 60000), self.cancel)
        self.chat_message.emit("Gemini", result["reply"])
        if result["entry_file"]:
            self.session.entry_file = result["entry_file"]
        tasks = result["tasks"]
        if not tasks:
            self.agent_log.emit("Gemini nu cere modificări de cod.", "info")
            return True, "Fără modificări de cod."
        self.agent_log.emit("Modificări planificate:\n" + "\n".join(f"• {t['path']} — {t['description']}" for t in tasks), "plan")
        failed = self._run_steps(tasks)
        summary = f"Modificări aplicate: {len(tasks) - len(failed)}/{len(tasks)} fișiere acceptate."
        if failed:
            summary += " Neacceptate: " + ", ".join(failed) + "."
        summary += self._maybe_push("AutoDev Studio: modificări după feedback")
        return not failed, summary

# --------------------------------------------------------------------------- #
# Worker + Dialog Chestionar de clarificare
# --------------------------------------------------------------------------- #
class ClarifyWorker(QThread):
    ready = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, architect: GeminiArchitect, requirements: str, project_type: str,
                 cancel: threading.Event) -> None:
        super().__init__()
        self.architect = architect
        self.requirements = requirements
        self.project_type = project_type
        self.cancel = cancel

    def run(self) -> None:
        try:
            data = self.architect.clarify(self.requirements, self.project_type, self.cancel)
            self.ready.emit(data)
        except Cancelled:
            self.failed.emit("Chestionar anulat.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))

class ExportWorker(QThread):
    console = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, project_type: str, project_dir: str, entry_file: str,
                 cancel: threading.Event) -> None:
        super().__init__()
        self.project_type = project_type
        self.project_dir = project_dir
        self.entry_file = entry_file
        self.cancel = cancel

    def run(self) -> None:
        try:
            res = export_windows_exe(
                self.project_type, self.project_dir, self.entry_file,
                self.console.emit, self.cancel,
            )
            self.done.emit(res.ok, res.summary)
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, f"Eroare la export: {exc}")

class QuestionnaireDialog(QDialog):
    def __init__(self, data: dict, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chestionar de clarificare — Gemini")
        self.resize(760, 680)
        self._entries: list[tuple[str, bool, list, QLineEdit]] = []
        self._groups: list[QButtonGroup] = []

        root = QVBoxLayout(self)
        intro = QLabel(data.get("intro", ""))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #8ab4f8; font-weight: bold; padding: 4px 0 8px 0;")
        root.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        form_layout = QVBoxLayout(container)
        form_layout.setSpacing(12)

        for idx, q in enumerate(data.get("questions", []), 1):
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background-color: #25272a; border: 1px solid #3c3f41; border-radius: 8px; padding: 8px; }"
            )
            c_layout = QVBoxLayout(card)
            c_layout.setSpacing(6)

            q_label = QLabel(f"{idx}. {q['question']}")
            q_label.setWordWrap(True)
            q_label.setStyleSheet("font-weight: bold; color: #dfe1e5; border: none; padding: 0;")
            c_layout.addWidget(q_label)

            multiple = bool(q.get("multiple", False))
            widgets: list = []
            if multiple:
                for opt in q["options"]:
                    cb = QCheckBox(opt)
                    cb.setStyleSheet("border: none;")
                    c_layout.addWidget(cb)
                    widgets.append(cb)
            else:
                group = QButtonGroup(self)
                self._groups.append(group)
                for opt in q["options"]:
                    rb = QRadioButton(opt)
                    rb.setStyleSheet("border: none;")
                    group.addButton(rb)
                    c_layout.addWidget(rb)
                    widgets.append(rb)

            custom = QLineEdit()
            custom.setPlaceholderText("Altă variantă sau detalii suplimentare (opțional)…")
            c_layout.addWidget(custom)

            self._entries.append((q["question"], multiple, widgets, custom))
            form_layout.addWidget(card)

        form_layout.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        ok_btn = buttons.addButton("Generează planul", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_btn = buttons.addButton("Anulează", QDialogButtonBox.ButtonRole.RejectRole)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        root.addWidget(buttons)

    def format_answers(self) -> str:
        lines: list[str] = []
        for idx, (question, _multiple, widgets, custom) in enumerate(self._entries, 1):
            chosen = [w.text() for w in widgets if w.isChecked()]
            extra = custom.text().strip()
            if extra:
                chosen.append(f"Altă variantă: {extra}")
            if chosen:
                lines.append(f"{idx}. {question}\n   Răspuns: {'; '.join(chosen)}")
        if not lines:
            return ""
        return "\n\n--- Răspunsuri la chestionarul de clarificare ---\n" + "\n".join(lines)

# --------------------------------------------------------------------------- #
# Dialog Setări
# --------------------------------------------------------------------------- #
class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("Setări")
        self.setMinimumWidth(460)

        self.gemini_edit = QLineEdit(settings.gemini_api_key)
        self.token_edit = QLineEdit(settings.github_token)
        self.user_edit = QLineEdit(settings.github_username)
        self.private_check = QCheckBox("Repo privat pe GitHub")
        self.private_check.setChecked(settings.github_private)
        self.show_check = QCheckBox("Arată cheile")
        for edit in (self.gemini_edit, self.token_edit):
            edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_check.toggled.connect(self._toggle_echo)

        form = QFormLayout()
        form.addRow("Gemini API key:", self.gemini_edit)
        form.addRow("GitHub token:", self.token_edit)
        form.addRow("GitHub username:", self.user_edit)
        form.addRow("", self.private_check)
        form.addRow("", self.show_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Se salvează în fișierul .env (permisiuni 600)."))
        layout.addWidget(buttons)

    def _toggle_echo(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self.gemini_edit.setEchoMode(mode)
        self.token_edit.setEchoMode(mode)

    def accept(self) -> None:
        self.settings.gemini_api_key = self.gemini_edit.text().strip()
        self.settings.github_token = self.token_edit.text().strip()
        self.settings.github_username = self.user_edit.text().strip()
        self.settings.github_private = self.private_check.isChecked()
        try:
            self.settings.save()
        except OSError as exc:
            QMessageBox.critical(self, "Eroare", f"Nu am putut salva .env: {exc}")
            return
        super().accept()


# --------------------------------------------------------------------------- #
# Fereastra principală
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AutoDev Studio")
        self.resize(1600, 900)
        self.settings = Settings.load()
        self.qsettings = QSettings("AutoDevStudio", "AutoDevStudio")
        self.session = Session()
        self.worker: Optional[PipelineWorker] = None
        self.cancel = threading.Event()
        self.attached_files: list[str] = []
        self.remote_api = RemoteAPIServer(port=8765, parent=self)
        self.remote_api.stop_requested.connect(self._stop)
        self.remote_api.resume_requested.connect(self._resume)
        self.remote_api.feedback_requested.connect(self._on_remote_feedback)
        self._build_ui()
        self._restore()
        self.clarify_worker: Optional[ClarifyWorker] = None
        self._pending_build: Optional[tuple[str, str, str, str]] = None

    # ----- UI -------------------------------------------------------------- #
    def _panel(self, title: str, body: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        label = QLabel(title)
        label.setObjectName("panelTitle")
        layout.addWidget(label)
        layout.addWidget(body, 1)
        return box

    def _log_widget(self, mono: bool = False) -> QTextEdit:
        widget = QTextEdit()
        widget.setReadOnly(True)
        widget.document().setMaximumBlockCount(6000)
        if mono:
            widget.setObjectName("mono")
        return widget

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # Bara de sus
        top = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(PROJECT_TYPES)
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        self.dir_edit.setPlaceholderText("Alege folderul proiectului…")
        self.dir_button = QPushButton("Alege folder…")
        self.push_check = QCheckBox("Auto GitHub Push (Python)")
        self.settings_button = QPushButton("Setări")
        self.resume_button = QPushButton("▶ Continuă")
        self.resume_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopBtn")
        self.stop_button.setEnabled(False)
        top.addWidget(QLabel("Tip proiect:"))
        top.addWidget(self.type_combo)
        top.addSpacing(12)
        top.addWidget(QLabel("Folder:"))
        top.addWidget(self.dir_edit, 1)
        top.addWidget(self.dir_button)
        top.addSpacing(12)
        top.addWidget(self.push_check)
        top.addWidget(self.settings_button)
        self.export_button = QPushButton("📦 Export .exe")
        self.export_button.setToolTip("Generează executabil Windows (.exe) pentru C++, Java sau Python")
        top.addWidget(self.export_button)
        top.addWidget(self.resume_button)
        top.addWidget(self.stop_button)
        root.addLayout(top)
        # Bara 2: Modele AI, thread-uri și temperatură CPU
        bar2 = QHBoxLayout()
        self.ollama_combo = QComboBox()
        self.ollama_combo.setMinimumWidth(220)
        self.refresh_ollama_btn = QPushButton("↻")
        self.refresh_ollama_btn.setToolTip("Reîncarcă lista de modele din Ollama")
        self.refresh_ollama_btn.setFixedWidth(34)

        self.gemini_combo = QComboBox()
        self.gemini_combo.addItems(GEMINI_MODELS)
        self.gemini_combo.setMinimumWidth(180)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 32)
        self.threads_spin.setSpecialValueText("Auto")
        self.threads_spin.setToolTip("Thread-uri CPU folosite de Ollama (0 = Auto, 4-6 = mai rece)")

        self.temp_label = QLabel("🌡️ --°C")
        self.temp_label.setToolTip("Temperatura CPU în timp real (k10temp)")

        self.api_check = QCheckBox("📡 API :8765")
        self.api_check.setToolTip("Pornește serverul API pentru telecomanda Android (Tailscale)")
        bar2.addSpacing(14)
        bar2.addWidget(self.api_check)
        bar2.addStretch(1)
        bar2.addWidget(self.temp_label)

        bar2.addWidget(QLabel("Model local (Ollama):"))
        bar2.addWidget(self.ollama_combo)
        bar2.addWidget(self.refresh_ollama_btn)
        bar2.addSpacing(14)
        bar2.addWidget(QLabel("Model Gemini:"))
        bar2.addWidget(self.gemini_combo)
        bar2.addSpacing(14)
        bar2.addWidget(QLabel("Thread-uri:"))
        bar2.addWidget(self.threads_spin)
        bar2.addStretch(1)
        bar2.addWidget(self.temp_label)
        root.addLayout(bar2)

        # Panoul 1: Tu <-> Gemini
        self.chat = self._log_widget()
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("Scrie cerințele sau feedback-ul… (Ctrl+Enter trimite)")
        self.input.setFixedHeight(110)

        attach_bar = QHBoxLayout()
        self.attach_button = QPushButton("📎 Atașează fișiere…")
        self.attach_label = QLabel("Niciun fișier atașat")
        self.attach_label.setStyleSheet("color: #9aa7b8; font-size: 9.5pt;")
        self.clear_attach_btn = QPushButton("✕")
        self.clear_attach_btn.setToolTip("Șterge fișierele atașate")
        self.clear_attach_btn.setFixedWidth(30)
        self.clear_attach_btn.setEnabled(False)
        attach_bar.addWidget(self.attach_button)
        attach_bar.addWidget(self.attach_label, 1)
        attach_bar.addWidget(self.clear_attach_btn)

        self.clarify_check = QCheckBox("📋 Chestionar de clarificare (înainte de plan)")
        self.clarify_check.setToolTip("Gemini îți pune 8-15 întrebări cu variante înainte să facă planul")
        self.send_button = QPushButton("Trimite către Gemini")

        self.send_button = QPushButton("Trimite către Gemini")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.chat, 1)
        left_layout.addWidget(self.input)
        left_layout.addLayout(attach_bar)
        left_layout.addWidget(self.clarify_check)
        left_layout.addWidget(self.send_button)
        left_layout.addWidget(self.send_button)

        # Panoul 2: Gemini <-> Qwen
        self.agent_view = self._log_widget()

        # Panoul 3: cod (sus) + consolă (jos)
        self.code_title = QLabel("Cod generat de Qwen")
        self.code_view = QPlainTextEdit()
        self.code_view.setObjectName("mono")
        self.code_view.setReadOnly(True)
        self.code_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        code_box = QWidget()
        code_layout = QVBoxLayout(code_box)
        code_layout.setContentsMargins(0, 0, 0, 0)
        code_layout.addWidget(self.code_title)
        code_layout.addWidget(self.code_view, 1)
        self.console_view = self._log_widget(mono=True)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(code_box)
        right.addWidget(self._panel("Consolă (salvare, validare, GitHub)", self.console_view))
        right.setSizes([520, 340])

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._panel("1 · Tu ↔ Gemini", left))
        splitter.addWidget(self._panel("2 · Gemini ↔ Qwen", self.agent_view))
        splitter.addWidget(self._panel("3 · Rezultat", right))
        splitter.setSizes([420, 480, 700])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)

        # Conexiuni
        self.dir_button.clicked.connect(self._choose_dir)
        self.settings_button.clicked.connect(self._open_settings)
        self.resume_button.clicked.connect(self._resume)
        self.stop_button.clicked.connect(self._stop)
        self.send_button.clicked.connect(self._send)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        for key in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(key), self.input).activated.connect(self._send)
        self.refresh_ollama_btn.clicked.connect(self._refresh_ollama_models)
        self.ollama_combo.currentTextChanged.connect(self._persist_choice)
        self.gemini_combo.currentTextChanged.connect(self._persist_choice)
        self.threads_spin.valueChanged.connect(self._persist_choice)
        self.api_check.toggled.connect(self._toggle_api)

        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self._update_temp)
        self.temp_timer.start(3000)
        self._update_temp()
        self.attach_button.clicked.connect(self._attach_files)
        self.clear_attach_btn.clicked.connect(self._clear_attachments)
        self.export_button.clicked.connect(self._export_exe)

    def _restore(self) -> None:
        self._refresh_ollama_models()
        self.gemini_combo.setCurrentText(self.settings.gemini_model)
        self.threads_spin.setValue(self.settings.ollama_threads)
        self.type_combo.setCurrentText(self.settings.last_type)
        if self.settings.last_dir and Path(self.settings.last_dir).is_dir():
            self.dir_edit.setText(self.settings.last_dir)
        self.push_check.setChecked(self.qsettings.value("auto_push", False, type=bool))
        self.push_check.toggled.connect(lambda v: self.qsettings.setValue("auto_push", v))
        self._on_type_changed(self.type_combo.currentText())
        self._chat_add("Sistem", "Alege tipul și folderul proiectului, apoi descrie ce vrei să construim.")
        self._update_resume_button()

    # ----- afișare --------------------------------------------------------- #
    @staticmethod
    def _append(widget: QTextEdit, text: str, color: str) -> None:
        safe = html.escape(text)
        widget.append(f'<span style="color:{color}; white-space:pre-wrap">{safe}</span>')

    def _chat_add(self, author: str, text: str) -> None:
        key = {"Tu": "user", "Gemini": "gemini"}.get(author, "sys")
        self.chat.append(f'<span style="color:{COLORS[key]}"><b>{html.escape(author)}</b></span>')
        self._append(self.chat, text, "#dfe1e5")
        self.chat.append("")

    def _on_agent_log(self, text: str, kind: str) -> None:
        self._append(self.agent_view, text, COLORS.get(kind, COLORS["info"]))
        self.agent_view.append("")
        self.remote_api.push_agent_log(text)

    def _on_console(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            if line.startswith("$ "):
                color = "#56b6c2"
            elif line.startswith(("✔", "[OK]")):
                color = COLORS["ok"]
            elif line.startswith(("✘", "[EȘEC]", "Traceback")) or "Error" in line[:40]:
                color = COLORS["err"]
            elif line.startswith(("⚠", "⛔")):
                color = COLORS["warn"]
            elif line.startswith(("▶", "🚀", "💾")):
                color = COLORS["user"]
            else:
                color = "#c9ccd1"
            self._append(self.console_view, line, color)
            self.remote_api.push_console_log(text)

    def _on_code_set(self, title: str, text: str) -> None:
        self.code_title.setText(f"Cod generat de Qwen — {title}")
        self.code_view.setPlainText(text)
        self.code_view.moveCursor(QTextCursor.MoveOperation.End)
        self.remote_api.update_status(current_file=title)

    def _on_code_chunk(self, chunk: str) -> None:
        self.code_view.moveCursor(QTextCursor.MoveOperation.End)
        self.code_view.insertPlainText(chunk)
        self.code_view.ensureCursorVisible()

    # ----- acțiuni --------------------------------------------------------- #
    def _choose_dir(self) -> None:
        start = self.dir_edit.text() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Alege folderul proiectului", start)
        if chosen:
            self.dir_edit.setText(chosen)
            self._persist_choice()
            self._update_resume_button()

    def _on_type_changed(self, project_type: str) -> None:
        self.push_check.setEnabled(project_type == "Python")
        self.export_button.setEnabled(project_type in {"Python", "C++", "Java"})
        self._persist_choice()

    def _refresh_ollama_models(self) -> None:
        current = self.ollama_combo.currentText() or self.settings.ollama_model
        models = list_ollama_models()
        if not models:
            models = [current]
        elif current and current not in models:
            models.insert(0, current)
        self.ollama_combo.blockSignals(True)
        self.ollama_combo.clear()
        self.ollama_combo.addItems(models)
        self.ollama_combo.setCurrentText(current)
        self.ollama_combo.blockSignals(False)

    def _update_temp(self) -> None:
        temp = read_cpu_temp()
        if temp is None:
            self.temp_label.setText("🌡️ N/A")
            return
        if temp >= 85:
            color = COLORS["err"]
        elif temp >= 75:
            color = COLORS["warn"]
        else:
            color = COLORS["ok"]
        self.temp_label.setText(f'<span style="color:{color}; font-weight:bold;">🌡️ {temp:.1f}°C</span>')

    def _persist_choice(self) -> None:
        self.settings.last_dir = self.dir_edit.text().strip()
        self.settings.last_type = self.type_combo.currentText()
        if self.ollama_combo.currentText():
            self.settings.ollama_model = self.ollama_combo.currentText()
        if self.gemini_combo.currentText():
            self.settings.gemini_model = self.gemini_combo.currentText()
        self.settings.ollama_threads = self.threads_spin.value()
        try:
            self.settings.save()
        except OSError:
            pass
        self._sync_api_status()

    def _open_settings(self) -> None:
        SettingsDialog(self.settings, self).exec()

    def _stop(self) -> None:
        self.cancel.set()
        self.stop_button.setEnabled(False)
        self._on_console("⛔ Stop cerut… aștept oprirea proceselor.")

    def _set_running(self, running: bool) -> None:
        self.send_button.setEnabled(not running)
        self.type_combo.setEnabled(not running)
        self.dir_button.setEnabled(not running)
        self.settings_button.setEnabled(not running)
        self.ollama_combo.setEnabled(not running)
        self.refresh_ollama_btn.setEnabled(not running)
        self.gemini_combo.setEnabled(not running)
        self.threads_spin.setEnabled(not running)
        self.attach_button.setEnabled(not running)
        self.clear_attach_btn.setEnabled((not running) and bool(self.attached_files))
        self.clarify_check.setEnabled(not running)
        self.export_button.setEnabled((not running) and self.type_combo.currentText() in {"Python", "C++", "Java"})
        self.stop_button.setEnabled(running)
        if running:
            self.resume_button.setEnabled(False)
        else:
            self._update_resume_button()
        self._sync_api_status()

    def _ensure_session(self, project_dir: str, project_type: str) -> bool:
        key = self.settings.gemini_api_key
        gmodel = self.settings.gemini_model
        s = self.session
        if (s.architect is not None and s.project_dir == project_dir
                and s.project_type == project_type and s.api_key == key
                and s.gemini_model == gmodel):
            return True
        try:
            architect = GeminiArchitect(key, model=gmodel)
        except AgentError as exc:
            QMessageBox.warning(self, "Gemini", str(exc))
            return False
        self.session = Session(architect=architect, api_key=key, gemini_model=gmodel,
                               project_dir=project_dir, project_type=project_type)
        return True

    def _attach_files(self) -> None:
        start = self.dir_edit.text() or str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Alege fișiere de referință pentru Gemini",
            start,
            "Fișiere text/cod și ZIP (*.py *.kt *.java *.xml *.json *.md *.txt *.gradle *.kts *.yml *.yaml *.ini *.cfg *.pro *.zip);;Toate fișierele (*)",
        )
        if not paths:
            return
        for p in paths:
            if p not in self.attached_files:
                self.attached_files.append(p)
        self._update_attach_ui()

    def _clear_attachments(self) -> None:
        self.attached_files.clear()
        self._update_attach_ui()

    def _update_attach_ui(self) -> None:
        if not self.attached_files:
            self.attach_label.setText("Niciun fișier atașat")
            self.clear_attach_btn.setEnabled(False)
            return
        names = [Path(p).name for p in self.attached_files]
        preview = ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else "")
        self.attach_label.setText(f"{len(names)} atașate: {preview}")
        self.clear_attach_btn.setEnabled(True)

    def _format_attachments(self) -> str:
        if not self.attached_files:
            return ""
        blocks: list[str] = []
        total_chars = 0
        max_total = 150000  # limită totală de siguranță (~40k tokeni)
        skip_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", "build", ".gradle", ".idea"}

        for raw_path in self.attached_files:
            if total_chars >= max_total:
                break
            p = Path(raw_path)
            if p.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(p, "r") as zf:
                        for info in zf.infolist():
                            if info.is_dir() or total_chars >= max_total:
                                continue
                            inner = Path(info.filename)
                            if any(part in skip_parts for part in inner.parts):
                                continue
                            if inner.suffix.lower() not in TEXT_SUFFIXES or info.file_size > 100000:
                                continue
                            content = zf.read(info).decode("utf-8", errors="replace")
                            if len(content) > 25000:
                                content = content[:25000] + "\n… [tăiat]"
                            blocks.append(f"### {p.name} :: {info.filename}\n```\n{content}\n```")
                            total_chars += len(content)
                except (OSError, zipfile.BadZipFile) as exc:
                    self._on_console(f"⚠ Nu am putut citi arhiva {p.name}: {exc}")
            else:
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 25000:
                        content = content[:25000] + "\n… [tăiat]"
                    blocks.append(f"### {p.name}\n```\n{content}\n```")
                    total_chars += len(content)
                except OSError as exc:
                    self._on_console(f"⚠ Nu am putut citi fișierul atașat {p.name}: {exc}")

        if not blocks:
            return ""
        return "\n\n--- Fișiere atașate de utilizator ca referință ---\n" + "\n\n".join(blocks)

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text:
            return
        if (self.worker is not None and self.worker.isRunning()) or (
            self.clarify_worker is not None and self.clarify_worker.isRunning()
        ):
            return
        if not self.settings.gemini_api_key:
            QMessageBox.information(self, "Setări", "Adaugă mai întâi cheia Gemini API.")
            self._open_settings()
            if not self.settings.gemini_api_key:
                return
        project_dir = self.dir_edit.text().strip()
        if not project_dir or not Path(project_dir).is_dir():
            QMessageBox.warning(self, "Folder", "Alege un folder de proiect valid.")
            return
        project_type = self.type_combo.currentText()
        if not self._ensure_session(project_dir, project_type):
            return

        mode = "feedback" if self.session.built else "build"
        attached_names = [Path(p).name for p in self.attached_files]
        attachments_block = self._format_attachments()
        full_prompt = text + attachments_block
        display_msg = text + (f"\n\n📎 Atașate: {', '.join(attached_names)}" if attached_names else "")

        if mode == "build" and self.clarify_check.isChecked():
            assert self.session.architect is not None
            self._pending_build = (text, full_prompt, display_msg, project_type)
            self.input.clear()
            self._clear_attachments()
            self._chat_add("Tu", display_msg)
            self._on_agent_log("📋 Gemini pregătește chestionarul de clarificare…", "info")
            self.cancel = threading.Event()
            cw = ClarifyWorker(self.session.architect, full_prompt, project_type, self.cancel)
            cw.ready.connect(self._on_clarify_ready)
            cw.failed.connect(self._on_clarify_failed)
            self.clarify_worker = cw
            self._set_running(True)
            cw.start()
            return

        self.input.clear()
        self._clear_attachments()
        self._chat_add("Tu", display_msg)
        self._start_pipeline(mode, full_prompt, project_type)

    def _on_clarify_ready(self, data: dict) -> None:
        self._set_running(False)
        pending = self._pending_build
        self._pending_build = None
        if not pending:
            return
        raw_text, full_prompt, _display_msg, project_type = pending
        dialog = QuestionnaireDialog(data, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            answers = dialog.format_answers()
            if answers:
                self._chat_add("Tu (Chestionar)", answers.replace("\n\n--- Răspunsuri la chestionarul de clarificare ---\n", ""))
            self._start_pipeline("build", full_prompt + answers, project_type)
        else:
            self._chat_add("Sistem", "Chestionar anulat. Îți poți edita cerința și trimite din nou.")
            self.input.setPlainText(raw_text)

    def _on_clarify_failed(self, msg: str) -> None:
        self._set_running(False)
        pending = self._pending_build
        self._pending_build = None
        self._on_agent_log(f"✘ Chestionar: {msg}", "err")
        self._chat_add("Sistem", f"⚠ {msg}")
        if pending:
            self.input.setPlainText(pending[0])

    def _start_pipeline(self, mode: str, full_prompt: str, project_type: str) -> None:
        self.cancel = threading.Event()
        worker = PipelineWorker(self.settings, self.session, mode, full_prompt,
                                self.push_check.isChecked() and project_type == "Python", self.cancel)
        worker.chat_message.connect(self._chat_add)
        worker.agent_log.connect(self._on_agent_log)
        worker.code_set.connect(self._on_code_set)
        worker.code_chunk.connect(self._on_code_chunk)
        worker.console.connect(self._on_console)
        worker.finished_run.connect(self._on_finished)
        self.worker = worker
        self._set_running(True)
        self._on_agent_log(f"━━ {'Proiect nou' if mode == 'build' else 'Feedback'} ━━", "plan")
        worker.start()

    def _on_finished(self, ok: bool, summary: str) -> None:
        self._set_running(False)
        self._chat_add("Sistem", ("✅ " if ok else "⚠ ") + summary)
        self._update_resume_button()

    def _update_resume_button(self) -> None:
        running = self.worker is not None and self.worker.isRunning()
        state = load_state(self.dir_edit.text().strip())
        if not running and state and not state.get("done") and state.get("steps"):
            done_cnt = len(state.get("completed", []))
            total_cnt = len(state["steps"])
            self.resume_button.setText(f"▶ Continuă ({done_cnt}/{total_cnt})")
            self.resume_button.setEnabled(True)
        else:
            self.resume_button.setText("▶ Continuă")
            self.resume_button.setEnabled(False)
            self._sync_api_status()

    def _export_exe(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        project_dir = self.dir_edit.text().strip()
        if not project_dir or not Path(project_dir).is_dir():
            QMessageBox.warning(self, "Folder", "Alege un folder de proiect valid.")
            return
        project_type = self.type_combo.currentText()
        self.cancel = threading.Event()
        ew = ExportWorker(project_type, project_dir, self.session.entry_file, self.cancel)
        ew.console.connect(self._on_console)
        ew.done.connect(lambda ok, msg: (self._set_running(False), self._chat_add("Sistem", ("✅ " if ok else "⚠ ") + msg)))
        self._export_worker = ew
        self._set_running(True)
        self._on_console(f"📦 Pornesc exportul Windows .exe ({project_type})…")
        ew.start()

    def _resume(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        if not self.settings.gemini_api_key:
            QMessageBox.information(self, "Setări", "Adaugă mai întâi cheia Gemini API.")
            self._open_settings()
            if not self.settings.gemini_api_key:
                return
        project_dir = self.dir_edit.text().strip()
        state = load_state(project_dir)
        if not state or not state.get("steps"):
            self._update_resume_button()
            return
        project_type = state.get("project_type") or self.type_combo.currentText()
        self.type_combo.setCurrentText(project_type)
        if not self._ensure_session(project_dir, project_type):
            return

        self.cancel = threading.Event()
        worker = PipelineWorker(self.settings, self.session, "resume", "",
                                self.push_check.isChecked() and project_type == "Python", self.cancel)
        worker.chat_message.connect(self._chat_add)
        worker.agent_log.connect(self._on_agent_log)
        worker.code_set.connect(self._on_code_set)
        worker.code_chunk.connect(self._on_code_chunk)
        worker.console.connect(self._on_console)
        worker.finished_run.connect(self._on_finished)
        self.worker = worker
        self._set_running(True)
        self._on_agent_log("━━ Reluare proiect din checkpoint ━━", "plan")
        worker.start()

    def _toggle_api(self, enabled: bool) -> None:
        if enabled:
            if self.remote_api.start():
                self._on_console("📡 Server API pornit pe portul 8765 (accesibil prin Tailscale).")
                self._sync_api_status()
            else:
                self.api_check.setChecked(False)
                self._on_console("✘ Nu am putut porni serverul API pe portul 8765 (port ocupat?).")
        else:
            self.remote_api.stop()
            self._on_console("📡 Server API oprit.")

    def _sync_api_status(self) -> None:
        project_dir = self.dir_edit.text().strip()
        state = load_state(project_dir)
        done_cnt = len(state.get("completed", [])) if state else 0
        total_cnt = len(state.get("steps", [])) if state else 0
        running = (self.worker is not None and self.worker.isRunning())
        self.remote_api.update_status(
            running=running,
            project_dir=project_dir,
            project_type=self.type_combo.currentText(),
            ollama_model=self.ollama_combo.currentText(),
            gemini_model=self.gemini_combo.currentText(),
            can_resume=self.resume_button.isEnabled(),
            progress_done=done_cnt,
            progress_total=total_cnt,
        )

    def _on_remote_feedback(self, message: str) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.input.setPlainText(message)
        self._send()

    def closeEvent(self, event) -> None:  # noqa: N802 - convenția Qt
        if self.worker is not None and self.worker.isRunning():
            self.cancel.set()
            self.worker.wait(4000)
        self.remote_api.stop()
        super().closeEvent(event)


# --------------------------------------------------------------------------- #
# Temă dark
# --------------------------------------------------------------------------- #
STYLESHEET = """
QWidget { background-color: #1e1f22; color: #dfe1e5; font-size: 10.5pt; }
QLabel#panelTitle { color: #8ab4f8; font-weight: bold; padding: 2px 0; }
QTextEdit, QPlainTextEdit, QLineEdit, QComboBox {
    background-color: #2b2d30; border: 1px solid #3c3f41; border-radius: 6px; padding: 4px;
    selection-background-color: #2f65ca;
}
QTextEdit#mono, QPlainTextEdit#mono {
    font-family: "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", monospace; font-size: 10pt;
}
QPushButton { background-color: #3574f0; border: none; border-radius: 6px; padding: 6px 14px; color: white; }
QPushButton:hover { background-color: #4a88ff; }
QPushButton:disabled { background-color: #3c3f41; color: #7a7e85; }
QPushButton#stopBtn { background-color: #c0392b; }
QPushButton#stopBtn:hover { background-color: #e04a3a; }
QPushButton#stopBtn:disabled { background-color: #3c3f41; color: #7a7e85; }
QComboBox QAbstractItemView { background-color: #2b2d30; selection-background-color: #2f65ca; }
QCheckBox::indicator { width: 16px; height: 16px; }
QSplitter::handle { background-color: #3c3f41; }
QToolTip { background-color: #2b2d30; color: #dfe1e5; border: 1px solid #3c3f41; }
"""


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: "#1e1f22", QPalette.ColorRole.WindowText: "#dfe1e5",
        QPalette.ColorRole.Base: "#2b2d30", QPalette.ColorRole.AlternateBase: "#25272a",
        QPalette.ColorRole.Text: "#dfe1e5", QPalette.ColorRole.Button: "#2b2d30",
        QPalette.ColorRole.ButtonText: "#dfe1e5", QPalette.ColorRole.ToolTipBase: "#2b2d30",
        QPalette.ColorRole.ToolTipText: "#dfe1e5", QPalette.ColorRole.Highlight: "#2f65ca",
        QPalette.ColorRole.HighlightedText: "#ffffff", QPalette.ColorRole.PlaceholderText: "#7a7e85",
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


def main() -> None:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
