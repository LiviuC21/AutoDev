"""app.py — AutoDev Studio: interfață PyQt6 (temă dark) care orchestrează Gemini (Arhitect) și Qwen (Muncitor)."""
from __future__ import annotations

import html
import sys
import threading
import time
import traceback
import re
from runner import deploy_and_capture_android, open_android_studio, prepare_android_project, run_gradle_build, validate_file
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPalette, QShortcut, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QSplitter, QTextEdit, QVBoxLayout, QWidget,
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
}
MAX_CONTEXT_FILE_BYTES = 20000

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
            if Path(rel).suffix.lower() not in TEXT_SUFFIXES:
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

    def _run_steps(self, steps: list[dict]) -> list[str]:
        failed: list[str] = []
        for index, step in enumerate(steps, 1):
            self._check()
            self.agent_log.emit(f"━━ Pasul {index}/{len(steps)}: {step['path']} ━━", "plan")
            if not self._produce_file(step):
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

        failed = self._run_steps(plan["files"])
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
        total = len(plan["files"])
        summary = f"Proiect generat: {total - len(failed)}/{total} fișiere acceptate."
        if failed:
            summary += " Neacceptate: " + ", ".join(failed) + "."
        summary += self._maybe_push("AutoDev Studio: proiect generat")
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
        self._build_ui()
        self._restore()

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
        self.send_button = QPushButton("Trimite către Gemini")
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.chat, 1)
        left_layout.addWidget(self.input)
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
        self.stop_button.clicked.connect(self._stop)
        self.send_button.clicked.connect(self._send)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        for key in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(key), self.input).activated.connect(self._send)
        self.refresh_ollama_btn.clicked.connect(self._refresh_ollama_models)
        self.ollama_combo.currentTextChanged.connect(self._persist_choice)
        self.gemini_combo.currentTextChanged.connect(self._persist_choice)
        self.threads_spin.valueChanged.connect(self._persist_choice)

        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self._update_temp)
        self.temp_timer.start(3000)
        self._update_temp()

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

    def _on_code_set(self, title: str, text: str) -> None:
        self.code_title.setText(f"Cod generat de Qwen — {title}")
        self.code_view.setPlainText(text)
        self.code_view.moveCursor(QTextCursor.MoveOperation.End)

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

    def _on_type_changed(self, project_type: str) -> None:
        self.push_check.setEnabled(project_type == "Python")
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
        self.stop_button.setEnabled(running)

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

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or (self.worker is not None and self.worker.isRunning()):
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
        self.input.clear()
        self._chat_add("Tu", text)
        self.cancel = threading.Event()
        worker = PipelineWorker(self.settings, self.session, mode, text,
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

    def closeEvent(self, event) -> None:  # noqa: N802 - convenția Qt
        if self.worker is not None and self.worker.isRunning():
            self.cancel.set()
            self.worker.wait(4000)
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
