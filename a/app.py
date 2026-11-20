import os
import sys
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox,
    QFileDialog, QCheckBox
)

from agents import GeminiArchitect, QwenWorker
from runner import save_file, test_python_file, git_push


class OrchestratorThread(QThread):
    log_user_chat = pyqtSignal(str)
    log_internal = pyqtSignal(str)
    qwen_chunk = pyqtSignal(str)
    qwen_clear = pyqtSignal()
    log_console = pyqtSignal(str)
    finished_run = pyqtSignal()

    def __init__(self, architect, worker, user_prompt, project_type, project_dir, auto_git):
        super().__init__()
        self.architect = architect
        self.worker = worker
        self.user_prompt = user_prompt
        self.project_type = project_type
        self.project_dir = project_dir
        self.auto_git = auto_git

    def run(self):
        try:
            self.log_internal.emit("🧠 [Gemini] Analizez cerința și împart proiectul în pași...")
            plan = self.architect.create_plan(self.user_prompt, self.project_type)
            self.log_internal.emit(
                f"📋 [Gemini] Plan generat ({len(plan)} fișiere):\n"
                + "\n".join(f"  - {s['fisier']}: {s['cerinta'][:80]}..." for s in plan)
            )

            for idx, step in enumerate(plan, 1):
                filename = step["fisier"]
                task = step["cerinta"]
                feedback = ""
                approved = False

                for attempt in range(1, 4):
                    self.log_internal.emit(f"\n➡️ [Gemini → Qwen] Pas {idx}/{len(plan)} (încercarea {attempt}): {filename}")
                    self.qwen_clear.emit()
                    code = self.worker.write_code_stream(
                        filename, task, feedback=feedback,
                        on_chunk=lambda c: self.qwen_chunk.emit(c)
                    )

                    saved_path = save_file(self.project_dir, filename, code)
                    self.log_console.emit(f"💾 Salvat: {saved_path}")

                    console_err = ""
                    if self.project_type == "Python" and filename.endswith(".py"):
                        self.log_console.emit(f"▶️ Testez {filename} în terminal...")
                        ok_run, out = test_python_file(self.project_dir, filename)
                        self.log_console.emit(out)
                        if not ok_run:
                            console_err = out

                    self.log_internal.emit(f"⬅️ [Qwen → Gemini] Am trimis `{filename}` la verificare...")
                    is_ok, verdict = self.architect.review_code(filename, task, code, console_err)
                    self.log_internal.emit(f"🔍 [Gemini Verdict]: {verdict}")

                    if is_ok and not console_err:
                        approved = True
                        break
                    feedback = verdict + (f"\nEroare consolă: {console_err}" if console_err else "")

                if not approved:
                    self.log_internal.emit(f"⚠️ `{filename}` a atins limita de 3 încercări. Trec la următorul pas.")

            if self.project_type == "Python" and self.auto_git:
                self.log_console.emit("🚀 Fac push pe GitHub...")
                self.log_console.emit(git_push(self.project_dir, f"AutoDev: {self.user_prompt[:40]}"))

            msg = (
                "✅ Gata! Poți testa proiectul în Android Studio."
                if self.project_type == "Android"
                else "✅ Proiectul Python este gata și verificat!"
            )
            self.log_user_chat.emit(f"<b>Gemini:</b> {msg} Spune-mi dacă îți place sau ce vrei să mai schimbăm.")
        except Exception as e:
            self.log_console.emit(f"❌ EROARE: {e}")
            self.log_user_chat.emit(f"<b>Eroare:</b> {e}")
        finally:
            self.finished_run.emit()


class AutoDevWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDev Studio — Gemini (Arhitect) + Qwen (Muncitor)")
        self.resize(1400, 820)

        api_key = os.getenv("GEMINI_API_KEY", "")
        self.architect = GeminiArchitect(api_key=api_key) if api_key else None
        self.worker = QwenWorker()
        self.thread = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Bara de sus: setări proiect
        top_bar = QHBoxLayout()
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Python", "Android"])
        self.dir_input = QLineEdit(os.path.expanduser("~/Proiecte/TestAutoDev"))
        btn_browse = QPushButton("Alege folder...")
        btn_browse.clicked.connect(self.pick_folder)
        self.git_check = QCheckBox("Auto GitHub Push (Python)")
        top_bar.addWidget(QLabel("Tip:"))
        top_bar.addWidget(self.type_combo)
        top_bar.addWidget(QLabel("Folder:"))
        top_bar.addWidget(self.dir_input)
        top_bar.addWidget(btn_browse)
        top_bar.addWidget(self.git_check)
        main_layout.addLayout(top_bar)

        # Cele 3 panouri principale
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panoul 1: Tu <-> Gemini
        p1 = QWidget()
        p1_lay = QVBoxLayout(p1)
        p1_lay.addWidget(QLabel("<b>1. Chat cu Gemini (Tu ↔ Arhitect)</b>"))
        self.user_chat = QTextEdit()
        self.user_chat.setReadOnly(True)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Cere aplicația sau modificările aici...")
        self.prompt_input.returnPressed.connect(self.start_task)
        self.btn_send = QPushButton("Trimite la Gemini")
        self.btn_send.clicked.connect(self.start_task)
        p1_lay.addWidget(self.user_chat)
        p1_lay.addWidget(self.prompt_input)
        p1_lay.addWidget(self.btn_send)
        splitter.addWidget(p1)

        # Panoul 2: Gemini <-> Qwen
        p2 = QWidget()
        p2_lay = QVBoxLayout(p2)
        p2_lay.addWidget(QLabel("<b>2. Discuție Internă (Gemini ↔ Qwen)</b>"))
        self.internal_log = QTextEdit()
        self.internal_log.setReadOnly(True)
        p2_lay.addWidget(self.internal_log)
        splitter.addWidget(p2)

        # Panoul 3: Munca lui Qwen (live) + Consolă
        p3 = QWidget()
        p3_lay = QVBoxLayout(p3)
        p3_lay.addWidget(QLabel("<b>3. Munca lui Qwen (Cod Live)</b>"))
        self.qwen_live = QTextEdit()
        self.qwen_live.setReadOnly(True)
        p3_lay.addWidget(self.qwen_live, stretch=3)
        p3_lay.addWidget(QLabel("<b>Consolă (Testare & Git)</b>"))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        p3_lay.addWidget(self.console, stretch=2)
        splitter.addWidget(p3)

        splitter.setSizes([380, 480, 540])
        main_layout.addWidget(splitter)

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Selectează folderul proiectului")
        if folder:
            self.dir_input.setText(folder)

    def start_task(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            return
        if not self.architect:
            self.user_chat.append("<b>Eroare:</b> Setează variabila GEMINI_API_KEY înainte să pornești aplicația!")
            return

        self.user_chat.append(f"<b>Tu:</b> {prompt}")
        self.prompt_input.clear()
        self.btn_send.setEnabled(False)

        self.thread = OrchestratorThread(
            self.architect, self.worker, prompt,
            self.type_combo.currentText(),
            self.dir_input.text().strip(),
            self.git_check.isChecked()
        )
        self.thread.log_user_chat.connect(self.user_chat.append)
        self.thread.log_internal.connect(self.internal_log.append)
        self.thread.qwen_clear.connect(self.qwen_live.clear)
        self.thread.qwen_chunk.connect(lambda c: self.qwen_live.insertPlainText(c))
        self.thread.log_console.connect(self.console.append)
        self.thread.finished_run.connect(lambda: self.btn_send.setEnabled(True))
        self.thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AutoDevWindow()
    win.show()
    sys.exit(app.exec())
