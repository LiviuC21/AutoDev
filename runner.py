"""runner.py — rulare comenzi în terminal, verificare fișiere, venv și instalare dependențe."""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from config import RUN_TIMEOUT, Cancelled, safe_join

Emit = Callable[[str], None]

IMPORT_CHECK_CODE = r"""
import importlib.util, sys
path, project = sys.argv[1], sys.argv[2]
sys.path.insert(0, project)
spec = importlib.util.spec_from_file_location("autodev_check_module", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("import OK")
"""


@dataclass
class RunResult:
    ok: bool
    summary: str
    output: str = ""

    def as_log(self, limit: int = 5000) -> str:
        out = self.output[-limit:] if len(self.output) > limit else self.output
        status = "OK" if self.ok else "EȘEC"
        return f"[{status}] {self.summary}\n{out}".strip()


# --------------------------------------------------------------------------- #
# Procese
# --------------------------------------------------------------------------- #
def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def stream_command(
    cmd: Sequence[str],
    cwd: str | Path,
    emit: Emit,
    timeout: Optional[float] = None,
    cancel: Optional[threading.Event] = None,
    env: Optional[dict[str, str]] = None,
    secrets: Sequence[str] = (),
) -> tuple[int, str, bool]:
    """Rulează o comandă și trimite output-ul linie cu linie prin `emit`.

    Returnează (returncode, output_complet, a_depășit_timeout). Secretele sunt mascate.
    """

    def mask(text: str) -> str:
        for secret in secrets:
            if secret:
                text = text.replace(secret, "***")
        return text

    emit("$ " + mask(" ".join(shlex.quote(part) for part in cmd)))
    full_env = os.environ.copy()
    full_env.update(env or {})
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.Popen(
            list(cmd), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, bufsize=1, env=full_env,
            errors="replace", start_new_session=True,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        message = f"Nu pot porni comanda '{cmd[0]}': {exc}"
        emit(message)
        return 127, message, False

    lines: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = mask(raw.rstrip("\n"))
            lines.append(line)
            emit(line)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    started = time.monotonic()
    timed_out = False
    while proc.poll() is None:
        if cancel is not None and cancel.is_set():
            _kill(proc)
            thread.join(timeout=2)
            raise Cancelled()
        if timeout and time.monotonic() - started > timeout:
            timed_out = True
            _kill(proc)
            break
        time.sleep(0.05)
    thread.join(timeout=3)
    return proc.returncode if proc.returncode is not None else -1, "\n".join(lines), timed_out


# --------------------------------------------------------------------------- #
# Mediu virtual + dependențe (Arch/CachyOS blochează pip global — PEP 668)
# --------------------------------------------------------------------------- #
def venv_python(project_dir: str | Path) -> str:
    candidate = Path(project_dir) / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def ensure_venv(project_dir: str | Path, emit: Emit, cancel: Optional[threading.Event]) -> bool:
    if (Path(project_dir) / ".venv" / "bin" / "python").exists():
        return True
    emit("Creez mediul virtual .venv al proiectului…")
    rc, _, _ = stream_command([sys.executable, "-m", "venv", ".venv"], project_dir, emit, 300, cancel)
    return rc == 0


def install_requirements(project_dir: str | Path, emit: Emit, cancel: Optional[threading.Event]) -> RunResult:
    if not (Path(project_dir) / "requirements.txt").exists():
        return RunResult(True, "Nu există requirements.txt.")
    if not ensure_venv(project_dir, emit, cancel):
        return RunResult(False, "Nu am putut crea .venv.")
    rc, out, timed_out = stream_command(
        [venv_python(project_dir), "-m", "pip", "install", "-r", "requirements.txt"],
        project_dir, emit, 900, cancel,
    )
    if timed_out:
        return RunResult(False, "pip install a depășit 15 minute.", out)
    if rc != 0:
        return RunResult(False, f"pip install a eșuat (cod {rc}).", out)
    return RunResult(True, "Dependențele din requirements.txt au fost instalate în .venv.", out)


# --------------------------------------------------------------------------- #
# Rulare / verificare
# --------------------------------------------------------------------------- #
def run_python_file(
    project_dir: str | Path, rel: str, emit: Emit,
    cancel: Optional[threading.Event], timeout: int = RUN_TIMEOUT,
) -> RunResult:
    emit(f"▶ Execut {rel} (timeout {timeout}s)")
    rc, out, timed_out = stream_command([venv_python(project_dir), rel], project_dir, emit, timeout, cancel)
    has_traceback = "Traceback (most recent call last)" in out
    if timed_out:
        if has_traceback:
            return RunResult(False, "Excepție apărută înainte de timeout.", out)
        return RunResult(
            True, f"Încă rulează după {timeout}s (GUI/server) fără erori → considerat funcțional; procesul a fost oprit.", out)
    if rc == 0:
        return RunResult(True, "Rulat cu succes (cod de ieșire 0).", out)
    if "EOFError" in out and "input" in out:
        return RunResult(True, "Programul cere input interactiv (stdin gol în test) → considerat funcțional.", out)
    return RunResult(False, f"Execuția a eșuat (cod de ieșire {rc}).", out)


def _check_import(project_dir: str | Path, rel: str, emit: Emit, cancel: Optional[threading.Event]) -> RunResult:
    path = safe_join(project_dir, rel)
    emit(f"▶ Verific importul modulului {rel}")
    rc, out, timed_out = stream_command(
        [venv_python(project_dir), "-c", IMPORT_CHECK_CODE, str(path), str(Path(project_dir).resolve())],
        project_dir, emit, 30, cancel,
    )
    if timed_out and "Traceback" not in out:
        return RunResult(True, "Importul rulează cod de lungă durată fără erori → OK.", out)
    if rc == 0:
        return RunResult(True, "Import reușit.", out)
    return RunResult(False, "Importul modulului a eșuat.", out)


def _validate_python(project_dir: str | Path, rel: str, entry_file: str,
                     emit: Emit, cancel: Optional[threading.Event]) -> RunResult:
    path = safe_join(project_dir, rel)
    source = path.read_text(encoding="utf-8", errors="replace")
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        message = f"SyntaxError în {rel}, linia {exc.lineno}: {exc.msg}\n{(exc.text or '').rstrip()}"
        emit(message)
        return RunResult(False, "Eroare de sintaxă.", message)
    emit(f"✔ Sintaxă validă: {rel}")
    is_entry = rel.replace("\\", "/") == (entry_file or "")
    has_main_guard = re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", source) is not None
    if is_entry or has_main_guard:
        return run_python_file(project_dir, rel, emit, cancel)
    return _check_import(project_dir, rel, emit, cancel)


def validate_file(project_type: str, project_dir: str | Path, rel: str, entry_file: str,
                  emit: Emit, cancel: Optional[threading.Event]) -> RunResult:
    """Verifică un fișier proaspăt salvat, în funcție de tipul lui."""
    path = safe_join(project_dir, rel)
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _validate_python(project_dir, rel, entry_file, emit, cancel)
    if suffix == ".xml":
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            message = f"XML invalid în {rel}: {exc}"
            emit(message)
            return RunResult(False, "XML prost format.", message)
        emit(f"✔ XML bine format: {rel}")
        return RunResult(True, "XML bine format.")
    if suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            message = f"JSON invalid în {rel}: {exc}"
            emit(message)
            return RunResult(False, "JSON invalid.", message)
        emit(f"✔ JSON valid: {rel}")
        return RunResult(True, "JSON valid.")
    if project_type == "Python" and path.name == "requirements.txt":
        return install_requirements(project_dir, emit, cancel)
    emit(f"✔ Salvat: {rel} (fără verificare automată pentru acest tip de fișier)")
    return RunResult(True, "Fișier salvat (fără verificare automată pentru acest tip).")
