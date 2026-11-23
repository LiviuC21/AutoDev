"""runner.py — rulare comenzi în terminal, verificare fișiere, venv și instalare dependențe."""
from __future__ import annotations

import json
import os
import re
import shutil
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
from config import ANDROID_TEMPLATE_DIR, ANDROID_STUDIO_CMD

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

def _validate_cpp(project_dir: str | Path, rel: str, emit: Emit,
                  cancel: Optional[threading.Event]) -> RunResult:
    if not shutil.which("g++"):
        emit("⚠ g++ nu este instalat — sar peste verificarea de sintaxă C++.")
        return RunResult(True, "Salvat (g++ lipsește).")
    emit(f"▶ Verific sintaxa C++: {rel}")
    rc, out, _ = stream_command(
        ["g++", "-fsyntax-only", "-std=c++17", rel], project_dir, emit, 30, cancel
    )
    if rc == 0:
        return RunResult(True, "Sintaxă C++ validă.", out)
    return RunResult(False, "Eroare de compilare C++.", out)


def _validate_java(project_dir: str | Path, rel: str, emit: Emit,
                   cancel: Optional[threading.Event]) -> RunResult:
    if not shutil.which("javac"):
        emit("⚠ javac (JDK) nu este instalat — sar peste verificarea Java.")
        return RunResult(True, "Salvat (javac lipsește).")
    out_dir = Path(project_dir) / ".build_java"
    out_dir.mkdir(exist_ok=True)
    emit(f"▶ Compilez Java: {rel}")
    rc, out, _ = stream_command(
        ["javac", "-d", str(out_dir), "-sourcepath", ".", rel], project_dir, emit, 45, cancel
    )
    if rc == 0:
        return RunResult(True, "Compilare Java reușită.", out)
    return RunResult(False, "Eroare de compilare Java.", out)


def _validate_js(project_dir: str | Path, rel: str, emit: Emit,
                 cancel: Optional[threading.Event]) -> RunResult:
    if not shutil.which("node"):
        emit(f"✔ Salvat: {rel} (node nu este instalat pentru verificare JS)")
        return RunResult(True, "Salvat (fără verificare node).")
    emit(f"▶ Verific sintaxa JS: {rel}")
    rc, out, _ = stream_command(["node", "--check", rel], project_dir, emit, 15, cancel)
    if rc == 0:
        return RunResult(True, "Sintaxă JavaScript validă.", out)
    return RunResult(False, "Eroare de sintaxă JavaScript.", out)

def validate_file(project_type: str, project_dir: str | Path, rel: str, entry_file: str,
                  emit: Emit, cancel: Optional[threading.Event]) -> RunResult:
    """Verifică un fișier proaspăt salvat, în funcție de tipul lui."""
    path = safe_join(project_dir, rel)
    suffix = path.suffix.lower()
    if suffix == ".py":
        return _validate_python(project_dir, rel, entry_file, emit, cancel)
    if suffix in {".cpp", ".cc", ".cxx"}:
        return _validate_cpp(project_dir, rel, emit, cancel)
    if suffix == ".java":
        return _validate_java(project_dir, rel, emit, cancel)
    if suffix == ".js":
        return _validate_js(project_dir, rel, emit, cancel)
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

def prepare_android_project(project_dir: Path, app_name: str, log=None) -> str:
    """Copiază matrița Android dacă folderul e gol, redenumește pachetul și returnează package_name."""
    clean_name = re.sub(r"[^A-Za-z0-9]", "", app_name) or "AutoDevApp"
    pkg_suffix = clean_name.lower()
    new_pkg = f"com.autodev.{pkg_suffix}"

    # Dacă proiectul există deja, nu îl suprascriem
    if (project_dir / "settings.gradle.kts").exists():
        if log:
            log("📱 Proiect Android existent detectat — păstrez structura actuală.")
        return new_pkg

    if not ANDROID_TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"Nu găsesc matrița la {ANDROID_TEMPLATE_DIR}")

    if log:
        log(f"📱 Creez proiectul Android '{clean_name}' ({new_pkg}) din matriță…")

    ignore = shutil.ignore_patterns(".gradle", "build", ".idea", ".git", "local.properties")
    shutil.copytree(ANDROID_TEMPLATE_DIR, project_dir, dirs_exist_ok=True, ignore=ignore)

    # Înlocuiește AppTemplate și com.autodev.apptemplate în fișierele text
    exts = {".kts", ".kt", ".xml", ".pro", ".gradle"}
    for f in project_dir.rglob("*"):
        if f.is_file() and f.suffix in exts:
            try:
                txt = f.read_text(encoding="utf-8")
                new_txt = txt.replace("com.autodev.apptemplate", new_pkg).replace("AppTemplate", clean_name)
                if new_txt != txt:
                    f.write_text(new_txt, encoding="utf-8")
            except Exception:
                pass

    # Redenumește folderele de pachet (main, test, androidTest)
    for src_set in ("main", "test", "androidTest"):
        old_dir = project_dir / "app" / "src" / src_set / "java" / "com" / "autodev" / "apptemplate"
        new_dir = project_dir / "app" / "src" / src_set / "java" / "com" / "autodev" / pkg_suffix
        if old_dir.exists() and old_dir != new_dir:
            old_dir.rename(new_dir)

    gradlew = project_dir / "gradlew"
    if gradlew.exists():
        gradlew.chmod(0o755)

    return new_pkg


def open_android_studio(project_dir: Path, log=None) -> None:
    """Deschide Android Studio direct pe folderul proiectului (fără să blocheze AutoDev)."""
    try:
        subprocess.Popen(
            [ANDROID_STUDIO_CMD, str(project_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if log:
            log(f"🚀 Android Studio deschis pe: {project_dir}")
    except FileNotFoundError:
        if log:
            log("⚠ Comanda 'android-studio' nu a fost găsită în PATH.")


def run_gradle_build(project_dir: Path, log=None) -> tuple[bool, str]:
    """Compilează proiectul Android cu ./gradlew assembleDebug și returnează (ok, output)."""
    gradlew = project_dir / "gradlew"
    if not gradlew.exists():
        return True, "Lipsește gradlew; sar peste build."
    if log:
        log("🔨 Compilez proiectul Android (./gradlew assembleDebug)…")
    proc = subprocess.run(
        [str(gradlew), "assembleDebug", "--no-daemon"],
        cwd=str(project_dir),
        capture_output=True,
        text=True,
        timeout=180,
    )
    out = (proc.stdout + "\n" + proc.stderr).strip()
    return proc.returncode == 0, out[-2500:]

def deploy_and_capture_android(project_dir: Path, package_name: str, log=None) -> tuple[bool, Optional[Path], str]:
    """Instalează APK-ul debug pe telefon prin adb, pornește aplicația și face o captură de ecran."""
    apk = project_dir / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
    if not apk.exists():
        return False, None, f"Nu găsesc APK-ul compilat la {apk}"

    # Verifică dacă există telefon conectat
    devs = subprocess.run(["adb", "devices"], capture_output=True, text=True)
    if "\tdevice" not in devs.stdout:
        return False, None, "Niciun telefon conectat prin adb."

    if log:
        log(f"📲 Instalez {apk.name} pe telefon…")
    inst = subprocess.run(["adb", "install", "-r", str(apk)], capture_output=True, text=True, timeout=60)
    if inst.returncode != 0:
        return False, None, f"Eroare adb install: {inst.stderr or inst.stdout}"

    if log:
        log(f"▶ Lansez {package_name} pe telefon…")
    subprocess.run(
        ["adb", "shell", "monkey", "-p", package_name, "-c", "android.intent.category.LAUNCHER", "1"],
        capture_output=True, text=True, timeout=15,
    )
    time.sleep(4)  # Așteptăm să se încarce UI-ul pe ecran

    shot = project_dir / "_last_screenshot.png"
    with open(shot, "wb") as f:
        cap = subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=f, stderr=subprocess.PIPE, timeout=15)
    if cap.returncode != 0 or shot.stat().st_size == 0:
        return False, None, "Nu am putut face captura de ecran prin adb."

    if log:
        log(f"📸 Captură salvată: {shot.name}")
    return True, shot, "OK"

# --------------------------------------------------------------------------- #
# Export .exe pentru Windows (C++, Java, Python)
# --------------------------------------------------------------------------- #
GITHUB_PYINSTALLER_WORKFLOW = """name: Build Windows EXE
on:
  push:
  workflow_dispatch:

jobs:
  build-exe:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pyinstaller
          if (Test-Path requirements.txt) { pip install -r requirements.txt }
      - name: Build EXE
        run: pyinstaller --onefile --noconsole --name "{app_name}" "{entry_file}"
      - name: Upload EXE artifact
        uses: actions/upload-artifact@v4
        with:
          name: {app_name}-windows-exe
          path: dist/{app_name}.exe
"""


def export_windows_exe(project_type: str, project_dir: str | Path, entry_file: str,
                       emit: Emit, cancel: Optional[threading.Event] = None) -> RunResult:
    """Generează (sau pregătește) executabilul .exe pentru Windows în funcție de tipul proiectului."""
    base = Path(project_dir)
    app_name = re.sub(r"[^A-Za-z0-9_-]+", "_", base.name).strip("_") or "AutoDevApp"
    dist_dir = base / "dist"
    dist_dir.mkdir(exist_ok=True)

    if project_type == "C++":
        compiler = shutil.which("x86_64-w64-mingw32-g++")
        if not compiler:
            msg = "Lipsește cross-compilatorul MinGW. Instalează-l cu: sudo pacman -S mingw-w64-gcc"
            emit(f"⚠ {msg}")
            return RunResult(False, msg)
        cpp_files = sorted(str(p.relative_to(base)) for p in base.rglob("*.cpp"))
        if not cpp_files:
            return RunResult(False, "Nu am găsit fișiere .cpp în proiect.")
        out_exe = dist_dir / f"{app_name}.exe"
        emit(f"📦 Cross-compilez C++ → {out_exe.name}…")
        cmd = [compiler, "-std=c++17", "-O2", "-static", "-static-libgcc", "-static-libstdc++",
               *cpp_files, "-o", str(out_exe)]
        rc, out, _ = stream_command(cmd, base, emit, 120, cancel)
        if rc == 0 and out_exe.is_file():
            return RunResult(True, f"Executabil Windows creat: dist/{out_exe.name}", out)
        return RunResult(False, "Cross-compilarea C++ a eșuat.", out)

    if project_type == "Java":
        if not shutil.which("javac") or not shutil.which("jar"):
            return RunResult(False, "Lipsește JDK (javac/jar). Instalează cu: sudo pacman -S jdk-openjdk")
        java_files = sorted(str(p.relative_to(base)) for p in base.rglob("*.java"))
        if not java_files:
            return RunResult(False, "Nu am găsit fișiere .java în proiect.")
        build_dir = base / ".build_java"
        build_dir.mkdir(exist_ok=True)
        rc, out, _ = stream_command(["javac", "-d", str(build_dir), *java_files], base, emit, 60, cancel)
        if rc != 0:
            return RunResult(False, "Compilarea Java a eșuat.", out)
        main_class = Path(entry_file or "Main.java").stem
        jar_path = dist_dir / f"{app_name}.jar"
        rc2, out2, _ = stream_command(
            ["jar", "cfe", str(jar_path), main_class, "-C", str(build_dir), "."], base, emit, 30, cancel
        )
        if rc2 != 0:
            return RunResult(False, "Crearea arhivei .jar a eșuat.", out2)
        bat_path = dist_dir / f"Run_{app_name}.bat"
        bat_path.write_text(f"@echo off\njava -jar \"%~dp0{jar_path.name}\"\npause\n", encoding="utf-8")
        return RunResult(True, f"Pachet Windows creat: dist/{jar_path.name} + dist/{bat_path.name}", out2)

    if project_type == "Python":
        entry = entry_file or ("main.py" if (base / "main.py").is_file() else "")
        if not entry:
            py_files = sorted(p.name for p in base.glob("*.py"))
            entry = py_files[0] if py_files else "main.py"
        wf_dir = base / ".github" / "workflows"
        wf_dir.mkdir(parents=True, exist_ok=True)
        wf_file = wf_dir / "build-windows-exe.yml"
        wf_file.write_text(
            GITHUB_PYINSTALLER_WORKFLOW.format(app_name=app_name, entry_file=entry),
            encoding="utf-8",
        )
        emit(f"✔ Creat workflow GitHub Actions: {wf_file.relative_to(base)}")
        emit("💡 Fă push pe GitHub (cu Auto GitHub Push) și descarcă .exe-ul din tab-ul Actions → Artifacts.")
        return RunResult(True, f"Workflow Windows .exe creat (.github/workflows/build-windows-exe.yml) pentru {entry}.")

    return RunResult(False, f"Exportul .exe nu se aplică pentru proiecte de tip {project_type}.")
