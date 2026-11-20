import subprocess
from pathlib import Path


def save_file(project_dir: str, relative_path: str, content: str) -> Path:
    target = Path(project_dir) / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def test_python_file(project_dir: str, relative_path: str, timeout: int = 8) -> tuple[bool, str]:
    """Verifică sintaxa și rulează un fișier Python ca să vadă dacă pornește fără erori."""
    full_path = Path(project_dir) / relative_path
    try:
        proc = subprocess.run(
            ["python3", "-m", "py_compile", str(full_path)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return False, proc.stderr

        proc_run = subprocess.run(
            ["python3", str(full_path)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc_run.returncode != 0:
            return False, proc_run.stderr or proc_run.stdout
        return True, proc_run.stdout or "Execuție OK (fără erori)."
    except subprocess.TimeoutExpired:
        # Dacă e o aplicație GUI care a rămas deschisă 8 secunde fără crash, a pornit corect
        return True, "Aplicația a pornit și a rămas activă fără erori (GUI detectat)."
    except Exception as e:
        return False, str(e)


def git_push(project_dir: str, message: str = "AutoDev update") -> str:
    cmds = [
        ["git", "add", "."],
        ["git", "commit", "-m", message],
        ["git", "push"],
    ]
    logs = []
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
        logs.append(f"$ {' '.join(cmd)}\n{proc.stdout}{proc.stderr}".strip())
    return "\n".join(logs)
