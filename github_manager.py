"""github_manager.py — creează repo-ul pe GitHub (REST), face commit și push cu token mascat în log."""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from config import Cancelled, Settings, slugify_repo
from runner import Emit, RunResult, stream_command

API = "https://api.github.com"
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
GITIGNORE_ENTRIES = (".venv/", "__pycache__/", ".env", "*.pyc")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AutoDev-Studio",
    }


def _api_message(response: requests.Response) -> str:
    try:
        return str(response.json().get("message", response.text[:200]))
    except ValueError:
        return response.text[:200]


def ensure_remote_repo(settings: Settings, repo: str, emit: Emit) -> tuple[bool, str, str]:
    """Verifică dacă repo-ul există; dacă nu, îl creează. Returnează (ok, mesaj, owner)."""
    token, user = settings.github_token.strip(), settings.github_username.strip()
    emit(f"Verific repo-ul {user}/{repo} pe GitHub…")
    try:
        response = requests.get(f"{API}/repos/{user}/{repo}", headers=_headers(token), timeout=20)
        if response.status_code == 200:
            emit("Repo-ul există deja pe GitHub.")
            return True, response.json().get("html_url", ""), response.json().get("owner", {}).get("login", user)
        if response.status_code == 401:
            return False, "Token GitHub invalid sau expirat (HTTP 401).", user
        if response.status_code != 404:
            return False, f"GitHub a răspuns HTTP {response.status_code}: {_api_message(response)}", user

        emit(f"Repo-ul nu există. Îl creez ({'privat' if settings.github_private else 'public'})…")
        response = requests.post(
            f"{API}/user/repos", headers=_headers(token), timeout=30,
            json={"name": repo, "private": settings.github_private,
                  "description": "Creat cu AutoDev Studio", "auto_init": False},
        )
    except requests.RequestException as exc:
        return False, f"Eroare de rețea către GitHub: {str(exc).replace(token, '***')}", user

    if response.status_code == 201:
        data = response.json()
        emit(f"✔ Repo creat: {data.get('html_url', '')}")
        return True, data.get("html_url", ""), data.get("owner", {}).get("login", user)
    if response.status_code in (401, 403):
        return False, (f"GitHub a refuzat crearea repo-ului (HTTP {response.status_code}): "
                       f"{_api_message(response)}. Token-ul are nevoie de permisiunea `repo`."), user
    if response.status_code == 422:
        return False, (f"Nu pot crea repo-ul: {_api_message(response)}. Dacă există deja un repo cu acest nume "
                       "la care token-ul nu are acces, verifică username-ul și permisiunile."), user
    return False, f"Crearea repo-ului a eșuat (HTTP {response.status_code}): {_api_message(response)}", user


def _git_quiet(project: Path, *args: str) -> subprocess.CompletedProcess:
    """Rulează git fără să afișeze nimic (pentru verificări interne)."""
    import os
    env = os.environ.copy()
    env.update(GIT_ENV)
    return subprocess.run(["git", *args], cwd=str(project), capture_output=True, text=True,
                          timeout=30, env=env, errors="replace")


def _ensure_gitignore(project: Path, emit: Emit) -> None:
    path = project / ".gitignore"
    existing = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    normalized = {line.strip().rstrip("/") for line in existing}
    missing = [entry for entry in GITIGNORE_ENTRIES if entry.rstrip("/") not in normalized]
    if not missing and path.exists():
        return
    lines = existing + missing
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    emit(f"✔ .gitignore actualizat ({', '.join(missing) if missing else 'creat'})")


def push_project(project_dir: str | Path, settings: Settings, emit: Emit,
                 cancel: Optional[threading.Event] = None,
                 commit_message: str = "AutoDev Studio: actualizare proiect") -> RunResult:
    """Creează repo-ul (dacă lipsește), face commit și `git push -u origin main`."""
    token, user = settings.github_token.strip(), settings.github_username.strip()
    if not token or not user:
        return RunResult(False, "Lipsește token-ul sau username-ul GitHub (butonul Setări).")
    if shutil.which("git") is None:
        return RunResult(False, "Comanda `git` nu este instalată (sudo pacman -S git).")
    project = Path(project_dir).resolve()
    if not project.is_dir():
        return RunResult(False, f"Folderul proiectului nu există: {project}")

    repo = slugify_repo(project.name)
    if cancel is not None and cancel.is_set():
        raise Cancelled()
    ok, info, owner = ensure_remote_repo(settings, repo, emit)
    if not ok:
        emit(f"✘ {info}")
        return RunResult(False, "Nu am putut pregăti repo-ul pe GitHub.", info)

    secrets = tuple({token, quote(token, safe="")})

    def git(*args: str, timeout: float = 120) -> tuple[int, str]:
        rc, out, timed_out = stream_command(
            ["git", *args], project, emit, timeout, cancel, env=GIT_ENV, secrets=secrets)
        return (124 if timed_out else rc), out

    _ensure_gitignore(project, emit)

    if not (project / ".git").exists():
        rc, out = git("init", "-b", "main")
        if rc != 0:  # git vechi, fără opțiunea -b
            rc, out = git("init")
            if rc == 0:
                rc, out = git("symbolic-ref", "HEAD", "refs/heads/main")
        if rc != 0:
            return RunResult(False, "git init a eșuat.", out)

    # Identitate locală doar dacă utilizatorul nu are una configurată.
    if not _git_quiet(project, "config", "user.name").stdout.strip():
        git("config", "user.name", user)
    if not _git_quiet(project, "config", "user.email").stdout.strip():
        git("config", "user.email", f"{user}@users.noreply.github.com")

    rc, out = git("add", ".")
    if rc != 0:
        return RunResult(False, "git add a eșuat.", out)

    has_changes = bool(_git_quiet(project, "status", "--porcelain").stdout.strip())
    has_head = _git_quiet(project, "rev-parse", "--verify", "HEAD").returncode == 0
    if has_changes:
        rc, out = git("commit", "-m", commit_message)
        if rc != 0:
            return RunResult(False, "git commit a eșuat.", out)
    elif not has_head:
        return RunResult(False, "Nu există nimic de comis (folder gol).")
    else:
        emit("Nicio modificare nouă de comis; trimit commit-urile existente.")

    rc, out = git("branch", "-M", "main")
    if rc != 0:
        return RunResult(False, "Nu am putut seta ramura `main`.", out)

    clean_url = f"https://github.com/{owner}/{repo}.git"
    auth_url = f"https://{quote(owner, safe='')}:{quote(token, safe='')}@github.com/{owner}/{repo}.git"
    has_origin = _git_quiet(project, "remote", "get-url", "origin").returncode == 0
    rc, out = git("remote", "set-url" if has_origin else "add", *(("origin", auth_url) if has_origin else ("origin", auth_url)))
    if rc != 0:
        return RunResult(False, "Nu am putut seta remote-ul `origin`.", out)

    try:
        rc, out = git("push", "-u", "origin", "main", timeout=300)
    finally:
        # Token-ul nu rămâne scris în .git/config: revenim la URL-ul curat.
        try:
            _git_quiet(project, "remote", "set-url", "origin", clean_url)
        except (OSError, subprocess.SubprocessError):
            pass

    if rc != 0:
        lowered = out.lower()
        if "rejected" in lowered or "non-fast-forward" in lowered or "fetch first" in lowered:
            hint = "Repo-ul de pe GitHub are commit-uri pe care nu le ai local; fă un `git pull --rebase` manual."
        elif "authentication" in lowered or "403" in lowered or "denied" in lowered:
            hint = "Autentificare refuzată; verifică token-ul (permisiunea `repo`) și username-ul."
        elif rc == 124:
            hint = "Push-ul a depășit 5 minute."
        else:
            hint = f"git push a eșuat (cod {rc})."
        emit(f"✘ {hint}")
        return RunResult(False, hint, out)

    url = f"https://github.com/{owner}/{repo}"
    emit(f"✔ Cod urcat pe GitHub: {url}")
    return RunResult(True, f"Cod urcat pe GitHub: {url}", out)
