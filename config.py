"""config.py — constante, setări (.env) și utilitare de cale pentru AutoDev Studio."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
OLLAMA_URL = "http://localhost:11434"
QWEN_MODEL = "qwen2.5-coder:14b"
QWEN_NUM_CTX = 8192
QWEN_NUM_PREDICT = 4096            # tokeni maximi generați de Qwen per fișier
QWEN_PROMPT_CHAR_BUDGET = 12000    # ~3500 tokeni de prompt, restul rămâne pentru răspuns
MAX_ATTEMPTS = 5                   # încercări per fișier
RUN_TIMEOUT = 10                   # secunde de rulare pentru testul unui fișier Python

PROJECT_TYPES = ("Python", "Android")

IGNORED_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "build",
    ".gradle", ".idea", ".mypy_cache", ".pytest_cache",
}


class Cancelled(Exception):
    """Ridicată când utilizatorul apasă Stop."""


# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #
def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def read_env_file() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    try:
        return _parse_env(ENV_PATH.read_text(encoding="utf-8"))
    except OSError:
        return {}


def write_env_values(updates: dict[str, str]) -> None:
    """Actualizează cheile date în .env, păstrând restul; fișierul primește chmod 600."""
    current = read_env_file()
    for key, value in updates.items():
        current[key] = str(value).replace("\n", "").replace("\r", "").strip()
    content = "".join(f"{k}={v}\n" for k, v in current.items())
    tmp = ENV_PATH.parent / ".env.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tmp, ENV_PATH)
    os.chmod(ENV_PATH, 0o600)


@dataclass
class Settings:
    gemini_api_key: str = ""
    github_token: str = ""
    github_username: str = ""
    github_private: bool = True
    last_dir: str = ""
    last_type: str = "Python"

    @classmethod
    def load(cls) -> "Settings":
        values = read_env_file()

        def pick(key: str) -> str:
            return values.get(key) or os.environ.get(key, "")

        last_type = values.get("LAST_PROJECT_TYPE", "Python")
        return cls(
            gemini_api_key=pick("GEMINI_API_KEY"),
            github_token=pick("GITHUB_TOKEN"),
            github_username=pick("GITHUB_USERNAME"),
            github_private=values.get("GITHUB_PRIVATE", "true").lower() != "false",
            last_dir=values.get("LAST_PROJECT_DIR", ""),
            last_type=last_type if last_type in PROJECT_TYPES else "Python",
        )

    def save(self) -> None:
        write_env_values({
            "GEMINI_API_KEY": self.gemini_api_key,
            "GITHUB_TOKEN": self.github_token,
            "GITHUB_USERNAME": self.github_username,
            "GITHUB_PRIVATE": "true" if self.github_private else "false",
            "LAST_PROJECT_DIR": self.last_dir,
            "LAST_PROJECT_TYPE": self.last_type,
        })


# --------------------------------------------------------------------------- #
# Utilitare de cale
# --------------------------------------------------------------------------- #
def safe_join(base: str | Path, rel: str) -> Path:
    """Îmbină base + rel și garantează că rezultatul rămâne în interiorul lui base."""
    base_path = Path(base).resolve()
    rel = rel.strip().replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    rel = rel.lstrip("/")
    if not rel:
        raise ValueError("Cale goală")
    target = (base_path / rel).resolve()
    if target != base_path and base_path not in target.parents:
        raise ValueError(f"Cale în afara proiectului: {rel}")
    return target


def list_project_files(project_dir: str | Path, limit: int = 300) -> list[str]:
    base = Path(project_dir)
    found: list[str] = []
    if not base.is_dir():
        return found
    for root, dirs, files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
        for name in sorted(files):
            found.append(Path(root, name).relative_to(base).as_posix())
            if len(found) >= limit:
                return found
    return found


def slugify_repo(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return slug or "autodev-project"

ANDROID_TEMPLATE_DIR = Path.home() / "Proiecte" / "_AndroidTemplate"
ANDROID_STUDIO_CMD = "android-studio"
