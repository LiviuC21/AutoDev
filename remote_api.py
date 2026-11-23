"""remote_api.py — Server HTTP JSON ușor pentru telecomanda Android prin Tailscale."""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal


def read_hardware_stats() -> dict[str, Optional[float | int]]:
    """Citește temperatura CPU (°C) și turația ventilatoarelor (RPM) din /sys/class/hwmon."""
    stats: dict[str, Optional[float | int]] = {
        "cpu_temp": None,
        "cpu_fan_rpm": None,
        "gpu_fan_rpm": None,
    }
    hwmon = Path("/sys/class/hwmon")
    if not hwmon.is_dir():
        return stats

    temp_priority = ("k10temp", "coretemp", "zenpower", "acpitz")
    temps: dict[str, float] = {}

    for entry in hwmon.iterdir():
        try:
            name = (entry / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue

        if name in temp_priority:
            try:
                raw = (entry / "temp1_input").read_text(encoding="utf-8").strip()
                temps[name] = round(int(raw) / 1000.0, 1)
            except (OSError, ValueError):
                pass

        # Ventilatoare ASUS / generice (fan1_input, fan2_input)
        for fan_idx, key in ((1, "cpu_fan_rpm"), (2, "gpu_fan_rpm")):
            fan_file = entry / f"fan{fan_idx}_input"
            if fan_file.is_file() and stats[key] is None:
                try:
                    stats[key] = int(fan_file.read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    pass

    for key in temp_priority:
        if key in temps:
            stats["cpu_temp"] = temps[key]
            break

    return stats


class RemoteAPIServer(QObject):
    """Server API pe 0.0.0.0:port (accesibil prin Tailscale) conectat prin semnale la PyQt6."""

    stop_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    feedback_requested = pyqtSignal(str)

    def __init__(self, port: int = 8765, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.port = port
        self._lock = threading.Lock()
        self._status: dict[str, Any] = {
            "running": False,
            "project_dir": "",
            "project_type": "",
            "ollama_model": "",
            "gemini_model": "",
            "can_resume": False,
            "progress_done": 0,
            "progress_total": 0,
            "current_file": "",
        }
        self._agent_logs: deque[str] = deque(maxlen=80)
        self._console_logs: deque[str] = deque(maxlen=80)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ----- actualizare stare din UI ---------------------------------------- #
    def update_status(self, **kwargs: Any) -> None:
        with self._lock:
            self._status.update(kwargs)

    def push_agent_log(self, text: str) -> None:
        with self._lock:
            self._agent_logs.append(text)

    def push_console_log(self, text: str) -> None:
        with self._lock:
            for line in text.splitlines():
                if line.strip():
                    self._console_logs.append(line)

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._status)
        data.update(read_hardware_stats())
        return data

    def get_logs(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                "agent": list(self._agent_logs),
                "console": list(self._console_logs),
            }

    # ----- pornire / oprire server ----------------------------------------- #
    def start(self) -> bool:
        if self._server is not None:
            return True
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # nu umplem terminalul cu loguri HTTP

            def _send_json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/status":
                    self._send_json(200, api.get_snapshot())
                elif self.path == "/api/logs":
                    self._send_json(200, api.get_logs())
                elif self.path == "/":
                    self._send_json(200, {"service": "AutoDev Studio API", "status": "online"})
                else:
                    self._send_json(404, {"error": "Endpoint inexistent"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw = self.rfile.read(length).decode("utf-8", errors="replace") if length > 0 else "{}"
                try:
                    body = json.loads(raw) if raw.strip() else {}
                except ValueError:
                    body = {}

                if self.path == "/api/stop":
                    api.stop_requested.emit()
                    self._send_json(200, {"ok": True, "action": "stop"})
                elif self.path == "/api/resume":
                    api.resume_requested.emit()
                    self._send_json(200, {"ok": True, "action": "resume"})
                elif self.path == "/api/feedback":
                    msg = str(body.get("message", "")).strip()
                    if not msg:
                        self._send_json(400, {"ok": False, "error": "Lipsește câmpul 'message'"})
                        return
                    api.feedback_requested.emit(msg)
                    self._send_json(200, {"ok": True, "action": "feedback"})
                else:
                    self._send_json(404, {"error": "Endpoint inexistent"})

        try:
            self._server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError:
            self._server = None
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            self._thread = None
