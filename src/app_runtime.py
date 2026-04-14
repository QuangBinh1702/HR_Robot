"""
Application Runtime - Shared state container for pipeline + API server.
Holds references to services shared between camera thread and API thread.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from src.database.repository import FaceRepository
from src.attendance.attendance_manager import AttendanceManager


@dataclass
class AppRuntime:
    """Shared runtime state between camera pipeline and API server."""
    repo: FaceRepository
    attendance: AttendanceManager
    _ws_clients: set = field(default_factory=set)
    _ws_lock: threading.Lock = field(default_factory=threading.Lock)

    def add_ws_client(self, ws):
        with self._ws_lock:
            self._ws_clients.add(ws)

    def remove_ws_client(self, ws):
        with self._ws_lock:
            self._ws_clients.discard(ws)

    @property
    def ws_clients(self) -> set:
        with self._ws_lock:
            return self._ws_clients.copy()
