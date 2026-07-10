from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class WorkerSignals(QObject):
    log = pyqtSignal(str)
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    draft_updated = pyqtSignal()
