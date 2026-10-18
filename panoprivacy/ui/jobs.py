"""Cooperative workers whose late results cannot mutate a different document."""
import threading
import uuid

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class _Signals(QObject):
    progress = Signal(str, object)
    result = Signal(str, object)
    error = Signal(str, str)
    done = Signal(str)


class _Worker(QRunnable):
    def __init__(self, token, function):
        super().__init__()
        self.token = token
        self.function = function
        self.cancel = threading.Event()
        self.signals = _Signals()

    def run(self):
        try:
            result = self.function(
                lambda event: self.signals.progress.emit(self.token, event), self.cancel)
            if not self.cancel.is_set():
                self.signals.result.emit(self.token, result)
        except Exception as error:
            if not self.cancel.is_set():
                self.signals.error.emit(self.token, str(error) or type(error).__name__)
        finally:
            self.signals.done.emit(self.token)


class JobController(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    progress = Signal(object)
    finished = Signal()
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.document_id = ''
        self._current = None
        self._workers = {}

    @property
    def running(self):
        return self._current is not None

    @property
    def active_count(self):
        return len(self._workers)

    def accepts_result(self, document_id):
        return document_id == self.document_id and self._current is not None

    def switch_document(self, document_id):
        self.cancel()
        self.document_id = document_id

    def cancel(self):
        if self._current in self._workers:
            self._workers[self._current].cancel.set()
        was_running = self.running
        self._current = None
        if was_running:
            self.busy_changed.emit(False)

    def start(self, function, document_id=None):
        self.cancel()
        if document_id is not None:
            self.document_id = document_id
        token = str(uuid.uuid4())
        worker = _Worker(token, function)
        worker.signals.result.connect(self._result)
        worker.signals.error.connect(self._error)
        worker.signals.progress.connect(self._progress)
        worker.signals.done.connect(self._done)
        self._workers[token] = worker
        self._current = token
        self.busy_changed.emit(True)
        QThreadPool.globalInstance().start(worker)
        return token

    @Slot(str, object)
    def _result(self, token, result):
        if token == self._current:
            self.succeeded.emit(result)

    @Slot(str, str)
    def _error(self, token, error):
        if token == self._current:
            self.failed.emit(error)

    @Slot(str, object)
    def _progress(self, token, event):
        if token == self._current:
            self.progress.emit(event)

    @Slot(str)
    def _done(self, token):
        self._workers.pop(token, None)
        if token == self._current:
            self._current = None
            self.busy_changed.emit(False)
            self.finished.emit()
