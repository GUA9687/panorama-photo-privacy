"""Subprocess helpers with cancellation even during blocked pipe IO."""
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager


@contextmanager
def kill_on_cancel(processes, cancel):
    done = threading.Event()
    def watch():
        while not done.wait(.1):
            if cancel is not None and cancel.is_set():
                for process in processes:
                    if process.poll() is None:
                        try: process.kill()
                        except OSError: pass
                return
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join(timeout=1)


def spawn(command, **kwargs):
    return subprocess.Popen([str(x) for x in command], creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0), **kwargs)


def run_checked(command, cancel=None, output=None, timeout=None):
    from .common import check_cancel
    check_cancel(cancel)
    with tempfile.TemporaryFile() as error:
        process = spawn(command, stdout=output or subprocess.DEVNULL, stderr=error)
        started = time.monotonic()
        try:
            with kill_on_cancel([process],cancel):
                while process.poll() is None:
                    check_cancel(cancel)
                    if timeout is not None and time.monotonic()-started > timeout:
                        raise TimeoutError(f'FFmpeg/FFprobe exceeded {timeout:g} seconds')
                    time.sleep(.05)
            check_cancel(cancel)
            if process.returncode:
                error.seek(0)
                raise RuntimeError('FFmpeg/FFprobe failed: ' + error.read()[-6000:].decode('utf8','replace'))
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def read_frame_bytes(stream, buffer):
    view=memoryview(buffer)
    received=0
    while received<len(buffer):
        count=stream.readinto(view[received:])
        if not count:
            return received
        received+=count
    return received
