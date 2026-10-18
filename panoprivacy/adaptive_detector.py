"""Measured GPU admission and bounded ordered work, independent of photo UI."""
from __future__ import annotations

import collections
import gc
import json
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass

import numpy as np

from .common import Cancelled, app_cache_dir, atomic_json, check_cancel, emit


@dataclass(frozen=True)
class ResourceSnapshot:
    ram_total: int | None
    ram_free: int | None
    gpu_total: int | None = None
    gpu_free: int | None = None
    device: str | None = None
    rss: int | None = None


_GPU_SAMPLE = None
_GPU_TIME = 0.
_RESOURCE_LOCK = threading.Lock()


def resources(include_gpu=True, refresh=False):
    global _GPU_SAMPLE, _GPU_TIME
    try:
        import psutil
        ram = psutil.virtual_memory()
        total, free, rss = int(ram.total), int(ram.available), int(psutil.Process().memory_info().rss)
    except Exception:
        total = free = rss = None
    gpu_total = gpu_free = device = None
    if include_gpu:
        with _RESOURCE_LOCK:
            if refresh or time.monotonic()-_GPU_TIME > 1:
                try:
                    process = subprocess.run(['nvidia-smi', '--id=0',
                        '--query-gpu=uuid,driver_version,memory.total,memory.free', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=1.5,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    parts = [x.strip() for x in process.stdout.strip().split(',')]
                    if process.returncode or len(parts) != 4:
                        raise ValueError('GPU memory unavailable')
                    _GPU_SAMPLE = (int(float(parts[2])*2**20), int(float(parts[3])*2**20), '|'.join(parts[:2]))
                except Exception:
                    _GPU_SAMPLE = None
                _GPU_TIME = time.monotonic()
            if _GPU_SAMPLE:
                gpu_total, gpu_free, device = _GPU_SAMPLE
    return ResourceSnapshot(total, free, gpu_total, gpu_free, device, rss)


def ram_reserve(live):
    return max(2*2**30, min(10*2**30, .15*(live.ram_total or 0)))


def under_pressure(live):
    return (None in (live.ram_total,live.ram_free,live.gpu_total,live.gpu_free) or
            (live.ram_free is not None and live.ram_free < ram_reserve(live)) or
            (live.gpu_total is not None and live.gpu_free is not None and
             live.gpu_free < max(2*2**30, .2*live.gpu_total)))


def admit_workers(live, gpu_per_worker, ram_per_worker, working_set=0):
    """Unknown allocation or telemetry permits serial execution only."""
    if (None in (live.ram_total, live.ram_free, live.gpu_total, live.gpu_free) or
            gpu_per_worker is None or ram_per_worker is None or min(gpu_per_worker, ram_per_worker) <= 0):
        return 1
    gpu_budget = max(0, live.gpu_free-max(2*2**30, .2*live.gpu_total))
    ram_budget = max(0, live.ram_free-ram_reserve(live)-working_set)
    allowed = 1
    for n in (2, 4, 8):
        if n*gpu_per_worker <= gpu_budget and 1.25*n*ram_per_worker <= ram_budget:
            allowed = n
    return allowed


def select_calibration(observations):
    selected = None
    for row in observations:
        if not row['equivalent'] or row['throughput'] <= 0:
            break
        if selected is not None and row['throughput'] / selected['throughput'] < 1.10:
            break
        selected = row
    return selected['workers'] if selected else 1


def _allocation_failure(error):
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(s in message for s in
        ('out of memory', 'cudaerrormemoryallocation', 'failed to allocate', 'bad allocation'))


class BoundedDetectorPool:
    """At most 2N pending tiles; independent sessions; join on every exit.

    Task rendering takes place inside each worker. Results are yielded in source
    order. Pressure drains the current batch and releases half the sessions;
    allocation failures retry only unfinished tasks in a smaller pool.
    """
    def __init__(self, detectors, cancel=None, pressure=None, progress=None):
        if not detectors:
            raise ValueError('At least one detector is required')
        self.detectors = list(detectors)
        self.cancel, self.pressure, self.progress = cancel, pressure, progress
        self._executor = None
        self._available = None
        self._open()

    @property
    def workers(self):
        return len(self.detectors)

    def _open(self):
        self._available = queue.Queue()
        for detector in self.detectors:
            self._available.put(detector)
        self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='photo-detector')

    def _shrink(self):
        if self.workers == 1:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = self._available = None
        self.detectors = self.detectors[:max(1,self.workers//2)]
        gc.collect()
        self._open()
        emit(self.progress, 'photo_adaptive', 0, f'可用资源减少，识别并行数已降至 {self.workers}')

    def _work(self, spec, render, threshold):
        check_cancel(self.cancel)
        detector = self._available.get()
        try:
            rgb = render(spec)
            check_cancel(self.cancel)
            result = detector.detect(rgb, min_score=threshold)
            check_cancel(self.cancel)
            return result
        finally:
            self._available.put(detector)

    def map(self, specs, render, threshold=.6):
        iterator = iter(specs)
        exhausted = False
        while not exhausted:
            check_cancel(self.cancel)
            if self.pressure and self.pressure():
                self._shrink()
            batch = []
            for _ in range(2*self.workers):
                check_cancel(self.cancel)
                try:
                    spec = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                batch.append((spec, self._executor.submit(self._work, spec, render, threshold)))
            if not batch:
                break
            results = []
            retry = []
            pressured = False
            for index, (spec, future) in enumerate(batch):
                while True:
                    check_cancel(self.cancel)
                    try:
                        value = future.result(timeout=.1)
                        results.append(value)
                        break
                    except TimeoutError:
                        pressured |= bool(self.pressure and self.pressure())
                    except Exception as error:
                        if not _allocation_failure(error) or self.workers == 1:
                            raise
                        # The Future owns this exception until the batch is
                        # released. Its worker traceback otherwise retains the
                        # failed CUDA session across resource backoff.
                        error.__traceback__ = None
                        results.append(None)
                        retry.append(index)
                        break
            if retry or pressured or (self.pressure and self.pressure()):
                self._shrink()
            # Only allocation failures are retryable. Completed results remain
            # in their original positions, avoiding duplicates after backoff.
            for index in retry:
                while True:
                    check_cancel(self.cancel)
                    try:
                        results[index] = self._work(batch[index][0], render, threshold)
                        break
                    except Exception as error:
                        if not _allocation_failure(error) or self.workers == 1:
                            raise
                        self._shrink()
            for (spec, _), value in zip(batch, results):
                check_cancel(self.cancel)
                yield spec, value

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = self._available = None
        self.detectors.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _same_results(left, right):
    if len(left) != len(right):
        return False
    return all(np.array_equal(np.asarray(a), np.asarray(b)) for a,b in zip(left,right))


def _raw_signature(detector, rgb):
    """Check raw boxes/logits even when calibration crops contain no faces."""
    import cv2
    from .detector import _MEAN, _STD
    resized = cv2.resize(rgb,(576,576),interpolation=cv2.INTER_LINEAR)
    tensor = (resized.astype(np.float32).transpose(2,0,1)/255.-_MEAN)/_STD
    return detector.session.run(['dets','labels'],
        {detector.session.get_inputs()[0].name: np.ascontiguousarray(tensor[None])})


def calibrated_pool(factory, representative, render, *, threshold=.6, cancel=None,
                    progress=None, working_set=0):
    """Warm/profile 1, then probe 2/4/8 while output and >=10% gain agree."""
    before = resources(refresh=True)
    first = factory()
    emit(progress, 'photo_adaptive', 0, '识别设备：'+first.provider + ('；'+getattr(first,'warning','') if getattr(first,'warning','') else ''))
    if first.provider != 'CUDAExecutionProvider':
        return BoundedDetectorPool([first], cancel=cancel, progress=progress)
    check_cancel(cancel)
    first.detect(render(representative[0]), min_score=threshold)
    after = resources(refresh=True)
    gpu_cost = None if before.gpu_free is None or after.gpu_free is None else max(0,before.gpu_free-after.gpu_free)*1.3
    ram_cost = None if before.rss is None or after.rss is None else max(0,after.rss-before.rss)*1.3
    allowed = admit_workers(before, gpu_cost, ram_cost, working_set)
    if allowed == 1:
        emit(progress, 'photo_adaptive', 1, '资源余量不足或测量不可用，使用单路识别')
        return BoundedDetectorPool([first], cancel=cancel, pressure=lambda: under_pressure(resources()), progress=progress)
    from .detector import EXPECTED_SHA256
    import onnxruntime
    key = '|'.join([EXPECTED_SHA256, first.provider, after.device or '', onnxruntime.__version__, '1,3,576,576', 'calibration-v2'])
    profile_path = app_cache_dir()/'photo-detector-profile.json'
    cached = None
    try:
        data = json.loads(profile_path.read_text(encoding='utf-8'))
        if (data['key'] == key and time.time()-data['created'] < 7*86400 and
                abs(before.gpu_free-data['gpu_free'])/max(1,data['gpu_free']) < .25 and
                abs(before.ram_free-data['ram_free'])/max(1,data['ram_free']) < .25 and
                data['working_set'] == working_set):
            cached = data
    except (OSError,ValueError,KeyError,TypeError):
        pass
    detectors = [first]
    observations = []
    expected = None
    best = 1
    signature_inputs = [representative[i] for i in (0,len(representative)//2,-1)]
    reference_signatures = [_raw_signature(first,render(spec)) for spec in signature_inputs]
    # Subsecond measurements are overly sensitive to desktop scheduling noise.
    # Repeat spatially varied specs, never prepared image arrays or tensors.
    trial_specs = [representative[i%len(representative)] for i in range(max(128,len(representative)))]

    def timed_trial(sessions):
        def guarded_render(spec):
            if under_pressure(resources()):
                raise RuntimeError('并行校准时资源余量不足或测量不可用。')
            return render(spec)
        start = time.perf_counter()
        with BoundedDetectorPool(sessions,cancel=cancel) as trial:
            actual = [value for _,value in trial.map(trial_specs,guarded_render,threshold)]
        return actual, time.perf_counter()-start

    try:
        for n in (1,2,4,8):
            check_cancel(cancel)
            if n > allowed or (cached and n > cached['workers']):
                break
            if n > 1:
                live = resources(refresh=True)
                # Existing allocations are already subtracted from live free.
                additional = (n-len(detectors))*gpu_cost
                if (under_pressure(live) or live.gpu_free is None or
                        live.gpu_free-additional < max(2*2**30,.2*live.gpu_total) or
                        live.ram_free-(n-len(detectors))*ram_cost*1.25-working_set < ram_reserve(live)):
                    break
                for _ in range(n-len(detectors)):
                    detector = factory()
                    if detector.provider != 'CUDAExecutionProvider':
                        del detector
                        raise RuntimeError('新增显卡会话不可用，保留已有显卡会话。')
                    detector.detect(render(representative[0]), min_score=threshold)
                    signatures = [_raw_signature(detector,render(spec)) for spec in signature_inputs]
                    if not all(_same_results(a,b) for a,b in zip(signatures,reference_signatures)):
                        del detector
                        raise RuntimeError('并行会话原始输出不一致，保留较小配置。')
                    detectors.append(detector)
            # Repeat the same native samples for every level. Timed work
            # includes native reads, preprocessing, inference and postprocessing.
            actual, elapsed = timed_trial(detectors)
            if expected is None:
                expected = actual
            equivalent = _same_results(actual,expected)
            row = {'workers':n,'throughput':len(trial_specs)/max(elapsed,1e-9),
                   'equivalent':equivalent,'seconds':elapsed,'tiles':len(trial_specs)}
            if observations and equivalent:
                previous = observations[-1]
                gain = row['throughput']/previous['throughput']-1
                if .05 <= gain <= .15:
                    # Repeat both adjacent configurations near the cutoff so
                    # one lucky/slow batch cannot decide the pool size.
                    prior_actual, prior_seconds = timed_trial(detectors[:previous['workers']])
                    actual, repeated_seconds = timed_trial(detectors)
                    previous['seconds'] += prior_seconds
                    previous['tiles'] += len(trial_specs)
                    previous['throughput'] = previous['tiles']/previous['seconds']
                    previous['equivalent'] &= _same_results(prior_actual,expected)
                    row['seconds'] += repeated_seconds
                    row['tiles'] += len(trial_specs)
                    row['throughput'] = row['tiles']/row['seconds']
                    row['equivalent'] &= _same_results(actual,expected)
            observations.append(row)
            selected = select_calibration(observations)
            best = selected
            emit(progress, 'photo_adaptive', n/8, f'自适应测试 {n} 路：{observations[-1]["throughput"]:.1f} 块/秒')
            if selected != n:
                break
    except Cancelled:
        detectors.clear()
        raise
    except Exception as error:
        emit(progress,'photo_adaptive',0,'并行探测已退回较小配置：'+str(error))
    detectors = detectors[:best]
    first = detector = None
    gc.collect()
    try:
        atomic_json(profile_path, {'key':key,'created':time.time(),'workers':best,
            'gpu_free':before.gpu_free,'ram_free':before.ram_free,'working_set':working_set,
            'gpu_per_worker':gpu_cost,'ram_per_worker':ram_cost,'observations':observations})
    except OSError:
        pass
    emit(progress,'photo_adaptive',1,f'使用 {best} 路独立显卡会话（按当前资源和实测速度选择）')
    return BoundedDetectorPool(detectors,cancel=cancel,pressure=lambda:under_pressure(resources()),progress=progress)
