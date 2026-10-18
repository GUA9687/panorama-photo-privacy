"""Photo-only context policy, bounded rendering and adaptive scheduling regressions."""
import threading
import time

import numpy as np
import pytest


def test_context_association_does_not_accept_neighbour_body():
    from panoprivacy.person_context import associated_person
    assert associated_person([40, 10, 60, 30], [30, 0, 70, 100])
    assert not associated_person([0, 10, 20, 30], [30, 0, 70, 100])
    assert not associated_person([40, 65, 60, 85], [30, 0, 70, 100])
    assert not associated_person([40, 10, 60, 30], [30, 20, 70, 21])


def test_person_output_rgb_nms_and_association_keep_small_head():
    from panoprivacy.person_context import decode_person_output
    output = np.zeros((1, 84, 3), np.float32)
    output[0, :4, :] = np.array([[50, 50, 300], [100, 100, 300], [80, 80, 20], [200, 200, 20]])
    output[0, 4, :] = [.9, .8, .1]
    detections = decode_person_output(output)
    assert len(detections) == 2
    assert detections[0]['box'] == [10, 0, 90, 200]
    assert detections[0]['score'] == pytest.approx(.9)


def test_context_a_then_conditional_b_and_manual_preserved():
    from panoprivacy.common import new_region
    from panoprivacy.person_context import filter_candidates
    regions = [new_region('erp', [.1, .2, .12, .24], origin='auto', enabled=True),
               new_region('erp', [.2, .2, .22, .24], origin='auto', enabled=False),
               new_region('erp', [.3, .2, .32, .24], origin='auto', enabled=True),
               new_region('erp', [.4, .2, .42, .24], origin='manual', enabled=False)]
    regions[0]['confirmation'] = {'confirmed': True}
    calls = []
    def evaluate(region, variant):
        calls.append((region['id'], variant))
        return {'keep': region is not regions[2], 'score': .8, 'provider': 'fixture'}
    result = filter_candidates({}, regions, evaluator=evaluate)
    assert [r['context_keep'] for r in result] == [True, True, False, True]
    assert [r['enabled'] for r in result] == [True, False, False, False]
    assert calls == [(regions[0]['id'], 'A'), (regions[1]['id'], 'A'),
                     (regions[1]['id'], 'B'), (regions[2]['id'], 'A')]
    assert 'context_keep' not in regions[0]


def test_context_failure_is_visible_and_never_silently_removes_confirmed_face():
    from panoprivacy.common import new_region
    from panoprivacy.person_context import filter_candidates
    r = new_region('erp', [.1, .2, .2, .3], origin='auto', enabled=True)
    r['confirmation'] = {'confirmed': True}
    messages = []
    def unavailable(region, variant):
        raise OSError('model unavailable')
    result = filter_candidates({}, [r], progress=messages.append, evaluator=unavailable)
    assert result[0]['context_keep'] is True and result[0]['enabled'] is True
    assert result[0]['context_reason'] == 'filter_unavailable'
    assert result[0]['context']['status'] == 'unavailable'
    assert any('不可用' in m['message'] for m in messages)


def test_context_cancel_is_not_misreported_as_unavailable():
    from panoprivacy.common import Cancelled, new_region
    from panoprivacy.person_context import filter_candidates
    def cancelled(region, variant):
        raise Cancelled('cancel')
    with pytest.raises(Cancelled):
        filter_candidates({}, [new_region('erp', [.1, .2, .2, .3], origin='auto')], evaluator=cancelled)


@pytest.mark.parametrize('pose', [{'yaw':179,'pitch':-30,'fov':85,'roll':-10},
                                 {'yaw':40,'pitch':85,'fov':100,'roll':20}])
def test_bounded_native_sampler_is_pixel_equal_at_seam_and_pole(tmp_path, pose):
    from PIL import Image
    from panoprivacy.photo_engine import prepare_photo, photo_view
    from panoprivacy.person_context import bounded_photo_view
    rgb = np.random.default_rng(8).integers(0, 256, (128, 256, 3), np.uint8)
    path = tmp_path / 'pano.jpg'
    Image.fromarray(rgb).save(path, quality=98)
    asset = prepare_photo(path, tmp_path / 'cache')
    expected = photo_view(asset, pose, (96, 64))
    assert np.array_equal(bounded_photo_view(asset, pose, (96, 64), block=16), expected)


def test_memory_admission_unknown_stays_serial_and_keeps_reserve():
    from panoprivacy.adaptive_detector import ResourceSnapshot, admit_workers
    unknown = ResourceSnapshot(64*2**30, 32*2**30)
    assert admit_workers(unknown, 512*2**20, 200*2**20) == 1
    rich = ResourceSnapshot(64*2**30, 32*2**30, 16*2**30, 12*2**30, 'device')
    assert admit_workers(rich, 2*2**30, 200*2**20) == 4
    tight = ResourceSnapshot(64*2**30, 2*2**30, 16*2**30, 12*2**30, 'device')
    assert admit_workers(tight, 512*2**20, 200*2**20) == 1


def test_profile_rejects_output_change_and_gain_below_ten_percent():
    from panoprivacy.adaptive_detector import select_calibration
    observations = [{'workers':1,'throughput':10,'equivalent':True},
                    {'workers':2,'throughput':14,'equivalent':True},
                    {'workers':4,'throughput':15,'equivalent':True},
                    {'workers':8,'throughput':30,'equivalent':True}]
    assert select_calibration(observations) == 2
    observations[1]['equivalent'] = False
    assert select_calibration(observations) == 1


def test_pool_order_queue_bound_and_independent_detectors():
    from panoprivacy.adaptive_detector import BoundedDetectorPool
    class Detector:
        provider = 'CUDAExecutionProvider'
        def __init__(self):
            self.lock = threading.Lock()
        def detect(self, value, min_score=.6):
            assert self.lock.acquire(blocking=False), 'concurrent use of one detector'
            try:
                time.sleep(.001 * (3 - value % 3))
                return value * 2
            finally:
                self.lock.release()
    produced, consumed = [0], [0]
    def tasks():
        for i in range(20):
            produced[0] += 1
            assert produced[0] - consumed[0] <= 4
            yield i
    with BoundedDetectorPool([Detector(), Detector()]) as pool:
        result = []
        for _, value in pool.map(tasks(), lambda i: i):
            result.append(value)
            consumed[0] += 1
    assert result == list(range(0, 40, 2))


def test_pool_cancel_stops_feeding_and_joins_workers():
    from panoprivacy.adaptive_detector import BoundedDetectorPool
    from panoprivacy.common import Cancelled
    cancel = threading.Event()
    active = [0]
    class Detector:
        def detect(self, value, min_score=.6):
            active[0] += 1
            try:
                cancel.set()
                time.sleep(.01)
                return value
            finally:
                active[0] -= 1
    with pytest.raises(Cancelled):
        with BoundedDetectorPool([Detector(), Detector()], cancel=cancel) as pool:
            list(pool.map(range(100), lambda i: i))
    assert active[0] == 0


def test_pool_pressure_shrinks_without_dropping_or_duplicating_tasks():
    from panoprivacy.adaptive_detector import BoundedDetectorPool
    class Detector:
        def detect(self, value, min_score=.6):
            return value
    with BoundedDetectorPool([Detector() for _ in range(4)], pressure=lambda: True) as pool:
        result = list(pool.map(range(13), lambda i: i))
        assert pool.workers == 1
    assert result == [(i, i) for i in range(13)]


def test_pool_retries_only_oom_tasks_and_keeps_complete_output():
    from panoprivacy.adaptive_detector import BoundedDetectorPool
    attempts = {}
    lock = threading.Lock()
    class Detector:
        def detect(self, value, min_score=.6):
            with lock:
                attempts[value] = attempts.get(value, 0) + 1
                count = attempts[value]
            if value == 2 and count == 1:
                raise MemoryError('CUDA out of memory')
            return value*3
    with BoundedDetectorPool([Detector() for _ in range(4)]) as pool:
        result = list(pool.map(range(11),lambda i:i))
        assert pool.workers == 2
    assert result == [(i,i*3) for i in range(11)]
    assert attempts[2] == 2
    assert all(count == 1 for key,count in attempts.items() if key != 2)


def test_pool_does_not_hide_nonallocation_failure():
    from panoprivacy.adaptive_detector import BoundedDetectorPool
    class Detector:
        def detect(self, value, min_score=.6):
            raise ValueError('corrupted model output')
    with pytest.raises(ValueError,match='corrupted'):
        with BoundedDetectorPool([Detector(),Detector()]) as pool:
            list(pool.map([0],lambda i:i))


def test_calibration_repeated_previous_level_failure_returns_serial(tmp_path, monkeypatch):
    """An equivalence failure in the repeated two-worker run revokes it."""
    import json
    from panoprivacy import adaptive_detector as adaptive
    before = adaptive.ResourceSnapshot(64*2**30,40*2**30,16*2**30,12*2**30,'test',2**30)
    after = adaptive.ResourceSnapshot(64*2**30,39*2**30,16*2**30,int(11.5*2**30),'test',2*2**30)
    snapshots = iter([before,after])
    monkeypatch.setattr(adaptive,'resources',lambda **kw:next(snapshots,after))
    monkeypatch.setattr(adaptive,'app_cache_dir',lambda:tmp_path)
    monkeypatch.setattr(adaptive,'_raw_signature',lambda detector,rgb:[np.array([1])])
    phase, calls = [0], [0]
    timings = iter([0,4,4,6,6,7.8,7.8,9.8,9.8,11.6])
    def measured_time():
        if calls[0]%2 == 0:
            phase[0] += 1
        calls[0] += 1
        return next(timings)
    monkeypatch.setattr(adaptive.time,'perf_counter',measured_time)
    class Detector:
        provider = 'CUDAExecutionProvider'
        def detect(self,value,min_score=.6):
            return [[1 if phase[0] == 4 else 0]]
    with adaptive.calibrated_pool(Detector,[0,1,2],lambda value:value) as pool:
        assert pool.workers == 1
    assert json.loads((tmp_path/'photo-detector-profile.json').read_text())['workers'] == 1
