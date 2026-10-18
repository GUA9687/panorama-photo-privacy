"""Compact ONNX person context with recoverable, conservative review decisions.

The A/B camera and association thresholds are frozen from the photo validation.
They filter proposals; they do not establish that a subject is a real person.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .common import Cancelled, check_cancel, emit, resource_path
from .geometry import region_center, region_to_view_box

MODEL_SHA256 = '757a583140b8736cbb2a59f8d2c9e6ae015c45a2bbe77baaa2ceec13f284f88b'


def associated_person(candidate, person):
    if candidate is None:
        return False
    a = np.asarray([*candidate, *person], dtype=float)
    if a.shape != (8,) or not np.isfinite(a).all():
        return False
    x0, y0, x1, y1, px0, py0, px1, py1 = a
    w, h, cw, ch = px1 - px0, py1 - py0, x1 - x0, y1 - y0
    if min(w, h, cw, ch) <= 0:
        return False
    overlap = max(0, min(x1, px1) - max(x0, px0)) * max(0, min(y1, py1) - max(y0, py0)) / (cw * ch)
    rx, ry = ((x0 + x1) / 2 - px0) / w, ((y0 + y1) / 2 - py0) / h
    return bool(overlap >= .6 and -.05 <= rx <= 1.05 and -.08 <= ry <= .42 and ch / h <= .65)


def context_pose(region, variant='A'):
    yaw, pitch = region_center(region)
    x0, y0, x1, y1 = region['box']
    if region['space'] == 'erp':
        angle = max((x1-x0)*360*max(.1, math.cos(math.radians(pitch))), (y1-y0)*180)
    else:
        angle = max(x1-x0, (y1-y0)*region['view_size'][1]/region['view_size'][0])*region['pose']['fov']
    scale, minimum, shift = {'A': (9, 3, 2.5), 'B': (14, 5, 4)}[variant]
    return {'yaw': yaw, 'pitch': max(-89, min(89, pitch-angle*shift)),
            'fov': min(100, max(minimum, angle*scale)), 'roll': 0}


def decode_person_output(output):
    """YOLOv8 one-to-many output; same best-class then class-0 NMS as prototype."""
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0:2] != (1, 84) or not np.isfinite(values).all():
        raise ValueError('人物模型返回无效结果。')
    rows = values[0].T
    scores = rows[:, 4:].max(axis=1)
    selected = np.flatnonzero((rows[:, 4:].argmax(axis=1) == 0) & (scores > .05))
    selected = selected[np.argsort(-scores[selected], kind='stable')][:30000]
    boxes = np.empty((len(rows), 4), np.float32)
    boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
    boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    detections = []
    while len(selected) and len(detections) < 300:
        index = selected[0]
        box = boxes[index]
        detections.append({'box': np.clip(box, 0, 640).tolist(), 'score': float(scores[index])})
        selected = selected[1:]
        others = boxes[selected]
        inter = np.maximum(0, np.minimum(box[2:], others[:, 2:]) - np.maximum(box[:2], others[:, :2])).prod(axis=1)
        area = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
        other_area = np.maximum(0, others[:, 2:]-others[:, :2]).prod(axis=1)
        selected = selected[inter / np.maximum(area + other_area - inter, 1e-9) <= .5]
    return detections


class PersonDetector:
    def __init__(self, model_path=None, prefer_gpu=True):
        from .detector import prepare_runtime
        path = Path(model_path) if model_path else resource_path('models/yolov8n-person.onnx')
        if not path.is_file():
            raise FileNotFoundError('缺少人物背景筛选模型。')
        data = path.read_bytes()  # bytes also support Windows Unicode model paths
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise ValueError('人物背景筛选模型校验失败。')
        ort = prepare_runtime()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        providers = ['CPUExecutionProvider']
        if prefer_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = [('CUDAExecutionProvider', {'gpu_mem_limit': 512*2**20,
                          'arena_extend_strategy': 'kSameAsRequested'}), 'CPUExecutionProvider']
        self.warning = ''
        try:
            self.session = ort.InferenceSession(data, sess_options=options, providers=providers)
        except Exception:
            if not prefer_gpu:
                raise
            self.warning = '人物筛选显卡运行时不可用，已改用 CPU。'
            self.session = ort.InferenceSession(data, sess_options=options, providers=['CPUExecutionProvider'])
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError('人物筛选模型规格不匹配。')

    def detect(self, rgb):
        if rgb.shape != (640, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError('人物上下文视图应为 640×640 RGB。')
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        return decode_person_output(self.session.run(None, {self.input_name: tensor})[0])


def bounded_photo_view(asset, pose, out_size=(640, 640), block=64, cancel=None):
    """Evaluate the identical native bilinear camera in bounded output blocks."""
    import pyvips
    from .photo_engine import _checked, _array
    width, height = map(int, out_size)
    if not 1 <= width <= 4096 or not 1 <= height <= 4096 or not 1 <= block <= 256:
        raise ValueError('视图尺寸或分块尺寸无效。')
    yaw, pitch, fov, roll = [float(pose.get(k, d)) for k, d in (('yaw',0),('pitch',0),('fov',90),('roll',0))]
    if not np.isfinite([yaw,pitch,fov,roll]).all() or not -90 <= pitch <= 90 or not 1 <= fov < 175:
        raise ValueError('视图方向或视野角度无效。')
    check_cancel(cancel)
    image = _checked(asset)
    image = None
    yaw, pitch, roll = np.deg2rad([yaw,pitch,roll])
    sy, cy, sp, cp = np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)
    right = np.array([cy,0,-sy]); up = np.array([-sy*sp,cp,-cy*sp]); forward = np.array([sy*cp,sp,cy*cp])
    old_right = right.copy()
    right = old_right*np.cos(roll)+up*np.sin(roll); up = up*np.cos(roll)-old_right*np.sin(roll)
    focal = width / (2*math.tan(math.radians(fov)/2))
    xx = (np.arange(width,dtype=np.float32)[None,:]+.5-width/2)/focal
    yy = (height/2-np.arange(height,dtype=np.float32)[:,None]-.5)/focal
    wx = forward[0]+xx*right[0]+yy*up[0]
    wy = forward[1]+xx*right[1]+yy*up[1]
    wz = forward[2]+xx*right[2]+yy*up[2]
    longitude = np.arctan2(wx,wz); latitude = np.arctan2(wy,np.hypot(wx,wz))
    mapping = np.empty((height,width,2),dtype=np.float32)
    mapping[:,:,0] = ((longitude/(2*np.pi)+.5)*asset['width']-.5)%asset['width']+asset['width']
    mapping[:,:,1] = np.clip((.5-latitude/np.pi)*asset['height']-.5,0,asset['height']-1)
    result = np.empty((height,width,3),dtype=np.uint8)
    for y in range(0,height,block):
        for x in range(0,width,block):
            check_cancel(cancel)
            m = np.ascontiguousarray(mapping[y:y+block,x:x+block]); h,w = m.shape[:2]
            image = pyvips.Image.new_from_file(str(asset['native_path']), access='random')
            index = pyvips.Image.new_from_memory(m.data,w,h,2,'float')
            output = image.replicate(3,1).mapim(index,interpolate=pyvips.Interpolate.new('bilinear'),extend='copy')
            result[y:y+h,x:x+w] = _array(output)
            output = index = image = None
    _checked(asset)
    return result


def memory_safe_view(asset, pose, out_size, cancel=None):
    from .adaptive_detector import resources, ram_reserve
    from .photo_engine import photo_view
    live = resources(include_gpu=False)
    # mapim can request a wide wrapped source ROI. Estimate conservatively from
    # source dimensions, including intermediates; never reduce source resolution.
    required = int(asset['width'])*int(asset['height'])*3*4 + out_size[0]*out_size[1]*64
    if live.ram_free is None or live.ram_total is None or live.ram_free - ram_reserve(live) < required:
        return bounded_photo_view(asset, pose, out_size, cancel=cancel)
    check_cancel(cancel)
    result = photo_view(asset, pose, out_size)
    check_cancel(cancel)
    return result


def filter_candidates(asset, regions, progress=None, cancel=None, evaluator=None):
    """Return every proposal; rejected automatic entries remain recoverable."""
    check_cancel(cancel)
    failure = None
    if evaluator is None:
        try:
            detector = PersonDetector()
            emit(progress, 'photo_context', 0, '人物背景筛选设备：'+detector.provider + ('；'+detector.warning if detector.warning else ''))
            def evaluator(region, variant):
                pose = context_pose(region, variant)
                rgb = memory_safe_view(asset, pose, (640,640), cancel)
                box = region_to_view_box(region, pose, (640,640))
                detections = detector.detect(rgb)
                score = max((d['score'] for d in detections if associated_person(box,d['box'])), default=0.)
                return {'keep': score >= .25, 'score': score, 'provider': detector.provider}
        except Exception as error:
            failure = str(error)
    result = []
    for i, region in enumerate(regions):
        check_cancel(cancel)
        candidate = dict(region)
        if region.get('origin') == 'manual':
            candidate.update(context_keep=True, context_reason='manual')
        else:
            try:
                if failure is not None:
                    raise RuntimeError(failure)
                a = evaluator(region, 'A')
                b = None
                if a['keep'] and not region.get('confirmation', {}).get('confirmed', False):
                    b = evaluator(region, 'B')
                keep = bool(a['keep'] and (b is None or b['keep']))
                candidate.update(context_keep=keep, context_reason='person_context' if keep else 'background_context')
                candidate['context'] = {'status': 'checked', 'policy': 'A_then_unconfirmed_B', 'A': a, 'B': b}
                if not keep:
                    candidate['enabled'] = False
            except Cancelled:
                raise
            except Exception as error:
                # A failed filter is not negative evidence. Keep the proposal
                # visible and preserve its existing enabled/review choice.
                failure = str(error)
                candidate.update(context_keep=True, context_reason='filter_unavailable')
                candidate['context'] = {'status': 'unavailable', 'message': failure}
                emit(progress, 'photo_context', i/max(1,len(regions)), '人物背景筛选不可用，候选已保留供人工检查：'+failure)
        result.append(candidate)
        emit(progress, 'photo_context', (i+1)/max(1,len(regions)), f'检查人物上下文 {i+1}/{len(regions)}')
    return result
