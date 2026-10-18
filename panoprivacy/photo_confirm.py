"""Small CPU face confirmation with explicit association to the RF proposal."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .common import resource_path

MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
THRESHOLD = .8


def associate_candidate(box, detections):
    """A neighbouring face must not confirm a different selected candidate.

    Detections are pixel-edge xyxy plus score. Association requires overlap,
    compatible centres, and the selected centre inside a modest face margin.
    An unconfirmed proposal is not evidence that it is a false positive.
    """
    matches, valid = [], []
    rf = np.asarray(box if box is not None else [], dtype=float)
    if rf.shape != (4,) or not np.isfinite(rf).all() or rf[2] <= rf[0] or rf[3] <= rf[1]:
        return {'confirmed': False, 'status': 'unconfirmed', 'score': 0., 'detections': []}
    cx, cy = (rf[0] + rf[2]) / 2, (rf[1] + rf[3]) / 2
    rw, rh = rf[2] - rf[0], rf[3] - rf[1]
    for detection in detections:
        values = np.asarray(detection, dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all() or values[4] < THRESHOLD:
            continue
        a, b, c, d, score = map(float, values)
        width, height = c - a, d - b
        if min(width, height) <= 0:
            continue
        inter = max(0., min(rf[2], c) - max(rf[0], a)) * max(0., min(rf[3], d) - max(rf[1], b))
        iou = inter / max(1e-9, rw * rh + width * height - inter)
        associated = bool(iou >= .1 and abs((a + c) / 2 - cx) <= rw * .6 and
                          abs((b + d) / 2 - cy) <= rh * .6 and
                          a - .2 * width <= cx <= c + .2 * width and
                          b - .2 * height <= cy <= d + .2 * height)
        valid.append({'box': [a, b, c, d], 'score': score, 'associated': associated})
        if associated:
            matches.append(score)
    return {'confirmed': bool(matches), 'status': 'confirmed' if matches else 'unconfirmed',
            'score': max(matches, default=0.), 'detections': valid}


class CandidateConfirmer:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else resource_path('models/face_detection_yunet_2023mar.onnx')
        if not path.is_file():
            raise FileNotFoundError('缺少二次人脸确认模型，请完整复制软件文件夹。')
        model_bytes = path.read_bytes()
        digest = hashlib.sha256(model_bytes).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError('二次人脸确认模型校验失败，请重新复制完整软件。')
        # OpenCV's Windows ONNX filename loader cannot open Unicode paths.
        # Python reads the verified small model; the buffer overload avoids it.
        self.detector = cv2.FaceDetectorYN.create('onnx', np.frombuffer(model_bytes, np.uint8),
                                                np.empty(0, np.uint8), (320, 320), THRESHOLD, .3, 5000,
                                                cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)

    def confirm(self, rgb, box):
        height, width = rgb.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        detections = []
        for face in [] if faces is None else faces:
            x, y, w, h = map(float, face[:4])
            detections.append([x, y, x + w, y + h, float(face[-1])])
        return associate_candidate(box, detections)
