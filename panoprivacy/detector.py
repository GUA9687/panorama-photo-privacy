"""Pinned RF-DETR face adapter; CUDA with explicit, observable CPU fallback."""
from __future__ import annotations
import hashlib
import os
import threading
from pathlib import Path
import cv2
import numpy as np
from .common import resource_path

EXPECTED_SHA256='b594500a99c3a4287805b7afa15eb74ee559be937eb00315768612846f93cc45'
_DLL_HANDLES=[]
_VERIFIED=set()
_MEAN=np.array([.485,.456,.406],np.float32)[:,None,None]
_STD=np.array([.229,.224,.225],np.float32)[:,None,None]


def prepare_runtime():
    folder=resource_path('runtime/cuda')
    if folder.is_dir() and hasattr(os,'add_dll_directory'):
        _DLL_HANDLES.append(os.add_dll_directory(str(folder)))
        os.environ['PATH']=str(folder)+os.pathsep+os.environ.get('PATH','')
    import onnxruntime as ort
    if folder.is_dir() and hasattr(ort,'preload_dlls'):
        ort.preload_dlls(directory=str(folder))
    return ort


class FaceDetector:
    def __init__(self, model_path=None, prefer_gpu=True):
        path=Path(model_path) if model_path else resource_path('models/rf-detr-medium-face.onnx')
        if not path.is_file():raise FileNotFoundError('缺少人脸模型，请完整复制软件文件夹。')
        stat=path.stat();key=(str(path.resolve()),stat.st_size,stat.st_mtime_ns)
        if key not in _VERIFIED:
            with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=EXPECTED_SHA256:raise ValueError('模型文件校验失败，请重新复制完整模型。')
            _VERIFIED.add(key)
        ort=prepare_runtime()
        options=ort.SessionOptions();options.intra_op_num_threads=4
        providers=['CPUExecutionProvider']
        if prefer_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers=[('CUDAExecutionProvider',{'gpu_mem_limit':2*1024**3,'arena_extend_strategy':'kSameAsRequested'}),'CPUExecutionProvider']
        self.warning=''
        try:self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        except Exception as error:
            if not prefer_gpu:raise
            self.warning='显卡运行时不可用，已改用CPU。'
            self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.provider=self.session.get_providers()[0]
        if self.provider!='CUDAExecutionProvider' and prefer_gpu:self.warning='当前使用CPU处理；更新NVIDIA驱动后可尝试显卡加速。'
        self._lock=threading.Lock()
        if self.session.get_inputs()[0].shape!=[1,3,576,576]:raise ValueError('模型输入规格不匹配')

    def detect(self,rgb,min_score=.6):
        if not 0<=float(min_score)<=1:raise ValueError('检测阈值无效')
        h,w=rgb.shape[:2]
        if h<1 or w<1:return []
        resized=cv2.resize(rgb,(576,576),interpolation=cv2.INTER_LINEAR)
        tensor=(resized.astype(np.float32).transpose(2,0,1)/255.-_MEAN)/_STD
        with self._lock:
            dets,labels=self.session.run(['dets','labels'],{self.session.get_inputs()[0].name:np.ascontiguousarray(tensor[None])})
        if dets.shape!=(1,300,4) or labels.shape!=(1,300,1) or not np.isfinite(dets).all() or not np.isfinite(labels).all():
            raise ValueError('模型返回无效检测结果')
        scores=1/(1+np.exp(-np.clip(labels[0,:,0],-80,80)))
        result=[]
        for i in np.argsort(-scores):
            score=float(scores[i])
            if score<min_score:break
            cx,cy,bw,bh=map(float,dets[0,i])
            x0,y0=max(0.,(cx-bw/2)*w),max(0.,(cy-bh/2)*h)
            x1,y1=min(float(w),(cx+bw/2)*w),min(float(h),(cy+bh/2)*h)
            if x1>x0 and y1>y0:result.append([x0,y0,x1,y1,score])
        return result
