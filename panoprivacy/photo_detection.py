"""Photo v0.2 scan: ordered native GPU jobs and recoverable person context."""
from __future__ import annotations

import math

from .common import check_cancel, emit, new_region


def scan_photo(asset, mode='obvious', progress=None, cancel=None):
    if mode not in ('obvious', 'more'):
        raise ValueError('识别模式需要为 obvious 或 more。')
    from . import photo_engine as engine
    from .detector import FaceDetector
    from .adaptive_detector import calibrated_pool
    from .person_context import filter_candidates, memory_safe_view
    check_cancel(cancel)
    image = engine._checked(asset)
    width, height = image.width, image.height
    threshold = .6 if mode == 'obvious' else .4
    xx = engine._starts(width, engine.DETECT_TILE, engine.DETECT_STRIDE)
    if width > engine.DETECT_TILE:
        xx.append(width-engine.DETECT_TILE//2)
    yy = engine._starts(height, engine.DETECT_TILE, engine.DETECT_STRIDE)
    views = [(yaw,pitch) for pitch in (-60,0,60) for yaw in range(-180,180,60)] + [(0,-90),(0,90)]
    native_count = len(xx)*len(yy)
    view_starts = engine._starts(1536,engine.DETECT_TILE,engine.DETECT_STRIDE)
    view_tile_count = len(view_starts)**2
    total = native_count+len(views)*view_tile_count

    def render(spec):
        if spec[0] == 'native':
            return engine._native_tile(image,spec[1],spec[2])[0]
        return spec[1][spec[4]:spec[4]+engine.DETECT_TILE,spec[3]:spec[3]+engine.DETECT_TILE]

    # Fixed spatial coverage, independent of candidate results; include a broad
    # native sample while keeping the calibration queue bounded.
    representative = [('native',xx[(i*7)%len(xx)],yy[(i*11)%len(yy)]) for i in range(24)]
    working_set = width*height*12+1536*1536*64
    pool = calibrated_pool(FaceDetector,representative,render,threshold=threshold,
                           cancel=cancel,progress=progress,working_set=working_set)

    def specs():
        for y in yy:
            for x in xx:
                check_cancel(cancel)
                yield ('native',x,y)
        for yaw,pitch in views:
            check_cancel(cancel)
            pose = {'yaw':yaw,'pitch':pitch,'fov':85,'roll':0}
            view = memory_safe_view(asset,pose,(1536,1536),cancel)
            for vy in view_starts:
                for vx in view_starts:
                    yield ('view',view,pose,vx,vy)

    regions = []
    with pool:
        for done,(spec,detections) in enumerate(pool.map(specs(),render,threshold),1):
            check_cancel(cancel)
            for x0,y0,x1,y1,score in detections:
                if score < threshold:
                    continue
                if spec[0] == 'native':
                    _,x,y = spec
                    w,h = min(engine.DETECT_TILE,width),min(engine.DETECT_TILE,height-y)
                    candidate = engine._candidate((x0,y0,x1,y1),score,mode,(w,h))
                    if candidate is None:
                        continue
                    box,fields = candidate
                    regions.append(new_region('erp',[(box[0]+x)/width,(box[1]+y)/height,
                                                     (box[2]+x)/width,(box[3]+y)/height],**fields))
                else:
                    _,_,pose,vx,vy = spec
                    candidate = engine._candidate((x0,y0,x1,y1),score,mode,(engine.DETECT_TILE,engine.DETECT_TILE))
                    if candidate is None:
                        continue
                    box,fields = candidate
                    native_per_pixel = width*math.radians(pose['fov'])/(2*math.pi*1536)
                    if min(box[2]-box[0],box[3]-box[1])*native_per_pixel < (24 if mode=='obvious' else 16):
                        continue
                    regions.append(new_region('view',[(box[0]+vx)/1536,(box[1]+vy)/1536,
                                                      (box[2]+vx)/1536,(box[3]+vy)/1536],
                                              pose=pose,view_size=(1536,1536),**fields))
            emit(progress,'photo_scan',done/total,f'原生照片识别 {done}/{total}，候选 {len(regions)}')
    image = None
    engine._checked(asset)
    result = engine._deduplicate(regions,cancel)
    if mode == 'obvious':
        result = engine.confirm_photo_candidates(asset,result,progress,cancel)
    result = filter_candidates(asset,result,progress,cancel)
    engine._checked(asset)
    visible = sum(r.get('context_keep') is not False for r in result)
    unavailable = any(r.get('context_reason') == 'filter_unavailable' for r in result)
    message = (f'识别完成：显示 {visible} 个候选，保留 {len(result)-visible} 个背景候选供恢复；请逐一检查'
               if not unavailable else f'识别完成，人物背景筛选不可用：{visible} 个可见候选需要人工检查')
    emit(progress,'photo_scan',1,message)
    return result
