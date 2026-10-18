"""Conservative local appearance tracking. Missing tracks stay missing until a seed."""
import copy
import math
import cv2
import numpy as np


def correction_bounds(seed_frames, changed_frame, count):
    earlier = [x for x in seed_frames if x < changed_frame]
    later = [x for x in seed_frames if x > changed_frame]
    return max(earlier, default=0), min(later, default=count - 1)


def _crop(rgb, box):
    x0, y0, x1, y1 = np.rint(box).astype(int)
    return rgb[max(0, y0):min(rgb.shape[0], y1), max(0, x0):min(rgb.shape[1], x1)].copy()


def match_identity(rgb, template, predicted, threshold=.66):
    """Return only a unique local match; proximity cannot override appearance."""
    x0, y0, x1, y1 = predicted
    width, height = x1 - x0, y1 - y0
    margin = max(12, int(max(width, height) * .85))
    left, top = max(0, int(x0) - margin), max(0, int(y0) - margin)
    right, bottom = min(rgb.shape[1], int(x1) + margin), min(rgb.shape[0], int(y1) + margin)
    search = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    if min(gray.shape) < 6 or gray.std() < 6:
        return None, 'uncertain', 0.
    candidates = []
    for scale in (.94, 1., 1.06):
        w, h = max(6, round(width * scale)), max(6, round(height * scale))
        if w > search.shape[1] or h > search.shape[0]:
            continue
        resized = cv2.resize(gray, (w, h))
        scores = cv2.matchTemplate(search, resized, cv2.TM_CCOEFF_NORMED)
        for _ in range(2):
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            candidates.append((float(score), [left+x, top+y, left+x+w, top+y+h]))
            scores[max(0, y-h//2):y+h//2+1, max(0, x-w//2):x+w//2+1] = -1
    candidates.sort(reverse=True, key=lambda item: item[0])
    if not candidates or candidates[0][0] < threshold:
        return None, 'missing', candidates[0][0] if candidates else 0.
    score, box = candidates[0]
    center = np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])
    for other_score, other in candidates[1:]:
        other_center = np.array([(other[0]+other[2])/2, (other[1]+other[3])/2])
        if np.linalg.norm(other_center-center) > min(width,height)*.6 and other_score > score-.08:
            return None, 'uncertain', score
    return box, 'ok', score


def _rotate_detect(detector, rgb):
    """Detector may see nadir faces sideways; rotate pixels and invert boxes."""
    h, w = rgb.shape[:2]
    result = []
    for k in range(4):
        rotated = np.ascontiguousarray(np.rot90(rgb, k))
        for x0, y0, x1, y1, score in detector.detect(rotated, min_score=.65):
            if k == 0: box = [x0, y0, x1, y1]
            elif k == 1: box = [w-y1, x0, w-y0, x1]
            elif k == 2: box = [w-x1, h-y1, w-x0, h-y0]
            else: box = [y0, h-x1, y1, h-x0]
            result.append(box + [score])
    return result


def unique_associated_face(detections, predicted):
    """Deduplicate rotations, then require a single overlapping local candidate."""
    x0,y0,x1,y1=predicted;w,h=x1-x0,y1-y0;cx,cy=(x0+x1)/2,(y0+y1)/2
    candidates=[]
    for d in sorted(detections,key=lambda row:row[4],reverse=True):
        area=(d[2]-d[0])*(d[3]-d[1])
        overlap=max(0,min(x1,d[2])-max(x0,d[0]))*max(0,min(y1,d[3])-max(y0,d[1]))
        union=area+w*h-overlap
        if not (.15<area/(w*h)<3 and abs((d[0]+d[2])/2-cx)<w*.3 and
                abs((d[1]+d[3])/2-cy)<h*.3 and overlap/max(union,1)>.3): continue
        duplicate=False
        for other in candidates:
            intersection=max(0,min(other[2],d[2])-max(other[0],d[0]))*max(0,min(other[3],d[3])-max(other[1],d[1]))
            union=area+(other[2]-other[0])*(other[3]-other[1])-intersection
            if intersection/max(union,1)>.4: duplicate=True;break
        if not duplicate:candidates.append(d)
    return list(candidates[0][:4]) if len(candidates)==1 else None


def _local_face(detector,rgb,box):
    w,h=box[2]-box[0],box[3]-box[1];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    radius=max(48,min(180,max(w,h)*2))
    left,top=max(0,int(cx-radius)),max(0,int(cy-radius))
    right,bottom=min(rgb.shape[1],int(cx+radius)),min(rgb.shape[0],int(cy+radius))
    detections=_rotate_detect(detector,rgb[top:bottom,left:right])
    mapped=[[x0+left,y0+top,x1+left,y1+top,score] for x0,y0,x1,y1,score in detections]
    return unique_associated_face(mapped,box)


def _angular_area(region):
    from .geometry import view_rays
    x0,y0,x1,y1=region['box'];cx,cy=(x0+x1)/2,(y0+y1)/2
    rays=view_rays(np.array([[x0,cy],[x1,cy],[cx,y0],[cx,y1]]),region['pose'],region['view_size'])
    rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    return float(np.arccos(np.clip(rays[0]@rays[1],-1,1))*np.arccos(np.clip(rays[2]@rays[3],-1,1)))


def _recenter(region, old_pose, view_size):
    """Project previous camera up onto the new tangent plane, including at poles."""
    from .geometry import camera, region_center
    yaw, pitch = region_center(region)
    pose = dict(old_pose, yaw=yaw, pitch=pitch, roll=0.)
    right,up,forward,_=camera(pose)
    previous_up=camera(old_pose)[1]
    transported=previous_up-np.dot(previous_up,forward)*forward
    if np.linalg.norm(transported)<1e-8:
        return old_pose
    transported/=np.linalg.norm(transported)
    pose['roll']=math.degrees(math.atan2(-np.dot(transported,right),np.dot(transported,up)))
    return pose


def track_direction(read_frame, seed, indices, detector=None, cancel=None, progress=None):
    from .common import check_cancel
    from .geometry import render_view, region_center, region_to_view_box, view_box_region
    region = copy.deepcopy(seed['region'])
    yaw, pitch = region_center(region)
    # A view seed already has a valid camera basis. Changing yaw/pitch here while
    # keeping its old roll rotates nadir rectangles and almost doubles their AABB.
    # Keep the original basis; normal projection handles a different view aspect.
    pose = dict(region.get('pose') or {}) if region['space']=='view' else {'yaw':yaw,'pitch':pitch}
    pose.setdefault('fov', 100.)
    pose.setdefault('roll', 0.)
    size = (640, 640)
    box = region_to_view_box(region, pose, size)
    if box is None or box[0]<0 or box[1]<0 or box[2]>size[0] or box[3]>size[1]:
        pose=_recenter(region,pose,size)
        box=region_to_view_box(region,pose,size)
    rgb = render_view(read_frame(seed['frame']), pose, size)
    if box is None or min(box[2]-box[0],box[3]-box[1]) < 8:
        raise ValueError('Target selection is too small or outside view')
    if (box[2]-box[0])*(box[3]-box[1]) > size[0]*size[1]*.3:
        raise ValueError('Target region is too large; select only the face and modest head margin')
    template = _crop(rgb, box)
    reference = template.copy()
    reference_area=_angular_area(view_box_region(box,pose,size))
    head_ratio=(1.,1.)
    if detector is not None:
        face=_local_face(detector,rgb,box)
        if face is not None:
            head_ratio=(np.clip((box[2]-box[0])/(face[2]-face[0]),1,1.5),
                        np.clip((box[3]-box[1])/(face[3]-face[1]),1,1.5))
    lost = False
    identity_uncertain = False
    records = []
    for step, frame in enumerate(indices):
        check_cancel(cancel)
        if lost:
            records.append({'frame':frame,'regions':[],'status':'missing','message':'目标已丢失；请添加校正锚点，未搜索其他人。'})
            if progress: progress(frame)
            continue
        rgb = render_view(read_frame(frame), pose, size)
        candidate, status, score = match_identity(rgb, template, box)
        refined=False
        if candidate is not None:
            crop = _crop(rgb, candidate)
            # Older appearance is confidence evidence, not proof of disappearance.
            comparison = cv2.resize(crop, (reference.shape[1],reference.shape[0]))
            similarity = float(cv2.matchTemplate(cv2.cvtColor(comparison,cv2.COLOR_RGB2GRAY),cv2.cvtColor(reference,cv2.COLOR_RGB2GRAY),cv2.TM_CCOEFF_NORMED)[0,0])
            if similarity < .40:
                identity_uncertain = True
            old_width,old_height=box[2]-box[0],box[3]-box[1]
            width,height=candidate[2]-candidate[0],candidate[3]-candidate[1]
            displacement=np.hypot((candidate[0]+candidate[2]-box[0]-box[2])/2,
                                  (candidate[1]+candidate[3]-box[1]-box[3])/2)
            if (not .65<width*height/(old_width*old_height)<1.55 or
                    displacement>max(old_width,old_height)*.9):
                candidate=None
                status='uncertain'
        if candidate is not None and detector is not None and step % 15 == 14:
            face=_local_face(detector,rgb,candidate)
            if face is None:
                status = 'uncertain'
            else:
                cx,cy=(face[0]+face[2])/2,(face[1]+face[3])/2
                fw,fh=(face[2]-face[0])*head_ratio[0],(face[3]-face[1])*head_ratio[1]
                candidate=[cx-fw/2,cy-fh/2,cx+fw/2,cy+fh/2]
                refined=True
        if candidate is not None and identity_uncertain:
            # Recentring can refresh the template but must not erase a review flag.
            status='uncertain'
        if candidate is None:
            lost = True
            records.append({'frame':frame,'regions':[],'status':status,'message':'目标外观不匹配或存在多个候选；请校正。'})
        else:
            box = candidate
            region = view_box_region(box,pose,size,id=seed['region']['id'],origin='track',score=score)
            area=_angular_area(region)
            if not .2<area/max(reference_area,1e-12)<2.25:
                lost=True
                records.append({'frame':frame,'regions':[],'status':'uncertain','message':'目标区域尺寸异常，可能发生累积漂移；请校正。'})
                if progress:progress(frame)
                continue
            if refined:
                # Validate against the previous reference before accepting a new one.
                reference_area=area
            region['enabled']=True
            records.append({'frame':frame,'regions':[region],'status':status,'message':'' if status=='ok' else '外观或检测置信度不足，请检查对象与完整脸部覆盖。'})
            template = _crop(rgb,box)
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            if max(abs(cx-320),abs(cy-320))>100:
                new_pose=_recenter(region,pose,size)
                if new_pose != pose:
                    pose=new_pose
                    rgb=render_view(read_frame(frame),pose,size)
                    box=region_to_view_box(region,pose,size)
                    if box is None:
                        lost=True
                    else:
                        template=_crop(rgb,box)
                        reference=template.copy()
        if progress: progress(frame)
    return records
