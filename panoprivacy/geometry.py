"""Pixel-center-consistent spherical views and bounded, seam-safe region masks."""
from __future__ import annotations

import math
import cv2
import numpy as np
from .common import default_style, new_region, validate_style


def camera(pose):
    yaw, pitch, roll = np.deg2rad([pose.get("yaw", 0), pose.get("pitch", 0), pose.get("roll", 0)])
    fov = float(pose.get("fov", 90))
    if not np.isfinite([yaw, pitch, roll, fov]).all() or not 1 <= fov < 175 or not -np.pi/2-1e-10 <= pitch <= np.pi/2+1e-10:
        raise ValueError("视角参数无效")
    right = np.array([np.cos(yaw), 0., -np.sin(yaw)])
    up = np.array([-np.sin(yaw)*np.sin(pitch), np.cos(pitch), -np.cos(yaw)*np.sin(pitch)])
    forward = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
    return right*np.cos(roll)+up*np.sin(roll), up*np.cos(roll)-right*np.sin(roll), forward, np.tan(np.deg2rad(fov)/2)


def view_rays(uv, pose, view_size):
    width, height = view_size
    right, up, forward, tangent = camera(pose)
    uv = np.asarray(uv, np.float64)
    return forward + (uv[..., 0:1]-.5)*(2*tangent)*right - (uv[..., 1:2]-.5)*(2*tangent*height/width)*up


def project_vectors(vectors, pose, view_size):
    right, up, forward, tangent = camera(pose)
    vectors = np.asarray(vectors, np.float64)
    depth = vectors @ forward
    safe = np.where(depth > 1e-9, depth, np.nan)
    width, height = view_size
    return np.stack([(.5 + (vectors @ right)/(2*tangent*safe))*width,
                     (.5 - (vectors @ up)*width/(2*tangent*safe*height))*height], axis=-1), depth


def _expanded_box(region, padding):
    x0, y0, x1, y1 = map(float, region["box"])
    dx, dy = (x1-x0)*padding, (y1-y0)*padding
    return [x0-dx, y0-dy, x1+dx, y1+dy]


def _region_vectors(region, padding=0):
    x0, y0, x1, y1 = _expanded_box(region, padding)
    uv = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    if region["space"] == "view":
        return view_rays(uv, region["pose"], region["view_size"])
    lon, lat = (uv[:, 0]-.5)*2*np.pi, (.5-np.clip(uv[:, 1],0,1))*np.pi
    return np.stack([np.cos(lat)*np.sin(lon), np.sin(lat), np.cos(lat)*np.cos(lon)], axis=-1)


def region_center(region):
    box = region["box"]
    uv = [(box[0]+box[2])/2, (box[1]+box[3])/2]
    if region["space"] == "erp":
        return (uv[0]*360-180+180)%360-180, 90-uv[1]*180
    ray = view_rays(np.asarray(uv), region["pose"], region["view_size"])
    return float(np.rad2deg(np.arctan2(ray[0], ray[2]))), float(np.rad2deg(np.arctan2(ray[1], np.hypot(ray[0],ray[2]))))


def view_box_region(box_pixels, pose, view_size, **fields):
    width, height = view_size
    box = np.asarray(box_pixels, np.float64)/[width,height,width,height]
    return new_region("view", box, pose, view_size, **fields)


def region_to_view_box(region, pose, view_size):
    pixels, depth = project_vectors(_region_vectors(region), pose, view_size)
    if not np.isfinite(pixels).all() or np.any(depth <= 1e-9):
        return None
    return [float(pixels[:,0].min()), float(pixels[:,1].min()), float(pixels[:,0].max()), float(pixels[:,1].max())]


def render_view(erp_rgb, pose, out_size=(960,640)):
    width, height = map(int, out_size)
    if width <= 0 or height <= 0 or width*height > 20_000_000:
        raise ValueError("预览尺寸无效或过大")
    h, w = erp_rgb.shape[:2]
    if h < 2 or w < 2:
        raise ValueError("全景预览为空")
    right, up, forward, tangent = camera(pose)
    xx = ((np.arange(width, dtype=np.float32)+.5)/width-.5)[None,:]*(2*tangent)
    yy = (.5-(np.arange(height, dtype=np.float32)+.5)/height)[:,None]*(2*tangent*height/width)
    wx = forward[0]+xx*right[0]+yy*up[0]
    wy = forward[1]+xx*right[1]+yy*up[1]
    wz = forward[2]+xx*right[2]+yy*up[2]
    lon = np.arctan2(wx,wz); lat=np.arctan2(wy,np.hypot(wx,wz))
    mx = np.remainder((lon/(2*np.pi)+.5)*w,w)-.5
    my = np.clip((.5-lat/np.pi)*h-.5,0,h-1)
    padded = np.concatenate([erp_rgb[:,-1:], erp_rgb, erp_rgb[:,:1]],axis=1)
    return cv2.remap(padded, (mx+1).astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _x_ranges(start, span, width):
    if span >= 1-1e-12:
        return [(0,width)]
    start = (start % 1)*width
    end = start+span*width
    ranges=[(max(0,int(math.floor(start))-1), min(width,int(math.ceil(end))+1))]
    if end > width:
        ranges.append((0,min(width,int(math.ceil(end-width))+1)))
    elif start < 1:
        ranges.append((max(0,width-1),width))
    return ranges


def _view_bounds(region, box):
    x0,y0,x1,y1=box
    rays=view_rays(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]),region["pose"],region["view_size"])
    samples=[]
    for a,b_end in zip(rays,np.roll(rays,-1,axis=0)):
        delta=b_end-a
        aa=float(a@a);ab=float(a@delta);bb=float(delta@delta)
        samples.extend(a+np.linspace(0,1,17,endpoint=False)[:,None]*delta)
        denom=delta[1]*ab-a[1]*bb
        if abs(denom)>1e-15:
            t=(a[1]*ab-delta[1]*aa)/denom
            if 0<t<1:samples.append(a+t*delta)
    pts=np.asarray(samples)
    lat=np.arctan2(pts[:,1],np.hypot(pts[:,0],pts[:,2]))
    # A hemisphere-contained camera rectangle may enclose a pole; longitude
    # then spans the whole panorama, while latitude still has finite bounds.
    pole=0
    for sign in (-1,1):
        pixels,depth=project_vectors(np.array([[0.,float(sign),0.]]),region["pose"],region["view_size"])
        if depth[0]>1e-9:
            u,v=pixels[0]/np.asarray(region["view_size"])
            if x0-1e-12<=u<=x1+1e-12 and y0-1e-12<=v<=y1+1e-12:pole=sign
    lo,hi=float(lat.min()),float(lat.max())
    if pole<0:lo=-np.pi/2
    if pole>0:hi=np.pi/2
    if pole:return 0.,1.,.5-hi/np.pi,.5-lo/np.pi
    # Perimeter ordering is required by unwrap; latitude extrema appended
    # above may interrupt it, so use edge samples separately for longitude.
    edge=np.concatenate([a+np.linspace(0,1,33,endpoint=False)[:,None]*(b-a) for a,b in zip(rays,np.roll(rays,-1,axis=0))])
    lon=np.unwrap(np.arctan2(edge[:,0],edge[:,2]))
    return float(lon.min()/(2*np.pi)+.5), min(1.,float(np.ptp(lon)/(2*np.pi))), .5-hi/np.pi, .5-lo/np.pi


def _shape_frame(region, padding, shape):
    """Return the shape's coordinate frame and padded bounding box.

    Small ERP selections use a local rectilinear frame so a circle remains
    visually round, including at the longitude seam. Very large ERP selections
    spanning a camera horizon retain an angular frame rather than dropping rays.
    """
    if shape not in {'rect', 'ellipse', 'circle'}:
        raise ValueError('未知遮挡形状')
    frame = region
    if shape != 'rect' and region['space'] == 'erp':
        yaw, pitch = region_center(region)
        pose = {'yaw': yaw, 'pitch': pitch, 'fov': 90., 'roll': 0.}
        x0,y0,x1,y1 = region['box']
        t = np.linspace(0.,1.,65)
        uv = np.concatenate([np.column_stack((x0+(x1-x0)*t,np.full_like(t,y0))),
                             np.column_stack((x0+(x1-x0)*t,np.full_like(t,y1))),
                             np.column_stack((np.full_like(t,x0),y0+(y1-y0)*t)),
                             np.column_stack((np.full_like(t,x1),y0+(y1-y0)*t))])
        lon,lat = (uv[:,0]-.5)*2*np.pi,(.5-uv[:,1])*np.pi
        rays = np.stack((np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)),axis=-1)
        pixels,depth = project_vectors(rays,pose,(1,1))
        if np.all(depth > .05) and x1-x0 < .5:
            frame = {**region,'space':'view','pose':pose,'view_size':[1,1],
                     'box':[float(pixels[:,0].min()),float(pixels[:,1].min()),
                            float(pixels[:,0].max()),float(pixels[:,1].max())]}
    box = _expanded_box(frame,padding)
    if shape == 'circle':
        sx,sy = frame['view_size'] if frame['space']=='view' else (2.,1.)
        cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
        radius = max((box[2]-box[0])*sx,(box[3]-box[1])*sy)/2
        box = [cx-radius/sx,cy-radius/sy,cx+radius/sx,cy+radius/sy]
    return frame,box


def _membership_prepared(frame, box, rays, shape):
    if frame['space']=='erp':
        lon=np.arctan2(rays[...,0],rays[...,2])
        lat=np.arctan2(rays[...,1],np.hypot(rays[...,0],rays[...,2]))
        u,v=lon/(2*np.pi)+.5,.5-lat/np.pi
        if shape=='rect':
            return (np.mod(u-box[0],1)<min(1.,box[2]-box[0])) & (v>=box[1]) & (v<box[3])
        cx=(box[0]+box[2])/2
        u=cx+np.remainder(u-cx+.5,1.)-.5
        front=np.ones(u.shape,bool)
    else:
        pixels,depth=project_vectors(rays,frame['pose'],frame['view_size'])
        u,v=pixels[...,0]/frame['view_size'][0],pixels[...,1]/frame['view_size'][1]
        front=depth>1e-9
    if shape=='rect':
        return front & (u>=box[0]) & (u<box[2]) & (v>=box[1]) & (v<box[3])
    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    rx,ry=(box[2]-box[0])/2,(box[3]-box[1])/2
    return front & (((u-cx)/rx)**2+((v-cy)/ry)**2 <= 1.+1e-12)


def region_membership(region, rays, padding=.25, shape='rect'):
    """Evaluate one region on arbitrary spherical rays; shared by every preview."""
    rays=np.asarray(rays,dtype=np.float64)
    if not region.get('enabled',True):return np.zeros(rays.shape[:-1],bool)
    if not np.isfinite(padding) or not 0<=padding<=1:raise ValueError('保护边缘无效')
    frame,box=_shape_frame(region,padding,shape)
    return _membership_prepared(frame,box,rays,shape)


def region_mask_tiles(region, pano_size, tile_size=256, padding=.25, shape='rect'):
    """Yield unique grid-aligned occupied masks, never a full-size panorama mask."""
    width,height=map(int,pano_size)
    if width<=0 or height<=0 or tile_size<1 or not np.isfinite(padding) or not 0<=padding<=1:
        raise ValueError('遮罩尺寸或保护边缘无效')
    if not region.get('enabled',True):return
    frame,box=_shape_frame(region,padding,shape)
    if frame['space']=='erp':
        xmin=box[0];span=min(1.,box[2]-box[0]);ymin=max(0.,box[1]);ymax=min(1.,box[3])
    else:
        xmin,span,ymin,ymax=_view_bounds(frame,box)
    low=max(0,int(math.floor(ymin*height))-1);high=min(height,int(math.ceil(ymax*height))+1)
    if high<=low:return
    seen=set()
    for xlow,xhigh in _x_ranges(xmin,span,width):
        for y in range((low//tile_size)*tile_size,high,tile_size):
            for x in range((xlow//tile_size)*tile_size,xhigh,tile_size):
                if (x,y) in seen:continue
                seen.add((x,y))
                th,tw=min(tile_size,height-y),min(tile_size,width-x)
                u=(x+np.arange(tw,dtype=np.float64)+.5)[None,:]/width
                v=(y+np.arange(th,dtype=np.float64)+.5)[:,None]/height
                lon=(u-.5)*2*np.pi;lat=(.5-v)*np.pi
                rays=np.stack([np.cos(lat)*np.sin(lon),np.broadcast_to(np.sin(lat),(th,tw)),
                               np.cos(lat)*np.cos(lon)],axis=-1)
                mask=_membership_prepared(frame,box,rays,shape)
                if mask.any():yield x,y,np.ascontiguousarray(mask)


def privacy_style_parameters(bounds, style, scale=1):
    """Region-relative privacy strength; scale is pixels per luma/source pixel."""
    x0,y0,x1,y1=bounds
    strength=float(style.get('strength',1.))
    shortest=max(1,min(x1-x0,y1-y0))
    return {'sigma':max(4.,shortest*.18*strength)*scale,
            'cells':max(2,round(6/strength))}


def gaussian_privacy_filter(patch, sigma, wrap_x=False):
    """One shared blur definition, with bounded coarse-grid sigma for large faces."""
    height,width=patch.shape[:2]
    reduction=max(1.,float(sigma)/12.)
    def blur(source,sx,sy):
        if not wrap_x:
            return cv2.GaussianBlur(source,(0,0),sigmaX=sx,sigmaY=sy,borderType=cv2.BORDER_REPLICATE)
        radius=max(1,int(np.ceil(4*sx)))
        horizontal=np.arange(-radius,source.shape[1]+radius)%source.shape[1]
        padded=source[:,horizontal]
        # A bounded horizontal halo is periodic; vertical Gaussian remains clamped.
        filtered=cv2.GaussianBlur(padded,(2*radius+1,2*max(1,int(np.ceil(4*sy)))+1),
                                  sigmaX=sx,sigmaY=sy,borderType=cv2.BORDER_REPLICATE)
        return filtered[:,radius:radius+source.shape[1]]
    if reduction==1:
        return blur(patch,sigma,sigma)
    small_size=(max(1,int(np.ceil(width/reduction))),max(1,int(np.ceil(height/reduction))))
    small=cv2.resize(patch,small_size,interpolation=cv2.INTER_AREA)
    blurred=blur(small,sigma*small_size[0]/width,sigma*small_size[1]/height)
    result=cv2.resize(blurred,(width,height),interpolation=cv2.INTER_LINEAR)
    if wrap_x:
        # AREA reduction partitions one complete period. Linear enlargement must
        # interpolate between its last/first cells, rather than copy either edge.
        edge=min(width,max(1,int(np.ceil(width/small_size[0]/2))))
        padded=np.concatenate((blurred[:,-1:],blurred,blurred[:,:1]),axis=1)
        for start,stop in ((0,edge),(max(edge,width-edge),width)):
            if stop<=start: continue
            mx=(np.arange(start,stop,dtype=np.float32)+.5)*small_size[0]/width-.5+1
            for top in range(0,height,256):
                bottom=min(height,top+256)
                my=np.clip((np.arange(top,bottom,dtype=np.float32)+.5)*small_size[1]/height-.5,0,small_size[1]-1)
                result[top:bottom,start:stop]=cv2.remap(padded,np.broadcast_to(mx,(bottom-top,stop-start)).copy(),
                    np.broadcast_to(my[:,None],(bottom-top,stop-start)).copy(),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return result


def apply_style_patch(patch, mask, style, wrap_x=False):
    style={**default_style(),**style};validate_style(style)
    if mask.shape!=patch.shape[:2]:raise ValueError("遮罩与图片尺寸不同")
    result=patch.copy();mask=np.asarray(mask,bool)
    positions=np.argwhere(mask)
    if len(positions)==0:return result
    y0,x0=positions.min(axis=0);y1,x1=positions.max(axis=0)+1
    parameters=privacy_style_parameters((x0,y0,x1,y1),style)
    if style["kind"]=="blur":
        filtered=gaussian_privacy_filter(patch,parameters['sigma'],wrap_x=wrap_x)
    elif style["kind"]=="mosaic":
        crop=patch[y0:y1,x0:x1]
        cells=parameters['cells']
        small=cv2.resize(crop,(min(cells,crop.shape[1]),min(cells,crop.shape[0])),interpolation=cv2.INTER_AREA)
        filtered=patch.copy();filtered[y0:y1,x0:x1]=cv2.resize(small,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
    else:
        color=[int(style["color"][i:i+2],16) for i in (1,3,5)]
        filtered=np.empty_like(patch)
        filtered[...] = color if patch.ndim==3 else style.get("solid_value",32)
    result[mask]=filtered[mask]
    return result


def apply_regions_rgb(rgb, regions, style):
    if rgb.shape[0]*rgb.shape[1]>40_000_000:
        raise ValueError("此函数仅用于小尺寸预览；原图请用分块导出。")
    style={**default_style(),**style};validate_style(style)
    result=rgb.copy();h,w=rgb.shape[:2]
    for region in regions:
        if not region.get("enabled",True):continue
        mask=np.zeros((h,w),bool)
        for x,y,tile in region_mask_tiles(region,(w,h),padding=float(style["padding"]),shape=style.get("shape","rect")):
            mask[y:y+tile.shape[0],x:x+tile.shape[1]] |= tile
        result=apply_style_patch(result,mask,style,wrap_x=True)
    return result

