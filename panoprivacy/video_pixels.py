"""Privacy rendering in source YUV planes, sharing preview strength definitions."""
import cv2
import numpy as np


def _solid_yuv(style, depth, limited):
    color=style.get('color','#202020').lstrip('#')
    r,g,b=[int(color[n:n+2],16)/255 for n in (0,2,4)]
    y=.2126*r+.7152*g+.0722*b
    u=(b-y)/1.8556; v=(r-y)/1.5748
    if limited:
        return [(16+219*y)*2**(depth-8),(128+224*u)*2**(depth-8),(128+224*v)*2**(depth-8)]
    maximum=(1<<depth)-1
    return [y*maximum,(u+.5)*maximum,(v+.5)*maximum]


def redact_planes(planes, regions, style, depth=10, limited=True, cancel=None):
    from .common import check_cancel,default_style,validate_style
    from .geometry import region_mask_tiles,privacy_style_parameters,gaussian_privacy_filter
    style={**default_style(),**style};validate_style(style)
    h,w=planes[0].shape
    color=_solid_yuv(style,depth,limited)
    for region in regions:
        if not region.get('enabled',True): continue
        check_cancel(cancel)
        tiles=list(region_mask_tiles(region,(w,h),tile_size=256,padding=style['padding']))
        if not tiles: continue
        bounds=[]
        for x,y,mask in tiles:
            yy,xx=np.nonzero(mask)
            bounds.append((x+xx.min(),y+yy.min(),x+xx.max()+1,y+yy.max()+1))
        x0=min(b[0] for b in bounds);y0=min(b[1] for b in bounds)
        x1=max(b[2] for b in bounds);y1=max(b[3] for b in bounds)
        for channel,plane in enumerate(planes):
            check_cancel(cancel)
            factor=1 if channel==0 else 2
            parameters=privacy_style_parameters((x0,y0,x1,y1),style,scale=1/factor)
            left,top=x0//factor,y0//factor
            right,bottom=(x1+factor-1)//factor,(y1+factor-1)//factor
            kind=style['kind']
            # Once per region/channel: no repeated per-tile large Gaussian halos.
            if kind=='blur':
                filtered=gaussian_privacy_filter(plane,parameters['sigma'],wrap_x=True)
            elif kind=='mosaic':
                crop=plane[top:bottom,left:right]
                cells=parameters['cells']
                small=cv2.resize(crop,(min(cells,crop.shape[1]),min(cells,crop.shape[0])),interpolation=cv2.INTER_AREA)
                filtered=cv2.resize(small,(crop.shape[1],crop.shape[0]),interpolation=cv2.INTER_NEAREST)
            else: filtered=None
            check_cancel(cancel)
            for x,y,mask in tiles:
                check_cancel(cancel)
                mh,mw=mask.shape
                px,py=x//factor,y//factor
                if factor==1: cmask=mask
                else:
                    cmask=np.zeros(((y+mh+1)//2-py,(x+mw+1)//2-px),bool)
                    yy,xx=np.nonzero(mask)
                    cmask[(yy+y)//2-py,(xx+x)//2-px]=True
                yy,xx=np.nonzero(cmask)
                gy,gx=yy+py,xx+px
                if kind=='solid': plane[gy,gx]=round(color[channel])
                elif kind=='blur': plane[gy,gx]=filtered[gy,gx]
                else: plane[gy,gx]=filtered[gy-top,gx-left]
