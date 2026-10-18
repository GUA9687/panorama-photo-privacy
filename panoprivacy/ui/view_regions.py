"""Small perspective masks evaluated against the original spherical regions.

There is no full-panorama mask and no bounding-box substitution: every output
pixel is tested in the region's original ERP or recorded perspective space.
"""
import numpy as np


def view_region_mask(region, pose, size, padding=.25, rays=None, shape='rect'):
    from panoprivacy.geometry import view_rays, region_membership
    width,height=size
    if rays is None:
        u,v=np.meshgrid((np.arange(width)+.5)/width,(np.arange(height)+.5)/height)
        rays=view_rays(np.stack((u,v),axis=-1),pose,size)
    return region_membership(region,rays,padding,shape)


def apply_view_regions(rgb, regions, style, pose, cancel=None):
    from panoprivacy.geometry import view_rays, apply_style_patch
    from panoprivacy.common import check_cancel
    height, width = rgb.shape[:2]
    u, v = np.meshgrid((np.arange(width)+.5)/width, (np.arange(height)+.5)/height)
    rays = view_rays(np.stack((u, v), axis=-1), pose, (width, height))
    result = rgb
    for region in regions:
        check_cancel(cancel)
        if region.get('enabled', True):
            mask = view_region_mask(region, pose, (width, height), style.get('padding', .25), rays, style.get('shape','rect'))
            if mask.any():
                result = apply_style_patch(result, mask, style)
    return result
