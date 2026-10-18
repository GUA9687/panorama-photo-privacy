import numpy as np
import pytest
from panoprivacy.common import new_region,default_style
from panoprivacy.geometry import apply_regions_rgb,region_mask_tiles,view_box_region
from panoprivacy.video_pixels import redact_planes
from panoprivacy.video_engine import validate_export_tracks

@pytest.mark.parametrize('kind',['blur','mosaic'])
@pytest.mark.parametrize('depth',[8,10])
@pytest.mark.parametrize('location',['small','large','seam','pole'])
def test_preview_and_native_planes_have_comparable_strength(kind,depth,location):
    w,h=512,256
    x=np.arange(w)[None,:];y=np.arange(h)[:,None]
    gray=np.clip(128+60*np.sin(2*np.pi*x/25)+30*np.sin(2*np.pi*y/35),0,255).astype(np.uint8)
    rgb=np.repeat(gray[:,:,None],3,axis=2)
    region={'small':new_region('erp',[.4,.4,.5,.6]),'large':new_region('erp',[.2,.2,.8,.8]),
            'seam':new_region('erp',[.95,.3,1.05,.6]),
            'pole':view_box_region([180,160,330,320],{'yaw':160,'pitch':-86,'roll':35,'fov':100},(512,512))}[location]
    style=dict(default_style(),kind=kind)
    preview=apply_regions_rgb(rgb,[region],style)[:,:,0]
    factor=2**(depth-8);dtype=np.uint16 if depth==10 else np.uint8
    planes=[(gray.astype(dtype)*factor),np.full((h//2,w//2),128*factor,dtype),np.full((h//2,w//2),128*factor,dtype)]
    original=[p.copy() for p in planes]
    redact_planes(planes,[region],style,depth)
    masks=[np.zeros(p.shape,bool) for p in planes]
    for tx,ty,mask in region_mask_tiles(region,(w,h),padding=.25):
        yy,xx=np.nonzero(mask);yy+=ty;xx+=tx
        masks[0][yy,xx]=True;masks[1][yy//2,xx//2]=True;masks[2][yy//2,xx//2]=True
    error=np.abs(planes[0].astype(float)/factor-preview)
    assert np.mean(error[masks[0]])<1.5
    for plane,before,mask in zip(planes,original,masks): assert np.array_equal(plane[~mask],before[~mask])

def test_uncertain_review_permission_is_explicit_and_never_bypasses_missing():
    region=new_region('erp',[.4,.3,.6,.6])
    analysis={'tracks':[{'frame':0,'regions':[region],'status':'uncertain'}]}
    with pytest.raises(ValueError): validate_export_tracks(analysis,1)
    analysis['reviewed_uncertain']=1
    with pytest.raises(ValueError): validate_export_tracks(analysis,1)
    analysis['reviewed_uncertain']=True
    assert validate_export_tracks(analysis,1)==analysis['tracks']
    assert analysis['tracks'][0]['status']=='uncertain'
    for status,regions in [('missing',[region]),('uncertain',[]),('uncertain',[dict(region,enabled=False)]),('ok',[])]:
        analysis['tracks'][0].update(status=status,regions=regions)
        with pytest.raises(ValueError): validate_export_tracks(analysis,1)

@pytest.mark.parametrize('depth',[8,10])
def test_chroma_blur_uses_half_luma_spatial_sigma(depth):
    from panoprivacy.geometry import gaussian_privacy_filter,privacy_style_parameters
    dtype=np.uint16 if depth==10 else np.uint8;factor=2**(depth-8)
    gray=np.random.default_rng(41).integers(40,210,(128,256),dtype=np.uint8)
    planes=[np.full((256,512),128*factor,dtype),gray.astype(dtype)*factor,gray.astype(dtype)*factor]
    before=planes[1].copy()
    region=new_region('erp',[.25,.25,.75,.75]);style=dict(default_style(),padding=0)
    sigma=privacy_style_parameters((128,64,384,192),style,scale=.5)['sigma']
    expected=gaussian_privacy_filter(before,sigma,wrap_x=True)
    redact_planes(planes,[region],style,depth)
    assert np.array_equal(planes[1][32:96,64:192],expected[32:96,64:192])

@pytest.mark.parametrize('kind',['blur','mosaic'])
def test_native_strength_scales_with_source_resolution(kind):
    import cv2
    width,height=512,256
    xx=np.arange(width)[None,:];yy=np.arange(height)[:,None]
    gray=(128+60*np.sin(2*np.pi*xx/25)+30*np.sin(2*np.pi*yy/35)).astype(np.uint8)
    region=new_region('erp',[.3,.2,.7,.8]);style=dict(default_style(),kind=kind)
    preview=apply_regions_rgb(np.repeat(gray[:,:,None],3,axis=2),[region],style)[:,:,0]
    native=cv2.resize(gray,(width*2,height*2),interpolation=cv2.INTER_LINEAR)
    planes=[native,np.full((height,width),128,np.uint8),np.full((height,width),128,np.uint8)]
    redact_planes(planes,[region],style,depth=8)
    down=cv2.resize(planes[0],(width,height),interpolation=cv2.INTER_AREA)
    assert np.mean(np.abs(down[80:180,180:330].astype(float)-preview[80:180,180:330]))<2

@pytest.mark.parametrize('sigma',[4.,13.68,55.])
def test_erp_gaussian_matches_independent_periodic_reference(sigma):
    from scipy.ndimage import gaussian_filter
    from panoprivacy.geometry import gaussian_privacy_filter
    source=np.full((128,512),40,np.uint8);source[:,256:]=210
    actual=gaussian_privacy_filter(source,sigma,wrap_x=True)
    expected=gaussian_filter(source.astype(np.float32),sigma=(sigma,sigma),mode=('nearest','wrap'))
    assert np.max(np.abs(actual[:,[0,1,-2,-1]].astype(float)-expected[:,[0,1,-2,-1]]))<3
    assert 110<int(actual[64,0])<140 and 110<int(actual[64,-1])<140
    ordinary=gaussian_privacy_filter(source,sigma,wrap_x=False)
    assert ordinary[64,0]==40 and ordinary[64,-1]==210


def test_native_seam_masks_mix_other_side_and_leave_outside_untouched():
    from scipy.ndimage import gaussian_filter
    source=np.full((256,512),40,np.uint8);source[:,256:]=210
    planes=[source.copy(),np.full((128,256),128,np.uint8),np.full((128,256),128,np.uint8)]
    region=new_region('erp',[.94,.35,1.06,.65]);style=dict(default_style(),padding=0)
    redact_planes(planes,[region],style,depth=8)
    expected=gaussian_filter(source.astype(np.float32),sigma=13.68,mode=('nearest','wrap'))
    assert np.max(np.abs(planes[0][110:140,[0,-1]].astype(float)-expected[110:140,[0,-1]]))<3
    assert np.array_equal(planes[0][:80],source[:80])

