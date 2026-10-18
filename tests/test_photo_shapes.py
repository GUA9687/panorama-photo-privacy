import numpy as np
import pytest
from panoprivacy.common import new_region, validate_style
from panoprivacy.geometry import region_mask_tiles, view_rays, region_membership
from panoprivacy.ui.view_regions import view_region_mask


@pytest.mark.parametrize('size', [(800,400),(400,800)])
def test_ellipse_default_margin_covers_original_box_but_not_expanded_corners(size):
    r=new_region('view',[.4,.35,.6,.65],view_size=size)
    points=np.array([[.4,.35],[.6,.35],[.4,.65],[.6,.65],[.35,.275],[.5,.5]])
    mask=region_membership(r,view_rays(points,r['pose'],size),padding=.25,shape='ellipse')
    assert mask.tolist()==[True,True,True,True,False,True]


def test_circle_is_round_in_recorded_pixel_space():
    r=new_region('view',[.45,.4,.55,.6],view_size=(800,400))
    rays=view_rays(np.array([[.55,.5],[.5,.6],[.56,.5],[.5,.62]]),r['pose'],r['view_size'])
    assert region_membership(r,rays,padding=0,shape='circle').tolist()==[True,True,False,False]


@pytest.mark.parametrize('space,box,pose',[
    ('view',[.3,.3,.7,.7],{'yaw':179,'pitch':0,'fov':70}),
    ('view',[.3,.3,.7,.7],{'yaw':25,'pitch':90,'fov':60}),
    ('erp',[.98,.4,1.02,.6],{}),
    ('erp',[.4,.01,.6,.08],{}),
])
@pytest.mark.parametrize('shape',['rect','ellipse','circle'])
def test_tiled_mask_equals_shared_spherical_membership(space,box,pose,shape):
    r=new_region(space,box,pose,(600,400));w,h=720,360
    u,v=np.meshgrid((np.arange(w)+.5)/w,(np.arange(h)+.5)/h)
    lon=(u-.5)*2*np.pi;lat=(.5-v)*np.pi
    rays=np.stack([np.cos(lat)*np.sin(lon),np.sin(lat),np.cos(lat)*np.cos(lon)],axis=-1)
    expected=region_membership(r,rays,padding=.25,shape=shape)
    actual=np.zeros((h,w),bool)
    for x,y,tile in region_mask_tiles(r,(w,h),tile_size=37,padding=.25,shape=shape):
        actual[y:y+tile.shape[0],x:x+tile.shape[1]]|=tile
    assert expected.any()
    np.testing.assert_array_equal(actual,expected)
    vw,vh=240,160
    u,v=np.meshgrid((np.arange(vw)+.5)/vw,(np.arange(vh)+.5)/vh)
    view=view_rays(np.stack([u,v],axis=-1),r['pose'],(vw,vh))
    np.testing.assert_array_equal(view_region_mask(r,r['pose'],(vw,vh),.25,shape=shape),
                                  region_membership(r,view,.25,shape))


def test_invalid_new_style_fields_rejected_and_legacy_valid():
    validate_style({'kind':'blur'})
    for fields in ({'shape':'triangle'},{'export_mode':'foo'},{'jpeg_quality':42}):
        with pytest.raises(ValueError):validate_style({'kind':'blur',**fields})
