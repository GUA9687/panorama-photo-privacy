import json
import numpy as np
import pytest


def test_native_seam_mask_does_not_fill_middle():
    from panoprivacy.common import new_region
    from panoprivacy.geometry import region_mask_tiles
    r = new_region('erp', [.98, .4, 1.02, .6])
    full = np.zeros((50, 100), bool)
    for x, y, m in region_mask_tiles(r, (100, 50), padding=0):
        full[y:y+m.shape[0], x:x+m.shape[1]] |= m
    assert full.sum() == 40
    assert not full[:, 10:90].any()


def test_pole_and_grid_masks():
    from panoprivacy.common import new_region
    from panoprivacy.geometry import region_mask_tiles
    r = new_region('view', [.35, .35, .65, .65], pose={'yaw': 0, 'pitch': -90, 'fov': 90, 'roll': 0}, view_size=(200, 200))
    tiles = list(region_mask_tiles(r, (600, 300), tile_size=64, padding=0))
    full = np.zeros((300, 600), bool)
    for x, y, m in tiles:
        assert x % 64 == y % 64 == 0
        assert m.shape[0] <= 64 and m.shape[1] <= 64
        full[y:y+m.shape[0], x:x+m.shape[1]] |= m
    assert full[-1].all() and not full[:200].any()


def test_view_box_round_trip():
    from panoprivacy.geometry import view_box_region, region_to_view_box
    p = {'yaw': 179, 'pitch': -65, 'fov': 100, 'roll': 23}
    r = view_box_region([150, 80, 260, 170], p, (400, 250))
    assert np.allclose(region_to_view_box(r, p, (400, 250)), [150, 80, 260, 170], atol=1e-7)


def test_disabled_and_padding_once():
    from panoprivacy.common import new_region, default_style
    from panoprivacy.geometry import apply_regions_rgb
    image = np.full((100, 200, 3), 255, np.uint8)
    r = new_region('erp', [.4, .4, .6, .6], enabled=False)
    style = default_style(); style.update(kind='solid', color='#000000', padding=0)
    assert np.array_equal(image, apply_regions_rgb(image, [r], style))
    r['enabled'] = True
    result = apply_regions_rgb(image, [r], style)
    assert (result[40:60, 80:120] == 0).all()
    assert (result[:40] == 255).all()


def test_project_and_output_guards(tmp_path):
    from panoprivacy.common import ensure_distinct_output, fingerprint, new_region, default_style
    from panoprivacy.project import save_project, load_project
    source=tmp_path/'原图.jpg'; source.write_bytes(b'original')
    with pytest.raises(ValueError): ensure_distinct_output(source,source)
    existing=tmp_path/'exists.jpg';existing.write_bytes(b'previous')
    with pytest.raises(FileExistsError):ensure_distinct_output(source,existing)
    asset={'kind':'photo','source':str(source),'width':200,'height':100,'fingerprint':fingerprint(source),'cache_dir':str(tmp_path/'cache'),'preview_path':'preview.jpg'}
    file=tmp_path/'工程.ppriv'
    save_project(file,'photo',asset,regions=[new_region('erp',[.1,.1,.2,.2])],style=default_style())
    data=load_project(file)
    assert data['kind']=='photo' and len(data['regions'])==1 and data['source_status']=='ok'
    source.write_bytes(b'changed-source')
    assert load_project(file)['source_status']=='changed'
    assert not list(tmp_path.glob('*.tmp'))


def test_invalid_regions_and_cancel():
    from panoprivacy.common import new_region, check_cancel, Cancelled
    import threading
    with pytest.raises(ValueError): new_region('erp',[0,0,float('nan'),1])
    with pytest.raises(ValueError):new_region('view',[.5,.5,.2,.7])
    event=threading.Event();event.set()
    with pytest.raises(Cancelled):check_cancel(event)
