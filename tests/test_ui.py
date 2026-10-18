from pathlib import Path
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import threading
import time
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPointF
from panoprivacy.ui.jobs import JobController
from panoprivacy.ui.viewport import PanoramaViewport
from panoprivacy.ui.state import DocumentState

@pytest.fixture(scope='session')
def app():
    return QApplication.instance() or QApplication([])

def pump(app, condition, seconds=3):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(.005)
    assert condition()

def test_cancel_and_switch_rejects_delayed_result(app):
    controller = JobController()
    delivered = []
    entered = threading.Event()
    release = threading.Event()
    controller.succeeded.connect(delivered.append)
    def old(progress, cancel):
        entered.set()
        release.wait(2)
        return 'old image'
    controller.switch_document('旧图片')
    controller.start(old)
    assert entered.wait(1)
    controller.cancel()
    controller.switch_document('新图片')
    assert not controller.accepts_result('旧图片')
    controller.start(lambda progress, cancel: 'new image')
    pump(app, lambda: delivered == ['new image'])
    release.set()
    pump(app, lambda: controller.active_count == 0)
    assert delivered == ['new image']

def test_worker_failure_carries_message_and_finishes(app):
    controller = JobController()
    errors = []
    controller.failed.connect(errors.append)
    def fail(progress, cancel):
        raise ValueError('无法读取：中文 路径.jpg')
    controller.start(fail)
    pump(app, lambda: bool(errors) and not controller.running)
    assert '中文 路径.jpg' in errors[0]

def test_view_drag_uses_pose_and_exact_display_size(app, monkeypatch):
    import sys
    import types
    geometry = types.ModuleType('panoprivacy.geometry')
    monkeypatch.setitem(sys.modules, 'panoprivacy.geometry', geometry)
    seen = {}
    def region(box, pose, size, **fields):
        seen.update(box=box, pose=pose, size=size)
        return {'box': box, 'pose': pose, 'view_size': size, **fields}
    monkeypatch.setattr(geometry, 'view_box_region', region, raising=False)
    viewport = PanoramaViewport()
    viewport.resize(800, 400)
    viewport.pose = {'yaw': 123., 'pitch': -32., 'fov': 75., 'roll': 0.}
    result = viewport.region_from_drag(QPointF(100, 80), QPointF(500, 320))
    assert seen['box'] == [100., 80., 500., 320.]
    assert seen['size'] == (800, 400)
    assert seen['pose']['yaw'] == 123.
    assert result['origin'] == 'manual'

def test_document_snapshot_preserves_manual_and_disabled_regions():
    state = DocumentState('photo')
    state.asset = {'source': 'C:/测试 照片.jpg', 'kind': 'photo'}
    state.regions = [{'id': 'r', 'enabled': False, 'origin': 'manual'}]
    state.style['kind'] = 'solid'
    snapshot = state.snapshot()
    restored = DocumentState.from_project(snapshot)
    snapshot['regions'][0]['enabled'] = True
    assert restored.regions[0]['enabled'] is False
    assert restored.style['kind'] == 'solid'
    assert restored.asset['source'] == 'C:/测试 照片.jpg'

def test_photo_queue_is_lazy_and_keeps_each_documents_regions(app, tmp_path, monkeypatch):
    import sys
    import types
    from PIL import Image
    from panoprivacy.ui.photo_window import PhotoWindow
    calls = []
    preview = tmp_path / '预览 空格.png'
    Image.fromarray(np.zeros((20, 40, 3), np.uint8)).save(preview)
    def prepare(source, cache_dir, progress, cancel):
        calls.append(source)
        return {'kind': 'photo', 'source': source, 'width': 40, 'height': 20,
                'preview_path': str(preview), 'cache_dir': cache_dir}
    monkeypatch.setitem(sys.modules, 'panoprivacy.common', types.SimpleNamespace(
        app_cache_dir=lambda: tmp_path, check_cancel=lambda cancel: None))
    monkeypatch.setitem(sys.modules, 'panoprivacy.photo_engine', types.SimpleNamespace(
        prepare_photo=prepare, photo_thumbnail=lambda asset, region, size: np.zeros((size,size,3), np.uint8)))
    monkeypatch.setitem(sys.modules, 'panoprivacy.geometry', types.SimpleNamespace(
        render_view=lambda rgb, pose, size: np.zeros((size[1],size[0],3),np.uint8),
        apply_regions_rgb=lambda rgb, regions, style: rgb))
    window = PhotoWindow()
    paths = [str(tmp_path/f'照片 {i}.jpg') for i in range(101)]
    for path in paths:
        Path(path).touch()
    window.add_paths(paths)
    pump(app, lambda: window._preview_rgb is not None and not window.jobs.running)
    assert len(calls) == 1
    assert window.queue.count() == 101
    window.add_region({'id':'manual', 'origin':'manual', 'enabled':False})
    pump(app, lambda: not window.jobs.running and bool(window._thumbnails))
    window.queue.setCurrentRow(1)
    pump(app, lambda: len(calls)==2 and window._preview_rgb is not None and not window.jobs.running)
    assert window.state.regions == []
    window.queue.setCurrentRow(0)
    pump(app, lambda: len(calls)==3 and window._preview_rgb is not None and not window.jobs.running)
    assert window.state.regions[0]['id'] == 'manual'
    assert window.state.regions[0]['enabled'] is False
    window.state.dirty = False
    window.close()
    pump(app, lambda: window.jobs.active_count == 0 and window.previews.active_count == 0)


def test_video_seed_belongs_to_displayed_frame_and_invalidates_export(app):
    from panoprivacy.ui.video_window import VideoWindow
    window = VideoWindow()
    window.state.asset = {'source':'视频 空格.mp4','frame_count':100,'fps_num':30,'fps_den':1}
    window.displayed_frame = 7
    window.frame = 28
    window._analysis_current = True
    window.add_seed({'id':'face', 'box':[.1,.2,.3,.4], 'enabled':True})
    assert window.state.seeds[0]['frame'] == 7
    assert window.frame == 7
    assert not window._analysis_current
    assert window.anchors.count() == 1
    window.preview_timer.stop()
    window.state.dirty = False
    window.close()

@pytest.mark.parametrize('cache_state', ['valid', 'missing', 'source_changed', 'relink'])
def test_video_project_reuses_only_unchanged_original_with_existing_cache(app, tmp_path, monkeypatch, cache_state):
    import sys
    import types
    from copy import deepcopy
    from panoprivacy.common import fingerprint
    from panoprivacy.ui.video_window import VideoWindow
    source = tmp_path/'original.mp4'
    source.write_bytes(b'original source')
    proxy = tmp_path/'proxy.mp4'
    preview = tmp_path/'preview.jpg'
    proxy.write_bytes(b'proxy')
    preview.write_bytes(b'preview')
    asset = {'kind':'video','source':str(source),'fingerprint':fingerprint(source),
             'proxy_path':str(proxy),'preview_path':str(preview),'width':7680,'height':3840,
             'fps_num':30,'fps_den':1,'frame_count':1492}
    state = DocumentState('video',asset=deepcopy(asset))
    state.seeds = [{'frame':5,'region':{'id':'selected'}}]
    if cache_state == 'missing':
        proxy.unlink()
    if cache_state == 'source_changed':
        source.write_bytes(b'changed original source')
    if cache_state == 'relink':
        source = tmp_path/'moved original.mp4'
        source.write_bytes(b'original source')
    calls = []
    def prepare(path, *args):
        calls.append(path)
        return {**asset,'source':path,'fingerprint':fingerprint(path)}
    monkeypatch.setitem(sys.modules,'panoprivacy.video_engine',types.SimpleNamespace(prepare_video=prepare))
    window = VideoWindow()
    results=[]
    window.run_job=lambda function, handler, message:results.append(function(lambda _:None,threading.Event()))
    if cache_state == 'source_changed':
        with pytest.raises(ValueError,match='源文件已改变'):
            window.load_source(str(source),state)
        assert not calls
    else:
        window.load_source(str(source),state,explicit_relink=cache_state=='relink')
        assert len(calls) == (0 if cache_state == 'valid' else 1)
        assert results[0]['fingerprint'] == fingerprint(source)
        assert window.state.seeds == state.seeds
    window.state.dirty=False
    window.close()

    pump(app, lambda: window.jobs.active_count == 0 and window.previews.active_count == 0)


def test_drag_on_resized_old_preview_uses_shown_pose_not_inflight_pose(app, monkeypatch):
    import sys
    import types
    seen = {}
    def map_region(box, pose, size, **fields):
        seen.update(box=box, pose=pose, size=size)
        return seen
    monkeypatch.setitem(sys.modules, 'panoprivacy.geometry', types.SimpleNamespace(view_box_region=map_region))
    viewport = PanoramaViewport()
    viewport.resize(800,400)
    viewport.set_rgb(np.zeros((200,400,3),np.uint8), {'yaw':45., 'pitch':0., 'fov':90., 'roll':0.})
    viewport.pose['yaw'] = 90.
    viewport.region_from_drag(QPointF(200,100),QPointF(600,300))
    assert seen['pose']['yaw'] == 45.
    assert seen['size'] == (400,200)
    assert seen['box'] == [100.,50.,300.,150.]

def test_saved_video_correction_does_not_reenable_stale_analysis(app):
    from panoprivacy.ui.video_window import VideoWindow
    window = VideoWindow()
    state = DocumentState('video')
    state.analysis = {'tracks':[], 'warnings':[], '_ui_stale':True, '_ui_changed_frames':[12]}
    restored = DocumentState.from_project(state.snapshot())
    window._set_analysis(restored.analysis)
    assert not window._analysis_current
    window.close()

def test_project_save_reopen_real_schema_and_real_preview_geometry(app, tmp_path, monkeypatch):
    from PIL import Image
    import sys
    import types
    from panoprivacy import common
    from panoprivacy.ui.photo_window import PhotoWindow
    from PySide6.QtWidgets import QFileDialog
    source = tmp_path / '原片 中文 空格.jpg'
    source.write_bytes(b'read-only fixture')
    preview = tmp_path / 'preview.jpg'
    Image.fromarray(np.full((64,128,3), 150, np.uint8)).save(preview)
    project = tmp_path / '工程 中文 空格.json'
    monkeypatch.setattr(common, 'app_cache_dir', lambda: tmp_path)
    def prepare(path, cache_dir, progress, cancel):
        return {'kind':'photo','source':path,'width':128,'height':64,
                'fingerprint':common.fingerprint(path), 'preview_path':str(preview),
                'native_path':str(tmp_path/'unavailable.v'), 'cache_dir':cache_dir}
    monkeypatch.setitem(sys.modules, 'panoprivacy.photo_engine', types.SimpleNamespace(
        prepare_photo=prepare, photo_thumbnail=lambda asset, region, size: np.zeros((size,size,3),np.uint8)))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(project),'JSON'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(project),'JSON'))
    window = PhotoWindow()
    errors = []
    window.jobs.failed.disconnect()
    window.jobs.failed.connect(errors.append)
    window.add_paths([str(source)])
    window.show()
    pump(app, lambda: window.viewport.image is not None and not window.jobs.running)
    region = common.new_region('erp',[.48,.4,.52,.6], origin='manual', enabled=False)
    window.add_region(region)
    pump(app, lambda: not window.jobs.running and bool(window._thumbnails))
    window.save_project()
    pump(app, lambda: project.is_file() and window.state.project_path is not None and not window.jobs.running)
    assert not window.state.dirty
    assert not errors
    previous_identity = window.state.identity
    window.open_project()
    pump(app, lambda: window.state.identity != previous_identity and
         window.state.project_path == str(project) and window.state.regions and not window.jobs.running)
    assert window.state.regions[0]['id'] == region['id']
    assert window.state.regions[0]['enabled'] is False
    assert source.read_bytes() == b'read-only fixture'
    window.state.dirty = False
    for state in window.queue_states:
        state.dirty = False
    window.close()
    pump(app, lambda: window.jobs.active_count == 0 and window.previews.active_count == 0)


def test_cache_eviction_only_deletes_exact_native_inside_app_cache(app, tmp_path, monkeypatch):
    from panoprivacy import common
    from panoprivacy.ui.photo_window import PhotoWindow
    cache = tmp_path / 'cache'
    cache.mkdir()
    native = cache/'native.v'
    native.write_bytes(b'cache')
    preview = cache/'preview.jpg'
    preview.write_bytes(b'preview stays')
    source = tmp_path/'source.jpg'
    source.write_bytes(b'original stays')
    outside = tmp_path/'other.v'
    outside.write_bytes(b'outside stays')
    monkeypatch.setattr(common, 'app_cache_dir', lambda: cache)
    window = PhotoWindow()
    window._eviction_candidates = [
        {'native_path':str(native),'source':str(source)},
        {'native_path':str(source),'source':str(source)},
        {'native_path':str(outside),'source':str(source)}]
    window.evict_unused_native()
    assert not native.exists()
    assert source.read_bytes() == b'original stays'
    assert outside.read_bytes() == b'outside stays'
    assert preview.exists()
    window.close()

def test_native_preview_masks_project_spherical_regions_and_padding_once():
    from panoprivacy.ui.view_regions import view_region_mask
    from panoprivacy.common import new_region
    pose = {'yaw':179., 'pitch':32., 'fov':80., 'roll':15.}
    region = new_region('view',[.35,.3,.65,.7], pose=pose, view_size=(200,100))
    mask = view_region_mask(region, pose, (200,100), .25)
    assert mask.shape == (100,200)
    assert mask[20:80,55:145].all()
    assert not mask[:19].any()
    assert not mask[:, :54].any()
    seam = new_region('erp',[.98,.3,1.02,.7])
    seam_mask = view_region_mask(seam, {'yaw':180.,'pitch':0.,'fov':80.,'roll':0.}, (200,100), 0)
    assert seam_mask[:,99:101].all()
    assert not seam_mask[:, :70].any()


def test_native_photo_idle_view_reads_real_native_and_rejects_old_pose(app, tmp_path, monkeypatch):
    from PIL import Image
    from panoprivacy import common, photo_engine
    from panoprivacy.ui.photo_window import PhotoWindow
    from copy import deepcopy
    source = tmp_path/'高清 原片.jpg'
    rng = np.random.default_rng(71)
    Image.fromarray(rng.integers(0,255,(256,512,3),np.uint8)).save(source)
    before = source.read_bytes()
    monkeypatch.setattr(common, 'app_cache_dir', lambda: tmp_path/'cache')
    real_view = photo_engine.photo_view
    entered, release = threading.Event(), threading.Event()
    calls = []
    def controlled_view(asset, pose, out_size=(960,640)):
        calls.append(deepcopy(pose))
        if len(calls) == 1:
            entered.set()
            release.wait(3)
        return real_view(asset, pose, out_size)
    monkeypatch.setattr(photo_engine, 'photo_view', controlled_view)
    window = PhotoWindow()
    window.viewport.setFixedSize(480,320)
    errors = []
    window.jobs.failed.disconnect()
    window.jobs.failed.connect(errors.append)
    window.show()
    window.add_paths([str(source)])
    pump(app, lambda: entered.is_set(), 5)
    old_revision = window._view_revision
    window.viewport.pose['yaw'] = 40.
    window.request_preview()
    window.native_timer.stop()
    release.set()
    pump(app, lambda: window.native_jobs.active_count == 0, 5)
    assert window._display_quality != ('native', old_revision)
    window.request_native_preview()
    pump(app, lambda: window._display_quality == ('native',window._view_revision), 5)
    assert len(calls) >= 2
    assert calls[-1]['yaw'] == 40.
    assert window.viewport.display_pose['yaw'] == 40.
    assert not errors
    assert source.read_bytes() == before
    window.state.dirty = False
    window.close()
    pump(app, lambda: window.jobs.active_count == 0 and window.previews.active_count == 0 and window.native_jobs.active_count == 0)

def test_native_preview_projection_agrees_with_spherical_export_masks():
    from panoprivacy.ui.view_regions import view_region_mask, apply_view_regions
    from panoprivacy.common import new_region, default_style
    from panoprivacy.geometry import region_mask_tiles, render_view
    region_pose = {'yaw':175.,'pitch':-70.,'fov':78.,'roll':23.}
    region = new_region('view',[.3,.2,.7,.75], pose=region_pose, view_size=(320,240))
    view_pose = {'yaw':-175.,'pitch':-65.,'fov':90.,'roll':-12.}
    native_mask = np.zeros((512,1024,3),np.uint8)
    for x,y,mask in region_mask_tiles(region,(1024,512),padding=.25):
        native_mask[y:y+mask.shape[0],x:x+mask.shape[1]][mask] = 255
    reference = render_view(native_mask,view_pose,(240,160))[:,:,0] > 127
    projected = view_region_mask(region,view_pose,(240,160),.25)
    assert np.mean(reference != projected) < .008
    style = {**default_style(),'kind':'solid','color':'#010203'}
    rgb = np.full((160,240,3),240,np.uint8)
    edited = apply_view_regions(rgb,[region],style,view_pose)
    assert np.all(edited[projected] == [1,2,3])
    assert np.all(edited[~projected] == 240)

def test_candidate_focus_fits_a_small_face_for_native_detail(app):
    from panoprivacy.common import new_region
    viewport = PanoramaViewport()
    viewport.resize(900,600)
    viewport.focus_region(new_region('erp',[.499,.499,.501,.501]))
    assert 2 <= viewport.pose['fov'] < 5

@pytest.mark.parametrize('window_type', ['photo','video'])
def test_window_save_does_not_clear_padding_edit_during_write(app,tmp_path,monkeypatch,window_type):
    from panoprivacy import project
    from panoprivacy.ui.photo_window import PhotoWindow
    from panoprivacy.ui.video_window import VideoWindow
    from PySide6.QtWidgets import QFileDialog
    window = PhotoWindow() if window_type == 'photo' else VideoWindow()
    window.state.asset = {'source':str(tmp_path/'source.jpg')}
    captured = []
    entered,release = threading.Event(),threading.Event()
    def delayed_save(path,**snapshot):
        captured.append(snapshot)
        entered.set()
        release.wait(3)
    monkeypatch.setattr(project,'save_project',delayed_save)
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(tmp_path/'project.json'),'JSON'))
    window.save_project()
    pump(app,entered.is_set)
    window.padding.setValue(50)
    release.set()
    pump(app,lambda: window.state.project_path is not None and not window.jobs.running)
    assert captured[0]['style']['padding'] == .25
    assert window.state.style['padding'] == .5
    assert window.state.dirty
    window.state.dirty = False
    window.close()
    pump(app,lambda: all(c.active_count == 0 for c in [window.jobs,window.previews,*window.extra_controllers]))


@pytest.mark.parametrize('replacement', [False,True])
def test_window_deferred_result_is_invalidated_by_cancel_or_next_operation(app,replacement):
    from panoprivacy.ui.photo_window import PhotoWindow
    window = PhotoWindow()
    applied = []
    window._result_handler = applied.append
    window._job_result('obsolete')
    if replacement:
        window.run_job(lambda progress,cancel:'fresh',applied.append,'new operation')
        pump(app,lambda: 'fresh' in applied)
        assert applied == ['fresh']
    else:
        window.cancel_job()
        app.processEvents()
        assert applied == []
    window.close()

def test_photo_more_mode_and_region_undo_for_add_remove_toggle_scan(app,monkeypatch):
    import sys
    import types
    from panoprivacy.common import new_region
    from panoprivacy.ui.photo_window import PhotoWindow
    window = PhotoWindow()
    window.state.asset = {'source':'photo.jpg'}
    window.load_thumbnails = lambda: None
    window.request_preview = lambda: None
    assert window.scan_mode.currentData() == 'obvious'
    r = new_region('erp',[.4,.4,.5,.6],origin='manual')
    window.add_region(r)
    window.candidates.setCurrentRow(0)
    window.remove_candidate()
    assert not window.state.regions
    window.undo_regions()
    assert window.state.regions[0]['id'] == r['id']
    window.candidates.item(0).setCheckState(__import__('PySide6.QtCore',fromlist=['Qt']).Qt.CheckState.Unchecked)
    assert not window.state.regions[0]['enabled']
    window.undo_regions()
    assert window.state.regions[0]['enabled']
    modes=[]
    automatic = new_region('erp',[.7,.4,.8,.6],origin='auto')
    def scan(asset,mode,progress,cancel):
        modes.append(mode)
        return [automatic]
    monkeypatch.setitem(sys.modules,'panoprivacy.photo_engine',types.SimpleNamespace(
        scan_photo=scan,photo_thumbnail=lambda *a,**k:np.zeros((20,20,3),np.uint8)))
    window.scan_mode.setCurrentIndex(1)
    window.scan()
    pump(app,lambda:len(window.state.regions)==2 and not window.jobs.running)
    assert modes == ['more']
    window.undo_regions()
    assert [region['id'] for region in window.state.regions] == [r['id']]
    for i in range(25):
        window.add_region(new_region('erp',[.1,.1,.2,.2],origin='manual'))
    for i in range(25):
        window.undo_regions()
    assert len(window.state.regions)==1
    assert window.state.dirty
    window.state.dirty=False
    window.close()


@pytest.mark.parametrize('status,regions,allowed',[
    ('uncertain','valid',True),('uncertain','empty',False),
    ('missing','valid',False),('ok','empty',False),('uncertain','disabled',False)])
def test_video_export_human_review_is_one_use_and_never_bypasses_empty_or_missing(app,tmp_path,monkeypatch,status,regions,allowed):
    import sys
    import types
    from copy import deepcopy
    from PySide6.QtWidgets import QMessageBox,QFileDialog
    from panoprivacy.common import new_region
    from panoprivacy.ui.video_window import VideoWindow
    window=VideoWindow()
    region=new_region('erp',[.4,.3,.6,.7],enabled=regions!='disabled')
    tracks=[{'frame':0,'status':status,'regions':[] if regions=='empty' else [region]}]
    window.state.asset={'source':str(tmp_path/'video.mp4'),'frame_count':1}
    window.state.analysis={'tracks':tracks,'warnings':[]}
    original=deepcopy(window.state.analysis)
    window._set_analysis(window.state.analysis)
    exported=[]
    prompts=[]
    monkeypatch.setattr(QMessageBox,'question',lambda *a,**k:(prompts.append(a[2]) or QMessageBox.StandardButton.Yes))
    monkeypatch.setattr(QMessageBox,'warning',lambda *a,**k:None)
    monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(str(tmp_path/'out.mp4'),'MP4'))
    def export(asset,analysis,style,path,progress,cancel):
        exported.append(deepcopy(analysis))
        return {'output':path}
    monkeypatch.setitem(sys.modules,'panoprivacy.video_engine',types.SimpleNamespace(export_video=export))
    window.export_done=lambda result:None
    window.export_current()
    if allowed:
        pump(app,lambda:len(exported)==1 and not window.jobs.running)
        assert exported[0]['reviewed_uncertain'] is True
        assert exported[0]['tracks'][0]['status']=='uncertain'
        assert prompts and '预览' in prompts[0]
    else:
        app.processEvents()
        assert not exported
    assert window.state.analysis==original
    window.close()

def test_photo_region_undo_is_scoped_to_each_document(app):
    from panoprivacy.common import new_region
    from panoprivacy.ui.photo_window import PhotoWindow
    window=PhotoWindow()
    window.load_thumbnails=lambda:None
    window.request_preview=lambda:None
    first=DocumentState('photo',asset={'source':'first.jpg'})
    second=DocumentState('photo',asset={'source':'second.jpg'})
    window.switch_state(first)
    window.add_region(new_region('erp',[.1,.1,.2,.2],origin='manual'))
    window.switch_state(second)
    window.add_region(new_region('erp',[.6,.1,.7,.2],origin='manual'))
    window.switch_state(first)
    window.undo_regions()
    assert first.regions == []
    assert len(second.regions)==1
    assert len(second.region_undo)==1
    first.dirty=False
    window.close()


def _video_follow_fixture():
    from panoprivacy.ui.video_window import VideoWindow
    window = VideoWindow()
    window.state.asset = {'source':'fixture.mp4', 'frame_count':8, 'fps_num':30, 'fps_den':1}
    window.viewport.resize(600, 400)
    window.mask_preview.blockSignals(True)
    window.mask_preview.setChecked(False)
    window.mask_preview.blockSignals(False)
    renders = []
    window.enqueue_preview = renders.append
    return window, renders


def _request_cached_video_preview(window, frame):
    window.frame = frame
    window._frame_rgb_index = frame
    window._frame_rgb = np.full((100, 200, 3), 120, np.uint8)
    window.request_preview()


def test_video_play_follow_keeps_moving_target_visible_through_nadir_without_changing_project(app):
    from copy import deepcopy
    from panoprivacy.common import new_region
    from panoprivacy.geometry import camera, region_to_view_box
    window, renders = _video_follow_fixture()
    targets = [new_region('view', [.46,.44,.54,.56],
        pose={'yaw':yaw,'pitch':pitch,'roll':0.,'fov':90.},view_size=(400,400))
        for yaw,pitch in [(0.,-7.),(10.,-45.),(0.,-85.),(0.,-90.),(180.,-85.),(180.,-60.)]]
    window.state.analysis = {'tracks':[{'frame':i,'status':'ok','regions':[region]}
                                       for i,region in enumerate(targets)], 'warnings':[]}
    window._set_analysis(window.state.analysis)
    original = deepcopy(window.state.snapshot())
    window.viewport.pose.update(fov=55.)
    previous_up = camera(window.viewport.pose)[1]
    window.toggle_play()
    for frame, region in enumerate(targets):
        _request_cached_video_preview(window, frame)
        result = renders.pop()(None, threading.Event())
        window._show_preview(result)
        pose = window.viewport.display_pose
        assert pose['fov'] == 55.
        box = region_to_view_box(region, pose, window.viewport.display_size)
        assert box is not None
        width, height = window.viewport.display_size
        assert 0 < box[0] < box[2] < width and 0 < box[1] < box[3] < height
        assert abs((box[0]+box[2])/2-width/2) < width*.02
        assert abs((box[1]+box[3])/2-height/2) < height*.02
        current_up = camera(pose)[1]
        assert previous_up @ current_up > .5  # crossing the pole must not flip camera up
        previous_up = current_up
        assert window.follow_target.isChecked()  # programmatic recentering keeps follow enabled
    assert window.state.snapshot() == original
    window.pause()
    window.close()


@pytest.mark.parametrize('space', ['view','erp'])
def test_video_follow_can_start_with_target_along_previous_camera_up(app, space):
    from panoprivacy.common import new_region
    from panoprivacy.geometry import region_to_view_box
    window, renders = _video_follow_fixture()
    if space == 'view':
        region = new_region('view', [.45,.45,.55,.55],
            pose={'yaw':30.,'pitch':-90.,'roll':20.,'fov':90.},view_size=(400,400))
        target_pitch = -90.
    else:
        region = new_region('erp', [.49,.74,.51,.76])
        window.viewport.pose['pitch'] = 45.
        target_pitch = -45.
    window.state.seeds = [{'frame':0,'region':region}]
    window.viewport.pose['fov'] = 70.
    window.toggle_play()
    _request_cached_video_preview(window, 0)
    result = renders.pop()(None, threading.Event())
    assert result['pose']['pitch'] == pytest.approx(target_pitch)
    assert result['pose']['fov'] == 70.
    box = region_to_view_box(region, result['pose'], (600,400))
    assert box is not None and 0 < box[0] < box[2] < 600 and 0 < box[1] < box[3] < 400
    window.pause()
    window.close()


@pytest.mark.parametrize('condition', ['paused','off','missing','empty','disabled','multiple','stale'])
def test_video_follow_leaves_view_unchanged_when_disabled_or_target_unusable(app, condition):
    from copy import deepcopy
    from panoprivacy.common import new_region
    window, renders = _video_follow_fixture()
    region = new_region('erp', [.7,.7,.72,.73], enabled=condition!='disabled')
    regions = [] if condition=='empty' else [region, deepcopy(region)] if condition=='multiple' else [region]
    window.state.analysis = {'tracks':[{'frame':0,'status':'missing' if condition=='missing' else 'ok',
                                       'regions':regions}], 'warnings':[], '_ui_stale':condition=='stale'}
    window._set_analysis(window.state.analysis)
    before = deepcopy(window.viewport.pose)
    if condition!='paused':
        window.toggle_play()
    if condition=='off':
        window.follow_target.setChecked(False)
    _request_cached_video_preview(window, 0)
    result = renders.pop()(None, threading.Event())
    assert result['pose'] == before
    assert window.viewport.pose == before
    window.pause()
    window.close()


def test_video_manual_pan_disables_follow_but_zoom_and_paused_pan_do_not(app):
    window, _ = _video_follow_fixture()
    window.viewport.pose['pitch'] = -20.
    window.viewport.view_changed.emit()
    assert window.follow_target.isChecked()
    window.toggle_play()
    _request_cached_video_preview(window, 0)
    window.viewport.pose['fov'] = 60.
    window.viewport.view_changed.emit()
    assert window.follow_target.isChecked()
    window.viewport.pose['yaw'] += 5.
    window.viewport.view_changed.emit()
    assert not window.follow_target.isChecked()
    window.pause()
    window.close()


def test_video_follow_inflight_pose_does_not_move_correction_to_unseen_frame(app):
    from copy import deepcopy
    from panoprivacy.common import new_region
    window, renders = _video_follow_fixture()
    window.state.analysis = {'tracks':[
        {'frame':0,'status':'ok','regions':[new_region('erp',[.49,.5,.51,.52])]},
        {'frame':1,'status':'ok','regions':[new_region('erp',[.7,.8,.72,.82])]}], 'warnings':[]}
    window._set_analysis(window.state.analysis)
    window.toggle_play()
    _request_cached_video_preview(window, 0)
    window._show_preview(renders.pop()(None, threading.Event()))
    shown_pose = deepcopy(window.viewport.display_pose)
    _request_cached_video_preview(window, 1)
    assert window.viewport.pose != shown_pose
    window.freeze_selection()
    region = window.viewport.region_from_drag(QPointF(250,150),QPointF(350,250))
    window.add_seed(region)
    assert window.state.seeds[0]['frame'] == 0
    assert window.state.seeds[0]['region']['pose'] == shown_pose
    assert not window.play_timer.isActive()
    window.state.dirty = False
    window.close()


@pytest.mark.parametrize('space,seed_roll,box', [
    ('view', 0., [.45,.45,.55,.55]),
    ('view', 32., [.45,.45,.55,.55]),
    ('view', 32., [.6,.35,.7,.45]),
    ('erp', 0., [.4,.45,.42,.47]),
])
def test_video_anchor_restores_seed_up_after_following_through_nadir(app, space, seed_roll, box):
    from copy import deepcopy
    from panoprivacy.common import new_region
    from panoprivacy.geometry import camera
    window, _ = _video_follow_fixture()
    region = new_region(space, box, pose={'yaw':15.,'pitch':-7.,'roll':seed_roll,'fov':90.},
                        view_size=(400,400))
    window.state.seeds = [{'frame':0,'region':region}]
    original = deepcopy(window.state.snapshot())
    window.refresh_anchors()
    window.viewport.pose.update(yaw=130.,pitch=-88.,roll=110.,fov=70.)
    window.frame = 6
    window.toggle_play()
    window.anchor_selected(window.anchors.item(0))
    assert window.frame == 0
    assert not window.play_timer.isActive()
    right, actual_up, forward, _ = camera(window.viewport.pose)
    if space == 'view':
        seed_up = camera(region['pose'])[1]
        expected_up = seed_up - np.dot(seed_up,forward)*forward
        expected_up /= np.linalg.norm(expected_up)
        assert actual_up @ expected_up > .999999
    else:
        assert window.viewport.pose['roll'] == 0.
    assert window.state.snapshot() == original
    window.close()
