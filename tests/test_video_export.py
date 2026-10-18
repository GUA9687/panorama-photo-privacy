"""CPU-only tiny real media tests; no user sample or GPU use."""
import hashlib
import subprocess
import threading
from pathlib import Path
import numpy as np
import pytest
from panoprivacy.common import Cancelled, default_style, new_region
from panoprivacy.media_tools import ffmpeg_path
from panoprivacy.mp4_metadata import pack, restore_spherical_metadata
from panoprivacy.video_engine import prepare_video, video_frame, export_video, _timing
from panoprivacy.video_pixels import redact_planes
from fractions import Fraction
from test_video_metadata import movie

@pytest.fixture
def tiny_asset(tmp_path):
    encoded=tmp_path/'plain.mp4'
    subprocess.run([str(ffmpeg_path()),'-v','error','-f','lavfi','-i','testsrc2=size=128x64:rate=12:duration=0.5',
                    '-f','lavfi','-i','sine=frequency=500:sample_rate=48000:duration=0.5',
                    '-c:v','libx265','-pix_fmt','yuv420p10le','-preset','ultrafast','-x265-params','pools=1:frame-threads=1:log-level=error',
                    '-color_range','tv','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                    '-c:a','aac',str(encoded)],check=True)
    reference=tmp_path/'metadata.mp4'; reference.write_bytes(pack(b'ftyp',b'isom0000')+movie())
    source=tmp_path/'源 视频.mp4'
    restore_spherical_metadata(reference,encoded,source)
    asset=prepare_video(source,tmp_path/'cache')
    asset['export_encoder']='libx265'  # CPU regression tests must not acquire GPU allocation.
    return asset

def tracks(asset):
    region=new_region('erp',[.4,.3,.6,.6])
    return {'tracks':[{'frame':i,'regions':[region],'status':'ok','message':''} for i in range(asset['frame_count'])]}

def test_real_tenbit_export_preserves_source_audio_timing_sphere(tiny_asset,tmp_path):
    asset=tiny_asset
    before=hashlib.sha256(Path(asset['source']).read_bytes()).hexdigest()
    assert asset['frame_count']==6 and video_frame(asset,5).shape==(64,128,3)
    result=export_video(asset,tracks(asset),dict(default_style(),kind='solid'),tmp_path/'输出.mp4')
    assert result['pix_fmt']=='yuv420p10le'
    assert result['audio_packets']>0 and result['audio_verified'] and result['full_decode_verified']
    assert hashlib.sha256(Path(asset['source']).read_bytes()).hexdigest()==before
    with pytest.raises(FileExistsError): export_video(asset,tracks(asset),default_style(),tmp_path/'输出.mp4')

def test_export_cancellation_removes_partial_and_does_not_publish(tiny_asset,tmp_path):
    cancel=threading.Event()
    def progress(event):
        if event['stage']=='export': cancel.set()
    output=tmp_path/'cancelled.mp4'
    with pytest.raises(Cancelled): export_video(tiny_asset,tracks(tiny_asset),default_style(),output,progress,cancel)
    assert not output.exists()
    assert not list(tmp_path.glob('.*.encoding.mp4'))
    assert not list(tmp_path.glob('.*.verified.mp4'))

def test_missing_tracks_and_changed_source_are_rejected(tiny_asset,tmp_path):
    analysis=tracks(tiny_asset); analysis['tracks'][2]['status']='missing'
    with pytest.raises(ValueError,match='Unresolved'): export_video(tiny_asset,analysis,default_style(),tmp_path/'out.mp4')
    with Path(tiny_asset['source']).open('ab') as file: file.write(b'x')
    with pytest.raises(ValueError,match='changed'): video_frame(tiny_asset,0)

def test_actual_irregular_frame_timestamps_are_rejected(tmp_path):
    output=tmp_path/'irregular.mp4'
    subprocess.run([str(ffmpeg_path()),'-v','error','-f','lavfi','-i','testsrc2=size=128x64:rate=12:duration=0.5',
                    '-vf','setpts=(N+gte(N\\,3))/(12*TB)','-fps_mode','passthrough','-c:v','libx264',str(output)],check=True)
    with pytest.raises(ValueError,match='CFR'): _timing(output,Fraction(12))

@pytest.mark.parametrize('kind',['blur','mosaic','solid'])
def test_native_plane_redaction_changes_only_mask_samples(kind):
    from panoprivacy.geometry import region_mask_tiles
    rng=np.random.default_rng(5)
    planes=[rng.integers(64,940,(h,w),dtype=np.uint16) for h,w in ((64,128),(32,64),(32,64))]
    original=[p.copy() for p in planes]
    region=new_region('erp',[.96,.3,1.04,.6])
    masks=[np.zeros(p.shape,bool) for p in planes]
    for x,y,mask in region_mask_tiles(region,(128,64),padding=.25):
        yy,xx=np.nonzero(mask); yy+=y;xx+=x
        masks[0][yy,xx]=True
        masks[1][yy//2,xx//2]=True;masks[2][yy//2,xx//2]=True
    redact_planes(planes,[region],dict(default_style(),kind=kind))
    assert np.any(planes[0]!=original[0])
    for plane,before,mask in zip(planes,original,masks):
        assert np.array_equal(plane[~mask],before[~mask])
        assert plane.max()<=1023

def test_reviewed_uncertain_region_can_export_without_changing_status(tiny_asset,tmp_path):
    analysis=tracks(tiny_asset)
    analysis['tracks'][2]['status']='uncertain'
    with pytest.raises(ValueError,match='Unresolved'):
        export_video(tiny_asset,analysis,default_style(),tmp_path/'unreviewed.mp4')
    analysis['reviewed_uncertain']=True
    result=export_video(tiny_asset,analysis,default_style(),tmp_path/'reviewed.mp4')
    assert result['full_decode_verified']
    assert analysis['tracks'][2]['status']=='uncertain'
