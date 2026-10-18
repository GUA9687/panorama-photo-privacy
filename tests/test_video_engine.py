import threading
import numpy as np
import pytest
from panoprivacy.video_tracking import match_identity, correction_bounds
from panoprivacy.video_engine import validate_video_profile

def test_identity_gate_does_not_choose_nearby_different_face():
    rng=np.random.default_rng(5)
    template=rng.integers(0,256,(32,32,3),dtype=np.uint8)
    scene=np.zeros((100,160,3),np.uint8)
    scene[35:67,65:97]=255-template
    assert match_identity(scene,template,[60,30,100,72])[0] is None

def test_identity_gate_flags_two_indistinguishable_candidates():
    rng=np.random.default_rng(5)
    template=rng.integers(0,256,(20,20,3),dtype=np.uint8)
    scene=np.zeros((100,160,3),np.uint8)
    scene[30:50,50:70]=template; scene[30:50,76:96]=template
    box, status, score=match_identity(scene,template,[60,30,80,50])
    assert box is None and status=='uncertain'

def test_correction_only_recomputes_neighbor_anchor_interval():
    assert correction_bounds([10,40,80],40,100)==(10,80)

def profile():
    return {'codec_type':'video','width':128,'height':64,'pix_fmt':'yuv420p10le','r_frame_rate':'30/1','avg_frame_rate':'30/1','time_base':'1/15360','start_time':'0.0','color_space':'bt709','side_data_list':[{'side_data_type':'Spherical Mapping','projection':'equirectangular'}]}

def test_profile_rejects_unverified_projection_and_vfr():
    p=profile(); assert validate_video_profile(p)['fps_num']==30
    p['side_data_list']=[]
    with pytest.raises(ValueError,match='ERP'): validate_video_profile(p)
    p=profile(); p['avg_frame_rate']='29/1'
    with pytest.raises(ValueError,match='CFR'): validate_video_profile(p)

def test_bidirectional_analysis_tracks_textured_target_from_middle_and_preserves_outside_correction(tmp_path,monkeypatch):
    import cv2
    from panoprivacy.common import fingerprint,new_region
    from panoprivacy.video_engine import analyze_video
    from panoprivacy.geometry import region_center
    import panoprivacy.detector as detector_module
    class NoDetector:
        provider='test/no-network'
        def detect(self,rgb,min_score=.6): return []
    monkeypatch.setattr(detector_module,'FaceDetector',NoDetector)
    path=tmp_path/'moving.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(256,128))
    target=np.random.default_rng(9).integers(40,230,(24,28,3),dtype=np.uint8)
    for i in range(11):
        rgb=np.full((128,256,3),25,np.uint8)
        rgb[52:76,100+i:128+i]=target
        rgb[25:49,170:198]=255-target
        writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    writer.release()
    asset={'source':str(path),'proxy_path':str(path),'fingerprint':fingerprint(path),'frame_count':11}
    seed={'frame':5,'region':new_region('erp',[105/256,52/128,133/256,76/128])}
    result=analyze_video(asset,[seed])
    assert [r['frame'] for r in result['tracks']]==list(range(11))
    assert all(r['status']=='ok' and len(r['regions'])==1 for r in result['tracks'])
    for frame in (0,10):
        expected=new_region('erp',[(100+frame)/256,52/128,(128+frame)/256,76/128])
        actual=result['tracks'][frame]['regions'][0]
        assert abs(region_center(actual)[0]-region_center(expected)[0])<3
    seeds=[{'frame':i,'region':new_region('erp',[(100+i)/256,52/128,(128+i)/256,76/128])} for i in (2,5,8)]
    result['tracks'][0]['message']='untouched sentinel'
    revised=analyze_video(asset,seeds,previous=result,changed_frame=5)
    assert revised['recomputed']==[2,8]
    assert revised['tracks'][0]['message']=='untouched sentinel'

def test_recenter_preserves_sphere_center_and_up_orientation_near_pole():
    from panoprivacy.geometry import camera,view_box_region,region_center
    from panoprivacy.video_tracking import _recenter
    pose={'yaw':150.,'pitch':-78.,'roll':25.,'fov':100.}
    region=view_box_region([410,220,470,280],pose,(640,640))
    recentered=_recenter(region,pose,(640,640))
    yaw,pitch=region_center(region)
    assert abs(recentered['yaw']-yaw)<1e-8
    assert abs(recentered['pitch']-pitch)<1e-8
    old_up=camera(pose)[1]
    _,new_up,forward,_=camera(recentered)
    expected=old_up-np.dot(old_up,forward)*forward; expected/=np.linalg.norm(expected)
    assert np.dot(expected,new_up)>.999

@pytest.mark.parametrize('move',[False,True])
def test_continuously_changing_target_keeps_strong_unique_box_as_uncertain(move):
    from panoprivacy.common import new_region
    from panoprivacy.video_tracking import track_direction
    from panoprivacy.geometry import region_center
    rng=np.random.default_rng(452)
    first=rng.integers(20,235,(24,28,3),dtype=np.uint8)
    final=rng.integers(20,235,(24,28,3),dtype=np.uint8)
    frames=[]
    for i in range(41):
        frame=np.full((128,256,3),25,np.uint8)
        left=100+(i if move else 0)
        frame[52:76,left:left+28]=np.rint(first.astype(float)*(1-min(i/15,1))+final.astype(float)*min(i/15,1)).astype(np.uint8)
        frames.append(frame)
    seed={'frame':0,'region':new_region('erp',[100/256,52/128,128/256,76/128])}
    records=track_direction(lambda i:frames[i],seed,range(1,41))
    assert all(len(r['regions'])==1 for r in records)
    assert any(r['status']=='uncertain' for r in records)
    errors=[abs(region_center(r['regions'][0])[0]-region_center(seed['region'])[0]-(r['frame']*360/256 if move else 0)) for r in records]
    assert max(errors)<(28/256*360)*.1, errors  # within10% of known target width
    first_uncertain=next(i for i,r in enumerate(records) if r['status']=='uncertain')
    assert all(r['status']=='uncertain' for r in records[first_uncertain:])





def test_cumulative_scale_growth_is_blocked_even_with_strong_temporal_matches(monkeypatch):
    import panoprivacy.video_tracking as tracking
    from panoprivacy.common import new_region
    rgb=np.random.default_rng(334).integers(20,235,(128,256,3),dtype=np.uint8)
    seed={'frame':0,'region':new_region('erp',[.45,.4,.55,.6])}
    def expanding(image,template,predicted,threshold=.66):
        x0,y0,x1,y1=predicted;cx=(x0+x1)/2;cy=(y0+y1)/2
        w=(x1-x0)*1.06;h=(y1-y0)*1.06
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2],'ok',.99
    monkeypatch.setattr(tracking,'match_identity',expanding)
    records=tracking.track_direction(lambda i:rgb,seed,range(1,21))
    assert any(not r['regions'] and '尺寸' in r['message'] for r in records)
    assert records[-1]['status']=='missing'

def test_face_refinement_requires_unique_associated_geometry():
    from panoprivacy.video_tracking import unique_associated_face
    predicted=[100,100,160,170]
    face=[110,108,154,163,.94]
    duplicate=[111,109,155,164,.91]
    assert unique_associated_face([face,duplicate,[250,100,300,160,.99]],predicted)==face[:4]
    assert unique_associated_face([[250,100,300,160,.99]],predicted) is None
    assert unique_associated_face([[90,105,120,150,.9],[140,105,170,150,.9]],predicted) is None

def test_neighbor_detection_cannot_replace_target_when_only_neighbor_is_detected():
    from panoprivacy.video_tracking import unique_associated_face
    assert unique_associated_face([[145,110,175,150,.99]],[100,100,160,170]) is None


def test_rf_refinement_must_pass_previous_angular_reference_before_reset(monkeypatch):
    import panoprivacy.video_tracking as tracking
    from panoprivacy.common import new_region
    rgb=np.random.default_rng(81).integers(20,235,(128,256,3),dtype=np.uint8)
    seed={'frame':0,'region':new_region('erp',[.47,.46,.53,.54])}
    calls=[0]
    def face(detector,image,box):
        calls[0]+=1
        if calls[0]==1:return list(box)
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        w,h=(box[2]-box[0])*1.7,(box[3]-box[1])*1.7
        return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
    monkeypatch.setattr(tracking,'_local_face',face)
    monkeypatch.setattr(tracking,'match_identity',lambda image,template,box:(list(box),'ok',.99))
    records=tracking.track_direction(lambda i:rgb,seed,range(1,17),detector=object())
    assert not records[14]['regions'] and '尺寸' in records[14]['message']
    assert records[15]['status']=='missing'

@pytest.mark.parametrize('view_size',[(1024,1024),(960,640)])
def test_stationary_nadir_seed_preserves_angular_area_at_tracking_initialization(view_size):
    from panoprivacy.geometry import view_box_region
    from panoprivacy.video_tracking import track_direction,_angular_area
    erp=np.random.default_rng(991).integers(20,235,(256,512,3),dtype=np.uint8)
    top=view_size[1]/2-15
    region=view_box_region([438,top,530,top+91],{'yaw':0,'pitch':-90,'roll':0,'fov':120},view_size)
    records=track_direction(lambda i:erp,{'frame':0,'region':region},range(1,3))
    assert all(r['regions'] for r in records)
    ratios=[_angular_area(r['regions'][0])/_angular_area(region) for r in records]
    assert all(.9<ratio<1.1 for ratio in ratios),ratios



def test_export_validation_rejects_missing_explicit_source_color_tags():
    from panoprivacy.video_engine import validate_export_stream
    source=dict(profile(),color_range='tv',color_transfer='bt709',color_primaries='bt709')
    asset=validate_video_profile(source)
    missing=dict(source);missing.pop('color_transfer');missing.pop('color_primaries')
    with pytest.raises(ValueError,match='color_transfer'):validate_export_stream(asset,missing)
    assert validate_export_stream(asset,source)['pix_fmt']=='yuv420p10le'
