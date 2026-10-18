"""CFR mono ERP MP4 preparation, target-only analysis, and native-depth export."""
from pathlib import Path
from fractions import Fraction
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import uuid
import cv2
import numpy as np

from .common import Cancelled, check_cancel, emit, fingerprint, ensure_distinct_output
from .media_tools import ffmpeg_path, ffprobe_path, probe
from .mp4_metadata import spherical_boxes, restore_spherical_metadata
from .video_process import run_checked, spawn, kill_on_cancel, read_frame_bytes


def validate_video_profile(stream):
    w,h=int(stream['width']),int(stream['height'])
    if w!=2*h or w%2 or h%2:
        raise ValueError('Only even-sized full mono ERP 2:1 video is supported')
    sides=stream.get('side_data_list',[])
    spherical=[x for x in sides if x.get('side_data_type')=='Spherical Mapping']
    if not spherical or spherical[0].get('projection')!='equirectangular':
        raise ValueError('Input is not verified ERP; spherical equirectangular metadata is required')
    if any(x.get('side_data_type')=='Stereo 3D' and x.get('type') not in (None,'2D') for x in sides):
        raise ValueError('Stereo ERP is not supported')
    if any(abs(float(spherical[0].get(k,0)))>.001 for k in ('yaw','pitch','roll')):
        raise ValueError('Non-zero spherical orientation metadata is not supported')
    if stream.get('sample_aspect_ratio','1:1') not in ('1:1','N/A'):
        raise ValueError('Non-square video pixels are not supported')
    if any(x.get('side_data_type')=='Display Matrix' and float(x.get('rotation',0))!=0 for x in sides):
        raise ValueError('Rotated video track is not supported')
    pix=stream.get('pix_fmt')
    if pix not in ('yuv420p','yuv420p10le'):
        raise ValueError('Only source 8-bit or 10-bit YUV420 is supported; no automatic depth conversion')
    try:
        fps=Fraction(stream['avg_frame_rate'])
        nominal=Fraction(stream['r_frame_rate'])
    except (ValueError,ZeroDivisionError,KeyError) as error:
        raise ValueError('CFR frame rate is unavailable') from error
    if fps<=0 or fps!=nominal:
        raise ValueError('Only CFR video is supported; variable frame timing detected')
    if abs(float(stream.get('start_time',0)))>1e-6:
        raise ValueError('Non-zero video start time is not supported')
    if stream.get('color_space','bt709') not in ('bt709','unknown'):
        raise ValueError('First profile supports BT709 video only')
    if stream.get('color_transfer','bt709') not in ('bt709','unknown'):
        raise ValueError('HDR or non-BT709 transfer is unsupported')
    return {'width':w,'height':h,'fps_num':fps.numerator,'fps_den':fps.denominator,
            'pix_fmt':pix,'depth':10 if pix=='yuv420p10le' else 8,
            'color_range':stream.get('color_range','tv'),'color_space':stream.get('color_space','bt709'),
            'color_transfer':stream.get('color_transfer','bt709'),'color_primaries':stream.get('color_primaries','bt709'),
            'time_base':stream.get('time_base'),'sphere':spherical[0]}


def _verify_source(asset):
    if fingerprint(asset['source'])!=asset['fingerprint']:
        raise ValueError('Source file changed after import; import it again before processing')


def _timing(path, fps, cancel=None):
    """Decode PTS to a disk spool, then verify every interval, not just averages."""
    with tempfile.TemporaryFile(mode='w+b') as output:
        run_checked([ffprobe_path(),'-v','error','-select_streams','v:0','-show_frames',
                     '-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',path],cancel,output)
        output.seek(0)
        count=0
        for raw in output:
            check_cancel(cancel)
            field=raw.decode('ascii','replace').strip().split(',')[0]
            if not field: continue
            try: timestamp=float(field)
            except ValueError: continue  # ffprobe side-data lines contain no timestamp.
            if abs(timestamp-count/float(fps))>max(2e-6,1/float(fps)*.001):
                raise ValueError(f'Only CFR starting at zero is supported; frame {count} timestamp={timestamp}')
            count+=1
    if count<1:
        raise ValueError('No decodable video frames')
    return count


def prepare_video(source,cache_dir,progress=None,cancel=None):
    check_cancel(cancel)
    source=Path(source).resolve()
    if source.suffix.lower() not in ('.mp4','.mov'):
        raise ValueError('First video profile accepts spherical MP4/MOV only')
    original=fingerprint(source)
    info=probe(source)
    videos=[s for s in info['streams'] if s.get('codec_type')=='video']
    if len(videos)!=1:
        raise ValueError('Exactly one video track is required')
    asset=validate_video_profile(videos[0])
    if not any(spherical_boxes(source)):
        raise ValueError('MP4 spherical metadata is absent')
    audio=[s for s in info['streams'] if s.get('codec_type')=='audio']
    if any(s.get('codec_name') not in ('aac','mp3','alac') for s in audio):
        raise ValueError('Audio codec cannot be copied into the supported MP4 output')
    cache=Path(cache_dir).resolve()/('video-'+uuid.uuid4().hex)
    cache.mkdir(parents=True)
    proxy=cache/'proxy.mp4'
    preview=cache/'preview.jpg'
    fps=Fraction(asset['fps_num'],asset['fps_den'])
    try:
        emit(progress,'probe',0,'正在验证全部帧时间和球面布局')
        count=_timing(source,fps,cancel)
        emit(progress,'proxy',.1,'正在生成保留逐帧对应关系的预览视频')
        width=min(2048,asset['width']); width-=width%4
        height=width//2
        run_checked([ffmpeg_path(),'-v','error','-nostdin','-i',source,'-map','0:v:0','-an',
                     '-vf',f'scale={width}:{height}:flags=lanczos','-fps_mode','passthrough',
                     '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
                     '-g','12','-keyint_min','12','-sc_threshold','0','-bf','0',
                     '-video_track_timescale',str(fps.numerator),'-movflags','+faststart',proxy],cancel)
        if _timing(proxy,fps,cancel)!=count:
            raise ValueError('Proxy frame mapping differs from source')
        asset.update(kind='video',source=str(source),fingerprint=original,cache_dir=str(cache),
                     proxy_path=str(proxy),preview_path=str(preview),frame_count=count,
                     duration=float(count/fps),audio=audio,proxy_width=width,proxy_height=height)
        first=video_frame(asset,0)
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(first,cv2.COLOR_RGB2BGR))
        if not ok: raise RuntimeError('Could not write preview')
        encoded.tofile(preview)
        _verify_source(asset)
        emit(progress,'ready',1,'视频导入完成；请选择目标脸部')
        return asset
    except BaseException:
        proxy.unlink(missing_ok=True); preview.unlink(missing_ok=True)
        try: cache.rmdir()
        except OSError: pass
        raise


def video_frame(asset,frame):
    _verify_source(asset)
    frame=int(frame)
    if not 0<=frame<asset['frame_count']: raise IndexError('Video frame out of range')
    cap=cv2.VideoCapture(str(asset['proxy_path']))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=cap.read()
        if not ok: raise RuntimeError(f'Cannot decode proxy frame {frame}')
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    finally: cap.release()


def analyze_video(asset,seeds,previous=None,changed_frame=None,progress=None,cancel=None):
    from .video_tracking import track_direction, correction_bounds
    from .detector import FaceDetector
    _verify_source(asset); check_cancel(cancel)
    count=asset['frame_count']
    seeds=sorted(copy.deepcopy(seeds),key=lambda seed:seed['frame'])
    if not seeds: raise ValueError('Select the target on a clear frame first')
    if len({s['frame'] for s in seeds})!=len(seeds): raise ValueError('Only one target anchor is allowed per frame')
    if any(not 0<=s['frame']<count for s in seeds): raise ValueError('Anchor frame out of range')
    start,end=0,count-1
    if previous is not None and changed_frame is not None:
        if len(previous.get('tracks',[]))!=count: raise ValueError('Previous analysis frame count differs')
        start,end=correction_bounds([s['frame'] for s in seeds],changed_frame,count)
        tracks=copy.deepcopy(previous['tracks'])
    else:
        tracks=[{'frame':i,'regions':[],'status':'missing','message':'尚未分析'} for i in range(count)]
    detector=FaceDetector()
    cap=cv2.VideoCapture(str(asset['proxy_path']))
    last=-2
    def read(index):
        nonlocal last
        check_cancel(cancel)
        if index!=last+1: cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,bgr=cap.read(); last=index
        if not ok: raise RuntimeError(f'Cannot decode proxy frame {index}')
        return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    done=0
    def advanced(frame):
        nonlocal done
        done+=1
        emit(progress,'tracking',min(1.,done/max(1,end-start+1)),f'正在跟踪第 {frame+1}/{count} 帧')
    try:
        # Midpoint partition ensures each frame is tracked from its closest user anchor.
        for pos,seed in enumerate(seeds):
            low=0 if pos==0 else (seeds[pos-1]['frame']+seed['frame'])//2+1
            high=count-1 if pos==len(seeds)-1 else (seed['frame']+seeds[pos+1]['frame'])//2
            low,high=max(start,low),min(end,high)
            if low>high: continue
            sf=seed['frame']
            if start<=sf<=end:
                tracks[sf]={'frame':sf,'regions':[seed['region']],'status':'ok','message':'用户校正锚点'}
                advanced(sf)
            for indices in (range(sf-1,low-1,-1),range(sf+1,high+1)):
                if not indices: continue
                records=track_direction(read,seed,indices,detector,cancel,advanced)
                for record in records:
                    if start<=record['frame']<=end: tracks[record['frame']]=record
    finally: cap.release()
    _verify_source(asset)
    warnings=[]
    for record in tracks:
        if record['status']=='ok': continue
        if warnings and warnings[-1]['end']==record['frame']-1:
            warnings[-1]['end']=record['frame']
        else: warnings.append({'start':record['frame'],'end':record['frame'],'message':'目标缺失或不确定，请逐段检查并添加校正锚点。'})
    return {'tracks':tracks,'warnings':warnings,'method':'local-template+local-RF-DETR-refinement+cumulative-angular-size-guard; bidirectional anchors',
            'provider':getattr(detector,'provider','unknown'),'recomputed':[start,end]}


def _audio_digest(path,cancel):
    with tempfile.TemporaryFile(mode='w+b') as output:
        run_checked([ffprobe_path(),'-v','error','-select_streams','a','-show_packets','-show_data_hash','sha256',
                     '-show_entries','packet=stream_index,pts_time,dts_time,duration_time,size,data_hash',
                     '-of','compact=p=0:nk=0',path],cancel,output)
        output.seek(0); digest=hashlib.sha256(); count=0
        for line in output:
            check_cancel(cancel)
            if b'data_hash=' not in line: continue
            # Video is stream0 in both accepted input and export; index may differ with other tracks.
            fields=[p for p in line.strip().split(b'|') if not p.startswith(b'stream_index=')]
            digest.update(b'|'.join(fields)+b'\n'); count+=1
        return count,digest.hexdigest()


def _probe_nvenc(asset,cancel):
    """Actually encode one synthetic frame with the current media specification."""
    rate=f"{asset['fps_num']}/{asset['fps_den']}"
    depth=asset.get('depth',10 if asset['pix_fmt']=='yuv420p10le' else 8)
    command=[ffmpeg_path(),'-v','error','-nostdin','-f','lavfi','-i',
             f"color=c=black:s={asset['width']}x{asset['height']}:r={rate}",
             '-frames:v','1','-an','-pix_fmt',asset['pix_fmt'],'-c:v','hevc_nvenc',
             '-profile:v','main10' if depth==10 else 'main','-preset','p6','-tune','hq',
             '-rc','vbr','-cq','14','-b:v','0','-f','null','-']
    run_checked(command,cancel=cancel,timeout=15)


def select_video_encoder(asset,progress=None,cancel=None):
    check_cancel(cancel)
    override=asset.get('export_encoder')
    if override is not None:
        if override not in ('libx265','hevc_nvenc'): raise ValueError('Unsupported video encoder')
        emit(progress,'encoder',0,f'使用指定编码器：{override}')
        return override
    emit(progress,'encoder',0,'正在用当前视频规格实测 NVENC 硬件编码（最多 15 秒）')
    reason='当前视频规格的一帧 NVENC 编码通过'
    fallback=False
    try:
        _probe_nvenc(asset,cancel)
        encoder='hevc_nvenc'
        message='NVENC 实测通过，将使用 NVIDIA 硬件编码'
    except Cancelled:
        raise
    except (RuntimeError,OSError,TimeoutError) as error:
        check_cancel(cancel)
        encoder='libx265'; fallback=True; reason=str(error)
        message='NVENC 当前不可用，已选择 CPU libx265（较慢）：'+reason[-600:]
    check_cancel(cancel)
    asset['encoder_probe']={key:asset[key] for key in ('width','height','pix_fmt','fps_num','fps_den')}
    asset['encoder_probe'].update(encoder=encoder,fallback=fallback,reason=reason)
    emit(progress,'encoder',0,message)
    return encoder


def validate_export_tracks(analysis,count):
    tracks=analysis.get('tracks',[])
    if len(tracks)!=count or [r.get('frame') for r in tracks]!=list(range(count)):
        raise ValueError('Analysis must contain every source frame in order')
    for record in tracks:
        status=record.get('status')
        regions=[r for r in record.get('regions',[]) if r.get('enabled',True)]
        if len(regions)!=1 or status not in ('ok','uncertain'):
            raise ValueError('Unresolved target frames remain; add corrections before export')
        box=regions[0].get('box',[])
        if len(box)!=4 or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Unresolved target frames remain; invalid target region')
        if status=='uncertain' and analysis.get('reviewed_uncertain') is not True:
            raise ValueError('Unresolved target frames remain; review uncertain regions or add corrections before export')
    return tracks


def hevc_color_filter(asset):
    """Explicit HEVC VUI values; some NVENC drivers omit primaries/transfer tags."""
    fields=[]
    for source,field in (('color_space','matrix_coefficients'),('color_primaries','colour_primaries'),('color_transfer','transfer_characteristics')):
        if asset.get(source)=='bt709':fields.append(f'{field}=1')
    if asset.get('color_range') in ('tv','pc'):
        fields.append('video_full_range_flag='+('1' if asset['color_range']=='pc' else '0'))
    return 'hevc_metadata='+':'.join(fields) if fields else None


def validate_export_stream(asset,stream):
    validated=validate_video_profile(stream)
    for key in ('width','height','fps_num','fps_den','pix_fmt'):
        if validated[key]!=asset[key]:raise ValueError(f'Export changed source {key}')
    for key in ('color_range','color_space','color_transfer','color_primaries'):
        expected=asset.get(key)
        if expected not in (None,'unknown') and stream.get(key)!=expected:
            raise ValueError(f'Export changed or omitted source {key}')
    return validated


def export_video(asset,analysis,style,destination,progress=None,cancel=None):
    from .video_pixels import redact_planes
    _verify_source(asset); check_cancel(cancel)
    ensure_distinct_output(asset['source'],destination)
    destination=Path(destination).resolve()
    if destination.suffix.lower()!='.mp4': raise ValueError('Video output must use .mp4')
    count=asset['frame_count']
    tracks=validate_export_tracks(analysis,count)
    if style.get('kind','blur') not in ('blur','mosaic','solid'): raise ValueError('Unsupported privacy style')
    source=Path(asset['source']); w,h=asset['width'],asset['height']
    depth=asset.get('depth',10 if asset['pix_fmt']=='yuv420p10le' else 8)
    pix=asset['pix_fmt']; rate=f"{asset['fps_num']}/{asset['fps_den']}"
    destination.parent.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    encoded=destination.parent/f'.{destination.stem}.{token}.encoding.mp4'
    spherical=destination.parent/f'.{destination.stem}.{token}.verified.mp4'
    encoder_name=select_video_encoder(asset,progress,cancel)
    decode=[ffmpeg_path(),'-v','error','-nostdin','-threads','4','-i',source,'-map','0:v:0','-an',
            '-fps_mode','passthrough','-pix_fmt',pix,'-f','rawvideo','pipe:1']
    encode=[ffmpeg_path(),'-v','error','-nostdin','-f','rawvideo','-pix_fmt',pix,'-s',f'{w}x{h}',
            '-r',rate,'-i','pipe:0','-i',source,'-map','0:v:0','-map','1:a?','-map_metadata','1',
            '-c:v',encoder_name,'-profile:v','main10' if depth==10 else 'main','-tag:v','hvc1','-c:a','copy']
    if encoder_name=='libx265': encode+=['-preset','medium','-crf','16','-x265-params','pools=4:frame-threads=2:log-level=error']
    else: encode+=['-preset','p6','-tune','hq','-rc','vbr','-cq','14','-b:v','0']
    for key,flag in (('color_range','-color_range'),('color_space','-colorspace'),('color_transfer','-color_trc'),('color_primaries','-color_primaries')):
        value=asset.get(key,'bt709' if key!='color_range' else 'tv')
        if value not in (None,'unknown'): encode += [flag,str(value)]
    color_filter=hevc_color_filter(asset)
    if color_filter:encode+=['-bsf:v',color_filter]
    encode+=['-video_track_timescale',str(asset['fps_num']),'-fps_mode','passthrough',encoded]
    bytes_per=2 if depth==10 else 1
    buffer=bytearray(w*h*3//2*bytes_per)
    try:
        with tempfile.TemporaryFile() as dlog,tempfile.TemporaryFile() as elog:
            decoder=spawn(decode,stdout=subprocess.PIPE,stderr=dlog,bufsize=1024*1024)
            encoder=None
            try:
                encoder=spawn(encode,stdin=subprocess.PIPE,stderr=elog,bufsize=1024*1024)
                with kill_on_cancel([decoder,encoder],cancel):
                    for i,record in enumerate(tracks):
                        check_cancel(cancel)
                        length=read_frame_bytes(decoder.stdout,buffer)
                        check_cancel(cancel)
                        if length!=len(buffer): raise RuntimeError(f'Unexpected source frame length at frame {i}')
                        flat=np.frombuffer(buffer,dtype='<u2' if depth==10 else np.uint8)
                        y=flat[:w*h].reshape(h,w)
                        u=flat[w*h:w*h*5//4].reshape(h//2,w//2)
                        v=flat[w*h*5//4:].reshape(h//2,w//2)
                        redact_planes((y,u,v),record['regions'],style,depth,asset.get('color_range','tv')!='pc',cancel)
                        encoder.stdin.write(buffer)
                        emit(progress,'export',(i+1)/count*.85,f'正在处理原始分辨率第 {i+1}/{count} 帧')
                    if decoder.stdout.read(1): raise RuntimeError('Unexpected extra source frame')
                    encoder.stdin.close()
                    while decoder.poll() is None or encoder.poll() is None:
                        check_cancel(cancel)
                        try:
                            decoder.wait(timeout=.2); encoder.wait(timeout=.2)
                        except subprocess.TimeoutExpired: pass
                    if decoder.returncode or encoder.returncode:
                        dlog.seek(0); elog.seek(0)
                        raise RuntimeError('Video codec failed: '+(dlog.read()+elog.read())[-6000:].decode('utf8','replace'))
            except BaseException:
                check_cancel(cancel)
                raise
            finally:
                for process in (decoder,encoder):
                    if process is not None:
                        if process.poll() is None: process.kill()
                        process.wait()
                        for stream in (process.stdout,process.stdin):
                            if stream is not None and not stream.closed:
                                # Windows buffered stdin may attempt a final flush to a killed child.
                                try: stream.close()
                                except OSError: pass
        emit(progress,'verify',.86,'正在恢复球面元数据并完整解码验证')
        restore_spherical_metadata(source,encoded,spherical,cancel)
        run_checked([ffmpeg_path(),'-v','error','-xerror','-nostdin','-i',spherical,'-map','0:v:0','-map','0:a?',
                     '-f','null','-'],cancel)
        result_info=probe(spherical)
        output_video=next(s for s in result_info['streams'] if s.get('codec_type')=='video')
        validate_export_stream(asset,output_video)
        if _timing(spherical,Fraction(asset['fps_num'],asset['fps_den']),cancel)!=count:
            raise ValueError('Export frame count differs from source')
        source_audio=_audio_digest(source,cancel); output_audio=_audio_digest(spherical,cancel)
        if source_audio!=output_audio: raise ValueError('Copied audio packet payload or timing differs')
        _verify_source(asset); check_cancel(cancel)
        # No-clobber on Windows; Linux hard-link also makes publication atomic and refuses an existing file.
        if os.name=='nt': os.rename(spherical,destination)
        else:
            os.link(spherical,destination); spherical.unlink()
        emit(progress,'done',1,'视频导出完成，帧、音频与球面元数据验证通过')
        return {'output':str(destination),'width':w,'height':h,'frame_count':count,'pix_fmt':pix,
                'audio_packets':source_audio[0],'audio_verified':True,'spherical_verified':True,'full_decode_verified':True,
                'encoder':encoder_name,'encoder_probe':asset.get('encoder_probe')}
    finally:
        encoded.unlink(missing_ok=True); spherical.unlink(missing_ok=True)
