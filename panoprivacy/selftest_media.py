"""Tiny generated media for portable-runtime diagnostics; never opens user media."""
from pathlib import Path
import struct


def video_roundtrip(folder):
    from .media_tools import ffmpeg_path
    from .video_process import run_checked
    from .mp4_metadata import pack, restore_spherical_metadata
    from .video_engine import prepare_video, export_video
    from .common import new_region, default_style
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    encoded, reference, source = [folder / name for name in ('generated.mp4','sphere.mp4','源 视频.mp4')]
    run_checked([ffmpeg_path(),'-v','error','-nostdin','-f','lavfi','-i',
        'testsrc2=size=128x64:rate=12:duration=0.5','-f','lavfi','-i',
        'sine=frequency=500:sample_rate=48000:duration=0.5',
        '-c:v','libx265','-pix_fmt','yuv420p10le','-preset','ultrafast',
        '-x265-params','pools=1:frame-threads=1:log-level=error',
        '-color_range','tv','-colorspace','bt709','-color_primaries','bt709',
        '-color_trc','bt709','-c:a','aac',encoded])
    sphere = pack(b'sv3d',pack(b'svhd',bytes(4)+b'PanoramaPrivacy\0')+
                  pack(b'proj',pack(b'prhd',bytes(16))+pack(b'equi',bytes(20))))
    entry = pack(b'hvc1',bytes(78)+sphere)
    stsd = pack(b'stsd',bytes(4)+struct.pack('>I',1)+entry)
    reference.write_bytes(pack(b'ftyp',b'isom0000')+
        pack(b'moov',pack(b'trak',pack(b'mdia',pack(b'minf',pack(b'stbl',stsd))))))
    restore_spherical_metadata(reference,encoded,source)
    asset = prepare_video(source,folder/'video-cache')
    asset['export_encoder'] = 'libx265'
    region = new_region('erp',[.96,.3,1.04,.6])
    analysis = {'tracks':[{'frame':i,'regions':[region],'status':'ok'}
                          for i in range(asset['frame_count'])]}
    result = export_video(asset,analysis,default_style(),folder/'视频 输出.mp4')
    return {key:result[key] for key in ('frame_count','pix_fmt','audio_verified',
                                      'spherical_verified','full_decode_verified','encoder')}
