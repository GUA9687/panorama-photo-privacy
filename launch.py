"""Two portable launchers share this entrypoint and their bundled runtime."""
import argparse
import logging
from pathlib import Path
import sys


def self_test(report, cpu_only=False, photo_only=False):
    import json
    import tempfile
    import numpy as np
    from PIL import Image
    from panoprivacy.common import atomic_json, default_style, new_region
    from panoprivacy.detector import FaceDetector
    from panoprivacy.photo_engine import prepare_photo, export_photo
    from panoprivacy.media_tools import ffmpeg_path, ffprobe_path
    from panoprivacy.ui.photo_window import PhotoWindow
    from PySide6.QtWidgets import QApplication
    import cv2
    import pyvips
    data={"complete":False,"executable":str(Path(sys.executable).resolve()),"checks":{},"errors":[]}
    try:
        if not photo_only:
            data['checks']['ffmpeg']=Path(ffmpeg_path()).is_file()
            data['checks']['ffprobe']=Path(ffprobe_path()).is_file()
        data['checks']['opencv_tracking']=hasattr(cv2,'TrackerCSRT_create')
        app=QApplication.instance() or QApplication([])
        windows=[PhotoWindow()]
        if not photo_only:
            from panoprivacy.ui.video_window import VideoWindow
            windows.append(VideoWindow())
        data['checks']['windows_constructed']=len(windows)==(1 if photo_only else 2)
        for window in windows:window.close()
        with tempfile.TemporaryDirectory(prefix='panoprivacy-selftest-') as tmp:
            tmp=Path(tmp);source=tmp/'source.jpg'
            arr=np.zeros((128,256,3),np.uint8);arr[:,:,0]=np.arange(256,dtype=np.uint8)
            Image.fromarray(arr).save(source)
            asset=prepare_photo(str(source),str(tmp/'cache'))
            style=default_style();style['kind']='solid'
            if photo_only:style.update(shape='ellipse',export_mode='auto',jpeg_quality=98)
            result=export_photo(asset,[new_region('erp',[.4,.3,.6,.7])],style,str(tmp/'output.jpg'))
            data['checks']['photo_native_export']=result['width']==256
            data['photo_export']=result
            if photo_only:
                data['checks']['photo_local_jpeg']=(result.get('export_method')=='local_jpeg'
                    and result.get('outside_coefficients_exact') is True
                    and result.get('outside_pixel_differences')==0)
            if not photo_only:
                from panoprivacy.selftest_media import video_roundtrip
                data['video_roundtrip']=video_roundtrip(tmp/'视频 自检')
                data['checks']['video_native_export']=all(data['video_roundtrip'][key] for key in
                    ('audio_verified','spherical_verified','full_decode_verified'))
            from panoprivacy.photo_confirm import CandidateConfirmer
            confirmation=CandidateConfirmer().confirm(np.zeros((320,320,3),np.uint8),[100,100,200,200])
            data['checks']['secondary_face_model']=confirmation['status']=='unconfirmed'
        detector=FaceDetector(prefer_gpu=not cpu_only)
        detections=detector.detect(np.zeros((576,576,3),np.uint8),.6)
        data['checks']['model_inference']=isinstance(detections,list)
        if photo_only:
            from panoprivacy.person_context import PersonDetector
            person=PersonDetector(prefer_gpu=not cpu_only)
            data['checks']['person_model_inference']=isinstance(person.detect(np.zeros((640,640,3),np.uint8)),list)
        data['provider']=detector.provider;data['provider_warning']=detector.warning
        data['vips_version']='.'.join(str(pyvips.version(i)) for i in range(3))
        data['complete']=all(data['checks'].values())
    except Exception as error:
        data['errors'].append(str(error));logging.exception('Portable self-test failed')
    atomic_json(report,data)
    return 0 if data['complete'] else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['photo','video'])
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--photo-only',action='store_true',help='Self-test only the photo release')
    parser.add_argument('--cpu-only',action='store_true',help='Run diagnostics without GPU inference')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    from panoprivacy.common import app_cache_dir
    logs=app_cache_dir().parent/'Logs';logs.mkdir(exist_ok=True)
    logging.basicConfig(filename=logs/'application.log',level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',encoding='utf-8')
    if args.self_test:
        if args.report is None:parser.error('--self-test requires --report')
        try:
            return self_test(args.report,cpu_only=args.cpu_only,photo_only=args.photo_only)
        except Exception as error:
            # Even an early bundled import failure must yield a report, not a
            # hidden windowed-bootloader dialog that leaves diagnostics hanging.
            from panoprivacy.common import atomic_json
            logging.exception('Portable self-test import failed')
            atomic_json(args.report,{'complete':False,'checks':{},'errors':[str(error)],
                                     'executable':str(Path(sys.executable).resolve())})
            return 1
    name=Path(sys.executable if getattr(sys,'frozen',False) else sys.argv[0]).stem.lower()
    mode=args.mode or ('video' if 'video' in name or '视频' in name else 'photo')
    try:
        from panoprivacy.ui.app import main as gui_main
        return gui_main(mode)
    except Exception:
        logging.exception('Application startup failed')
        from PySide6.QtWidgets import QApplication,QMessageBox
        app=QApplication.instance() or QApplication([])
        QMessageBox.critical(None,'无法启动','程序未能启动。请完整复制软件文件夹。\n诊断日志：'+str(logs/'application.log'))
        return 1


if __name__=='__main__':
    raise SystemExit(main())

