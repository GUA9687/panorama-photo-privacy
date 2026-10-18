"""Build photo v0.2 into a separate folder, preserving the previous video release."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    'rf-detr-medium-face.onnx':
        'b594500a99c3a4287805b7afa15eb74ee559be937eb00315768612846f93cc45',
    'yolov8n-person.onnx':
        '757a583140b8736cbb2a59f8d2c9e6ae015c45a2bbe77baaa2ceec13f284f88b',
    'face_detection_yunet_2023mar.onnx':
        '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4',
}


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def snapshot_sources():
    """Build and deliver the same snapshot even if working files change later."""
    stage = ROOT / 'build' / 'source-snapshots' / uuid.uuid4().hex
    stage.mkdir(parents=True)
    paths = [ROOT / 'launch.py', ROOT / 'requirements-lock.txt',
             ROOT / 'tools' / 'build_portable.py']
    paths.extend(sorted((ROOT / 'panoprivacy').rglob('*.py')))
    paths.extend(path for path in sorted((ROOT / 'native').rglob('*')) if path.is_file())
    for path in paths:
        destination = stage / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', action='store_true')
    args = parser.parse_args()
    if (ROOT / 'dist/PanoramaPhoto-Windows-0.2.0').exists():
        raise RuntimeError('Photo0.2 output exists; choose a new release directory or archive this build explicitly.')
    for filename, expected in MODELS.items():
        path = ROOT / 'models' / filename
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Missing or invalid pinned model: {filename}')
    for relative in ['runtime/jpeg/jpeg_local.exe',
                     'runtime/cuda/cudart64_13.dll', 'runtime/cuda/cudnn64_9.dll',
                     'licenses/inventory.json',
                     'licenses/YuNet/LICENSE', 'licenses/YOLO-person/LICENSE']:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f'Missing offline release resource: {relative}')
    site = Path(importlib.metadata.distribution('pyvips-binary').locate_file(''))
    vips = list(site.glob('libvips*.dll'))
    if not vips:
        raise RuntimeError('Bundled libvips binary not found')
    windows = Path(os.environ.get('SystemRoot', r'C:\Windows'))
    stage = snapshot_sources()
    cuda_names = {path.name.lower() for path in (ROOT / 'runtime/cuda').glob('*.dll')}
    datas = [(str(ROOT / 'models' / filename), 'models') for filename in MODELS]
    datas += [(str(ROOT / 'models/yolov8n-person.provenance.json'), 'models')]
    datas += [(str(ROOT / 'runtime/jpeg'), 'runtime/jpeg'),
              (str(ROOT / 'runtime/cuda'), 'runtime/cuda')]
    spec = ROOT / 'build/Photo02.spec'
    content = f'''from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata
a = Analysis([{str(stage / 'launch.py')!r}], pathex=[{str(stage)!r}],
    binaries={[(str(p), '.') for p in vips]!r},
    datas={datas!r}+collect_data_files('pyvips')+copy_metadata('pyvips-binary'),
    hiddenimports=['_libvips', 'panoprivacy.photo_engine', 'panoprivacy.photo_confirm',
                   'panoprivacy.photo_detection', 'panoprivacy.person_context',
                   'panoprivacy.adaptive_detector', 'panoprivacy.photo_export',
                   'panoprivacy.local_jpeg', 'panoprivacy.ui.photo_window'],
    excludes=['panoprivacy.video_engine', 'panoprivacy.ui.video_window',
              'panoprivacy.selftest_media', 'tkinter', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
              'PySide6.QtQml', 'PySide6.QtQuick', 'PySide6.QtQuickWidgets'],
    noarchive=False)
# Analysis follows CUDA provider imports and also copies their dependencies to
# the bundle root. The application registers runtime/cuda before importing ORT,
# so keep one complete CUDA set at that explicit location instead of two.
cuda_names = {cuda_names!r}
a.binaries = [entry for entry in a.binaries if entry[0].lower() not in cuda_names]
allowed_roots = [Path(p).resolve() for p in {[str(ROOT), str(Path(sys.base_prefix)), str(windows)]!r}]
unexpected = [entry[1] for entry in a.binaries
              if not any(Path(entry[1]).resolve().is_relative_to(root) for root in allowed_roots)]
if unexpected:
    raise RuntimeError('Dependency outside declared build roots: ' + repr(unexpected))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='PanoramaPhoto', debug=False,
    bootloader_ignore_signals=False, strip=False, upx=False, console=False,
    disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name='PanoramaPhoto-Windows-0.2.0')
'''
    spec.write_text(content, encoding='utf-8')
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm',
               '--distpath', str(ROOT / 'dist'), '--workpath', str(ROOT / 'build'), str(spec)]
    if args.clean:
        command.insert(3, '--clean')
    # Inherited desktop PATH may contain unrelated native libraries (for example
    # Poppler's ICU, which has different exports from the Windows ICU used by Qt).
    # Resolve dependencies only from this environment and its explicit runtimes.
    search = [Path(sys.executable).parent, windows / 'System32', windows,
              Path(sys.base_prefix) / 'DLLs', Path(sys.base_prefix),
              site / 'PySide6', site / 'shiboken6', ROOT / 'runtime/cuda']
    build_env = os.environ.copy()
    build_env['PATH'] = os.pathsep.join(str(path) for path in search if path.is_dir())
    build_env.pop('PYTHONPATH', None)
    build_env.pop('PYTHONHOME', None)
    subprocess.run(command, cwd=stage, env=build_env, check=True)
    target = ROOT / 'dist/PanoramaPhoto-Windows-0.2.0'
    shutil.copy2(ROOT / 'docs/照片版0.2使用说明.txt', target / '使用说明.txt')
    shutil.copy2(ROOT / 'THIRD_PARTY_NOTICES.txt', target / 'THIRD_PARTY_NOTICES.txt')
    shutil.copytree(ROOT / 'licenses', target / 'licenses', dirs_exist_ok=True)
    shutil.copytree(stage, target / 'application-source', dirs_exist_ok=True)
    (target / 'application-source' / 'REBUILD.txt').write_text(
        'This source snapshot is the one used for the photo 0.2 executable.\n'
        'To rebuild on Windows x64, create a Python 3.13 virtual environment and install requirements-lock.txt.\n'
        'Copy models/ and runtime/ from ../_internal/ into this source folder; copy licenses/,\n'
        'THIRD_PARTY_NOTICES.txt from the portable folder into it.\n'
        'Copy 使用说明.txt into docs/照片版0.2使用说明.txt in this source folder.\n'
        'Run python tools/build_portable.py. Keep the original pinned model hashes.\n'
        'No user media, cache, project files, credentials or machine-specific environments are needed.\n',
        encoding='utf-8')
    manifest = {'version': '0.2.0', 'launchers': ['PanoramaPhoto.exe'],
                'source_snapshot': stage.name, 'files': []}
    for path in sorted(target.rglob('*')):
        if path.is_file() and path.name != '文件校验.json':
            manifest['files'].append({'path': path.relative_to(target).as_posix(),
                                      'bytes': path.stat().st_size, 'sha256': sha256(path)})
    (target / '文件校验.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'folder': str(target), 'source_snapshot': str(stage),
                      'files': len(manifest['files']),
                      'bytes': sum(f['bytes'] for f in manifest['files'])}), flush=True)


if __name__ == '__main__':
    main()
