from __future__ import annotations
import json
import subprocess
from .common import resource_path


def _tool(name):
    path=resource_path('runtime/ffmpeg/'+name+'.exe')
    if not path.is_file():raise FileNotFoundError(f"程序资源缺少 {name}.exe，请完整复制软件文件夹。")
    return str(path)


def ffmpeg_path():return _tool('ffmpeg')
def ffprobe_path():return _tool('ffprobe')


def probe(path):
    result=subprocess.run([ffprobe_path(),'-v','error','-show_streams','-show_format','-of','json',str(path)],
                          capture_output=True,text=True,encoding='utf-8',errors='replace',
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=120)
    if result.returncode:raise ValueError('无法读取媒体文件：'+result.stderr[-1500:])
    return json.loads(result.stdout)

