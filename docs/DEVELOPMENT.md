# 环境、源码运行与构建

## 已验证环境

- Windows11 x64；Python3.13.5。
- RTX5080 Laptop16GB；约64GB系统RAM；CUDA测试时NVIDIA驱动617.14。
- PySide6 6.11.0、NumPy2.3.4、OpenCV-contrib4.13.0.92、Pillow12.0.0。
- ONNX Runtime GPU1.31.0；CUDA13与cuDNN9运行库。
- pyvips3.2.0、pyvips-binary8.18.7、PyInstaller6.17.0。
- native JPEG helper：libjpeg-turbo3.2.0，MinGW-w64 GCC静态链接。

完整依赖见根目录 `requirements-lock.txt`。以上是已经测试的精确组合，不代表其他版本一定兼容。Linux/macOS不属于本次交付范围；JPEG原生工具使用Windows API。

## 从源码运行

先从Release下载并完整解压便携包，假设放在 `C:\Apps\PanoramaPhoto-Windows-0.2.0`。从源码运行仍需要相同模型和原生运行库，这些大型文件不在Git仓库。

在仓库根目录执行PowerShell：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$portable = 'C:\Apps\PanoramaPhoto-Windows-0.2.0'
New-Item -ItemType Directory -Force models, runtime | Out-Null
Copy-Item "$portable\_internal\models\*" models -Recurse -Force
Copy-Item "$portable\_internal\runtime\*" runtime -Recurse -Force
.\.venv\Scripts\python.exe launch.py --mode photo
```

路径可自行更换。运行时不需要PyTorch、Ultralytics Python包或SAM权重。Python环境安装需联网；便携版日常使用不依赖联网。

## 模型及完整性

| 模型文件 | SHA-256 |
|---|---|
| rf-detr-medium-face.onnx | b594500a99c3a4287805b7afa15eb74ee559be937eb00315768612846f93cc45 |
| face_detection_yunet_2023mar.onnx | 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 |
| yolov8n-person.onnx | 757a583140b8736cbb2a59f8d2c9e6ae015c45a2bbe77baaa2ceec13f284f88b |

模型来源、固定revision和许可证分别见 `licenses/RF-DETR-face/`、`licenses/YuNet/`、`licenses/YOLO-person/`。YOLO ONNX导出参数见 `models/yolov8n-person.provenance.json`；直接复制已验证Release模型最容易保证hash一致。重新导出可能因依赖变化得到不同二进制，不应随意移除哈希校验。

## 测试和自检

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe launch.py --self-test --photo-only --cpu-only --report cpu-check.json
.\.venv\Scripts\python.exe launch.py --self-test --photo-only --report gpu-check.json
Remove-Item Env:QT_QPA_PLATFORM
```

照片专项测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_core.py tests/test_photo_engine.py tests/test_photo_confirmation.py tests/test_photo_detection_v2.py tests/test_photo_shapes.py tests/test_photo_export_integration.py tests/test_local_jpeg.py tests/test_photo_ui_v2.py -q
```

完整228项测试包含早期共享视频回归，部分需要 `runtime/ffmpeg/ffmpeg.exe` 和 `ffprobe.exe`。照片0.2便携包没有分发FFmpeg；只验证照片时使用专项命令和 `--photo-only`。原完整测试结果不能解释为照片包中包含视频运行环境。

便携EXE也支持同样自检参数：`PanoramaPhoto.exe --self-test --photo-only --cpu-only --report C:\Temp\photo-check.json`。检查JSON中complete和provider；GPU自检只有provider为CUDAExecutionProvider才说明实际启用CUDA。照片自检还要求真正完成local_jpeg导出及系数/像素验证，回退重编码不能代替该项。

## 构建JPEG原生工具（可选）

普通源码运行可复制Release里的 `runtime/jpeg/jpeg_local.exe`。需要重编译时，准备libjpeg-turbo3.2.0源代码、同一MinGW-w64 GCC工具链、CMake、Ninja和NASM。以下为标准构建路径示例；本次Release的原生工具已实际构建并验证，新的工具链可能产生不同哈希。

```powershell
cmake -S C:\src\libjpeg-turbo -B C:\build\libjpeg -G Ninja -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build C:\build\libjpeg
.\native\build-jpeg.ps1 -LibjpegSource C:\src\libjpeg-turbo -LibjpegBuild C:\build\libjpeg
```

脚本要求构建目录中的 `libjpeg.a`，会把工具写入 `runtime/jpeg/jpeg_local.exe`。系统依赖为bcrypt、KERNEL32、msvcrt和SHELL32。原生代码及许可证随源码提供。

## 打包

补齐模型和runtime后：

```powershell
.\.venv\Scripts\python.exe tools/build_portable.py
```

输出是 `dist/PanoramaPhoto-Windows-0.2.0`。已存在时脚本会停止，请先将旧产物归档到其他目录，再构建。脚本固定校验三个模型哈希，创建不可变源码快照，限制构建DLL搜索路径，并为发行文件生成 `文件校验.json`。不会覆盖旧的 `PanoramaPrivacy-Windows` 发行目录。

运行库和模型被Git忽略；不应把个人照片、工程、缓存、日志或认证信息提交到仓库。修改程序后需重新测试和构建，不能只改ZIP中的源码就声称EXE已更新。
