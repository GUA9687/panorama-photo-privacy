# 组件、来源和发布边界

本仓库发布项目源码及照片版便携包，不包含用户测试媒体、项目、访问令牌或SAM受限权重。项目自有代码尚未单独指定整体授权许可证；第三方权利不因上传此仓库而改变。

- RF-DETR Face ONNX：固定来源和模型卡位于 licenses/RF-DETR-face，按所收录模型卡声明为Apache-2.0。
- YuNet：OpenCV Zoo，MIT，见 licenses/YuNet。
- YOLOv8n COCO导出ONNX：Ultralytics，AGPL-3.0，见 licenses/YOLO-person。只分发ONNX模型，不依赖Ultralytics或PyTorch运行时。
- Qt/PySide6：动态链接；官方包许可证和元数据见licenses。发行包提供应用源码。
- libjpeg-turbo3.2.0：native/jpeg_local.c静态链接该库；许可证及IJG说明见licenses/libjpeg-turbo。上游源码：https://github.com/libjpeg-turbo/libjpeg-turbo/tree/3.2.0 。
- libvips/pyvips、OpenCV、NumPy、Pillow、ORT等：完整组件清单和上游声明见THIRD_PARTY_NOTICES.txt及licenses目录。
- NVIDIA CUDA/cuDNN：使用各组件运行库条款，许可证随包附带；显卡驱动不在包内。
- 部分licenses文件记录早期视频构建使用的FFmpeg组件；照片0.2不包含FFmpeg二进制。

要对外再分发或修改，请先阅读实际组件许可证，不应仅依赖本摘要。不能把全部依赖统称为MIT或宣称不存在分发限制。
