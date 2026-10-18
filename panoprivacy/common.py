from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path


class Cancelled(Exception):
    """A user canceled a processing job; sources remain untouched."""


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled("操作已取消")


def emit(progress, stage, fraction, message):
    if progress is not None:
        progress({"stage": str(stage), "fraction": max(0., min(1., float(fraction))), "message": str(message)})


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path):
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def ensure_distinct_output(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if os.path.normcase(str(source)) == os.path.normcase(str(destination)):
        raise ValueError("输出不能覆盖原文件，请选择另一个文件名。")
    if destination.exists():
        raise FileExistsError("输出文件已经存在，请选择新的文件名。")
    if not destination.parent.is_dir():
        raise ValueError("输出文件夹不存在。")
    return destination


def app_cache_dir():
    root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "PanoramaPrivacy" / "Cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def resource_path(relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("资源路径必须位于程序目录中")
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
        roots.append(Path(sys._MEIPASS))
    roots.append(Path(__file__).resolve().parent.parent)
    for root in roots:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return roots[0] / relative


def default_style():
    return {"kind": "blur", "strength": 1.0, "padding": .25, "color": "#202020"}


def validate_style(style):
    if style.get("kind") not in {"blur", "mosaic", "solid"}:
        raise ValueError("未知遮挡方式")
    if style.get('shape', 'rect') not in {'rect', 'ellipse', 'circle'}:
        raise ValueError('未知遮挡形状')
    if style.get('export_mode', 'auto') not in {'auto', 'jpeg_reencode'}:
        raise ValueError('未知照片导出方式')
    if style.get('jpeg_quality', 98) not in {95, 98, 100}:
        raise ValueError('JPEG 质量需要为 95、98 或 100')
    for key, low, high in [("strength", .2, 3.), ("padding", 0., 1.)]:
        value = float(style.get(key, default_style()[key]))
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 超出允许范围")
    color = style.get("color", "#202020")
    if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
        raise ValueError("颜色需要 #RRGGBB 格式")
    try:
        int(color[1:], 16)
    except ValueError as error:
        raise ValueError("颜色需要 #RRGGBB 格式") from error


def new_region(space, box, pose=None, view_size=(1, 1), **fields):
    if space not in {"erp", "view"}:
        raise ValueError("未知全景坐标类型")
    box = [float(v) for v in box]
    if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("遮挡框坐标无效")
    if space == "erp" and not (0 <= box[1] < box[3] <= 1 and 0 < box[2]-box[0] <= 1):
        raise ValueError("全景遮挡框超出范围")
    if space == "view" and (max(abs(v) for v in box) > 8):
        raise ValueError("透视遮挡框超出范围")
    view_size = [int(v) for v in view_size]
    if len(view_size) != 2 or min(view_size) <= 0:
        raise ValueError("预览尺寸无效")
    pose = {**{"yaw": 0., "pitch": 0., "fov": 90., "roll": 0.}, **(pose or {})}
    pose = {k: float(pose[k]) for k in ("yaw", "pitch", "fov", "roll")}
    if not all(math.isfinite(v) for v in pose.values()) or not -90 <= pose["pitch"] <= 90 or not 1 <= pose["fov"] < 175:
        raise ValueError("全景视角无效")
    result = {"id": uuid.uuid4().hex, "space": space, "box": box, "pose": pose,
              "view_size": view_size, "enabled": True, "score": 1., "origin": "manual", "note": ""}
    result.update(fields)
    result["enabled"] = bool(result["enabled"])
    result["score"] = float(result["score"])
    if not math.isfinite(result["score"]):
        raise ValueError("检测得分无效")
    return result

