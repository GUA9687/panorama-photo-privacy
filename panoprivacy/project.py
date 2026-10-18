from __future__ import annotations
import json
import os
from pathlib import Path
from .common import atomic_json, fingerprint, default_style, new_region, validate_style


def _validate_region(region):
    return new_region(region["space"],region["box"],region.get("pose"),region.get("view_size",(1,1)),
                      **{k:v for k,v in region.items() if k not in {"space","box","pose","view_size"}})


def save_project(path,kind,asset,regions=None,seeds=None,analysis=None,style=None):
    if kind not in {"photo","video"}:raise ValueError("未知工程类型")
    if os.path.normcase(str(Path(path).resolve()))==os.path.normcase(str(Path(asset["source"]).resolve())):
        raise ValueError("工程文件不能覆盖原片")
    style={**default_style(),**(style or {})};validate_style(style)
    regions=[_validate_region(r) for r in (regions or [])]
    for seed in seeds or []:
        if int(seed["frame"])<0:raise ValueError("校正帧不能为负")
        _validate_region(seed["region"])
    data={"schema_version":1,"kind":kind,"asset":asset,"regions":regions,"seeds":seeds or [],"analysis":analysis,"style":style}
    atomic_json(path,data)


def load_project(path):
    if Path(path).stat().st_size>150_000_000:raise ValueError("工程文件过大")
    data=json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("schema_version")!=1 or data.get("kind") not in {"photo","video"}:
        raise ValueError("不支持此工程版本")
    asset=data["asset"]
    if min(int(asset["width"]),int(asset["height"]))<1:raise ValueError("工程尺寸无效")
    data["regions"]=[_validate_region(r) for r in data.get("regions",[])]
    for seed in data.get("seeds",[]):
        if not 0<=int(seed["frame"])<int(asset.get("frame_count",1)):raise ValueError("工程校正帧超出视频范围")
        _validate_region(seed["region"])
    data["style"]={**default_style(),**data.get("style",{})};validate_style(data["style"])
    source=Path(asset["source"])
    data["source_status"]="missing" if not source.is_file() else ("ok" if fingerprint(source)==asset["fingerprint"] else "changed")
    data["requires_relink"]=data["source_status"]!="ok"
    return data

