"""Disk-backed panorama review and export; NumPy is limited to views/tiles.

The native .v cache is deliberately uncompressed and random-access. JPEG input
is decoded once, rather than decoded from its first scanline for every face.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np
import pyvips

# Cached C images retain open files on Windows after Python wrappers disappear.
# Photo queues must be able to evict native.v, and export temporaries must close
# before deletion. Lazy pixel evaluation still streams without this graph cache.
pyvips.cache_set_max(0)

from .common import (check_cancel, emit, ensure_distinct_output, fingerprint,
                     new_region, validate_style)
from .geometry import (region_center, region_mask_tiles, region_to_view_box,
                       privacy_style_parameters)

TILE_SIZE = 256
DETECT_TILE = 576
DETECT_STRIDE = 448
METADATA_FIELDS = ("exif-data", "xmp-data", "icc-profile-data", "iptc-data")


def _rgb(image):
    loader = str(image.get('vips-loader')) if image.get_typeof('vips-loader') else ''
    if image.bands != 3 or image.format != "uchar" or image.interpretation == "cmyk" or not loader.startswith('jpeg'):
        raise ValueError("本版仅支持 8 位 RGB JPEG 全景照片；请先将灰度、CMYK、透明或高位深素材转换为 RGB JPEG。")
    return image.copy(interpretation="srgb")


def _array(image):
    return np.frombuffer(image.write_to_memory(), dtype=np.uint8).reshape(image.height, image.width, image.bands).copy()


def _checked(asset):
    source = Path(asset["source"])
    if fingerprint(source) != asset["fingerprint"]:
        raise ValueError("源文件已改变，请重新导入照片后再处理。")
    image = pyvips.Image.new_from_file(str(asset["native_path"]), access="random")
    if (image.width, image.height, image.bands, image.format) != (asset["width"], asset["height"], 3, "uchar"):
        raise ValueError("照片缓存损坏，请重新导入。")
    return image


def _evaluate(image, action, progress, cancel, stage, message):
    """Poll cancellation without unsafe cached-image GObject callbacks.

    pyvips 3.2 signal handlers outlive a collected Python image wrapper when
    its C image remains in the operation cache. A later evaluation can then
    call a freed ffi handle. A bounded worker and set_kill avoid those signals.
    The worker is always joined before its output files can be removed.
    """
    try:
        check_cancel(cancel)
        emit(progress, stage, 0, message)
        check_cancel(cancel)
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="photo-vips") as executor:
            future = executor.submit(action)
            while True:
                try:
                    result = future.result(timeout=.1)
                    break
                except FutureTimeout:
                    if cancel is not None and cancel.is_set():
                        image.set_kill(True)
        check_cancel(cancel)
        emit(progress, stage, 1, message)
        return result
    except BaseException as error:
        # Future exceptions retain codec frames and their file-backed `self`.
        # Preserve the error message/type, but release those C image wrappers
        # before export's cleanup runs while this exception is still displayed.
        error.__traceback__ = None
        if isinstance(error, pyvips.Error):
            check_cancel(cancel)
        raise
    finally:
        image.set_kill(False)
        # Cancelled tracebacks remain live while export's finally cleans up.
        # Do not let that traceback retain an open rawload file via this frame.
        image = action = future = None


def prepare_photo(source, cache_dir, progress=None, cancel=None):
    """Decode read-only input to a unique disk-native cache and small preview."""
    check_cancel(cancel)
    source = Path(source).resolve(strict=True)
    before = fingerprint(source)
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    job_dir = Path(tempfile.mkdtemp(prefix="photo-", dir=cache_dir))
    image = cached = preview = None
    try:
        image = _rgb(pyvips.Image.new_from_file(str(source), access="sequential", fail_on="warning", revalidate=True))
        orientation = int(image.get("orientation")) if image.get_typeof("orientation") else 1
        if orientation != 1:
            raise ValueError("照片 EXIF 方向不是标准方向（Orientation=1）；请先规范方向并检查 GPano 元数据后重新导入。")
        if abs(image.width / image.height - 2) > .04:
            raise ValueError("照片不是约 2:1 的等距柱状全景图，请检查输入素材。")
        needed = image.width * image.height * 3
        if shutil.disk_usage(job_dir).free < needed + 128 * 1024 ** 2:
            raise OSError("缓存磁盘空间不足：原生照片缓存需要约 %.1f GB。" % (needed / 1e9))
        native = job_dir / "native.v"
        _evaluate(image, lambda: image.write_to_file(str(native)), progress, cancel,
                  "photo_prepare", "正在创建可随机读取的原生照片缓存")
        cached = pyvips.Image.new_from_file(str(native), access="random")
        preview = cached.thumbnail_image(2048, height=1024, size="down", no_rotate=True)
        preview_path = job_dir / "preview.jpg"
        _evaluate(preview, lambda: preview.jpegsave(str(preview_path), Q=92, subsample_mode="off"),
                  progress, cancel, "photo_preview", "正在生成全景预览")
        if fingerprint(source) != before:
            raise ValueError("源文件在导入期间改变，请重新导入。")
        asset = {"kind": "photo", "source": str(source), "width": image.width,
                 "height": image.height, "fingerprint": before,
                 "cache_dir": str(job_dir), "native_path": str(native),
                 "preview_path": str(preview_path),
                 "metadata_fields": [key for key in METADATA_FIELDS if cached.get_typeof(key)]}
        emit(progress, "photo_prepare", 1, "照片已导入，原文件保持不变")
        return asset
    except BaseException:
        image = cached = preview = None
        shutil.rmtree(job_dir, ignore_errors=True)
        raise


def _native_view(image, pose, out_size):
    """Bounded mapim view samples native .v directly, including seam and pole."""
    width, height = map(int, out_size)
    if width < 1 or height < 1 or width * height > 4096 * 4096:
        raise ValueError("视图尺寸需要在 1 至 4096×4096 像素范围内。")
    yaw, pitch, fov, roll = [float(pose.get(key, default)) for key, default in
                             (("yaw", 0), ("pitch", 0), ("fov", 90), ("roll", 0))]
    if not np.isfinite([yaw, pitch, fov, roll]).all() or not -90 <= pitch <= 90 or not 1 <= fov < 175:
        raise ValueError("视图方向或视野角度无效。")
    yaw, pitch, roll = np.deg2rad([yaw, pitch, roll])
    sy, cy, sp, cp = np.sin(yaw), np.cos(yaw), np.sin(pitch), np.cos(pitch)
    right = np.array([cy, 0, -sy])
    up = np.array([-sy * sp, cp, -cy * sp])
    forward = np.array([sy * cp, sp, cy * cp])
    old_right = right.copy()
    right = old_right * np.cos(roll) + up * np.sin(roll)
    up = up * np.cos(roll) - old_right * np.sin(roll)
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    xx = (np.arange(width, dtype=np.float32)[None, :] + .5 - width / 2) / focal
    yy = (height / 2 - np.arange(height, dtype=np.float32)[:, None] - .5) / focal
    wx = forward[0] + xx * right[0] + yy * up[0]
    wy = forward[1] + xx * right[1] + yy * up[1]
    wz = forward[2] + xx * right[2] + yy * up[2]
    longitude = np.arctan2(wx, wz)
    latitude = np.arctan2(wy, np.hypot(wx, wz))
    mapping = np.empty((height, width, 2), dtype=np.float32)
    mapping[:, :, 0] = ((longitude / (2 * np.pi) + .5) * image.width - .5) % image.width + image.width
    mapping[:, :, 1] = np.clip((.5 - latitude / np.pi) * image.height - .5, 0, image.height - 1)
    index = pyvips.Image.new_from_memory(mapping.data, width, height, 2, "float")
    output = image.replicate(3, 1).mapim(index, interpolate=pyvips.Interpolate.new("bilinear"), extend="copy")
    return _array(output)


def photo_view(asset, pose, out_size=(960, 640)):
    return _native_view(_checked(asset), pose, out_size)


def _thumbnail_pose(region):
    yaw, pitch = region_center(region)
    if region["space"] == "erp":
        x0, y0, x1, y1 = region["box"]
        angle = max((x1 - x0) * 360 * max(.1, math.cos(math.radians(pitch))), (y1 - y0) * 180)
    else:
        box = region["box"]
        angle = max(box[2] - box[0], (box[3] - box[1]) * region["view_size"][1] / region["view_size"][0]) * region["pose"]["fov"]
    return {"yaw": yaw, "pitch": pitch, "fov": min(150, max(3, angle * 2)), "roll": 0}


def photo_thumbnail(asset, region, size=144):
    image = _checked(asset)
    size = int(size)
    if not 1 <= size <= 2048:
        raise ValueError("缩略图尺寸需要在 1 到 2048 之间。")
    return _native_view(image, _thumbnail_pose(region), (size, size))


def confirm_photo_candidates(asset, regions, progress=None, cancel=None):
    """Conservatively disable unconfirmed proposals, retaining every region."""
    from .photo_confirm import CandidateConfirmer
    check_cancel(cancel)
    confirmer = CandidateConfirmer()
    result = []
    for index, region in enumerate(regions):
        check_cancel(cancel)
        candidate = dict(region)
        if region.get('enabled', True):
            rgb = photo_thumbnail(asset, region, size=320)
            box = region_to_view_box(region, _thumbnail_pose(region), (320, 320))
            confirmation = confirmer.confirm(rgb, box)
            candidate['confirmation'] = {'method': 'yunet_cpu', 'threshold': .8, **confirmation}
            if not confirmation['confirmed']:
                candidate['enabled'] = False
                candidate['review_reason'] = 'unconfirmed'
                candidate['note'] = (region.get('note', '') + ' 未确认：可能为背景或侧脸、低头；可手动启用。').strip()
        result.append(candidate)
        emit(progress, 'photo_confirm', (index + 1) / max(1, len(regions)), f'进一步检查明显人脸 {index + 1}/{len(regions)}')
    return result


def _starts(length, tile, stride):
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def _native_tile(image, x, y, tile=DETECT_TILE):
    height = min(tile, image.height - y)
    width = min(tile, image.width)
    wrapped = image.replicate(2, 1).crop(x, y, width, height)
    rgb = _array(wrapped)
    padded = np.zeros((tile, tile, 3), dtype=np.uint8)
    padded[:height, :width] = rgb
    return padded, width, height


def _candidate(box, score, mode, dimensions, pose=None, view_size=None):
    x0, y0, x1, y1 = map(float, box)
    width, height = dimensions
    x0, y0, x1, y1 = max(0., x0), max(0., y0), min(width, x1), min(height, y1)
    bw, bh = x1 - x0, y1 - y0
    if min(bw, bh) < (24 if mode == "obvious" else 16):
        return None
    suspicious = bw * bh > width * height * .25 or max(bw, bh) / min(bw, bh) > 3
    suggestion = score < .6 or min(bw, bh) < 24
    note = "区域过大或形状异常，可能为背景；请核实后启用" if suspicious else ("更多模式补充建议，请核实" if suggestion else "")
    fields = {"enabled": not (suspicious or suggestion), "score": float(score), "origin": "auto", "note": note,
              "review_reason": "implausible" if suspicious else ("suggestion" if suggestion else "")}
    if pose is None:
        return (x0, y0, x1, y1), fields
    return new_region("view", [x0 / width, y0 / height, x1 / width, y1 / height], pose=pose, view_size=view_size, **fields)


def _overlap(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-12, aa + ab - inter) > .35 or inter / max(1e-12, min(aa, ab)) > .78


def _deduplicate(regions, cancel=None):
    kept = []
    for region in sorted(regions, key=lambda r: (r["enabled"], r["score"]), reverse=True):
        check_cancel(cancel)
        duplicate = False
        for old in kept:
            if old["enabled"] != region["enabled"]:
                continue
            if old.get("review_reason", "") != region.get("review_reason", ""):
                continue
            if old["space"] == region["space"] == "erp":
                a, b = old["box"], region["box"]
                duplicate = any(_overlap(a, [b[0] + shift, b[1], b[2] + shift, b[3]]) for shift in (-1, 0, 1))
            else:
                base = old if old["space"] == "view" else region
                other = region if base is old else old
                size = base["view_size"]
                mapped = region_to_view_box(other, base["pose"], size)
                box = base["box"]
                duplicate = mapped is not None and _overlap([box[0] * size[0], box[1] * size[1], box[2] * size[0], box[3] * size[1]], mapped)
            if duplicate:
                break
        if not duplicate:
            kept.append(region)
    return kept


def scan_photo(asset, mode="obvious", progress=None, cancel=None):
    from .photo_detection import scan_photo as scan
    return scan(asset, mode, progress, cancel)


def _mosaic_cells(image, bounds, cells, cancel=None):
    """Exact box-area reduction to a tiny grid, holding at most 128 ROI rows."""
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top
    nx, ny = min(cells, width), min(cells, height)
    dx, dy = width / nx, height / ny
    totals = np.zeros((ny, nx, 3), dtype=np.float64)
    for y in range(0, height, 128):
        check_cancel(cancel)
        data = _array(image.crop(left, top + y, width, min(128, height - y)))
        reduced = np.empty((data.shape[0], nx, 3), dtype=np.float64)
        for column in range(nx):
            a, b = column * dx, (column + 1) * dx
            low, high = math.floor(a), min(width, math.ceil(b))
            values = data[:, low:high]
            summed = values.sum(axis=1, dtype=np.float64)
            summed -= (a - low) * values[:, 0]
            summed -= (high - b) * values[:, -1]
            reduced[:, column] = summed / dx
        for row in range(ny):
            a, b = max(y, row * dy), min(y + len(data), (row + 1) * dy)
            if b <= a:
                continue
            low, high = math.floor(a) - y, math.ceil(b) - y
            summed = reduced[low:high].sum(axis=0)
            summed -= (a - math.floor(a)) * reduced[low]
            summed -= (math.ceil(b) - b) * reduced[high - 1]
            totals[row] += summed / dy
    return np.rint(totals).clip(0, 255).astype(np.uint8)


def _tile_style(image, x, y, mask, style, parameters):
    height, width = mask.shape
    original = _array(image.crop(x, y, width, height))
    kind = style["kind"]
    if kind == "solid":
        color = style.get("color", "#202020")
        if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
            raise ValueError("纯色需要 #RRGGBB 格式。")
        modified = np.empty_like(original)
        modified[:] = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    elif kind == "blur":
        # A shared native source and fixed sigma across tiles avoid patch seams.
        sigma = parameters['sigma']
        reduction = max(1., sigma / 12.)
        if reduction > 1:
            sw, sh = math.ceil(image.width / reduction), math.ceil(image.height / reduction)
            small = image.resize(sw / image.width, vscale=sh / image.height, kernel='linear')
            filtered = small.replicate(3, 1).gaussblur(sigma * sw / image.width, min_ampl=.001, precision='float')
            blurred = filtered.resize(image.width / sw, vscale=image.height / sh, kernel='linear').crop(image.width + x, y, width, height).cast('uchar')
        else:
            blurred = image.replicate(3, 1).gaussblur(sigma, min_ampl=.001, precision='float').crop(image.width + x, y, width, height).cast('uchar')
        modified = _array(blurred)
    elif kind == "mosaic":
        left, top, right, bottom = parameters['bounds']
        small = parameters['mosaic']
        xx = np.clip(((x + np.arange(width) - left) * small.shape[1] / (right - left)).astype(int), 0, small.shape[1] - 1)
        yy = np.clip(((y + np.arange(height) - top) * small.shape[0] / (bottom - top)).astype(int), 0, small.shape[0] - 1)
        modified = small[yy[:, None], xx[None, :]]
    else:
        raise ValueError("未知遮挡样式：" + str(kind))
    original[mask] = modified[mask]
    return original


def _copy_metadata(source, image):
    image = image.copy(interpretation="srgb", xres=source.xres, yres=source.yres)
    for key in METADATA_FIELDS:
        if source.get_typeof(key):
            image.set_type(pyvips.GValue.blob_type, key, source.get(key))
    return image


def _write_native_raw(asset, path, progress, cancel):
    """Copy 128-row strips using short-lived native file mappings.

    A single .v rawsave reader can retain gigabytes of touched mmap pages in
    the Windows working set despite lazy pixel processing. Closing each strip
    reader releases its mapping; the largest Python buffer is width×128×3.
    """
    width, height = asset["width"], asset["height"]
    with Path(path).open("xb") as handle:
        for y in range(0, height, 128):
            check_cancel(cancel)
            reader = stripe = buffer = None
            try:
                reader = pyvips.Image.new_from_file(str(asset["native_path"]), access="sequential")
                stripe = reader.crop(0, y, width, min(128, height - y))
                buffer = stripe.write_to_memory()
                if handle.write(buffer) != len(buffer):
                    raise OSError("原生输出磁盘写入不完整，请检查磁盘空间。")
            finally:
                reader = stripe = buffer = None
            emit(progress, "photo_export_native", min(height, y + 128) / height, "正在准备原生尺寸输出")
    check_cancel(cancel)


def _export_photo_reencode(asset, regions, style, destination, progress=None, cancel=None):
    """Render native RGB via bounded tiles, then validate and publish JPEG.

    A sparse collection of per-tile masks is kept on disk. The export raw file
    is copied through 128-row readers, then patched only at selected tiles.
    There is no full panorama ndarray, mask, or in-memory encoded JPEG.
    """
    check_cancel(cancel)
    ensure_distinct_output(asset["source"], destination)
    image = _checked(asset)
    style = {"kind": "blur", "strength": 1., "padding": .25, "color": "#202020", **style}
    validate_style(style)
    destination = Path(destination).resolve()
    if destination.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("照片导出请使用 .jpg 或 .jpeg 扩展名。")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < image.width * image.height * 3 + 128 * 1024 ** 2:
        raise OSError("输出磁盘空间不足，原生临时文件需要约 %.1f GB。" % (image.width * image.height * 3 / 1e9))
    enabled = [region for region in regions if region.get("enabled", True)]
    job_dir = Path(tempfile.mkdtemp(prefix=".photo-export-", dir=destination.parent))
    temporary = destination.parent / ("." + destination.name + "." + uuid.uuid4().hex + ".part.jpg")
    output = checked = raw_image = None
    try:
        jobs = []
        for index, region in enumerate(enabled):
            check_cancel(cancel)
            emit(progress, "photo_export_masks", index / max(1, len(enabled)), f"准备遮挡区域 {index + 1}/{len(enabled)}")
            check_cancel(cancel)
            region_tiles = []
            bounds = [image.width, image.height, 0, 0]
            for x, y, mask in region_mask_tiles(region, (image.width, image.height), tile_size=TILE_SIZE, padding=float(style["padding"]), shape=style.get("shape", "rect")):
                check_cancel(cancel)
                if not mask.any():
                    continue
                yy, xx = np.nonzero(mask)
                bounds = [min(bounds[0], x + int(xx.min())), min(bounds[1], y + int(yy.min())),
                          max(bounds[2], x + int(xx.max()) + 1), max(bounds[3], y + int(yy.max()) + 1)]
                path = job_dir / f"mask-{index}-{x}-{y}.bin"
                path.write_bytes(np.asarray(mask, dtype=np.uint8).tobytes())
                region_tiles.append((int(x), int(y), path, mask.shape))
            if region_tiles:
                jobs.append((region_tiles, {**privacy_style_parameters(bounds, style), 'bounds': bounds}))
        check_cancel(cancel)
        raw_path = job_dir / "render.rgb"
        emit(progress, "photo_export_native", 0, "正在准备原生尺寸输出")
        _write_native_raw(asset, raw_path, progress, cancel)
        tile_total = sum(len(tiles) for tiles, _ in jobs)
        completed = 0
        for region_index, (tiles, parameters) in enumerate(jobs):
            raw_image = pyvips.Image.rawload(str(raw_path), image.width, image.height, 3)
            if style['kind'] == 'mosaic':
                parameters['mosaic'] = _mosaic_cells(raw_image, parameters['bounds'], parameters['cells'], cancel)
            # Freeze every modified tile before writing this region, so its
            # blur cannot feed earlier tiles back into later tile halos.
            staged = []
            for x, y, path, shape in tiles:
                check_cancel(cancel)
                mask = np.frombuffer(path.read_bytes(), dtype=np.uint8).reshape(shape).astype(bool)
                patch = _tile_style(raw_image, x, y, mask, style, parameters)
                patch_path = job_dir / f"patch-{region_index}-{x}-{y}.rgb"
                patch_path.write_bytes(patch.tobytes())
                staged.append((x, y, patch_path, shape))
            raw_image = None
            with raw_path.open("r+b", buffering=0) as handle:
                for x, y, patch_path, shape in staged:
                    check_cancel(cancel)
                    patch = np.frombuffer(patch_path.read_bytes(), np.uint8).reshape(*shape, 3)
                    for row in range(shape[0]):
                        handle.seek(((y + row) * image.width + x) * 3)
                        handle.write(patch[row].tobytes())
                    patch_path.unlink()
                    completed += 1
                    emit(progress, "photo_export_render", completed / max(1, tile_total), f"正在渲染遮挡分块 {completed}/{tile_total}")
        output = _copy_metadata(image, pyvips.Image.rawload(str(raw_path), image.width, image.height, 3))
        if style.get('_strip_exif_preview'):
            for key in list(output.get_fields()):
                if key == 'exif-data' or key.startswith('exif-'):
                    output.remove(key)
        quality = int(style.get('jpeg_quality', 98))
        # Huffman optimisation buffers a full panorama's DCT coefficient array
        # in libjpeg (17 GB observed on a 50k photo). Disable it to stream MCU
        # rows with unchanged Q98/4:4:4 pixels and metadata preservation.
        _evaluate(output, lambda: output.jpegsave(str(temporary), Q=quality, subsample_mode="off", optimize_coding=False, keep=pyvips.enums.ForeignKeep.ALL),
                  progress, cancel, "photo_export_encode", f"正在写入 JPEG Q{quality} / 4:4:4")
        checked = pyvips.Image.new_from_file(str(temporary), access="sequential", fail_on="warning")
        if (checked.width, checked.height) != (image.width, image.height):
            raise RuntimeError("输出尺寸验证失败。")
        _evaluate(checked, checked.avg, progress, cancel, "photo_export_verify", "正在完整解码验证输出")
        for key in ("xmp-data", "icc-profile-data", "iptc-data"):
            if image.get_typeof(key) and (not checked.get_typeof(key) or image.get(key) != checked.get(key)):
                raise RuntimeError("输出元数据验证失败：" + key)
        if not style.get("_strip_exif_preview") and image.get_typeof("exif-data") and not checked.get_typeof("exif-data"):
            raise RuntimeError("输出 EXIF 元数据丢失。")
        # JPEG writers may rebuild EXIF offsets; verify meaningful camera tags
        # rather than comparing those rewritten binary pointer bytes.
        for key in ("exif-ifd0-Make", "exif-ifd0-Model", "exif-ifd0-DateTime",
                    "exif-ifd0-Orientation", "exif-ifd2-DateTimeOriginal"):
            if not style.get("_strip_exif_preview") and image.get_typeof(key) and (not checked.get_typeof(key) or image.get(key) != checked.get(key)):
                raise RuntimeError("输出 EXIF 标签验证失败：" + key)
        _checked(asset)
        check_cancel(cancel)
        # Windows rename is atomic and refuses existing files, including SMB.
        # POSIX rename replaces files, so use its no-clobber hard-link instead.
        if os.name == "nt":
            os.rename(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
        result = {"output": str(destination), "width": image.width, "height": image.height,
                  "masked_regions": len(enabled), "jpeg_quality": quality, "chroma": "4:4:4",
                  "export_method": "jpeg_reencode", "shape": style.get("shape", "rect"),
                  "metadata_removed": "EXIF (embedded preview)" if style.get("_strip_exif_preview") else None,
                  "jpeg_optimize_coding": False,
                  "bytes": destination.stat().st_size, "validated": True}
        emit(progress, "photo_export", 1, "照片导出并验证完成")
        return result
    finally:
        # Release native rawload/verification file handles before Windows delete.
        output = checked = raw_image = None
        temporary.unlink(missing_ok=True)
        shutil.rmtree(job_dir, ignore_errors=True)


def export_photo(asset, regions, style, destination, progress=None, cancel=None):
    from .photo_export import export_photo as export
    # Existing projects retain their export behavior; new photo UI writes auto.
    return export(asset, regions, {'export_mode':'jpeg_reencode', **style}, destination, progress, cancel)
