"""Photo export selection, local coefficient edits, and independent validation."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyvips

from . import local_jpeg
from .common import check_cancel, emit, ensure_distinct_output, fingerprint, validate_style
from .geometry import region_mask_tiles, privacy_style_parameters


def _publish(temporary, destination):
    if os.name == 'nt':
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)
        temporary.unlink()


def _prepare_patches(asset, regions, style, info, directory, progress, cancel):
    from .photo_engine import _checked, _tile_style, _mosaic_cells
    width, height = info['size']
    mw = 8 * max(c[1] for c in info['sampling'])
    mh = 8 * max(c[2] for c in info['sampling'])
    patches = {}
    for index, region in enumerate(regions):
        emit(progress, 'photo_export_masks', index / max(1, len(regions)), f'准备遮挡区域 {index + 1}/{len(regions)}')
        check_cancel(cancel)
        bounds = [width, height, 0, 0]
        tiles = []
        for x, y, mask in region_mask_tiles(region, (width, height), tile_size=256,
                                           padding=float(style['padding']), shape=style['shape']):
            check_cancel(cancel)
            yy, xx = np.nonzero(mask)
            if not len(yy):
                continue
            bounds = [min(bounds[0], x + int(xx.min())), min(bounds[1], y + int(yy.min())),
                      max(bounds[2], x + int(xx.max()) + 1), max(bounds[3], y + int(yy.max()) + 1)]
            mask_path = directory / f'mask-{index}-{x}-{y}.bin'
            with mask_path.open('xb') as handle:
                handle.write(mask.tobytes())
            tiles.append((x, y, mask.shape, mask_path))
        if not tiles:
            continue
        parameters = {**privacy_style_parameters(bounds, style), 'bounds': bounds}
        image = _checked(asset)
        staged = []
        try:
            # Freeze prior regions as lazy disk-backed inserts. This preserves
            # the same sequential overlap semantics as the high-quality path.
            for old in patches.values():
                old_image = pyvips.Image.rawload(str(old['path']) + '.rgb', old['w'], old['h'], 3)
                image = image.insert(old_image, old['x'], old['y'])
                old_image = None
            if style['kind'] == 'mosaic':
                parameters['mosaic'] = _mosaic_cells(image, bounds, parameters['cells'], cancel)
            for x, y, shape, mask_path in tiles:
                check_cancel(cancel)
                mask = np.frombuffer(mask_path.read_bytes(), np.bool_).reshape(shape)
                # The shared native renderer samples composited pixels and fixes one
                # filter strength per region, so tile boundaries stay invisible.
                current = _tile_style(image, x, y, mask, style, parameters)
                patch_path = directory / f'patch-{x}-{y}.jpg'
                raw_path = Path(str(patch_path) + '.rgb')
                old = patches.get((x, y))
                rows, cols = (shape[0] + mh - 1) // mh, (shape[1] + mw - 1) // mw
                padded = np.zeros((rows * mh, cols * mw), bool)
                padded[:shape[0], :shape[1]] = mask
                blocks = padded.reshape(rows, mh, cols, mw).any(axis=(1, 3)).astype(np.uint8)
                if old is not None:
                    blocks |= np.frombuffer(old['mask'], np.uint8).reshape(rows, cols)
                patches[(x, y)] = dict(x=x, y=y, w=shape[1], h=shape[0], path=patch_path, mask=blocks.tobytes())
                if len(patches) > 65536:
                    raise local_jpeg.UnsupportedJPEG('遮挡区域数量超出局部 JPEG 限制，使用高质量重新编码。')
                stage_path = directory / f'stage-{index}-{x}-{y}.rgb'
                with stage_path.open('xb') as handle:
                    handle.write(current.tobytes())
                staged.append((stage_path, raw_path))
                mask_path.unlink()
                emit(progress, 'photo_export_render', (index + 1) / max(1, len(regions)), '正在准备局部遮挡分块')
        finally:
            image = None
        # Close all lazy readers before replacing raw patch files on Windows.
        # Publishing only after the entire region is rendered prevents one tile
        # in this region feeding its blur back into a later tile's halo.
        for stage_path, raw_path in staged:
            check_cancel(cancel)
            os.replace(stage_path, raw_path)
    return [patches[key] for key in sorted(patches, key=lambda value: (value[1], value[0]))]


def _local_export(asset, regions, style, destination, info, progress, cancel):
    source = Path(asset['source'])
    from .photo_engine import _checked
    image = _checked(asset)
    if (image.width, image.height) != info['size']:
        raise local_jpeg.LocalJPEGError('照片尺寸与源 JPEG 不一致，请重新导入。')
    image = None
    if shutil.disk_usage(destination.parent).free < source.stat().st_size * 2 + 128 * 1024 ** 2:
        raise OSError('输出磁盘空间不足。')
    source_hash = local_jpeg.sha256_file(source, cancel)
    with tempfile.TemporaryDirectory(prefix='.photo-local-', dir=destination.parent) as temporary_dir:
        directory = Path(temporary_dir)
        patches = _prepare_patches(asset, regions, style, info, directory, progress, cancel)
        manifest = directory / 'patches.bin'
        local_jpeg.write_manifest(manifest, patches)
        emit(progress, 'photo_export_native', 0, '保持原始量化表和色度采样，编码局部遮挡')
        check_cancel(cancel)
        local_jpeg.run_helper(['encode', source, manifest], cancel)
        emit(progress, 'photo_export_encode', 0, '正在替换受遮挡影响的 JPEG 系数块')
        check_cancel(cancel)
        temporary = directory / 'validated.jpg'
        _require_memory(info)
        local_jpeg.run_helper(['patch', source, temporary, manifest], cancel)
        emit(progress, 'photo_export_verify', 0, '正在独立验证未遮挡系数及完整解码')
        check_cancel(cancel)
        _require_memory(info)
        before = local_jpeg.run_helper(['hash', source, manifest], cancel)
        _require_memory(info)
        after = local_jpeg.run_helper(['hash', temporary, manifest], cancel)
        if before != after:
            raise local_jpeg.LocalJPEGError('未遮挡区域 JPEG 系数验证失败。')
        margin = 2 if info['sampling'][0][1:3] == (2, 2) else 0
        decoded = local_jpeg.run_helper(['pixels', source, temporary, manifest, margin], cancel)
        if not decoded.get('full_decode_completed') or decoded.get('outside_channel_differences') != 0:
            raise local_jpeg.LocalJPEGError('输出完整解码或未遮挡像素验证失败。')
        output_info = local_jpeg.inspect_jpeg(temporary)
        for field in ('size', 'sampling', 'qtables', 'markers'):
            if info[field] != output_info[field]:
                raise local_jpeg.LocalJPEGError('JPEG 输出元数据/参数验证失败：' + field)
        if fingerprint(source) != asset['fingerprint'] or local_jpeg.sha256_file(source, cancel) != source_hash:
            raise ValueError('源文件在导出期间改变，请重新导入。')
        check_cancel(cancel)
        _publish(temporary, destination)
    emit(progress, 'photo_export', 1, '局部 JPEG 导出并验证完成')
    return {'output': str(destination), 'width': info['size'][0], 'height': info['size'][1],
            'masked_regions': len(regions), 'export_method': 'local_jpeg', 'shape': style['shape'],
            'jpeg_quality': 'original', 'bytes': destination.stat().st_size, 'validated': True,
            'outside_coefficients_exact': True, 'outside_pixel_differences': 0,
            'jpeg_affected_mcus': sum(sum(patch['mask']) for patch in patches),
            'source_sha256': source_hash}


def _require_memory(info):
    if local_jpeg.available_memory() < local_jpeg.memory_required(info) + 1024 ** 3:
        raise local_jpeg.UnsupportedJPEG('可用内存不足以保留原始 JPEG 系数，使用流式高质量重新编码。')


def export_photo(asset, regions, style, destination, progress=None, cancel=None):
    check_cancel(cancel)
    style = {'kind': 'blur', 'strength': 1., 'padding': .25, 'color': '#202020',
             'shape': 'rect', 'export_mode': 'auto', 'jpeg_quality': 98, **style}
    validate_style(style)
    destination = ensure_distinct_output(asset['source'], destination)
    if destination.suffix.lower() not in ('.jpg', '.jpeg'):
        raise ValueError('照片导出请使用 .jpg 或 .jpeg 扩展名。')
    if fingerprint(asset['source']) != asset['fingerprint']:
        raise ValueError('源文件已改变，请重新导入照片后再处理。')
    info = local_jpeg.inspect_jpeg(asset['source'])
    reason = local_jpeg.eligibility_reason(info)
    if info['unsafe_exif_preview']:
        style['_strip_exif_preview'] = True
    if style['export_mode'] == 'jpeg_reencode':
        reason = reason or '已选择高质量重新编码。'
    elif reason is None:
        available = local_jpeg.available_memory()
        if available < local_jpeg.memory_required(info) + 1024 ** 3:
            reason = '可用内存不足以保留原始 JPEG 系数，使用流式高质量重新编码。'
        else:
            try:
                return _local_export(asset, [r for r in regions if r.get('enabled', True)],
                                     style, destination, info, progress, cancel)
            except local_jpeg.UnsupportedJPEG as error:
                reason = str(error)
    from .photo_engine import _export_photo_reencode
    emit(progress, 'photo_export_fallback', 0, reason)
    check_cancel(cancel)
    result = _export_photo_reencode(asset, regions, style, destination, progress, cancel)
    return {**result, 'export_method': 'jpeg_reencode', 'shape': style['shape'],
            'jpeg_quality': int(style['jpeg_quality']), 'fallback_reason': reason}
