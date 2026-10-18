"""Bounded JPEG inspection and a strict, cancellable native codec boundary."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
import subprocess
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ExifTags

from .common import check_cancel, resource_path


class LocalJPEGError(RuntimeError):
    """Corruption or failed verification; callers must not hide this as fallback."""


class UnsupportedJPEG(ValueError):
    """Known input/resource limitation for which high-quality reencoding is safe."""


def _read_exact(handle, count):
    data = handle.read(count)
    if len(data) != count:
        raise LocalJPEGError('JPEG 文件被截断。')
    return data


def _xmp_has_preview(payload):
    if payload.startswith(b'http://ns.adobe.com/xmp/extension/\0'):
        # Extended XMP can split a namespace or base64 image across markers.
        # This release conservatively rejects it instead of preserving opaque
        # chunks without a complete semantic preview sanitizer.
        return True
    prefix = b'http://ns.adobe.com/xap/1.0/\0'
    if not payload.startswith(prefix):
        return False
    try:
        root = ET.fromstring(payload[len(prefix):])
    except (ET.ParseError, ValueError):
        return True
    image_namespaces = ('{http://ns.adobe.com/xap/1.0/g/img/}',
                        '{http://ns.google.com/photos/1.0/image/}')
    for node in root.iter():
        for name in (node.tag, *node.attrib):
            if (isinstance(name, str) and
                    (name.startswith(image_namespaces) or
                     name.rsplit('}', 1)[-1].lower() in {'thumbnails', 'thumbnailimage', 'previewimage'})):
                return True
    return False


def inspect_jpeg(path):
    """Read marker/header data without allocating the decoded panorama."""
    info = {'markers': [], 'qtables': {}, 'unsafe_exif_preview': False,
            'unsafe_other_preview': False, 'sof': None}
    metadata_bytes = 0
    with Path(path).open('rb') as handle:
        if _read_exact(handle, 2) != b'\xff\xd8':
            raise LocalJPEGError('输入不是 JPEG 文件。')
        while True:
            if _read_exact(handle, 1) != b'\xff':
                raise LocalJPEGError('JPEG 标记无效。')
            code = _read_exact(handle, 1)[0]
            while code == 255:
                code = _read_exact(handle, 1)[0]
            if code in (0xDA, 0xD9):
                break
            length = int.from_bytes(_read_exact(handle, 2), 'big')
            if length < 2:
                raise LocalJPEGError('JPEG 标记长度无效。')
            payload = _read_exact(handle, length - 2)
            if code in set(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}:
                if info['sof'] is not None or len(payload) < 6:
                    raise LocalJPEGError('JPEG 帧头无效。')
                components = payload[5]
                if len(payload) != 6 + 3 * components:
                    raise LocalJPEGError('JPEG 分量头无效。')
                info.update(sof=code, precision=payload[0],
                            size=(int.from_bytes(payload[3:5], 'big'), int.from_bytes(payload[1:3], 'big')),
                            sampling=[(payload[6+i*3], payload[7+i*3] >> 4,
                                       payload[7+i*3] & 15, payload[8+i*3]) for i in range(components)])
            elif code == 0xDB:
                offset = 0
                while offset < len(payload):
                    table, precision = payload[offset] & 15, payload[offset] >> 4
                    offset += 1
                    if table > 3 or precision > 1 or offset + 64 * (precision + 1) > len(payload):
                        raise LocalJPEGError('JPEG 量化表无效。')
                    values = struct.unpack('>64H' if precision else '64B', payload[offset:offset + 64 * (precision + 1)])
                    if not all(values):
                        raise LocalJPEGError('JPEG 量化表含零值。')
                    info['qtables'][table] = values
                    offset += 64 * (precision + 1)
            if 0xE0 <= code <= 0xEF or code == 0xFE:
                metadata_bytes += len(payload)
                if metadata_bytes > 32 * 1024 ** 2:
                    raise LocalJPEGError('JPEG 元数据超出安全读取范围。')
                info['markers'].append((code, len(payload), hashlib.sha256(payload).hexdigest()))
                if code == 0xE1 and payload.startswith(b'Exif\0\0'):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('error')
                            exif = Image.Exif()
                            exif.load(payload)
                            ifd1 = exif.get_ifd(ExifTags.IFD.IFD1)
                            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                        info['unsafe_exif_preview'] |= bool(ifd1) or 37500 in exif_ifd or 37500 in exif
                    except (ValueError, TypeError, KeyError, IndexError, struct.error, SyntaxError, Warning) as error:
                        raise LocalJPEGError('EXIF 结构损坏，不能确认内嵌预览安全。') from error
                elif code == 0xE0:
                    info['unsafe_other_preview'] |= (payload.startswith(b'JFXX\0') or
                        payload.startswith(b'JFIF\0') and len(payload) >= 14 and bool(payload[12] or payload[13]))
                elif code == 0xE2 and payload.startswith(b'MPF\0'):
                    info['unsafe_other_preview'] = True
                elif code == 0xED and payload.startswith(b'Photoshop 3.0\0'):
                    info['unsafe_other_preview'] |= b'8BIM\x04\x09' in payload or b'8BIM\x04\x0c' in payload
                elif code == 0xE1:
                    info['unsafe_other_preview'] |= _xmp_has_preview(payload)
        if info['sof'] is None or not all(info['size']):
            raise LocalJPEGError('JPEG 缺少有效帧头。')
    return info


def eligibility_reason(info):
    if info['unsafe_other_preview']:
        # The legacy writer preserves XMP/IPTC. Do not send potentially visible
        # unredacted image data to that writer until a sanitizer is implemented.
        raise LocalJPEGError('JPEG 含有额外内嵌预览或多图数据，请先移除预览后重新导入。')
    if info['unsafe_exif_preview']:
        return 'EXIF 含内嵌预览或 MakerNote；重新编码时移除该 EXIF 数据以避免泄露原图。'
    if info['sof'] != 0xC0 or info['precision'] != 8:
        return '该 JPEG 不是 8 位基线格式，使用高质量重新编码。'
    sampling = [tuple(layer[1:3]) for layer in info['sampling']]
    if sampling not in ([(1, 1)] * 3, [(2, 2), (1, 1), (1, 1)]):
        return '该 JPEG 分量或色度采样不支持局部 DCT 编辑，使用高质量重新编码。'
    return None


def coefficient_bytes(info):
    width, height = info['size']
    max_h = max(layer[1] for layer in info['sampling'])
    max_v = max(layer[2] for layer in info['sampling'])
    if max_h < 1 or max_v < 1:
        raise LocalJPEGError('JPEG 采样因子无效。')
    return sum(((width * h + 8 * max_h - 1) // (8 * max_h)) *
               ((height * v + 8 * max_v - 1) // (8 * max_v)) * 128
               for _, h, v, _ in info['sampling'])


def available_memory():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        if os.name == 'nt':
            class Status(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total',
                    'page_available', 'virtual_total', 'virtual_available', 'extended')]
            status = Status()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
        return 0


def memory_required(info):
    # Coefficient arrays are held once. Allow row alignment, library overhead,
    # bounded patch work, and 15% headroom; the caller keeps an OS reserve.
    return int(coefficient_bytes(info) * 1.15) + 256 * 1024 ** 2


def run_helper(arguments, cancel=None):
    helper = resource_path('runtime/jpeg/jpeg_local.exe')
    if not helper.is_file():
        raise UnsupportedJPEG('局部 JPEG 组件不可用，使用高质量重新编码。')
    check_cancel(cancel)
    command = [str(helper), *map(str, arguments)]
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    try:
        while True:
            check_cancel(cancel)
            try:
                output, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                continue
        check_cancel(cancel)
        if process.returncode:
            raise LocalJPEGError('局部 JPEG 验证/处理失败：' + error.decode('utf-8', 'replace').strip())
        try:
            return json.loads(output)
        except (ValueError, UnicodeDecodeError) as error:
            raise LocalJPEGError('局部 JPEG 组件返回无效验证结果。') from error
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def write_manifest(path, patches):
    with Path(path).open('xb') as handle:
        handle.write(b'PJ01' + struct.pack('<I', len(patches)))
        for patch in patches:
            name = str(Path(patch['path']).resolve()).encode('utf-8')
            mask = bytes(patch['mask'])
            handle.write(struct.pack('<6I', patch['x'], patch['y'], patch['w'], patch['h'], len(name), len(mask)))
            handle.write(name)
            handle.write(mask)


def sha256_file(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4 * 1024 ** 2):
            check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()
