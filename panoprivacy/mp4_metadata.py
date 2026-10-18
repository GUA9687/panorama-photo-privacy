"""Bounded MP4 spherical metadata copying without relocating media chunks."""
from pathlib import Path
import struct

SPHERICAL_UUID = bytes.fromhex('ffcc8263f8554a938814587a02521fdd')
CONTAINERS = {b'moov', b'trak', b'mdia', b'minf', b'stbl'}
VIDEO_ENTRIES = {b'hvc1', b'hev1', b'avc1', b'avc3'}
MAX_MOOV = 128 * 1024 * 1024


def pack(kind, payload):
    size = len(payload) + 8
    return (struct.pack('>I4s', size, kind) if size <= 0xffffffff else
            struct.pack('>I4sQ', 1, kind, size + 8)) + payload


def boxes(data):
    position = 0
    while position < len(data):
        if len(data) - position < 8:
            raise ValueError('Truncated MP4 box header')
        size, kind = struct.unpack_from('>I4s', data, position)
        header = 8
        if size == 1:
            if len(data) - position < 16:
                raise ValueError('Truncated MP4 extended header')
            size = struct.unpack_from('>Q', data, position + 8)[0]
            header = 16
        elif size == 0:
            size = len(data) - position
        if size < header or position + size > len(data):
            raise ValueError('Invalid MP4 box bounds')
        yield kind, data[position + header:position + size], data[position:position + size]
        position += size


def read_moov(path):
    path = Path(path)
    length = path.stat().st_size
    found = []
    with path.open('rb') as file:
        while file.tell() < length:
            start = file.tell()
            header = file.read(8)
            if len(header) != 8:
                raise ValueError('Truncated MP4 header')
            size, kind = struct.unpack('>I4s', header)
            header_size = 8
            if size == 1:
                extended = file.read(8)
                if len(extended) != 8:
                    raise ValueError('Truncated extended MP4 header')
                size = struct.unpack('>Q', extended)[0]
                header_size = 16
            if size == 0 or size < header_size or start + size > length:
                raise ValueError('Unsupported open-ended or invalid MP4 box')
            if kind == b'moov':
                if size > MAX_MOOV:
                    raise ValueError('MP4 metadata exceeds supported 128 MiB limit')
                file.seek(start)
                found.append((start, size, file.read(size)))
            file.seek(start + size)
    if len(found) != 1:
        raise ValueError('Expected exactly one MP4 moov')
    return found[0]


def _collect(data, spherical, legacy):
    for kind, payload, raw in boxes(data):
        if kind in {b'sv3d', b'st3d'}:
            spherical.append(raw)
        elif kind == b'uuid' and payload[:16] == SPHERICAL_UUID:
            legacy.append(raw)
        elif kind in CONTAINERS:
            _collect(payload, spherical, legacy)
        elif kind == b'stsd':
            _collect(payload[8:], spherical, legacy)
        elif kind in VIDEO_ENTRIES:
            _collect(payload[78:], spherical, legacy)


def spherical_boxes(path):
    sphere, legacy = [], []
    _collect(read_moov(path)[2], sphere, legacy)
    return sphere, legacy


def _rewrite(data, sphere, legacy):
    result, count = [], 0
    for kind, payload, raw in boxes(data):
        n = 0
        if kind in CONTAINERS:
            content, n = _rewrite(payload, sphere, legacy)
            if kind == b'trak' and n:
                content = b''.join(r for k, p, r in boxes(content)
                                   if not (k == b'uuid' and p[:16] == SPHERICAL_UUID))
                content += b''.join(legacy)
            raw = pack(kind, content)
        elif kind == b'stsd':
            content, n = _rewrite(payload[8:], sphere, legacy)
            raw = pack(kind, payload[:8] + content)
        elif kind in VIDEO_ENTRIES:
            content = b''.join(r for k, _, r in boxes(payload[78:]) if k not in {b'sv3d', b'st3d'})
            raw = pack(kind, payload[:78] + content + b''.join(sphere))
            n = 1
        result.append(raw)
        count += n
    return b''.join(result), count


def restore_spherical_metadata(reference, encoded, destination, cancel=None):
    from .common import check_cancel
    reference, encoded, destination = map(Path, (reference, encoded, destination))
    if destination.exists():
        raise FileExistsError(destination)
    if destination.resolve() in {reference.resolve(), encoded.resolve()}:
        raise ValueError('Output must be separate from source')
    sphere, legacy = spherical_boxes(reference)
    if not sphere and not legacy:
        raise ValueError('Source has no spherical metadata')
    position, size, moov = read_moov(encoded)
    rewritten, count = _rewrite(moov, sphere, legacy)
    if count != 1:
        raise ValueError('Expected one supported video sample entry')
    try:
        with encoded.open('rb') as src, destination.open('xb') as dst:
            # Replacing moov with same-sized free keeps stco/co64 offsets valid.
            remaining = position
            while remaining:
                check_cancel(cancel)
                block = src.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError('Unexpected MP4 EOF')
                dst.write(block)
                remaining -= len(block)
            dst.write(pack(b'free', bytes(size - 8)))
            src.seek(position + size)
            while block := src.read(8 * 1024 * 1024):
                check_cancel(cancel)
                dst.write(block)
            dst.write(rewritten)
        if spherical_boxes(destination) != (sphere, legacy):
            raise ValueError('Spherical metadata verification failed')
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
