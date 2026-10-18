"""Real codec tests for the coefficient-preserving export boundary."""
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def make_asset(tmp_path, subsampling=0, progressive=False, keep_rgb=False):
    from panoprivacy.photo_engine import prepare_photo
    rgb = np.random.default_rng(39).integers(20, 240, (131, 261, 3), dtype=np.uint8)
    source = tmp_path / '源 照片.jpg'
    exif = Image.Exif()
    exif[271] = 'Metadata Test'
    Image.fromarray(rgb).save(source, quality=93, subsampling=subsampling,
                             progressive=progressive, keep_rgb=keep_rgb, exif=exif,
                             comment=b'keep comments')
    return prepare_photo(source, tmp_path / '缓存'), rgb


def test_module_contract():
    from panoprivacy import local_jpeg, photo_export
    assert callable(local_jpeg.inspect_jpeg)
    assert callable(photo_export.export_photo)


@pytest.mark.parametrize('subsampling', [0, 2])
def test_memory_uses_component_block_counts(tmp_path, subsampling):
    from panoprivacy.local_jpeg import inspect_jpeg, coefficient_bytes
    asset, _ = make_asset(tmp_path, subsampling)
    info = inspect_jpeg(asset['source'])
    # ceil component dimensions/8, not source dimensions multiplied by sampling.
    expected = (33 * 17 * 3 if subsampling == 0 else 33 * 17 + 17 * 9 * 2) * 128
    assert coefficient_bytes(info) == expected


@pytest.mark.parametrize('subsampling,shape', [(0, 'rect'), (0, 'ellipse'), (2, 'ellipse'), (2, 'circle')])
def test_real_export_odd_dimensions_shapes_overlap_seam(tmp_path, subsampling, shape):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.common import new_region
    from panoprivacy.local_jpeg import inspect_jpeg
    asset, _ = make_asset(tmp_path, subsampling)
    before = hashlib.sha256(Path(asset['source']).read_bytes()).hexdigest()
    regions = [new_region('erp', [.3, .3, .48, .65]),
               new_region('erp', [.38, .4, .53, .7]),
               new_region('erp', [.97, .35, 1.03, .65])]
    output = tmp_path / '输出 带空格.jpg'
    result = export_photo(asset, regions, {'kind': 'solid', 'color': '#202020', 'padding': .25,
                                         'shape': shape, 'export_mode': 'auto'}, output)
    assert result['export_method'] == 'local_jpeg'
    assert result['validated'] and result['outside_coefficients_exact']
    assert result['outside_pixel_differences'] == 0
    assert result['shape'] == shape and result['masked_regions'] == 3
    assert hashlib.sha256(Path(asset['source']).read_bytes()).hexdigest() == before
    source_info, output_info = inspect_jpeg(asset['source']), inspect_jpeg(output)
    for field in ('size', 'sampling', 'qtables', 'markers'):
        assert source_info[field] == output_info[field]
    assert Image.open(output).getexif()[271] == 'Metadata Test'
    decoded = np.asarray(Image.open(output))
    assert np.abs(decoded[65, 0].astype(int) - 32).max() < 8
    assert np.abs(decoded[65, 110].astype(int) - 32).max() < 8
    assert not list(tmp_path.glob('.photo-local-*'))


def test_rgb_jpeg_supported(tmp_path):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.common import new_region
    asset, _ = make_asset(tmp_path, keep_rgb=True)
    result = export_photo(asset, [new_region('erp', [.3, .3, .5, .7])],
                          {'kind': 'solid', 'shape': 'rect'}, tmp_path / 'rgb.jpg')
    assert result['export_method'] == 'local_jpeg'


@pytest.mark.parametrize('stage', ['photo_export_masks', 'photo_export_native', 'photo_export_encode', 'photo_export_verify'])
def test_cancel_cleans_temps(tmp_path, stage):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.common import new_region, Cancelled
    asset, _ = make_asset(tmp_path)
    event = threading.Event()
    def progress(record):
        if record['stage'] == stage:
            event.set()
    with pytest.raises(Cancelled):
        export_photo(asset, [new_region('erp', [.2, .3, .4, .6])], {'kind': 'blur'},
                     tmp_path / 'cancel.jpg', progress, event)
    assert not (tmp_path / 'cancel.jpg').exists()
    assert not list(tmp_path.glob('.photo-local-*'))


def test_native_rejects_malformed_manifest_and_existing_output(tmp_path):
    from panoprivacy.local_jpeg import run_helper, LocalJPEGError
    asset, _ = make_asset(tmp_path)
    manifest = tmp_path / 'bad.manifest'
    manifest.write_bytes(b'PJ01\x01\x00\x00\x00truncated')
    output = tmp_path / 'keep.jpg'
    output.write_bytes(b'preserve')
    with pytest.raises(LocalJPEGError):
        run_helper(['patch', asset['source'], output, manifest])
    assert output.read_bytes() == b'preserve'


def test_corrupt_source_is_error_not_fallback(tmp_path):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.common import fingerprint
    asset, _ = make_asset(tmp_path)
    source = Path(asset['source'])
    source.write_bytes(source.read_bytes()[:-200])
    asset['fingerprint'] = fingerprint(source)
    with pytest.raises((RuntimeError, ValueError, OSError)):
        export_photo(asset, [], {'kind': 'solid'}, tmp_path / 'bad.jpg')
    assert not (tmp_path / 'bad.jpg').exists()


def test_explicit_fallback_and_low_memory_reason(tmp_path, monkeypatch):
    from panoprivacy.photo_export import export_photo
    from panoprivacy import local_jpeg
    asset, _ = make_asset(tmp_path)
    monkeypatch.setattr(local_jpeg, 'available_memory', lambda: 1)
    result = export_photo(asset, [], {'kind': 'solid', 'jpeg_quality': 95}, tmp_path / 'low.jpg')
    assert result['export_method'] == 'jpeg_reencode'
    assert result['jpeg_quality'] == 95 and '内存' in result['fallback_reason']


def test_progressive_fallback_is_explicit(tmp_path):
    from panoprivacy.photo_export import export_photo
    asset, _ = make_asset(tmp_path, progressive=True)
    result = export_photo(asset, [], {'kind': 'solid', 'jpeg_quality': 100}, tmp_path / 'fallback.jpg')
    assert result['export_method'] == 'jpeg_reencode'
    assert result['jpeg_quality'] == 100 and result['fallback_reason']


def _one_patch(tmp_path, asset, quantization_match=True):
    from panoprivacy.local_jpeg import write_manifest
    patch = tmp_path / '遮挡 patch.jpg'
    with Image.open(asset['source']) as original:
        pixels = original.crop((16, 16, 48, 48))
        if quantization_match:
            pixels.save(patch, qtables=original.quantization, subsampling=0)
        else:
            pixels.save(patch, quality=20, subsampling=0)
    manifest = tmp_path / 'manifest.bin'
    write_manifest(manifest, [dict(x=16, y=16, w=32, h=32, path=patch, mask=b'\x01' * 16)])
    return manifest


def test_native_quantization_mismatch_fails_without_output(tmp_path):
    from panoprivacy.local_jpeg import run_helper, LocalJPEGError
    asset, _ = make_asset(tmp_path)
    manifest = _one_patch(tmp_path, asset, False)
    output = tmp_path / 'badquant.jpg'
    with pytest.raises(LocalJPEGError, match='quantization'):
        run_helper(['patch', asset['source'], output, manifest])
    assert not output.exists()


def test_native_pixel_comparison_returns_error_on_mismatch(tmp_path):
    from panoprivacy.local_jpeg import run_helper, LocalJPEGError, write_manifest
    asset, _ = make_asset(tmp_path)
    wrong = tmp_path / 'wrong.jpg'
    Image.new('RGB', (261, 131), 'red').save(wrong, quality=93, subsampling=0)
    manifest = tmp_path / 'empty.bin'
    write_manifest(manifest, [])
    with pytest.raises(LocalJPEGError, match='pixels outside'):
        run_helper(['pixels', asset['source'], wrong, manifest, 0])


def test_verification_failure_never_falls_back_or_publishes(tmp_path, monkeypatch):
    from panoprivacy.photo_export import export_photo
    from panoprivacy import local_jpeg, photo_engine
    asset, _ = make_asset(tmp_path)
    original = local_jpeg.run_helper
    def helper(arguments, cancel=None):
        result = original(arguments, cancel)
        if arguments[0] == 'hash' and str(arguments[1]) != asset['source']:
            result['coefficients'][0]['outside_sha256'] = 'mismatch'
        return result
    monkeypatch.setattr(local_jpeg, 'run_helper', helper)
    def must_not_fallback(*args, **kwargs):
        pytest.fail('verification errors must not be hidden')
    monkeypatch.setattr(photo_engine, '_export_photo_reencode', must_not_fallback)
    with pytest.raises(local_jpeg.LocalJPEGError, match='系数验证失败'):
        export_photo(asset, [], {'kind': 'solid'}, tmp_path / 'error.jpg')
    assert not (tmp_path / 'error.jpg').exists()
    assert not list(tmp_path.glob('.photo-local-*'))


def test_alias_output_cannot_modify_source(tmp_path):
    from panoprivacy.local_jpeg import run_helper, LocalJPEGError, write_manifest
    from panoprivacy.photo_export import export_photo
    asset, _ = make_asset(tmp_path)
    source = Path(asset['source'])
    before = source.read_bytes()
    alias = tmp_path / 'hardlink.jpg'
    alias.hardlink_to(source)
    manifest = tmp_path / 'empty.bin'
    write_manifest(manifest, [])
    with pytest.raises((ValueError, FileExistsError)):
        export_photo(asset, [], {'kind': 'solid'}, alias)
    with pytest.raises(LocalJPEGError, match='exists'):
        run_helper(['patch', source, alias, manifest])
    assert source.read_bytes() == before


def test_semantically_equal_grouped_quantization_tables(tmp_path):
    from panoprivacy.local_jpeg import inspect_jpeg
    asset, _ = make_asset(tmp_path)
    source = Path(asset['source'])
    blob = source.read_bytes()
    # Pillow emits one DQT marker per table; JPEG also permits a combined DQT.
    offset = 2
    tables = []
    others = []
    while blob[offset:offset + 2] != b'\xff\xda':
        length = int.from_bytes(blob[offset+2:offset+4], 'big')
        segment = blob[offset:offset+2+length]
        if segment[:2] == b'\xff\xdb':
            tables.append(segment[4:])
        else:
            others.append(segment)
        offset += 2 + length
    combined = b''.join(tables)
    grouped = tmp_path / 'grouped.jpg'
    grouped.write_bytes(b'\xff\xd8' + b''.join(others) + b'\xff\xdb' +
                        (len(combined)+2).to_bytes(2, 'big') + combined + blob[offset:])
    assert inspect_jpeg(source)['qtables'] == inspect_jpeg(grouped)['qtables']


def test_overlapping_blur_composes_sequentially_without_unmasked_changes(tmp_path):
    from panoprivacy.photo_export import _prepare_patches
    from panoprivacy.local_jpeg import inspect_jpeg
    from panoprivacy.common import new_region
    asset, _ = make_asset(tmp_path)
    region = new_region('erp', [.2, .25, .7, .75])
    style = {'kind': 'blur', 'padding': 0, 'shape': 'ellipse', 'strength': 1, 'color': '#202020'}
    once, twice = tmp_path / 'once', tmp_path / 'twice'
    once.mkdir()
    twice.mkdir()
    info = inspect_jpeg(asset['source'])
    first = _prepare_patches(asset, [region], style, info, once, None, None)
    second = _prepare_patches(asset, [region, region], style, info, twice, None, None)
    a = np.frombuffer(Path(str(first[0]['path']) + '.rgb').read_bytes(), np.uint8)
    b = np.frombuffer(Path(str(second[0]['path']) + '.rgb').read_bytes(), np.uint8)
    assert np.any(a != b), 'A second overlapping blur must consume the previous region result'


def test_memory_rechecked_after_patch_rendering(tmp_path, monkeypatch):
    from panoprivacy.photo_export import export_photo
    from panoprivacy import local_jpeg
    asset, _ = make_asset(tmp_path)
    memory = [100 * 1024 ** 3]
    monkeypatch.setattr(local_jpeg, 'available_memory', lambda: memory[0])
    def progress(event):
        if event['stage'] == 'photo_export_encode':
            memory[0] = 1
    result = export_photo(asset, [], {'kind': 'solid'}, tmp_path / 'latepressure.jpg', progress)
    assert result['export_method'] == 'jpeg_reencode'
    assert '内存' in result['fallback_reason']


@pytest.mark.parametrize('payload', [
    b'http://ns.adobe.com/xap/1.0/\0<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:thumb="http://ns.adobe.com/xap/1.0/g/img/"><thumb:image>private base64</thumb:image></rdf:RDF>',
    b'http://ns.adobe.com/xmp/extension/\0extended-data',
    b'http://ns.adobe.com/xap/1.0/\0<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:p="http://ns.google.com/photos/1.0/image/"><p:Data>private base64</p:Data></rdf:RDF>',
])
def test_embedded_xmp_preview_rejected_for_both_export_modes(tmp_path, payload):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.local_jpeg import LocalJPEGError, inspect_jpeg
    from panoprivacy.common import fingerprint
    asset, _ = make_asset(tmp_path)
    source = Path(asset['source'])
    blob = source.read_bytes()
    source.write_bytes(blob[:2] + b'\xff\xe1' + (len(payload)+2).to_bytes(2, 'big') + payload + blob[2:])
    asset['fingerprint'] = fingerprint(source)
    assert inspect_jpeg(source)['unsafe_other_preview']
    for mode in ('auto', 'jpeg_reencode'):
        with pytest.raises(LocalJPEGError, match='内嵌预览'):
            export_photo(asset, [], {'kind': 'solid', 'export_mode': mode}, tmp_path / f'{mode}.jpg')
        assert not (tmp_path / f'{mode}.jpg').exists()


def test_makernote_forces_preview_removing_fallback(tmp_path):
    from panoprivacy.photo_export import export_photo
    from panoprivacy.local_jpeg import inspect_jpeg
    from panoprivacy.common import fingerprint
    asset, rgb = make_asset(tmp_path)
    source = Path(asset['source'])
    exif = Image.Exif()
    exif[34665] = {37500: b'unknown maker note may contain preview'}
    Image.fromarray(rgb).save(source, quality=93, subsampling=0, exif=exif)
    asset['fingerprint'] = fingerprint(source)
    assert inspect_jpeg(source)['unsafe_exif_preview']
    result = export_photo(asset, [], {'kind': 'solid'}, tmp_path / 'sanitized.jpg')
    assert result['export_method'] == 'jpeg_reencode'
    assert result['metadata_removed']
    assert not inspect_jpeg(result['output'])['unsafe_exif_preview']


def test_corrupt_exif_is_not_preserved_as_preview_safe(tmp_path):
    from panoprivacy.local_jpeg import LocalJPEGError, inspect_jpeg
    asset, _ = make_asset(tmp_path)
    source = Path(asset['source'])
    payload = b'Exif\0\0II*\0' + (4000).to_bytes(4, 'little') + b'private embedded image'
    blob = source.read_bytes()
    source.write_bytes(blob[:2] + b'\xff\xe1' + (len(payload)+2).to_bytes(2, 'big') + payload + blob[2:])
    with pytest.raises(LocalJPEGError, match='EXIF'):
        inspect_jpeg(source)
