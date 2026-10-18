"""Photo regressions use real files/codecs; only inference is replaced."""
import hashlib
import threading
from pathlib import Path

import numpy as np
import pytest
import pyvips
from PIL import Image, ImageCms


def test_photo_engine_contract_is_available():
    from panoprivacy import photo_engine
    assert callable(photo_engine.prepare_photo)
    assert callable(photo_engine.scan_photo)
    assert callable(photo_engine.photo_thumbnail)
    assert callable(photo_engine.photo_view)
    assert callable(photo_engine.export_photo)


@pytest.fixture
def panorama(tmp_path):
    # Fine texture exposes unredacted holes; marker colours expose seam mistakes.
    rng = np.random.default_rng(31)
    rgb = rng.integers(30, 230, (384, 768, 3), dtype=np.uint8)
    source = tmp_path / "原始 全景.jpg"
    exif = Image.Exif()
    exif[271] = "Panorama Privacy Test Camera"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(rgb).save(source, quality=98, subsampling=0, exif=exif, icc_profile=icc)
    image = pyvips.Image.new_from_file(str(source)).copy()
    image.set_type(pyvips.GValue.blob_type, "xmp-data", b'<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:RDF><rdf:Description xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" GPano:ProjectionType="equirectangular" GPano:FullPanoWidthPixels="768" GPano:FullPanoHeightPixels="384"/></rdf:RDF></x:xmpmeta>')
    temporary = tmp_path / "metadata.jpg"
    image.jpegsave(str(temporary), Q=98, subsample_mode="off")
    temporary.replace(source)
    return source


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_prepare_export_preserves_source_and_metadata(panorama, tmp_path):
    from panoprivacy.common import new_region, default_style
    from panoprivacy.photo_engine import prepare_photo, export_photo, photo_thumbnail, photo_view
    before = _hash(panorama)
    asset = prepare_photo(panorama, tmp_path / "缓存")
    assert asset["width"] == 768 and asset["height"] == 384
    assert Path(asset["native_path"]).suffix == ".v"
    assert Path(asset["preview_path"]).is_file()
    r = new_region("erp", [.2, .2, .3, .4])
    assert photo_thumbnail(asset, r, 80).shape == (80, 80, 3)
    assert photo_view(asset, {"yaw": 0, "pitch": 0, "fov": 90, "roll": 0}, (120, 80)).shape == (80, 120, 3)
    destination = tmp_path / "结果.jpg"
    result = export_photo(asset, [r], default_style(), destination)
    assert result["width"] == 768 and result["height"] == 384
    assert _hash(panorama) == before
    source_vips = pyvips.Image.new_from_file(str(panorama), revalidate=True)
    output_vips = pyvips.Image.new_from_file(str(destination))
    assert output_vips.get("xmp-data") == source_vips.get("xmp-data")
    assert output_vips.get("icc-profile-data") == source_vips.get("icc-profile-data")
    assert Image.open(destination).getexif()[271] == "Panorama Privacy Test Camera"
    assert result["masked_regions"] == 1


def test_solid_wrap_pole_disabled_and_overlap(panorama, tmp_path):
    from panoprivacy.common import new_region
    from panoprivacy.photo_engine import prepare_photo, export_photo
    asset = prepare_photo(panorama, tmp_path / "cache")
    regions = [new_region("erp", [.97, .4, 1.03, .6]),
               new_region("erp", [.02, .45, .08, .55]),
               new_region("erp", [.4, .4, .6, .6], enabled=False),
               new_region("view", [.4, .4, .6, .6], pose={"yaw": 0, "pitch": 90, "fov": 80, "roll": 0}, view_size=(100, 100))]
    destination = tmp_path / "wrap.jpg"
    export_photo(asset, regions, {"kind": "solid", "strength": 1, "padding": 0, "color": "#202020"}, destination)
    output = np.array(Image.open(destination))
    assert np.max(np.abs(output[192, 0].astype(int) - 32)) <= 3
    assert np.max(np.abs(output[192, -1].astype(int) - 32)) <= 3
    assert np.max(np.abs(output[192, 40].astype(int) - 32)) <= 3
    assert np.max(np.abs(output[0, 100].astype(int) - 32)) <= 5
    # Disabled centre cannot become a mask; original texture remains.
    assert output[170:210, 350:410].std() > 20


def test_refuses_overwrite_and_changed_source(panorama, tmp_path):
    from panoprivacy.photo_engine import prepare_photo, export_photo
    from panoprivacy.common import default_style
    asset = prepare_photo(panorama, tmp_path / "cache")
    with pytest.raises((ValueError, FileExistsError)):
        export_photo(asset, [], default_style(), panorama)
    destination = tmp_path / "existing.jpg"
    destination.write_bytes(b"keep")
    with pytest.raises((ValueError, FileExistsError)):
        export_photo(asset, [], default_style(), destination)
    assert destination.read_bytes() == b"keep"
    with panorama.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises((ValueError, RuntimeError), match="来源|源文件|source|changed"):
        export_photo(asset, [], default_style(), tmp_path / "changed.jpg")


@pytest.mark.parametrize("cancel_stage", ["photo_export_masks", "photo_export_native", "photo_export_encode", "photo_export_verify"])
def test_cancel_before_and_during_export_leaves_no_output(panorama, tmp_path, cancel_stage):
    from panoprivacy.common import Cancelled, default_style, new_region
    from panoprivacy.photo_engine import prepare_photo, export_photo
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        prepare_photo(panorama, tmp_path / "precancel", cancel=cancel)
    cancel.clear()
    asset = prepare_photo(panorama, tmp_path / "cache")
    def progress(event):
        if event["stage"] == cancel_stage:
            cancel.set()
    destination = tmp_path / "cancelled.jpg"
    with pytest.raises(Cancelled):
        export_photo(asset, [new_region("erp", [.1, .2, .2, .4])], default_style(), destination, progress, cancel)
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part*"))
    assert not list(tmp_path.glob(".photo-export-*"))


@pytest.mark.parametrize("kind", ["blur", "mosaic"])
def test_styles_remove_texture_inside_entire_region(panorama, tmp_path, kind):
    from panoprivacy.common import new_region
    from panoprivacy.photo_engine import prepare_photo, export_photo
    asset = prepare_photo(panorama, tmp_path / "cache")
    destination = tmp_path / f"{kind}.jpg"
    export_photo(asset, [new_region("erp", [.2, .2, .4, .6])], {"kind": kind, "padding": 0, "strength": 1, "color": "#202020"}, destination)
    output = np.array(Image.open(destination))
    assert output[90:210, 170:285].std() < 12
    assert output[90:210, 400:500].std() > 20


@pytest.mark.parametrize("mode,expected", [("obvious", 2), ("more", 3)])
def test_scan_keeps_reviewable_giant_but_disables_suspect_and_more_suggestions(panorama, tmp_path, monkeypatch, mode, expected):
    from panoprivacy import detector, photo_engine, photo_confirm, person_context
    class SyntheticInference:
        provider = "CPU synthetic inference fixture"
        def __init__(self):
            self.calls = 0
        def detect(self, rgb, min_score=.6):
            self.calls += 1
            if self.calls != 1:
                return []
            return [[20, 30, 80, 90, .91], [220, 20, 240, 40, .45], [0, 0, 550, 350, .85]]
    # Inference is the external expensive boundary; preparation/projection/NMS
    # remain the production implementation, operating on the real fixture.
    monkeypatch.setattr(detector, "FaceDetector", SyntheticInference)
    monkeypatch.setattr(person_context, "filter_candidates", lambda asset, regions, progress=None, cancel=None: regions)
    class ConfirmedInference:
        def confirm(self, rgb, box):
            return {'confirmed': True, 'status': 'confirmed', 'score': .99, 'detections': []}
    monkeypatch.setattr(photo_confirm, 'CandidateConfirmer', ConfirmedInference)
    asset = photo_engine.prepare_photo(panorama, tmp_path / "cache")
    result = photo_engine.scan_photo(asset, mode)
    assert len(result) == expected
    assert sum(r["enabled"] for r in result) == 1
    suspect = next(r for r in result if r["score"] == .85)
    assert suspect["enabled"] is False and suspect["note"]
    assert all(isinstance(r["id"], str) for r in result)
    if mode == "more":
        extra = next(r for r in result if r["score"] == .45)
        assert extra["enabled"] is False and extra["note"]


def test_native_view_convention_and_roll_agree_with_shared_preview(tmp_path):
    from panoprivacy import photo_engine
    from panoprivacy.geometry import render_view
    # A coordinate ramp detects sign flips in pitch, roll, and seam wrapping.
    yy, xx = np.indices((128, 256))
    rgb = np.stack([xx, yy * 2, np.full_like(xx, 90)], axis=2).astype(np.uint8)
    image = pyvips.Image.new_from_memory(rgb.data, 256, 128, 3, "uchar")
    for pose in ({"yaw": 40, "pitch": 30, "fov": 70, "roll": 20},
                 {"yaw": 179, "pitch": -30, "fov": 85, "roll": -10}):
        actual = photo_engine._native_view(image, pose, (96, 64))
        expected = render_view(rgb, pose, (96, 64))
        # Ignore the wrapped colour discontinuity when comparing interpolators.
        difference = np.abs(actual.astype(int) - expected.astype(int))
        assert np.percentile(difference, 95) <= 2


def test_repeated_exports_clean_temporary_files_and_allow_native_eviction(panorama, tmp_path):
    from panoprivacy.common import new_region, default_style
    from panoprivacy.photo_engine import prepare_photo, export_photo, photo_thumbnail
    asset = prepare_photo(panorama, tmp_path / "cache")
    region = new_region("erp", [.2, .2, .4, .5])
    photo_thumbnail(asset, region)
    for index, kind in enumerate(("blur", "mosaic", "solid", "blur")):
        export_photo(asset, [region], {**default_style(), "kind": kind}, tmp_path / f"round-{index}.jpg")
        assert not list(tmp_path.glob(".photo-export-*"))
        assert not list(tmp_path.glob("*.part.jpg"))
    # On Windows this raises if libvips still caches an open native file handle.
    native = Path(asset["native_path"])
    native.unlink()
    assert not native.exists()
    rebuilt = prepare_photo(panorama, tmp_path / "cache")
    assert Path(rebuilt["native_path"]).is_file()
    assert rebuilt["fingerprint"] == asset["fingerprint"]


def test_failed_encoder_cleans_disk_work_before_exception_is_released(panorama, tmp_path, monkeypatch):
    from panoprivacy.common import default_style
    from panoprivacy.photo_engine import prepare_photo, export_photo
    asset = prepare_photo(panorama, tmp_path / "cache")
    def failing_save(self, *args, **kwargs):
        raise pyvips.Error("synthetic codec failure")
    monkeypatch.setattr(pyvips.Image, "jpegsave", failing_save, raising=False)
    destination = tmp_path / "failed.jpg"
    with pytest.raises(pyvips.Error) as caught:
        export_photo(asset, [], default_style(), destination)
    assert "synthetic codec failure" in str(caught.value)
    assert not destination.exists()
    assert not list(tmp_path.glob(".photo-export-*"))


def test_cancel_during_preview_releases_native_cache(panorama, tmp_path):
    from panoprivacy.common import Cancelled
    from panoprivacy.photo_engine import prepare_photo
    event = threading.Event()
    cache = tmp_path / "cancelled-cache"
    def progress(update):
        if update["stage"] == "photo_preview":
            event.set()
    with pytest.raises(Cancelled):
        prepare_photo(panorama, cache, progress, event)
    assert not list(cache.glob("photo-*"))


@pytest.mark.parametrize("orientation", [2, 3, 6, 8])
def test_rejects_oriented_source_before_preview_geometry_can_diverge(tmp_path, orientation):
    from panoprivacy.photo_engine import prepare_photo
    source = tmp_path / f"orientation-{orientation}.jpg"
    rgb = np.zeros((200, 400, 3), dtype=np.uint8)
    rgb[:100, :200] = [220, 20, 40]
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(rgb).save(source, exif=exif)
    before = _hash(source)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="方向|orientation|Orientation"):
        prepare_photo(source, cache)
    assert _hash(source) == before
    assert not list(cache.glob("photo-*"))


@pytest.mark.parametrize("kind", ["blur", "mosaic"])
@pytest.mark.parametrize("box", [[.12, .2, .18, .32], [.2, .2, .4, .6], [.45, .25, .8, .85]])
def test_native_styles_match_same_resolution_preview_on_sine_and_ramp(tmp_path, kind, box):
    from panoprivacy.common import new_region
    from panoprivacy.geometry import apply_regions_rgb, region_mask_tiles
    from panoprivacy.photo_engine import prepare_photo, export_photo
    import cv2
    yy, xx = np.indices((512, 1024))
    rgb = np.stack([127 + 95 * np.sin(xx / 13), 127 + 90 * np.sin(yy / 9), xx / 1023 * 255], axis=2).clip(0, 255).astype(np.uint8)
    source = tmp_path / "pattern.jpg"
    Image.fromarray(rgb).save(source, quality=98, subsampling=0)
    decoded = np.array(Image.open(source))
    region = new_region("erp", box)
    style = {"kind": kind, "strength": 1., "padding": .25, "color": "#202020"}
    expected = apply_regions_rgb(decoded, [region], style)
    asset = prepare_photo(source, tmp_path / "cache")
    output = tmp_path / "styled.jpg"
    export_photo(asset, [region], style, output)
    actual = np.array(Image.open(output))
    mask = np.zeros((512, 1024), np.uint8)
    for x, y, tile in region_mask_tiles(region, (1024, 512), padding=.25):
        mask[y:y + tile.shape[0], x:x + tile.shape[1]] |= tile.astype(np.uint8)
    interior = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
    error = np.abs(actual.astype(float) - expected.astype(float))[interior]
    assert error.mean() < 2.5
    assert np.percentile(error, 99) <= 9


@pytest.mark.parametrize("mode", ["L", "CMYK"])
def test_non_rgb_jpeg_is_explicitly_rejected_without_reusing_wrong_icc(tmp_path, mode):
    from panoprivacy.photo_engine import prepare_photo
    source = tmp_path / f"{mode}.jpg"
    Image.new(mode, (400, 200)).save(source)
    before = _hash(source)
    with pytest.raises(ValueError, match="RGB"):
        prepare_photo(source, tmp_path / "cache")
    assert _hash(source) == before


def test_blur_filters_across_erp_seam_instead_of_leaving_a_sharp_split(tmp_path):
    from panoprivacy.common import new_region
    from panoprivacy.photo_engine import prepare_photo, export_photo
    source = tmp_path / 'seam-step.jpg'
    rgb = np.full((200, 400, 3), 40, dtype=np.uint8)
    rgb[:, 200:] = 210
    Image.fromarray(rgb).save(source, quality=98, subsampling=0)
    asset = prepare_photo(source, tmp_path / 'cache')
    destination = tmp_path / 'seam-blur.jpg'
    region = new_region('erp', [.9, .2, 1.1, .8])
    export_photo(asset, [region], {'kind': 'blur', 'strength': 1, 'padding': 0, 'color': '#202020'}, destination)
    result = np.array(Image.open(destination))
    assert 100 < float(result[100, 0].mean()) < 145
    assert 100 < float(result[100, -1].mean()) < 145
    assert abs(float(result[100, 0].mean()) - float(result[100, -1].mean())) < 15
