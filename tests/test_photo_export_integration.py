from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from panoprivacy.common import new_region
from panoprivacy.photo_engine import prepare_photo, export_photo


@pytest.mark.parametrize('quality',[95,98,100])
def test_requested_quality_and_ellipse_reencode(tmp_path,quality):
    source=tmp_path/'输入.jpg'
    rng=np.random.default_rng(42)
    Image.fromarray(rng.integers(0,256,(128,256,3),dtype=np.uint8)).save(source,quality=100,subsampling=0)
    asset=prepare_photo(source,tmp_path/'cache')
    out=tmp_path/f'output-{quality}.jpg'
    result=export_photo(asset,[new_region('erp',[.4,.3,.6,.7])],
                        {'kind':'solid','shape':'ellipse','export_mode':'jpeg_reencode','jpeg_quality':quality},out)
    assert result['jpeg_quality']==quality and result['shape']=='ellipse'
    assert result['export_method']=='jpeg_reencode'
    with Image.open(out) as img:
        expected=Image.new('RGB',(16,16))
        reference=tmp_path/'q.jpg';expected.save(reference,quality=quality,subsampling=0)
        with Image.open(reference) as ref:assert img.quantization==ref.quantization


def test_exif_makernote_fallback_removes_opaque_original_preview(tmp_path):
    source=tmp_path/'带隐式预览.jpg'
    exif=Image.Exif();exif[271]='Test camera';exif[37500]=b'opaque maker data'
    Image.new('RGB',(256,128),(160,120,80)).save(source,exif=exif)
    asset=prepare_photo(source,tmp_path/'cache');out=tmp_path/'sanitized.jpg'
    result=export_photo(asset,[new_region('erp',[.4,.3,.6,.7])],
                        {'kind':'solid','shape':'ellipse','export_mode':'auto'},out)
    assert result['export_method']=='jpeg_reencode' and result['metadata_removed']
    with Image.open(out) as img:
        assert 37500 not in img.getexif()
        assert 37500 not in img.getexif().get_ifd(34665)
