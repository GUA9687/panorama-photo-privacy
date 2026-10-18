import numpy as np
import pytest


def test_real_confirmation_model_loads_from_unicode_install_folder(tmp_path):
    from panoprivacy.common import resource_path
    from panoprivacy.photo_confirm import CandidateConfirmer
    folder = tmp_path/'全景工具 含空格'
    folder.mkdir()
    model = folder/'人脸确认.onnx'
    model.write_bytes(resource_path('models/face_detection_yunet_2023mar.onnx').read_bytes())
    result = CandidateConfirmer(model).confirm(np.zeros((320,320,3),np.uint8),[100,100,200,200])
    assert result['status'] == 'unconfirmed'
    assert result['detections'] == []


@pytest.mark.parametrize("detections,confirmed", [
    ([[90, 90, 170, 170, .95]], True),
    ([], False),
    ([[90, 90, 170, 170, .79]], False),
    ([[220, 100, 280, 160, .99]], False),
    ([[90, 90, 170, 170, float('nan')]], False),
])
def test_confirmation_requires_target_overlap_and_threshold_not_any_neighbour(detections, confirmed):
    from panoprivacy.photo_confirm import associate_candidate
    result = associate_candidate([100, 100, 160, 160], detections)
    assert result['confirmed'] is confirmed
    assert result['status'] == ('confirmed' if confirmed else 'unconfirmed')


def test_unconfirmed_candidate_stays_reviewable_without_being_called_false(panorama_asset, monkeypatch):
    from panoprivacy.common import new_region
    from panoprivacy import photo_engine, photo_confirm
    class NoFaceInference:
        def confirm(self, rgb, box):
            return {'confirmed': False, 'status': 'unconfirmed', 'score': 0., 'detections': []}
    monkeypatch.setattr(photo_confirm, 'CandidateConfirmer', NoFaceInference)
    region = new_region('erp', [.2, .2, .3, .4], origin='auto', score=.9)
    result = photo_engine.confirm_photo_candidates(panorama_asset, [region])
    assert len(result) == 1 and result[0]['id'] == region['id']
    assert result[0]['box'] == region['box'] and result[0]['score'] == .9
    assert result[0]['enabled'] is False and '未确认' in result[0]['note']
    assert '误报' not in result[0]['note'] and '假' not in result[0]['note']
    assert region['enabled'] is True


@pytest.fixture
def panorama_asset(tmp_path):
    from PIL import Image
    from panoprivacy.photo_engine import prepare_photo
    source = tmp_path/'plain.jpg'
    Image.fromarray(np.full((200, 400, 3), 100, np.uint8)).save(source)
    return prepare_photo(source, tmp_path/'cache')
