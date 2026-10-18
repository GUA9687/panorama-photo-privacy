"""Photo interactions use real Qt events without loading a detector or GPU."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from copy import deepcopy
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from panoprivacy.common import new_region
from panoprivacy.ui.photo_window import PhotoWindow
from panoprivacy.ui.state import DocumentState


@pytest.fixture(scope='session')
def app():
    return QApplication.instance() or QApplication([])


def pump(app, condition, seconds=3):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        app.processEvents()
        time.sleep(.005)
    assert condition()


@pytest.fixture
def window(app):
    widget = PhotoWindow()
    yield widget
    widget._closing = True
    widget.preview_timer.stop()
    widget.native_timer.stop()
    widget._eviction_timer.stop()
    widget.close()
    pump(app, lambda: all(not c.active_count for c in
        [widget.jobs, widget.previews, widget.native_jobs]))


def drop_urls(app, widget, urls):
    mime = QMimeData()
    mime.setUrls(urls)
    enter = QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(widget.viewport(), enter)
    drop = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    app.sendEvent(widget.viewport(), drop)
    return enter.isAccepted(), drop.isAccepted()


def fixture_regions(count=8):
    regions = [new_region('erp', [.05+i*.06, .3, .08+i*.06, .4],
                          id=f'face-{i}', origin='auto', enabled=i % 2 == 0)
               for i in range(count)]
    thumbnails = {region['id']: np.full((96, 96, 3), 80+i*8, np.uint8)
                  for i, region in enumerate(regions)}
    return regions, thumbnails


def test_drop_filters_local_existing_jpegs_and_deduplicates_queue(app, window, tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(window, '_prepare', lambda source, **kwargs: opened.append(source))
    photo = tmp_path/'照片 空格.JPEG'
    photo.write_bytes(b'JPEG decoding belongs to the preparation worker')
    other = tmp_path/'not-supported.png'
    other.write_bytes(b'png')
    folder = tmp_path/'folder.jpg'
    folder.mkdir()
    urls = [QUrl.fromLocalFile(str(photo)), QUrl.fromLocalFile(str(photo)),
            QUrl.fromLocalFile(str(other)), QUrl.fromLocalFile(str(folder)),
            QUrl.fromLocalFile(str(tmp_path/'missing.jpg')),
            QUrl('https://example.invalid/photo.jpg')]
    window.show()
    app.processEvents()
    assert drop_urls(app, window.queue, urls) == (True, True)
    assert [state.asset['source'] for state in window.queue_states] == [str(photo.resolve())]
    assert opened == [str(photo.resolve())]
    assert window.queue.count() == 1
    drop_urls(app, window.queue, [QUrl.fromLocalFile(str(photo))])
    assert window.queue.count() == 1
    assert len(opened) == 1
    assert drop_urls(app, window.queue, [QUrl('https://example.invalid/photo.jpg')]) == (False, False)


def test_drop_and_add_paths_refuse_to_mutate_busy_queue(app, window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, '_prepare', lambda *args, **kwargs: None)
    source = tmp_path/'photo.jpg'
    source.write_bytes(b'fixture')
    window.show()
    app.processEvents()
    release = threading.Event()
    entered = threading.Event()
    def job(progress, cancel):
        entered.set()
        release.wait(2)
    window.run_job(job, lambda _: None, 'busy')
    assert entered.wait(1)
    try:
        assert drop_urls(app, window.queue, [QUrl.fromLocalFile(str(source))]) == (False, False)
        window.add_paths([str(source)])
        assert window.queue_states == []
        assert window.queue.count() == 0
    finally:
        release.set()
        pump(app, lambda: not window.jobs.active_count)


def test_grid_reflows_when_candidate_pane_resizes_without_losing_selection(app, window):
    regions, thumbs = fixture_regions(12)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('grid'))
    window.show()
    app.processEvents()
    window.splitter.setSizes([220, 800, 270])
    app.processEvents()
    window.candidates.setCurrentRow(2)
    narrow = [window.candidates.visualItemRect(window.candidates.item(i)) for i in range(8)]
    window.splitter.setSizes([220, 500, 570])
    app.processEvents()
    wide = [window.candidates.visualItemRect(window.candidates.item(i)) for i in range(8)]
    narrow_columns = len({r.x() for r in narrow if r.y() == narrow[0].y()})
    wide_columns = len({r.x() for r in wide if r.y() == wide[0].y()})
    assert narrow_columns >= 2
    assert wide_columns > narrow_columns
    assert all(window.candidates.item(i).text() == '' for i in range(12))
    assert window.candidates.currentItem().data(Qt.ItemDataRole.UserRole) == 'face-2'
    assert [r['enabled'] for r in window.state.regions] == [i % 2 == 0 for i in range(12)]
    assert window.state.region_undo == []
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('list'))
    assert all(window.candidates.item(i).text() for i in range(12))
    assert window.candidates.currentItem().data(Qt.ItemDataRole.UserRole) == 'face-2'


def test_grid_check_overlay_updates_state_and_undo_and_double_click_focuses(app, window):
    regions, thumbs = fixture_regions(2)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('grid'))
    window.show()
    app.processEvents()
    item = window.candidates.item(0)
    rect = window.candidates.visualItemRect(item)
    QTest.mouseClick(window.candidates.viewport(), Qt.MouseButton.LeftButton,
                     pos=rect.topLeft()+QPoint(17, 17))
    assert window.state.regions[0]['enabled'] is False
    assert len(window.state.region_undo) == 1
    assert '人脸候选' in item.toolTip()
    window.undo_regions()
    assert window.state.regions[0]['enabled'] is True
    item = window.candidates.item(0)
    original_pose = deepcopy(window.viewport.pose)
    # Qt's helper emits only the second-click event; a user double click has
    # an ordinary first click before it.
    QTest.mouseClick(window.candidates.viewport(), Qt.MouseButton.LeftButton,
                     pos=window.candidates.visualItemRect(item).center())
    QTest.mouseDClick(window.candidates.viewport(), Qt.MouseButton.LeftButton,
                     pos=window.candidates.visualItemRect(item).center())
    assert window.viewport.pose != original_pose
    assert window.state.regions[0]['id'] == 'face-0'


def test_context_toggle_hides_only_background_and_preserves_disabled_choices(window):
    regions, thumbs = fixture_regions(3)
    regions[0].update(context_keep=True)
    regions[1].update(context_keep=False, context_reason='no_person_in_context')
    regions[2].update(origin='manual', context_keep=False)
    window.state.regions = deepcopy(regions)
    window.refresh_candidates(thumbs)
    assert window.candidates.count() == 2
    assert [window.candidates.item(i).data(Qt.ItemDataRole.UserRole) for i in range(2)] == ['face-0', 'face-2']
    assert len(window.state.regions) == 3
    window.show_background.setChecked(True)
    assert window.candidates.count() == 3
    item = window.candidates.item(1)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert 'no_person_in_context' in item.toolTip()
    item.setCheckState(Qt.CheckState.Checked)
    window.show_background.setChecked(False)
    window.show_background.setChecked(True)
    assert window.state.regions[1]['enabled'] is True
    assert window.candidates.item(1).checkState() == Qt.CheckState.Checked
    assert len(window.state.region_undo) == 1


def test_thumbnail_relayout_does_not_schedule_new_panorama_renders(window):
    regions, thumbs = fixture_regions(8)
    window.state.regions = regions
    window._preview_rgb = np.full((20, 40, 3), 140, np.uint8)
    revision = window._view_revision
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('grid'))
    window.candidates.resize(550, 400)
    window.candidates.reflow()
    assert window._view_revision == revision
    assert not window.state.dirty


def test_unavailable_context_filter_stays_visible_with_explicit_warning(window):
    regions, thumbs = fixture_regions(2)
    regions[0].update(context_keep=True, context_reason='filter_unavailable',
                      context={'status': 'unavailable', 'message': '模型文件缺失'})
    regions[1].update(context_keep=False, context_reason='background_context')
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    assert window.candidates.count() == 1
    assert window.state.regions[0]['enabled'] is True
    assert '不可用' in window.candidate_summary.text()
    assert '人工' in window.candidate_summary.text()


def test_new_photo_defaults_and_legacy_project_style_are_separate(window, tmp_path, monkeypatch):
    monkeypatch.setattr(window, '_prepare', lambda *args, **kwargs: None)
    source = tmp_path/'original.jpg'
    source.write_bytes(b'fixture')
    window.add_paths([str(source)])
    assert window.state.style['shape'] == 'ellipse'
    assert window.state.style['export_mode'] == 'auto'
    assert window.state.style['jpeg_quality'] == 98
    assert window.shape_combo.currentData() == 'ellipse'
    legacy = {'kind': 'photo', 'asset': {'kind': 'photo', 'source': str(source)},
              'style': {'kind': 'mosaic', 'strength': 1., 'padding': .25, 'color': '#202020'},
              'regions': [new_region('erp', [.1, .2, .2, .4], enabled=False)]}
    window.restore_project(legacy, str(tmp_path/'legacy.json'))
    assert window.state.style.get('shape', 'rect') == 'rect'
    assert window.shape_combo.currentData() == 'rect'
    assert window.export_mode.currentData() == 'jpeg_reencode'
    assert window.state.regions[0]['enabled'] is False
    window.shape_combo.setCurrentIndex(window.shape_combo.findData('circle'))
    window.export_mode.setCurrentIndex(window.export_mode.findData('jpeg_reencode'))
    window.jpeg_quality.setCurrentIndex(window.jpeg_quality.findData(100))
    assert window.state.style['shape'] == 'circle'
    assert window.state.style['export_mode'] == 'jpeg_reencode'
    assert window.state.style['jpeg_quality'] == 100
    snapshot = window.state.snapshot()
    window.switch_state(DocumentState.from_project(snapshot))
    assert window.shape_combo.currentData() == 'circle'
    assert window.export_mode.currentData() == 'jpeg_reencode'
    assert window.jpeg_quality.currentData() == 100
    assert window.state.regions[0]['enabled'] is False


def test_export_completion_explains_method_and_fallback(window, monkeypatch):
    dialogs = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: dialogs.append(args[2]))
    window.export_done({'output': 'result.jpg', 'export_method': 'jpeg_reencode',
                        'jpeg_quality': 98, 'fallback_reason': '可用内存不足'})
    assert '98' in window.status.text()
    assert '重新编码' in window.status.text()
    assert '可用内存不足' in window.status.text()
    assert '可用内存不足' in dialogs[-1]
    window.export_done({'output': 'local.jpg', 'export_method': 'local_jpeg', 'jpeg_quality': 98})
    assert '局部' in window.status.text()
    assert '可用内存不足' not in window.status.text()


def test_export_completion_discloses_removed_embedded_previews(window, monkeypatch):
    dialogs = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *args: dialogs.append(args[2]))
    window.export_done({'output': 'result.jpg', 'export_method': 'local_jpeg',
                        'jpeg_quality': 'original', 'metadata_removed': ['EXIF 内嵌缩略图']})
    assert 'EXIF 内嵌缩略图' in window.status.text()
    assert 'EXIF 内嵌缩略图' in dialogs[-1]


def test_loaded_thumbnails_clear_loading_status_and_complete_progress(app, window, monkeypatch):
    from panoprivacy import photo_engine
    # Thumbnail decoding is the external worker boundary. Keep the real Qt
    # task controller and success callback that update the visible status.
    monkeypatch.setattr(photo_engine, 'photo_thumbnail',
                        lambda asset, region, size: np.full((size, size, 3), 120, np.uint8))
    window.state.asset = {'source': 'fixture.jpg'}
    window.state.regions, _ = fixture_regions(2)
    window.load_thumbnails()
    pump(app, lambda: len(window._thumbnails) == 2 and not window.jobs.running)
    assert not window.status.text().startswith('正在')
    assert window.progress.value() == 100
    assert window.candidates.count() == 2


def selected_ids(window):
    return {item.data(Qt.ItemDataRole.UserRole) for item in window.candidates.selectedItems()}


def click_candidate(window, index, modifier=Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(window.candidates.viewport(), Qt.MouseButton.LeftButton, modifier,
                     window.candidates.visualItemRect(window.candidates.item(index)).center())


@pytest.mark.parametrize('mode', ['list', 'grid'])
def test_ctrl_shift_selection_is_independent_of_enabled_masks(app, window, mode):
    regions, thumbs = fixture_regions(6)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData(mode))
    window.show()
    app.processEvents()
    click_candidate(window, 0)
    click_candidate(window, 2, Qt.KeyboardModifier.ControlModifier)
    assert selected_ids(window) == {'face-0', 'face-2'}
    click_candidate(window, 0, Qt.KeyboardModifier.ControlModifier)
    assert selected_ids(window) == {'face-2'}
    # Re-anchor the range after testing Ctrl deselection.
    click_candidate(window, 2)
    click_candidate(window, 4, Qt.KeyboardModifier.ShiftModifier)
    assert selected_ids(window) == {'face-2', 'face-3', 'face-4'}
    click_candidate(window, 3, Qt.KeyboardModifier.ControlModifier)
    assert selected_ids(window) == {'face-2', 'face-4'}
    assert [r['enabled'] for r in window.state.regions] == [True, False, True, False, True, False]
    assert not window.state.region_undo


def test_grid_empty_space_rubber_band_selects_without_changing_masks(app, window):
    regions, thumbs = fixture_regions(8)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('grid'))
    window.show()
    window.splitter.setSizes([220, 500, 570])
    app.processEvents()
    viewport = window.candidates.viewport()
    start, end = viewport.rect().bottomRight()-QPoint(8, 8), QPoint(1, 1)
    assert window.candidates.itemAt(start) is None
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    for point in (QPoint(start.x(), start.y()-20), end):
        event = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(point),
                            QPointF(viewport.mapToGlobal(point)), Qt.MouseButton.NoButton,
                            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        app.sendEvent(viewport, event)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
    assert selected_ids(window) == {f'face-{i}' for i in range(8)}
    assert [r['enabled'] for r in window.state.regions] == [i % 2 == 0 for i in range(8)]
    assert not window.state.region_undo


def test_bulk_buttons_make_one_undo_and_preserve_selection_across_refresh_modes(app, window):
    regions, thumbs = fixture_regions(4)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.show()
    app.processEvents()
    click_candidate(window, 0)
    click_candidate(window, 2, Qt.KeyboardModifier.ControlModifier)
    window.disable_selected_button.click()
    assert [r['enabled'] for r in window.state.regions] == [False, False, False, False]
    assert len(window.state.region_undo) == 1
    assert selected_ids(window) == {'face-0', 'face-2'}
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData('grid'))
    assert selected_ids(window) == {'face-0', 'face-2'}
    window.undo_regions()
    assert [r['enabled'] for r in window.state.regions] == [True, False, True, False]
    assert not window.state.region_undo
    assert selected_ids(window) == {'face-0', 'face-2'}
    window.disable_selected_button.click()
    window.enable_selected_button.click()
    assert [r['enabled'] for r in window.state.regions] == [True, False, True, False]
    assert len(window.state.region_undo) == 2


@pytest.mark.parametrize('mode', ['list', 'grid'])
def test_selected_checkbox_changes_whole_selection_once_and_unselected_only_itself(app, window, mode):
    regions, thumbs = fixture_regions(4)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidate_mode.setCurrentIndex(window.candidate_mode.findData(mode))
    window.show()
    app.processEvents()
    click_candidate(window, 0)
    click_candidate(window, 1, Qt.KeyboardModifier.ControlModifier)
    rect = window.candidates.visualItemRect(window.candidates.item(0))
    point = rect.topLeft()+QPoint(17, 17) if mode == 'grid' else QPoint(rect.left()+12, rect.center().y())
    QTest.mouseClick(window.candidates.viewport(), Qt.MouseButton.LeftButton,
                     pos=point)
    assert selected_ids(window) == {'face-0', 'face-1'}
    assert [r['enabled'] for r in window.state.regions] == [False, False, True, False]
    assert len(window.state.region_undo) == 1
    rect = window.candidates.visualItemRect(window.candidates.item(2))
    point = rect.topLeft()+QPoint(17, 17) if mode == 'grid' else QPoint(rect.left()+12, rect.center().y())
    QTest.mouseClick(window.candidates.viewport(), Qt.MouseButton.LeftButton,
                     pos=point)
    assert [r['enabled'] for r in window.state.regions] == [False, False, False, False]
    assert len(window.state.region_undo) == 2


def test_space_sets_selection_uniformly_from_current_checkbox(app, window):
    regions, thumbs = fixture_regions(4)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.show()
    app.processEvents()
    click_candidate(window, 0)
    click_candidate(window, 1, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window.candidates, Qt.Key.Key_Space)
    assert [r['enabled'] for r in window.state.regions] == [True, True, True, False]
    assert len(window.state.region_undo) == 1
    QTest.keyClick(window.candidates, Qt.Key.Key_Space)
    assert [r['enabled'] for r in window.state.regions] == [False, False, True, False]
    assert len(window.state.region_undo) == 2


def test_bulk_actions_do_not_include_hidden_background_candidates(app, window):
    regions, thumbs = fixture_regions(3)
    regions[1].update(context_keep=False, enabled=False)
    window.state.regions = regions
    window.show_background.setChecked(True)
    window.refresh_candidates(thumbs)
    window.candidates.selectAll()
    assert selected_ids(window) == {'face-0', 'face-1', 'face-2'}
    window.show_background.setChecked(False)
    assert selected_ids(window) == {'face-0', 'face-2'}
    window.disable_selected_button.click()
    window.enable_selected_button.click()
    assert window.state.regions[1]['enabled'] is False
    assert window.candidates.count() == 2


def test_bulk_controls_and_space_are_disabled_during_foreground_job(app, window):
    regions, thumbs = fixture_regions(3)
    window.state.regions = regions
    window.refresh_candidates(thumbs)
    window.candidates.selectAll()
    release = threading.Event()
    window.run_job(lambda progress, cancel: release.wait(2), lambda _: None, 'busy')
    try:
        assert not window.enable_selected_button.isEnabled()
        assert not window.disable_selected_button.isEnabled()
        window.disable_selected_button.click()
        QTest.keyClick(window.candidates, Qt.Key.Key_Space)
        assert [r['enabled'] for r in window.state.regions] == [True, False, True]
        assert not window.state.region_undo
    finally:
        release.set()
        pump(app, lambda: not window.jobs.active_count)
