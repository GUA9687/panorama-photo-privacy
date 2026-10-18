from copy import deepcopy
from pathlib import Path
import uuid

from PySide6.QtCore import QItemSelectionModel, Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QLabel, QListWidgetItem, QFileDialog,
                              QHBoxLayout, QMessageBox, QComboBox, QCheckBox)

from .base import BaseWindow, button
from .state import DocumentState
from .viewport import rgb_image
from .jobs import JobController
from .photo_widgets import PhotoCandidates, PhotoQueue, path_key, photo_path


class PhotoWindow(BaseWindow):
    def __init__(self):
        super().__init__('photo', '全景照片 · 人脸遮挡 0.2')
        self.state.style.update(shape='ellipse', export_mode='auto', jpeg_quality=98)
        self.splitter.setHandleWidth(8)
        self.splitter.setStyleSheet('QSplitter::handle { width: 8px; background: #cbd4db; } '
                                   'QSplitter::handle:hover { background: #8aaba6; }')
        self.queue_states = []
        self._queue_index = -1
        self._preview_rgb = None
        self._thumbnails = {}
        self._view_revision = 0
        self._display_quality = None
        self.native_jobs = JobController(self)
        self.extra_controllers.append(self.native_jobs)
        self.native_jobs.switch_document(self.state.identity)
        self.native_jobs.succeeded.connect(self.accept_native_preview)
        self.native_jobs.failed.connect(lambda message: self.detail_label.setText('原图高清读取失败：'+message))
        self.native_timer = QTimer(self)
        self.native_timer.setSingleShot(True)
        self.native_timer.setInterval(250)
        self.native_timer.timeout.connect(self.request_native_preview)
        self.detail_label = QLabel('拖动时快速预览；停止后读取原图高清。')
        self.middle_layout.insertWidget(1, self.detail_label)
        self.mask_preview.setText('遮挡预览（取消勾选看原图）')
        self.mask_preview.setToolTip('拖动使用快速预览；停止250毫秒后直接读取原图高清视角。遮挡区域按原始球面坐标投影。')
        self.viewport.selection_started.connect(self.freeze_native_selection)
        self._eviction_candidates = []
        self._eviction_timer = QTimer(self)
        self._eviction_timer.setInterval(250)
        self._eviction_timer.timeout.connect(self.evict_unused_native)
        self.left_layout.addWidget(QLabel('照片队列'))
        self.add_button = button('添加照片', self.add_photos, self.left_layout)
        self.queue = PhotoQueue(lambda: not self.jobs.running)
        self.queue.paths_dropped.connect(self.add_paths)
        self.queue.setWordWrap(True)
        self.queue.currentRowChanged.connect(self.select_photo)
        self.left_layout.addWidget(self.queue, 1)
        self.scan_mode = QComboBox()
        self.scan_mode.addItem('明显人脸（默认）', 'obvious')
        self.scan_mode.addItem('更多候选（需要仔细核对）', 'more')
        self.scan_mode.setToolTip('默认处理显眼的正脸。远处很小的人脸和侧脸可以忽略；更多模式会增加待检查候选。')
        self.left_layout.addWidget(self.scan_mode)
        self.scan_button = button('查找人脸', self.scan, self.left_layout)
        self.next_button = button('下一张', self.next_photo, self.left_layout)
        candidate_controls = QHBoxLayout()
        candidate_controls.addWidget(QLabel('检查候选区域'))
        candidate_controls.addStretch()
        self.candidate_mode = QComboBox()
        self.candidate_mode.addItem('列表', 'list')
        self.candidate_mode.addItem('缩略图网格', 'grid')
        candidate_controls.addWidget(self.candidate_mode)
        self.right_layout.addLayout(candidate_controls)
        self.show_background = QCheckBox('显示疑似背景候选')
        self.show_background.setToolTip('疑似背景仍保存在项目中；显示后可重新勾选。手动区域始终显示。')
        self.show_background.toggled.connect(lambda _: self.refresh_candidates())
        self.right_layout.addWidget(self.show_background)
        self.candidate_summary = QLabel('每个候选都可取消勾选；双击定位。')
        self.candidate_summary.setWordWrap(True)
        self.right_layout.addWidget(self.candidate_summary)
        self.candidates = PhotoCandidates()
        self.candidates.itemChanged.connect(self.candidate_changed)
        self.candidates.itemDoubleClicked.connect(self.focus_candidate)
        self.candidates.itemSelectionChanged.connect(self.candidate_selection_changed)
        self.candidates.space_requested.connect(self.toggle_selected_candidates)
        self.right_layout.addWidget(self.candidates, 1)
        self.candidate_mode.currentIndexChanged.connect(self.candidate_mode_changed)
        batch_row = QHBoxLayout()
        self.enable_selected_button = button('勾选所选', lambda: self.set_selected_enabled(True), batch_row)
        self.disable_selected_button = button('取消所选', lambda: self.set_selected_enabled(False), batch_row)
        self.enable_selected_button.setToolTip('统一启用当前选中候选的遮挡；一次批量修改可一次撤销。')
        self.disable_selected_button.setToolTip('统一取消当前选中候选的遮挡；空格键可切换所选。')
        self.right_layout.addLayout(batch_row)
        row = QHBoxLayout()
        button('定位', lambda: self.focus_candidate(self.candidates.currentItem()), row)
        self.remove_button = button('移除当前', self.remove_candidate, row)
        self.right_layout.addLayout(row)
        self.undo_button = button('撤销区域编辑  Ctrl+Z', self.undo_regions, self.right_layout)
        self.undo_button.setShortcut('Ctrl+Z')
        self.export_button = button('导出当前照片', self.export_current, self.right_layout)
        self._add_photo_style_controls()
        self.viewport.region_drawn.connect(self.add_region)
        self._busy(False)

    def set_media_busy(self, busy):
        for name in ('add_button', 'scan_button', 'next_button', 'remove_button', 'export_button'):
            widget = getattr(self, name, None)
            if widget:
                widget.setEnabled(not busy and (name == 'add_button' or self.state.asset is not None))
        if hasattr(self, 'candidates'):
            self.candidates.setEnabled(not busy)
            self.queue.setEnabled(not busy)
            self.scan_mode.setEnabled(not busy)
            self.candidate_mode.setEnabled(not busy)
            self.show_background.setEnabled(not busy)
            self.update_selection_controls(busy)
        for name in ('shape_combo', 'export_mode', 'jpeg_quality'):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not busy)
        if hasattr(self, 'undo_button'):
            self.undo_button.setEnabled(not busy and bool(self.state.region_undo))

    def add_photos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '添加全景照片', '',
            '全景 JPEG 照片 (*.jpg *.jpeg);;所有文件 (*)')
        if paths:
            self.add_paths(paths)

    def add_paths(self, paths):
        if self.jobs.running:
            return
        known = {path_key(state.asset['source']) for state in self.queue_states if state.asset}
        accepted, skipped = [], 0
        for source in paths:
            path = photo_path(source)
            if path is None or path_key(path) in known:
                skipped += 1
                continue
            known.add(path_key(path))
            accepted.append(path)
        self.queue.blockSignals(True)
        for source in accepted:
            state = DocumentState('photo', asset={'kind': 'photo', 'source': str(Path(source).absolute())})
            state.style.update(shape='ellipse', export_mode='auto', jpeg_quality=98)
            self.queue_states.append(state)
            item = QListWidgetItem(Path(source).name)
            item.setToolTip(str(source))
            self.queue.addItem(item)
        self.queue.blockSignals(False)
        if skipped:
            self.status.setText(f'已添加 {len(accepted)} 张照片；跳过 {skipped} 个重复、不存在或不支持的项目。仅支持本地 JPG / JPEG 文件。')
        if self._queue_index < 0 and self.queue_states:
            self.queue.setCurrentRow(0)

    def select_photo(self, index):
        if index < 0 or index == self._queue_index or self.jobs.running:
            return
        # Each queue item retains its own regions and style, so changing items loses no edits.
        if self._queue_index >= 0:
            self.queue_states[self._queue_index] = self.state
            if self.state.asset and self.state.asset.get('native_path'):
                self._eviction_candidates.append(deepcopy(self.state.asset))
                self._eviction_timer.start()
        self._queue_index = index
        self._preview_rgb = None
        self._thumbnails = {}
        self.switch_state(self.queue_states[index])
        self.candidates.clear()
        self._prepare(self.state.asset['source'])

    def evict_unused_native(self):
        if self.jobs.active_count or self.previews.active_count or self.native_jobs.active_count:
            return
        from panoprivacy.common import app_cache_dir
        cache_root = Path(app_cache_dir()).resolve()
        current = (self.state.asset or {}).get('native_path')
        current_path = Path(current).resolve() if current else None
        keep = []
        for asset in self._eviction_candidates:
            raw = asset.get('native_path')
            if not raw or raw == current:
                continue
            native = Path(raw).resolve()
            # Only this exact engine-generated cache file may be deleted; no directory removal.
            if (native == current_path or not native.is_relative_to(cache_root)
                    or native == Path(asset['source']).resolve()):
                continue
            try:
                native.unlink(missing_ok=True)
            except OSError:
                keep.append(asset)
        self._eviction_candidates = keep
        if not keep:
            self._eviction_timer.stop()

    def confirm_discard(self):
        if not self.state.dirty and not any(state.dirty for state in self.queue_states):
            return True
        return QMessageBox.question(self, '队列中有未保存的修改',
            '继续将离开照片队列。要放弃尚未保存的修改吗？',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def _prepare(self, source, restored=None, explicit_relink=False):
        expected = deepcopy(self.state.asset)
        def prepare(progress, cancel):
            from panoprivacy.common import app_cache_dir
            from panoprivacy.photo_engine import prepare_photo
            from PIL import Image
            import numpy as np
            if expected and expected.get('fingerprint') and not explicit_relink:
                from panoprivacy.common import fingerprint
                if fingerprint(source) != expected['fingerprint']:
                    raise ValueError('源文件已改变。请重新添加照片，或打开项目后重新关联原来的同一份素材。')
            reusable = expected and all(Path(expected.get(k, '')).is_file() for k in ('native_path', 'preview_path'))
            if reusable:
                from panoprivacy.common import fingerprint
                reusable = fingerprint(source) == expected.get('fingerprint')
            asset = expected if reusable else prepare_photo(
                source, str(Path(app_cache_dir())/'photos'/str(uuid.uuid4())), progress, cancel)
            if expected and any(asset[k] != expected[k] for k in ('width', 'height') if k in expected):
                raise ValueError('重新关联的照片尺寸不同。请选择原来的照片，避免区域位置错误。')
            with Image.open(asset['preview_path']) as image:
                rgb = np.asarray(image.convert('RGB')).copy()
            return asset, rgb
        def prepared(result):
            self.state.asset, self._preview_rgb = result
            self.source_label.setText(f"{source}\n{self.state.asset['width']} × {self.state.asset['height']}")
            self.refresh_candidates()
            self.request_preview()
            self._busy(False)
            self.status.setText('照片已打开。查找明显人脸，或手动框选遗漏区域。')
            if self.state.regions:
                self.load_thumbnails()
        self.run_job(prepare, prepared, '正在准备照片预览…')

    def next_photo(self):
        if self._queue_index+1 < len(self.queue_states):
            self.queue.setCurrentRow(self._queue_index+1)
        else:
            self.status.setText('已是队列中的最后一张。可继续添加照片。')

    def scan(self):
        if not self.state.asset:
            return
        asset = deepcopy(self.state.asset)
        mode = self.scan_mode.currentData()
        def scan(progress, cancel):
            from panoprivacy.photo_engine import scan_photo, photo_thumbnail
            from panoprivacy.common import check_cancel
            regions = scan_photo(asset, mode=mode, progress=progress, cancel=cancel)
            thumbnails = {}
            for index, region in enumerate(regions):
                check_cancel(cancel)
                thumbnails[region['id']] = photo_thumbnail(asset, region, size=144)
                progress({'fraction': (index+1)/max(1, len(regions)), 'message': f'准备候选缩略图 {index+1}/{len(regions)}'})
            return regions, thumbnails
        def scanned(result):
            regions, thumbnails = result
            manual = [r for r in self.state.regions if r.get('origin') == 'manual']
            self.state.remember_regions()
            self.state.regions = manual + regions
            self.state.mark_edited()
            self.refresh_candidates(thumbnails)
            self.request_preview()
            self.status.setText(f'找到 {len(regions)} 个候选区域。'+self.context_warning()+'请逐个检查，再导出。')
        self.run_job(scan, scanned, '正在查找明显人脸…')

    def refresh_candidates(self, thumbnails=None):
        self._thumbnails.update(thumbnails or {})
        thumbnails = self._thumbnails
        current = self.candidates.currentItem()
        current_id = current.data(Qt.ItemDataRole.UserRole) if current else None
        selected_ids = {item.data(Qt.ItemDataRole.UserRole) for item in self.candidates.selectedItems()}
        scroll = self.candidates.verticalScrollBar().value()
        self.candidates.blockSignals(True)
        self.candidates.clear()
        for index, region in enumerate(self.state.regions):
            manual = region.get('origin') == 'manual'
            background = region.get('context_keep') is False and not manual
            if background and not self.show_background.isChecked():
                continue
            label = f"{index+1} · {'手动区域' if manual else '人脸候选'}"
            if background:
                label += ' · 疑似背景'
            if region.get('note'):
                label += '\n'+region['note']
            if region.get('context_reason'):
                reason = region['context_reason']
                label += '\n'+{'person_context': '人物环境检查通过',
                    'background_context': '环境检查倾向背景，可手动恢复',
                    'filter_unavailable': '人体环境筛选不可用，请人工确认',
                    'manual': '手动区域'}.get(reason, str(reason))
            item = QListWidgetItem('' if self.candidates.grid else label)
            item.setToolTip(label+'\n勾选决定是否遮挡；双击定位。\nCtrl 增减选择，Shift 连续选择；空格切换所选遮挡。')
            item.setData(Qt.ItemDataRole.AccessibleTextRole, label)
            item.setData(Qt.ItemDataRole.UserRole, region['id'])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if region.get('enabled', True) else Qt.CheckState.Unchecked)
            if region['id'] in thumbnails:
                item.setIcon(QIcon(QPixmap.fromImage(rgb_image(thumbnails[region['id']]))))
            self.candidates.addItem(item)
            if region['id'] == current_id:
                self.candidates.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
            item.setSelected(region['id'] in selected_ids)
        self.candidates.blockSignals(False)
        self.candidates.reflow()
        self.candidates.verticalScrollBar().setValue(scroll)
        self.update_candidate_summary()
        self.update_selection_controls()
        self.viewport.regions = self.state.regions
        self.undo_button.setEnabled(bool(self.state.region_undo) and not self.jobs.running)

    def update_candidate_summary(self):
        active = sum(r.get('enabled', True) for r in self.state.regions)
        background = [r for r in self.state.regions if r.get('context_keep') is False and r.get('origin') != 'manual']
        hidden = len(background) if not self.show_background.isChecked() else 0
        hidden_active = sum(r.get('enabled', True) for r in background) if hidden else 0
        summary = f'共 {len(self.state.regions)} 个区域，启用 {active} 个'
        if hidden:
            summary += f'\n已隐藏 {hidden} 个疑似背景'
            if hidden_active:
                summary += f'（其中 {hidden_active} 个仍启用遮挡）'
        summary += f'\n已选 {len(self.candidates.selectedItems())} 个 · Ctrl/Shift 多选'
        summary += '\n空白处拖框；空格切换所选遮挡；双击定位。'
        warning = self.context_warning()
        if warning:
            summary += '\n'+warning
        self.candidate_summary.setText(summary)

    def context_warning(self):
        count = sum(r.get('context_reason') == 'filter_unavailable' or
                    (r.get('context') or {}).get('status') == 'unavailable'
                    for r in self.state.regions)
        return f'人体环境筛选不可用：{count} 个候选保留待人工确认。' if count else ''

    def candidate_mode_changed(self, *_):
        self.candidates.set_mode(self.candidate_mode.currentData() == 'grid')
        self.refresh_candidates()

    def update_selection_controls(self, busy=None):
        busy = self.jobs.running if busy is None else busy
        enabled = not busy and bool(self.candidates.selectedItems())
        for name in ('enable_selected_button', 'disable_selected_button'):
            control = getattr(self, name, None)
            if control is not None:
                control.setEnabled(enabled)

    def candidate_selection_changed(self):
        self.update_candidate_summary()
        self.update_selection_controls()

    def set_selected_enabled(self, enabled):
        self._set_candidate_checks(self.candidates.selectedItems(), enabled)

    def toggle_selected_candidates(self):
        selected = self.candidates.selectedItems()
        if not selected:
            return
        current = self.candidates.currentItem()
        anchor = current if current is not None and current.isSelected() else selected[0]
        self._set_candidate_checks(selected, anchor.checkState() != Qt.CheckState.Checked)

    def _set_candidate_checks(self, items, enabled):
        if self.jobs.running:
            return
        ids = {item.data(Qt.ItemDataRole.UserRole) for item in items}
        changed = [region for region in self.state.regions
                   if region['id'] in ids and region.get('enabled', True) != enabled]
        if not changed:
            return
        self.state.remember_regions()
        for region in changed:
            region['enabled'] = enabled
        self.state.mark_edited()
        blocked = self.candidates.blockSignals(True)
        try:
            for item in items:
                item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
        finally:
            self.candidates.blockSignals(blocked)
        self.undo_button.setEnabled(True)
        self.update_candidate_summary()
        self.request_preview()

    def _add_photo_style_controls(self):
        row = QHBoxLayout()
        row.addWidget(QLabel('遮挡形状'))
        self.shape_combo = QComboBox()
        for label, value in [('椭圆（默认）', 'ellipse'), ('圆形', 'circle'), ('矩形', 'rect')]:
            self.shape_combo.addItem(label, value)
        self.shape_combo.setToolTip('形状以选定视角为准；请用遮挡预览检查覆盖范围。')
        row.addWidget(self.shape_combo)
        row.addWidget(QLabel('JPEG 导出'))
        self.export_mode = QComboBox()
        self.export_mode.addItem('自动：优先局部处理', 'auto')
        self.export_mode.addItem('整图高质量重新编码', 'jpeg_reencode')
        self.export_mode.setToolTip('局部处理保留未修改区域的 JPEG 系数；格式或资源不支持时采用所选质量重新编码，并说明原因。')
        row.addWidget(self.export_mode)
        row.addWidget(QLabel('重新编码质量'))
        self.jpeg_quality = QComboBox()
        for quality in (95, 98, 100):
            self.jpeg_quality.addItem(str(quality), quality)
        self.jpeg_quality.setCurrentIndex(self.jpeg_quality.findData(98))
        self.jpeg_quality.setToolTip('仅用于整图重新编码及自动回退；质量 100 仍是有损 JPEG。局部处理保留原量化表。')
        row.addWidget(self.jpeg_quality)
        row.addStretch()
        outer = self.centralWidget().layout()
        outer.insertLayout(outer.count()-1, row)
        for control in (self.shape_combo, self.export_mode, self.jpeg_quality):
            control.currentIndexChanged.connect(self.photo_style_changed)

    def photo_style_changed(self, *_):
        self.state.style.update(shape=self.shape_combo.currentData(),
            export_mode=self.export_mode.currentData(), jpeg_quality=self.jpeg_quality.currentData())
        self.state.mark_edited()
        self.preview_timer.start()

    def load_thumbnails(self):
        asset, regions = deepcopy(self.state.asset), deepcopy(self.state.regions)
        missing = [region for region in regions if region['id'] not in self._thumbnails]
        if not missing:
            return
        def load(progress, cancel):
            from panoprivacy.photo_engine import photo_thumbnail
            from panoprivacy.common import check_cancel
            result = {}
            for region in missing:
                check_cancel(cancel)
                result[region['id']] = photo_thumbnail(asset, region, size=144)
            return result
        def loaded(thumbnails):
            self.refresh_candidates(thumbnails)
            self.progress.setValue(100)
            self.status.setText(f'照片已就绪，{self.candidates.count()} 个候选可供检查。'+self.context_warning())
        self.run_job(load, loaded, '正在准备原图区域缩略图…')

    def candidate_changed(self, item):
        enabled = item.checkState() == Qt.CheckState.Checked
        targets = self.candidates.selectedItems() if item.isSelected() else [item]
        self._set_candidate_checks(targets, enabled)

    def focus_candidate(self, item):
        if item:
            for region in self.state.regions:
                if region['id'] == item.data(Qt.ItemDataRole.UserRole):
                    self.viewport.focus_region(region)

    def remove_candidate(self):
        item = self.candidates.currentItem()
        if item:
            self.state.remember_regions()
            self.state.regions = [r for r in self.state.regions if r['id'] != item.data(Qt.ItemDataRole.UserRole)]
            self.state.mark_edited()
            self.refresh_candidates()
            self.request_preview()

    def add_region(self, region):
        if not self.jobs.running and self.state.asset:
            self.state.remember_regions()
            self.state.regions.append(region)
            self.state.mark_edited()
            self.refresh_candidates()
            self.request_preview()
            self.load_thumbnails()

    def undo_regions(self):
        if not self.jobs.running and self.state.undo_regions():
            self.refresh_candidates()
            self.request_preview()
            self.load_thumbnails()
            self.status.setText('已撤销上一步区域编辑。')

    def request_preview(self):
        if self._preview_rgb is None:
            return
        self._view_revision += 1
        revision, identity = self._view_revision, self.state.identity
        self._display_quality = None
        self.detail_label.setText('快速预览 · 停止拖动后读取原图高清')
        self.native_timer.start(250)
        rgb, regions, style = self._preview_rgb, deepcopy(self.state.regions), deepcopy(self.state.style)
        pose, size = deepcopy(self.viewport.pose), (self.viewport.width(), self.viewport.height())
        show_mask = self.mask_preview.isChecked()
        self.viewport.regions = regions
        def render(progress, cancel):
            from panoprivacy.geometry import apply_regions_rgb, render_view
            edited = apply_regions_rgb(rgb, regions, style) if show_mask else rgb
            return {'rgb': render_view(edited, pose, size), 'pose': pose,
                    'revision': revision, 'identity': identity, 'size': size}
        self.enqueue_preview(render)

    def switch_state(self, state):
        self.native_timer.stop()
        self.native_jobs.switch_document(state.identity)
        self._view_revision += 1
        self._display_quality = None
        super().switch_state(state)
        for control, value in ((self.shape_combo, state.style.get('shape', 'rect')),
                               (self.export_mode, state.style.get('export_mode', 'jpeg_reencode')),
                               (self.jpeg_quality, state.style.get('jpeg_quality', 98))):
            control.blockSignals(True)
            control.setCurrentIndex(max(0, control.findData(value)))
            control.blockSignals(False)

    def _matches_view(self, result):
        return (not self._closing and result['identity'] == self.state.identity
                and result['revision'] == self._view_revision
                and result['pose'] == self.viewport.pose
                and result['size'] == (self.viewport.width(), self.viewport.height()))

    def _show_preview(self, result):
        if (self._matches_view(result)
                and self._display_quality != ('native', self._view_revision)):
            self._display_quality = ('proxy', self._view_revision)
            super()._show_preview(result)

    def request_native_preview(self):
        if self._closing or not self.state.asset or self._preview_rgb is None:
            return
        if not Path(self.state.asset.get('native_path', '')).is_file():
            return
        if self.native_jobs.active_count or self.jobs.active_count:
            self.native_timer.start(75)
            return
        asset, pose = deepcopy(self.state.asset), deepcopy(self.viewport.pose)
        regions, style = deepcopy(self.state.regions), deepcopy(self.state.style)
        size = (self.viewport.width(), self.viewport.height())
        revision, identity = self._view_revision, self.state.identity
        show_mask = self.mask_preview.isChecked()
        def render(progress, cancel):
            from panoprivacy.photo_engine import photo_view
            from panoprivacy.common import check_cancel
            rgb = photo_view(asset, pose, out_size=size)
            check_cancel(cancel)
            if show_mask and regions:
                from .view_regions import apply_view_regions
                rgb = apply_view_regions(rgb, regions, style, pose, cancel)
            return {'rgb': rgb, 'pose': pose, 'size': size, 'identity': identity,
                    'revision': revision, 'masked': show_mask}
        self.native_jobs.start(render)

    def accept_native_preview(self, result):
        if self._matches_view(result):
            self._display_quality = ('native', self._view_revision)
            self.detail_label.setText('原图高清 · '+('遮挡预览' if result['masked'] else '原图与区域轮廓'))
            super()._show_preview(result)

    def freeze_native_selection(self):
        self.preview_timer.stop()
        self.native_timer.stop()
        self.previews.cancel()
        self.native_jobs.cancel()
        self._pending_preview = None

    def closeEvent(self, event):
        self.native_timer.stop()
        super().closeEvent(event)
        if not event.isAccepted() and not self._closing:
            self.native_timer.start(250)

    def export_current(self):
        if not self.state.asset:
            return
        if not any(r.get('enabled', True) for r in self.state.regions):
            QMessageBox.information(self, '没有启用的区域', '请先查找人脸或手动框选要遮挡的区域。')
            return
        source = Path(self.state.asset['source'])
        path, _ = QFileDialog.getSaveFileName(self, '导出到新照片', str(source.with_name(source.stem+'_遮挡.jpg')),
                                             'JPEG 照片 (*.jpg)')
        if not path:
            return
        snapshot = self.state.snapshot()
        def export(progress, cancel):
            from panoprivacy.photo_engine import export_photo
            return export_photo(snapshot['asset'], snapshot['regions'], snapshot['style'], path, progress, cancel)
        self.run_job(export, self.export_done, '正在导出完整分辨率照片…')

    def export_done(self, result):
        self.progress.setValue(100)
        output = result.get('output') or result.get('destination') or result.get('path', '')
        method = result.get('export_method', 'jpeg_reencode')
        if method == 'local_jpeg':
            detail = 'JPEG 局部处理：未修改区域保留原系数和量化表。'
        else:
            detail = f"整图高质量重新编码 · JPEG 质量 {result.get('jpeg_quality', 98)}。"
        if result.get('fallback_reason'):
            detail += '\n自动回退原因：'+str(result['fallback_reason'])
        removed = result.get('metadata_removed')
        if removed:
            description = '、'.join(map(str, removed)) if isinstance(removed, (list, tuple)) else str(removed)
            detail += '\n已移除可能含原图的内嵌预览元数据：'+description
        self.status.setText('导出完成，已验证输出。'+detail+'\n'+str(output))
        QMessageBox.information(self, '导出完成',
            detail+'\n请继续抽查人脸覆盖情况。\n'+str(output))

    def restore_project(self, project, path, source=None):
        state = DocumentState.from_project(project)
        state.project_path = path
        if source:
            state.asset['source'] = source
            state.mark_edited()
        self._preview_rgb = None
        self._thumbnails = {}
        self.switch_state(state)
        self.queue.blockSignals(True)
        self.queue.clear()
        self.queue.addItem(Path(state.asset['source']).name)
        self.queue.setCurrentRow(0)
        self.queue.blockSignals(False)
        self.queue_states = [state]
        self._queue_index = 0
        self._prepare(state.asset['source'], restored=state, explicit_relink=source is not None)
