from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import math
import time
import uuid

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QHBoxLayout, QSlider, QComboBox, QSpinBox, QMessageBox, QCheckBox)

from .base import BaseWindow, button
from .state import DocumentState


class VideoWindow(BaseWindow):
    def __init__(self):
        super().__init__('video', '全景视频 · 指定人物遮挡')
        self.frame = 0
        self.displayed_frame = None
        self._play_started = None
        self._play_start_frame = 0
        self._frame_rgb = None
        self._frame_rgb_index = None
        self._tracks = {}
        self._analysis_current = False
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(15)
        self.play_timer.timeout.connect(self.play_tick)
        self.follow_target = QCheckBox('播放时跟随人物')
        self.follow_target.setChecked(True)
        self.follow_target.setToolTip('播放时让已跟踪的人脸留在画面中央，保留当前缩放。拖动画面会关闭跟随；暂停时可自由查看。')
        self.view_controls.insertWidget(self.view_controls.count()-1, self.follow_target)
        self._follow_orientation = self._view_orientation()
        self.viewport.view_changed.connect(self._preview_view_changed)
        self.left_layout.addWidget(QLabel('指定同一位人物'))
        self.open_button = button('打开全景视频', self.open_video, self.left_layout)
        help_text = QLabel('先找到清楚的一帧，再框选整张脸。\n跟踪后检查提示片段，必要时在该帧重新框选。')
        help_text.setWordWrap(True)
        self.left_layout.addWidget(help_text)
        self.analyze_button = button('跟踪选中人物', self.analyze, self.left_layout)
        self.left_layout.addWidget(QLabel('选择 / 修正锚点'))
        self.anchors = QListWidget()
        self.anchors.itemClicked.connect(self.anchor_selected)
        self.left_layout.addWidget(self.anchors, 1)
        self.delete_anchor_button = button('移除选中锚点', self.delete_anchor, self.left_layout)
        self.right_layout.addWidget(QLabel('需要复查的片段'))
        self.warning_summary = QLabel('跟踪完成后列出不确定或未找到人物的片段。')
        self.warning_summary.setWordWrap(True)
        self.right_layout.addWidget(self.warning_summary)
        self.warnings = QListWidget()
        self.warnings.setWordWrap(True)
        self.warnings.itemClicked.connect(self.warning_selected)
        self.right_layout.addWidget(self.warnings, 1)
        self.export_button = button('导出完整视频', self.export_current, self.right_layout)
        timeline = QHBoxLayout()
        self.play_button = button('播放', self.toggle_play, timeline)
        self.previous_button = button('上一帧', lambda: self.step_frame(-1), timeline)
        self.next_button = button('下一帧', lambda: self.step_frame(1), timeline)
        self.rate = QComboBox()
        for label, value in [('0.25 倍', .25), ('0.5 倍', .5), ('1 倍', 1.), ('2 倍', 2.)]:
            self.rate.addItem(label, value)
        self.rate.setCurrentIndex(2)
        self.rate.currentIndexChanged.connect(self.rate_changed)
        timeline.addWidget(self.rate)
        self.frame_spin = QSpinBox()
        self.frame_spin.setPrefix('帧 ')
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self.seek)
        timeline.addWidget(self.frame_spin)
        self.frame_label = QLabel('尚无视频')
        self.frame_label.setObjectName('frameReadout')
        timeline.addWidget(self.frame_label, 1)
        self.middle_layout.addLayout(timeline)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.sliderPressed.connect(self.pause)
        self.timeline.valueChanged.connect(self.seek)
        self.middle_layout.addWidget(self.timeline)
        self.viewport.region_drawn.connect(self.add_seed)
        self.viewport.selection_started.connect(self.freeze_selection)
        self._busy(False)

    def set_media_busy(self, busy):
        for name in ('open_button', 'analyze_button', 'delete_anchor_button', 'export_button'):
            widget = getattr(self, name, None)
            if widget:
                widget.setEnabled(not busy and (name == 'open_button' or self.state.asset is not None))
        if hasattr(self, 'anchors'):
            self.anchors.setEnabled(not busy)

    def toggle_draw(self):
        self.pause()
        super().toggle_draw()

    def freeze_selection(self):
        self.pause()
        self.preview_timer.stop()
        self.previews.cancel()
        self._pending_preview = None

    def fps(self):
        if not self.state.asset or 'fps_num' not in self.state.asset:
            return 1.
        return float(Fraction(self.state.asset['fps_num'], self.state.asset['fps_den']))

    def open_video(self):
        if not self.confirm_discard():
            return
        source, _ = QFileDialog.getOpenFileName(self, '打开全景视频', '',
            '全景 MP4 视频 (*.mp4);;所有文件 (*)')
        if source:
            self.load_source(source)

    def load_source(self, source, state=None, explicit_relink=False):
        self.pause()
        self.frame = 0
        self.displayed_frame = None
        self._frame_rgb = self._frame_rgb_index = None
        self._tracks = {}
        self._analysis_current = False
        self.switch_state(state or DocumentState('video'))
        expected = deepcopy(self.state.asset)
        def prepare(progress, cancel):
            from panoprivacy.common import app_cache_dir, fingerprint, check_cancel
            from panoprivacy.video_engine import prepare_video
            check_cancel(cancel)
            current = fingerprint(source)
            if expected and expected.get('fingerprint') and not explicit_relink:
                if current != expected['fingerprint']:
                    raise ValueError('源文件已改变。请重新打开原片，或重新关联原来的同一份素材。')
            reusable = (expected and not explicit_relink
                and current == expected.get('fingerprint')
                and Path(source).resolve() == Path(expected['source']).resolve()
                and all(Path(expected.get(key, '')).is_file() for key in ('proxy_path','preview_path'))
                and all(key in expected for key in ('width','height','fps_num','fps_den','frame_count')))
            if reusable:
                progress({'fraction':1.,'message':'已复用原视频的预览缓存，无需重新转码。'})
                asset = expected
            else:
                asset = prepare_video(source, str(Path(app_cache_dir())/'videos'/str(uuid.uuid4())), progress, cancel)
            if expected:
                for key in ('width', 'height', 'fps_num', 'fps_den', 'frame_count'):
                    if key in expected and asset[key] != expected[key]:
                        raise ValueError('重新关联的视频尺寸、帧率或帧数不同。请选择原来的视频。')
            return asset
        def prepared(asset):
            self.state.asset = asset
            self.timeline.setRange(0, asset['frame_count']-1)
            self.frame_spin.setRange(0, asset['frame_count']-1)
            self.source_label.setText(f"{source}\n{asset['width']} × {asset['height']} · {self.fps():.3f} 帧/秒")
            self._set_analysis(self.state.analysis)
            self.refresh_anchors()
            self._busy(False)
            self.seek(0)
            self.request_preview()
            self.status.setText('视频已打开。找到清楚的一帧，使用“手动框选”指定人物。')
        self.run_job(prepare, prepared, '正在验证视频并生成预览…')

    def seek(self, frame):
        if not self.state.asset or 'frame_count' not in self.state.asset:
            return
        self.frame = max(0, min(self.state.asset['frame_count']-1, int(frame)))
        for widget in (self.timeline, self.frame_spin):
            widget.blockSignals(True)
            widget.setValue(self.frame)
            widget.blockSignals(False)
        self.preview_timer.start()

    def step_frame(self, direction):
        self.pause()
        self.seek(self.frame+direction)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.pause()
        elif self.state.asset and 'frame_count' in self.state.asset:
            if self.frame >= self.state.asset['frame_count']-1:
                self.seek(0)
            self._play_started = time.monotonic()
            self._play_start_frame = self.frame
            self.play_timer.start()
            self.play_button.setText('暂停')

    def pause(self):
        self.play_timer.stop()
        if hasattr(self, 'play_button'):
            self.play_button.setText('播放')

    def rate_changed(self):
        if self.play_timer.isActive():
            self._play_started = time.monotonic()
            self._play_start_frame = self.frame

    def play_tick(self):
        if not self.state.asset:
            return
        frame = self._play_start_frame+int((time.monotonic()-self._play_started)*self.fps()*self.rate.currentData())
        if frame != self.frame:
            self.seek(frame)
            # Playback coalesces decode requests; slow frames do not reset a debounce forever.
            self.request_preview()
        if frame >= self.state.asset['frame_count']-1:
            self.pause()

    def regions_at(self, frame):
        seeds = [s['region'] for s in self.state.seeds if s['frame'] == frame]
        if seeds:
            return seeds
        return self._tracks.get(frame, {}).get('regions', [])

    def _view_orientation(self):
        return tuple(self.viewport.pose.get(key, 0.) for key in ('yaw', 'pitch', 'roll'))

    def _preview_view_changed(self):
        orientation = self._view_orientation()
        if self.play_timer.isActive() and orientation != self._follow_orientation:
            self.follow_target.setChecked(False)
        self._follow_orientation = orientation

    def _preview_pose(self, frame, regions, size):
        pose = deepcopy(self.viewport.pose)
        if self.play_timer.isActive() and self.follow_target.isChecked():
            enabled = [region for region in regions if region.get('enabled', True)]
            seeded = any(seed['frame'] == frame for seed in self.state.seeds)
            status = self._tracks.get(frame, {}).get('status')
            tracked = self._analysis_current and status in ('ok', 'uncertain')
            if len(enabled) == 1 and (seeded or tracked):
                # Transport the previous camera up vector across the sphere so
                # following a target through the nadir does not flip the view.
                from panoprivacy.video_tracking import _recenter
                try:
                    centered = _recenter(enabled[0], pose, size)
                    if centered is pose:
                        # A jump exactly along the old up vector has no unique
                        # transported up direction. Use the target's recorded
                        # view basis for that first frame, retaining zoom.
                        from panoprivacy.geometry import region_center
                        yaw, pitch = region_center(enabled[0])
                        reference = dict(pose, yaw=yaw, pitch=pitch, roll=0.)
                        if enabled[0].get('space') == 'view':
                            reference.update(enabled[0].get('pose') or {})
                        reference['fov'] = pose['fov']
                        centered = _recenter(enabled[0], reference, size)
                    if all(math.isfinite(value) for value in centered.values()):
                        pose = centered
                        self.viewport.pose.update(pose)
                except (ValueError, KeyError, TypeError):
                    pass  # An unusable target must never redirect the view to another person.
        self._follow_orientation = self._view_orientation()
        return pose

    def request_preview(self):
        if not self.state.asset or 'frame_count' not in self.state.asset:
            return
        asset, frame = deepcopy(self.state.asset), self.frame
        regions, style = deepcopy(self.regions_at(frame)), deepcopy(self.state.style)
        size = (self.viewport.width(), self.viewport.height())
        pose = self._preview_pose(frame, regions, size)
        mask = self.mask_preview.isChecked()
        cached = self._frame_rgb if self._frame_rgb_index == frame else None
        def render(progress, cancel):
            from panoprivacy.video_engine import video_frame
            from panoprivacy.geometry import render_view, apply_regions_rgb
            rgb = cached if cached is not None else video_frame(asset, frame)
            edited = apply_regions_rgb(rgb, regions, style) if mask else rgb
            return {'rgb': render_view(edited, pose, size), 'frame': frame,
                    'erp': rgb, 'regions': regions, 'pose': pose}
        self.enqueue_preview(render)

    def _show_preview(self, result):
        self._frame_rgb, self._frame_rgb_index = result['erp'], result['frame']
        self.viewport.regions = result['regions']
        super()._show_preview(result)

    def preview_frame_displayed(self, frame):
        self.displayed_frame = frame
        status = self._tracks.get(frame, {}).get('status')
        note = {'ok': '已跟踪', 'uncertain': '需要复查', 'missing': '未找到人物'}.get(status, '尚未跟踪')
        self.frame_label.setText(f'显示帧 {frame} · {frame/self.fps():.3f} 秒 · {note}')

    def add_seed(self, region):
        self.pause()
        if self.jobs.running or self.displayed_frame is None:
            return
        # Geometry belongs to the frame the user actually saw, even if a seek is decoding.
        frame = self.displayed_frame
        self.seek(frame)
        self.state.seeds = [s for s in self.state.seeds if s['frame'] != frame]
        self.state.seeds.append({'frame': frame, 'region': region})
        self.state.seeds.sort(key=lambda seed: seed['frame'])
        self.state.mark_edited()
        self._analysis_current = False
        if self.state.analysis is not None:
            changed = set(self.state.analysis.get('_ui_changed_frames', []))
            changed.add(frame)
            self.state.analysis['_ui_changed_frames'] = sorted(changed)
            self.state.analysis['_ui_stale'] = True
        self.refresh_anchors()
        self.request_preview()
        self.status.setText(f'已记录第 {frame} 帧的选择。请点击“跟踪选中人物”更新结果。')

    def refresh_anchors(self):
        self.anchors.clear()
        for seed in self.state.seeds:
            item = QListWidgetItem(f"第 {seed['frame']} 帧 · {seed['frame']/self.fps():.3f} 秒")
            item.setData(Qt.ItemDataRole.UserRole, seed['frame'])
            self.anchors.addItem(item)

    def anchor_selected(self, item):
        self.pause()
        frame = item.data(Qt.ItemDataRole.UserRole)
        self.seek(frame)
        for seed in self.state.seeds:
            if seed['frame'] == frame:
                region = seed['region']
                if region['space'] == 'view':
                    from panoprivacy.video_tracking import _recenter
                    self.viewport.pose.update(_recenter(region, deepcopy(region['pose']), region['view_size']))
                else:
                    self.viewport.pose['roll'] = 0.
                self.viewport.focus_region(region)

    def delete_anchor(self):
        item = self.anchors.currentItem()
        if item:
            frame = item.data(Qt.ItemDataRole.UserRole)
            self.state.seeds = [s for s in self.state.seeds if s['frame'] != frame]
            self.state.mark_edited()
            self._analysis_current = False
            if self.state.analysis is not None:
                changed = set(self.state.analysis.get('_ui_changed_frames', []))
                changed.add(frame)
                self.state.analysis['_ui_changed_frames'] = sorted(changed)
                self.state.analysis['_ui_stale'] = True
            self.refresh_anchors()
            self.request_preview()

    def analyze(self):
        if not self.state.seeds:
            QMessageBox.information(self, '请先指定人物', '在清楚的一帧使用“手动框选”，框住整张脸。')
            return
        self.pause()
        snapshot = self.state.snapshot()
        changed = (snapshot['analysis'] or {}).get('_ui_changed_frames', [])
        changed_frame = changed[0] if len(changed) == 1 else None
        def analyze(progress, cancel):
            from panoprivacy.video_engine import analyze_video
            return analyze_video(snapshot['asset'], snapshot['seeds'], previous=snapshot['analysis'],
                                 changed_frame=changed_frame, progress=progress, cancel=cancel)
        def analyzed(result):
            self.state.analysis = result
            self.state.mark_edited()
            self._set_analysis(result)
            self.request_preview()
            self.status.setText(f"跟踪完成，有 {self.warnings.count()} 个片段需要复查。")
        self.run_job(analyze, analyzed, '正在跟踪指定人物…')

    def _set_analysis(self, analysis):
        self._tracks = {track['frame']: track for track in (analysis or {}).get('tracks', [])}
        self._analysis_current = analysis is not None and not analysis.get('_ui_stale', False)
        self.warnings.clear()
        warnings = list((analysis or {}).get('warnings', []))
        # Engines may have no segment warnings: derive contiguous ranges from per-frame state.
        if not warnings:
            for track in self._tracks.values():
                if track['status'] != 'ok':
                    if warnings and warnings[-1]['end'] == track['frame']-1:
                        warnings[-1]['end'] = track['frame']
                    else:
                        warnings.append({'start': track['frame'], 'end': track['frame'],
                                         'message': track.get('message') or '人物位置不确定，请复查并修正。'})
        for warning in warnings:
            item = QListWidgetItem(f"帧 {warning['start']}–{warning['end']}\n{warning['message']}")
            item.setData(Qt.ItemDataRole.UserRole, warning['start'])
            self.warnings.addItem(item)
        self.warning_summary.setText(f'共 {len(warnings)} 个片段需要复查。\n点击跳转；画新框添加修正锚点。')

    def warning_selected(self, item):
        self.pause()
        self.seek(item.data(Qt.ItemDataRole.UserRole))

    def export_current(self):
        if not self.state.analysis or not self._analysis_current:
            QMessageBox.information(self, '需要更新跟踪结果', '请先跟踪人物。添加或移除锚点后需要重新跟踪。')
            return
        def usable(track):
            import math
            regions = [region for region in track.get('regions', []) if region.get('enabled', True)]
            if len(regions) != 1 or track.get('status') not in ('ok', 'uncertain'):
                return False
            box = regions[0].get('box', [])
            return (len(box) == 4 and all(math.isfinite(float(value)) for value in box)
                    and box[2] > box[0] and box[3] > box[1])
        missing = sum(not usable(t) for t in self._tracks.values())
        incomplete = len(self._tracks) != self.state.asset['frame_count']
        if missing or incomplete:
            QMessageBox.warning(self, '仍有未找到人物的帧',
                '存在缺失帧或没有有效遮挡区域的帧。请检查提示片段、添加修正锚点并重新跟踪后再导出。')
            return
        uncertain = any(t['status'] == 'uncertain' for t in self._tracks.values())
        reviewed_uncertain = False
        if self.warnings.count() or uncertain:
            answer = QMessageBox.question(self, '确认复查结果',
                '仍有片段标记为需要复查。请先逐段观看遮挡预览，确认始终覆盖同一人物的整张脸。\n\n'
                '确认已经看过并检查这些片段，允许本次导出继续吗？此确认不会清除项目中的风险标记。',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
            reviewed_uncertain = uncertain
        self.pause()
        source = Path(self.state.asset['source'])
        path, _ = QFileDialog.getSaveFileName(self, '导出到新视频', str(source.with_name(source.stem+'_遮挡.mp4')),
                                            'MP4 视频 (*.mp4)')
        if not path:
            return
        snapshot = self.state.snapshot()
        snapshot['analysis'].pop('reviewed_uncertain', None)
        if reviewed_uncertain:
            snapshot['analysis']['reviewed_uncertain'] = True
        def export(progress, cancel):
            from panoprivacy.video_engine import export_video
            return export_video(snapshot['asset'], snapshot['analysis'], snapshot['style'], path, progress, cancel)
        self.run_job(export, self.export_done, '正在导出并验证完整视频…')

    def restore_project(self, project, path, source=None):
        state = DocumentState.from_project(project)
        state.project_path = path
        if source:
            state.mark_edited()
        self.load_source(source or state.asset['source'], state, explicit_relink=source is not None)

    def closeEvent(self, event):
        self.pause()
        super().closeEvent(event)
