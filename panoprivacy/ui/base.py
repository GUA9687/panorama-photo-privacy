from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSlider, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QSplitter)

from .jobs import JobController
from .state import DocumentState
from .viewport import PanoramaViewport


def button(text, action, layout):
    widget = QPushButton(text)
    widget.clicked.connect(action)
    layout.addWidget(widget)
    return widget


class BaseWindow(QMainWindow):
    """Shared project, styles and async task boundary. No media decoding here."""
    def __init__(self, kind, title):
        super().__init__()
        self.kind = kind
        self.state = DocumentState(kind)
        self._result_handler = None
        self._foreground_generation = 0
        self._pending_preview = None
        self._closing = False
        self.setWindowTitle(title)
        self.resize(1420, 920)
        self.setMinimumSize(1080, 700)
        self.jobs = JobController(self)
        self.previews = JobController(self)
        self.extra_controllers = []
        self.jobs.succeeded.connect(self._job_result)
        self.jobs.failed.connect(self.show_error)
        self.jobs.progress.connect(self._progress)
        self.jobs.busy_changed.connect(self._busy)
        self.previews.succeeded.connect(self._show_preview)
        self.previews.failed.connect(lambda message: self.status.setText('预览失败：'+message))
        self.previews.finished.connect(self._preview_next)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(45)
        self.preview_timer.timeout.connect(self.request_preview)
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 16, 18, 14)
        outer.setSpacing(12)
        top = QHBoxLayout()
        self.heading = QLabel(title)
        self.heading.setObjectName('heading')
        top.addWidget(self.heading)
        top.addStretch()
        button('打开项目', self.open_project, top)
        self.save_button = button('保存项目', self.save_project, top)
        outer.addLayout(top)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(self.splitter, 1)
        self.left = QWidget()
        self.left_layout = QVBoxLayout(self.left)
        self.left_layout.setContentsMargins(0, 0, 10, 0)
        self.left.setMinimumWidth(210)
        self.splitter.addWidget(self.left)
        middle = QWidget()
        self.middle_layout = QVBoxLayout(middle)
        self.middle_layout.setContentsMargins(0, 0, 0, 0)
        self.source_label = QLabel('尚未打开文件')
        self.source_label.setWordWrap(True)
        self.middle_layout.addWidget(self.source_label)
        self.viewport = PanoramaViewport()
        self.middle_layout.addWidget(self.viewport, 1)
        self.viewport.view_changed.connect(lambda: self.preview_timer.start())
        view_controls = QHBoxLayout()
        self.view_controls = view_controls
        self.draw_button = button('手动框选', self.toggle_draw, view_controls)
        self.draw_button.setCheckable(True)
        button('重置视角', self.reset_view, view_controls)
        self.mask_preview = QCheckBox('显示遮挡效果')
        self.mask_preview.setChecked(True)
        self.mask_preview.setToolTip('交互预览使用缩小图；照片候选缩略图读取原图，导出使用完整分辨率。')
        self.mask_preview.toggled.connect(lambda _: self.request_preview())
        view_controls.addWidget(self.mask_preview)
        view_controls.addStretch()
        self.middle_layout.addLayout(view_controls)
        self.splitter.addWidget(middle)
        self.right = QWidget()
        self.right.setMinimumWidth(260)
        self.right_layout = QVBoxLayout(self.right)
        self.right_layout.setContentsMargins(10, 0, 0, 0)
        self.splitter.addWidget(self.right)
        self.splitter.setSizes([235, 900, 285])
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel('遮挡方式'))
        self.style_combo = QComboBox()
        for label, value in [('强高斯模糊', 'blur'), ('马赛克', 'mosaic'), ('纯色遮挡', 'solid')]:
            self.style_combo.addItem(label, value)
        self.style_combo.currentIndexChanged.connect(self.style_changed)
        style_row.addWidget(self.style_combo)
        outer.addLayout(style_row)
        padding_row = QHBoxLayout()
        self.padding_label = QLabel('每侧留白 25%')
        padding_row.addWidget(self.padding_label)
        self.padding = QSlider(Qt.Orientation.Horizontal)
        self.padding.setRange(0, 75)
        self.padding.setValue(25)
        self.padding.valueChanged.connect(self.style_changed)
        padding_row.addWidget(self.padding, 1)
        outer.addLayout(padding_row)
        footer = QHBoxLayout()
        self.status = QLabel('源文件保持原样；完成检查后导出到新文件。')
        self.status.setWordWrap(True)
        footer.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(230)
        self.progress.setRange(0, 100)
        footer.addWidget(self.progress)
        self.cancel_button = button('取消任务', self.cancel_job, footer)
        self.cancel_button.setEnabled(False)
        outer.addLayout(footer)
        self.jobs.switch_document(self.state.identity)
        self.previews.switch_document(self.state.identity)

    def _busy(self, busy):
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(not busy and self.state.asset is not None)
        self.draw_button.setEnabled(not busy and self.state.asset is not None)
        self.set_media_busy(busy)

    def set_media_busy(self, busy):
        pass

    def run_job(self, function, result_handler, message):
        if self.jobs.running:
            return False
        self._foreground_generation += 1
        self._result_handler = result_handler
        self.status.setText(message)
        self.progress.setValue(0)
        self.jobs.start(function)
        return True

    def _job_result(self, result):
        handler, identity = self._result_handler, self.state.identity
        generation = self._foreground_generation
        if handler:
            QTimer.singleShot(0, lambda: handler(result)
                if self.state.identity == identity and generation == self._foreground_generation
                and not self._closing else None)

    def _progress(self, event):
        self.status.setText(event.get('message') or event.get('stage', '正在处理'))
        self.progress.setValue(int(max(0., min(1., event.get('fraction', 0)))*100))

    def cancel_job(self):
        self._foreground_generation += 1
        self.jobs.cancel()
        self.status.setText('已请求取消；正在运行的步骤会尽快退出。')

    def switch_state(self, state):
        self._foreground_generation += 1
        self.state = state
        self.jobs.switch_document(state.identity)
        self.previews.switch_document(state.identity)
        self._pending_preview = None
        self.viewport.image = None
        self.viewport.regions = []
        self.style_combo.blockSignals(True)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(state.style['kind'])))
        self.style_combo.blockSignals(False)
        self.padding.blockSignals(True)
        self.padding.setValue(round(state.style.get('padding', .25)*100))
        self.padding.blockSignals(False)
        self.padding_label.setText(f'每侧留白 {self.padding.value()}%')
        self.source_label.setText(state.asset['source'] if state.asset else '正在打开…')
        self.source_label.setToolTip(self.source_label.text())
        self._busy(False)

    def style_changed(self, *_):
        self.state.style.update(kind=self.style_combo.currentData(), padding=self.padding.value()/100)
        self.state.mark_edited()
        self.padding_label.setText(f'每侧留白 {self.padding.value()}%')
        self.preview_timer.start()

    def toggle_draw(self):
        self.viewport.draw_mode = self.draw_button.isChecked()
        self.status.setText('在画面拖出矩形，包含整张脸和少量头部。' if self.viewport.draw_mode
                            else '拖动画面查看四周；滚轮缩放。')

    def reset_view(self):
        self.viewport.pose.update(yaw=0., pitch=0., fov=90., roll=0.)
        self.request_preview()

    def enqueue_preview(self, function):
        self._pending_preview = function
        self._preview_next()

    def _preview_next(self):
        if self._pending_preview is not None and not self.previews.running:
            function, self._pending_preview = self._pending_preview, None
            self.previews.start(function)

    def _show_preview(self, result):
        self.viewport.set_rgb(result['rgb'], result.get('pose'))
        if 'frame' in result:
            self.preview_frame_displayed(result['frame'])

    def preview_frame_displayed(self, frame):
        pass

    def request_preview(self):
        raise NotImplementedError

    def show_error(self, message):
        self.status.setText('处理失败：'+message)
        QMessageBox.warning(self, '无法完成操作',
                            message+'\n\n请检查源文件、输出目录和可用磁盘空间；源文件不会被替换。')

    def save_project(self):
        if not self.state.asset:
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存项目',
            self.state.project_path or str(Path(self.state.asset['source']).with_suffix('.panoprivacy.json')),
            '遮挡项目 (*.json)')
        if not path:
            return
        snapshot = self.state.snapshot()
        saved_revision = self.state.edit_revision
        def save(progress, cancel):
            from panoprivacy.project import save_project
            save_project(path, **snapshot)
            return path
        def saved(result):
            self.state.project_path = result
            if self.state.edit_revision == saved_revision:
                self.state.dirty = False
                self.status.setText('项目已保存：'+result)
            else:
                self.status.setText('已保存此前版本；当前修改尚未保存：'+result)
        self.run_job(save, saved, '正在保存项目…')

    def confirm_discard(self):
        if not self.state.dirty:
            return True
        return QMessageBox.question(self, '尚有未保存的修改',
            '继续将离开当前文件。要放弃尚未保存的修改吗？',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def open_project(self):
        if self.jobs.running or not self.confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, '打开项目', '', '遮挡项目 (*.json)')
        if not path:
            return
        def load(progress, cancel):
            from panoprivacy.project import load_project
            project = load_project(path)
            if project.get('requires_relink'):
                reason = '原来的源文件已移动或内容已改变，请重新关联同一份素材。'
                return {'relink': project, 'reason': reason}
            return {'project': project}
        self.run_job(load, lambda data: self._opened_project(path, data), '正在打开项目…')

    def _opened_project(self, path, data):
        project = data.get('project') or data.get('relink')
        if project['kind'] != self.kind:
            self.show_error('项目类型不匹配，请使用对应的照片或视频程序打开。')
            return
        if 'relink' in data:
            QMessageBox.information(self, '需要重新关联源文件',
                data['reason']+'\n\n请选择原来的同一份照片或视频；更换内容会使已画区域失效。')
            source, _ = QFileDialog.getOpenFileName(self, '重新关联源文件')
            if not source:
                return
            self.restore_project(project, path, source)
        else:
            self.restore_project(project, path)

    def restore_project(self, project, path, source=None):
        raise NotImplementedError

    def export_done(self, result):
        self.progress.setValue(100)
        output = result.get('destination') or result.get('path') or result.get('output', '')
        self.status.setText('导出完成，已验证输出。'+str(output))
        QMessageBox.information(self, '导出完成', '文件已导出。请继续抽查人脸覆盖情况。\n'+str(output))

    def closeEvent(self, event):
        if not self._closing and not self.confirm_discard():
            event.ignore()
            return
        self.preview_timer.stop()
        self._pending_preview = None
        controllers = [self.jobs, self.previews, *self.extra_controllers]
        if any(controller.active_count for controller in controllers):
            for controller in controllers:
                controller.cancel()
            self._pending_preview = None
            self._closing = True
            self.status.setText('正在取消并等待任务退出，随后关闭窗口…')
            QTimer.singleShot(150, self.close)
            event.ignore()
            return
        self._closing = True
        event.accept()
