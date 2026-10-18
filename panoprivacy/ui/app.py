import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication


STYLESHEET = '''
QWidget { background: #e6eaed; color: #23333f; font-size: 13px; }
QMainWindow { background: #e6eaed; }
QLabel#heading { font-size: 22px; font-weight: 600; }
QLabel#frameReadout { font-family: Consolas; color: #516574; }
QPushButton { background: #f8fafb; border: 1px solid #bbc7cf; border-radius: 5px;
              padding: 8px 13px; min-height: 18px; }
QPushButton:hover { background: #f0f9f8; border-color: #398e86; }
QPushButton:checked { background: #c9e6e1; border-color: #247c73; }
QPushButton:disabled { color: #89969f; background: #dce1e5; }
QPushButton:focus, QComboBox:focus, QListWidget:focus { border: 2px solid #398e86; }
QListWidget { background: #f3f6f8; border: 1px solid #c2ccd3; border-radius: 4px; }
QListWidget::item { padding: 8px 6px; border-bottom: 1px solid #dce3e8; }
QListWidget::item:selected { background: #cfe5e1; color: #173d38; }
QComboBox, QSpinBox { background: #f7f9fb; border: 1px solid #bac7cf;
                     border-radius: 4px; padding: 6px; min-height: 22px; }
QSlider::groove:horizontal { height: 5px; background: #bdc9d1; border-radius: 2px; }
QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #277e75; border-radius: 7px; }
QProgressBar { border: 1px solid #b6c4cd; border-radius: 4px; text-align: center; min-height: 20px; }
QProgressBar::chunk { background: #78b5ac; }
QSplitter::handle { background: #cbd4db; width: 2px; }
QToolTip { background: #f8fafb; color: #23333f; border: 1px solid #aebdc8; padding: 5px; }
'''


def main(mode: str) -> int:
    if mode not in ('photo', 'video'):
        raise ValueError('mode 必须为 photo 或 video')
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('全景人脸遮挡')
    app.setOrganizationName('PanoramaPrivacy')
    app.setFont(QFont('Microsoft YaHei UI', 10))
    app.setStyle('Fusion')
    app.setStyleSheet(STYLESHEET)
    if mode == 'photo':
        from .photo_window import PhotoWindow
        window = PhotoWindow()
    else:
        from .video_window import VideoWindow
        window = VideoWindow()
    window.show()
    return app.exec()
