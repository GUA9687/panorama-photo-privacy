"""Perspective panorama canvas. Drag to orbit; explicit drawing mode adds a box."""
from copy import deepcopy
import numpy as np
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QImage, QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget


def rgb_image(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width = rgb.shape[:2]
    return QImage(rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888).copy()


class PanoramaViewport(QWidget):
    view_changed = Signal()
    region_drawn = Signal(object)
    selection_started = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pose = {'yaw': 0., 'pitch': 0., 'fov': 90., 'roll': 0.}
        self.draw_mode = False
        self.image = None
        self.display_pose = None
        self.display_size = None
        self.regions = []
        self._start = None
        self._end = None
        self.setMinimumSize(360, 260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

    def set_rgb(self, rgb, pose=None):
        self.image = rgb_image(rgb)
        self.display_pose = deepcopy(pose or self.pose)
        self.display_size = (self.image.width(), self.image.height())
        self.update()

    def region_from_drag(self, start, end):
        from panoprivacy.geometry import view_box_region
        box = [min(start.x(), end.x()), min(start.y(), end.y()),
               max(start.x(), end.x()), max(start.y(), end.y())]
        size = self.display_size if self.image is not None else (self.width(), self.height())
        pose = self.display_pose if self.image is not None else self.pose
        box = [box[0]*size[0]/self.width(), box[1]*size[1]/self.height(),
               box[2]*size[0]/self.width(), box[3]*size[1]/self.height()]
        return view_box_region(box, deepcopy(pose), size,
                               origin='manual', enabled=True, score=1.)

    def focus_region(self, region):
        import math
        from panoprivacy.geometry import region_center, region_to_view_box
        self.pose['yaw'], self.pose['pitch'] = region_center(region)
        self.pose['fov'] = 90.
        box = region_to_view_box(region, self.pose, (1000, 1000))
        if box:
            span = max((box[2]-box[0])/1000,
                       (box[3]-box[1])/1000*self.width()/max(1, self.height()))
            self.pose['fov'] = max(2., min(110., math.degrees(2*math.atan(span*3))))
        else:
            self.pose['fov'] = 65.
        self.view_changed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#131b22'))
        if self.image is None:
            painter.setPen(QColor('#c6d0d8'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             '打开全景文件开始\n拖动查看四周 · 滚轮缩放')
            return
        painter.drawImage(self.rect(), self.image)
        from panoprivacy.geometry import region_to_view_box
        for region in self.regions:
            try:
                size = self.display_size or (self.width(), self.height())
                box = region_to_view_box(region, self.display_pose or self.pose, size)
                if box:
                    box = [box[0]*self.width()/size[0], box[1]*self.height()/size[1],
                           box[2]*self.width()/size[0], box[3]*self.height()/size[1]]
                    painter.setPen(QPen(QColor('#56c4b6') if region.get('enabled', True)
                                        else QColor('#b4bec6'), 2))
                    painter.drawRect(QRectF(box[0], box[1], box[2]-box[0], box[3]-box[1]))
            except (ValueError, KeyError):
                pass
        if self.draw_mode and self._start is not None and self._end is not None:
            painter.setPen(QPen(QColor('#f2cc80'), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self._start, self._end).normalized())
        painter.setPen(QColor('#e0e7eb'))
        painter.drawText(14, self.height()-14,
                         f"方向 {self.pose['yaw']:.0f}°  俯仰 {self.pose['pitch']:.0f}°  视角 {self.pose['fov']:.0f}°")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.image is not None:
            if self.draw_mode:
                self.selection_started.emit()
            self._start = event.position()
            self._end = self._start
            self.setCursor(Qt.CursorShape.CrossCursor if self.draw_mode else Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._start is None:
            return
        current = event.position()
        if not self.draw_mode:
            delta = current - self._end
            self.pose['yaw'] = (self.pose['yaw'] - delta.x()*self.pose['fov']/self.width()+180)%360-180
            self.pose['pitch'] = max(-89., min(89., self.pose['pitch']+delta.y()*self.pose['fov']/self.width()))
            self.view_changed.emit()
        self._end = current
        self.update()

    def mouseReleaseEvent(self, event):
        if self._start is None:
            return
        if self.draw_mode:
            point = QPointF(max(0., min(self.width(), event.position().x())),
                            max(0., min(self.height(), event.position().y())))
            if abs(point.x()-self._start.x()) >= 6 and abs(point.y()-self._start.y()) >= 6:
                self.region_drawn.emit(self.region_from_drag(self._start, point))
        self._start = self._end = None
        self.unsetCursor()
        self.update()

    def wheelEvent(self, event):
        self.pose['fov'] = max(2., min(125., self.pose['fov']-event.angleDelta().y()/120*5))
        self.view_changed.emit()

    def resizeEvent(self, event):
        self.view_changed.emit()
        super().resizeEvent(event)
