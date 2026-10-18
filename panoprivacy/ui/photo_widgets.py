"""Photo-only queue drop target and compact, responsive candidate thumbnails."""
import os
from pathlib import Path

from PySide6.QtCore import (QEvent, QItemSelectionModel, QModelIndex,
                           QPersistentModelIndex, QRect, QSize, Qt, Signal)
from PySide6.QtGui import QColor, QIcon, QPainter
from PySide6.QtWidgets import (QApplication, QAbstractItemView, QListView,
                              QListWidget, QStyle, QStyledItemDelegate,
                              QStyleOptionButton, QStyleOptionViewItem)


def photo_path(source):
    """Cheap admission check; full image decoding stays in the worker."""
    try:
        path = Path(source).resolve()
        if path.suffix.lower() in {'.jpg', '.jpeg'} and path.is_file():
            return path
    except (OSError, ValueError, TypeError):
        pass
    return None


def path_key(source):
    return os.path.normcase(str(Path(source).resolve()))


class PhotoQueue(QListWidget):
    paths_dropped = Signal(list)

    def __init__(self, can_add, parent=None):
        super().__init__(parent)
        self.can_add = can_add
        self.setAcceptDrops(True)
        self.setDragEnabled(False)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setToolTip('把本地 JPG / JPEG 照片拖到这里；重复文件会自动跳过。')

    def _paths(self, event):
        if not self.isEnabled() or not self.can_add() or not event.mimeData().hasUrls():
            return []
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = photo_path(url.toLocalFile())
                if path is not None:
                    paths.append(str(path))
        return paths

    def dragEnterEvent(self, event):
        if self._paths(event):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = self._paths(event)
        if paths:
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self.paths_dropped.emit(paths)
        else:
            event.ignore()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.count():
            painter = QPainter(self.viewport())
            painter.setPen(self.palette().placeholderText().color())
            painter.drawText(self.viewport().rect().adjusted(12, 12, -12, -12),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             '拖入 JPG / JPEG 照片\n或点击上方“添加照片”')


class ThumbnailDelegate(QStyledItemDelegate):
    """Image-only grid cells with a generously sized checkbox overlay."""
    @staticmethod
    def check_rect(rect):
        return QRect(rect.left()+7, rect.top()+7, 24, 24)

    def paint(self, painter, option, index):
        painter.save()
        area = option.rect.adjusted(3, 3, -3, -3)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QColor('#277e75') if selected else QColor('#bdc9d1'))
        painter.setBrush(QColor('#cfe5e1') if selected else QColor('#edf1f3'))
        painter.drawRoundedRect(area, 4, 4)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        if isinstance(icon, QIcon) and not icon.isNull():
            icon.paint(painter, area.adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignCenter)
        checkbox = QStyleOptionButton()
        checkbox.rect = self.check_rect(option.rect)
        checkbox.state = QStyle.StateFlag.State_Enabled if option.state & QStyle.StateFlag.State_Enabled else QStyle.StateFlag.State_None
        checkbox.state |= (QStyle.StateFlag.State_On
            if index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
            else QStyle.StateFlag.State_Off)
        painter.fillRect(checkbox.rect.adjusted(-2, -2, 2, 2), QColor(247, 250, 251, 235))
        QApplication.style().drawControl(QStyle.ControlElement.CE_CheckBox, checkbox, painter)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not index.flags() & Qt.ItemFlag.ItemIsEnabled:
            return False
        if event.type() == QEvent.Type.MouseButtonRelease:
            if event.button() != Qt.MouseButton.LeftButton or not self.check_rect(option.rect).contains(event.position().toPoint()):
                return False
        elif event.type() == QEvent.Type.KeyPress:
            if event.key() != Qt.Key.Key_Space:
                return False
        else:
            return False
        checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
        return model.setData(index, Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked,
                             Qt.ItemDataRole.CheckStateRole)


class PhotoCandidates(QListWidget):
    space_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = False
        self._list_delegate = QStyledItemDelegate(self)
        self._thumbnail_delegate = ThumbnailDelegate(self)
        self.setDragEnabled(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionRectVisible(True)
        self._check_press = None
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setWordWrap(True)
        self.setSpacing(4)
        self.set_mode(False)

    def _checkbox_at(self, point):
        index = self.indexAt(point)
        if not index.isValid():
            return QModelIndex()
        rect = self.visualRect(index)
        if self.grid:
            check = self._thumbnail_delegate.check_rect(rect)
        else:
            option = QStyleOptionViewItem()
            option.initFrom(self.viewport())
            self._list_delegate.initStyleOption(option, index)
            option.rect = rect
            check = self.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator,
                                                option, self)
        return index if check.contains(point) else QModelIndex()

    def mousePressEvent(self, event):
        self._check_press = None
        index = self._checkbox_at(event.position().toPoint())
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton and index.isValid():
            item = self.itemFromIndex(index)
            # Checking an already selected row must not collapse a Windows-style
            # multiple selection before its enabled state is applied to the group.
            if not item.isSelected():
                self.clearSelection()
                item.setSelected(True)
            self.setCurrentItem(item, QItemSelectionModel.SelectionFlag.NoUpdate)
            self._check_press = QPersistentModelIndex(index)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        pressed, self._check_press = self._check_press, None
        if pressed is not None:
            index = self._checkbox_at(event.position().toPoint())
            if self.isEnabled() and pressed.isValid() and QModelIndex(pressed) == index:
                item = self.itemFromIndex(index)
                item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked
                                   else Qt.CheckState.Checked)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            if self.isEnabled():
                self.space_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def set_mode(self, grid):
        self.grid = bool(grid)
        self.setViewMode(QListView.ViewMode.IconMode if grid else QListView.ViewMode.ListMode)
        self.setFlow(QListView.Flow.LeftToRight if grid else QListView.Flow.TopToBottom)
        self.setWrapping(grid)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setItemDelegate(self._thumbnail_delegate if grid else self._list_delegate)
        self.setIconSize(QSize(144, 144) if grid else QSize(100, 90))
        self.reflow()

    def reflow(self):
        if self.grid:
            # A 260px sidebar can already show two columns. Wider panes add
            # columns instead of stretching thumbnails into one oversized row.
            available = max(1, self.viewport().width()-8)
            columns = max(1, available//110)
            cell = max(64, available//columns)
            self.setGridSize(QSize(cell, cell))
            hint = QSize(max(56, cell-8), max(56, cell-8))
        else:
            self.setGridSize(QSize())
            hint = QSize(200, 106)
        blocked = self.blockSignals(True)
        try:
            for index in range(self.count()):
                self.item(index).setSizeHint(hint)
        finally:
            self.blockSignals(blocked)
        self.doItemsLayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()
