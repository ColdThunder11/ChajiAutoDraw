"""独立图片预览窗口：默认 100% 显示，滚轮缩放、拖拽平移查看细节。"""

from __future__ import annotations

from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)

#: 缩放范围：10% ~ 500%
_MIN_ZOOM = 0.1
_MAX_ZOOM = 5.0


class _ZoomableGraphicsView(QGraphicsView):
    """滚轮仅用于以鼠标位置为中心缩放，绝不滚动视图。"""

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._zoom = 1.0

    def apply_zoom(self, factor: float) -> None:
        """在 10%~500% 范围内叠加缩放系数。"""
        new_zoom = self._zoom * factor
        if not (_MIN_ZOOM <= new_zoom <= _MAX_ZOOM):
            return
        self.scale(factor, factor)
        self._zoom = new_zoom

    def reset_zoom(self) -> None:
        """恢复 100%（1:1 像素）并居中显示图片。"""
        self.resetTransform()
        self._zoom = 1.0
        self.centerOn(self.sceneRect().center())

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt 签名
        """滚轮只缩放；刻意不调用父类实现，避免滚轮变成上下滚动。"""
        delta = event.angleDelta().y()
        if delta:
            self.apply_zoom(1.25 if delta > 0 else 0.8)


class PreviewWindow(QWidget):
    """独立图片预览窗口。

    默认 100%（1:1 像素）显示并居中；滚轮以鼠标位置为中心在 10%~500%
    之间缩放（滚轮仅缩放，不用于滚动）；按住左键拖动平移；双击重置
    100%；Esc 关闭。
    """

    def __init__(
        self, title: str, pixmap: QPixmap, parent: QWidget | None = None
    ) -> None:
        """创建预览窗口并显示图片。

        Args:
            title: 窗口标题。
            pixmap: 要显示的图片。
            parent: Qt 父对象（拥有者，窗口本身仍为独立顶层窗口）。
        """
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        screen = QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRect(0, 0, 1280, 720)
        self.resize(min(1000, avail.width() - 60), min(900, avail.height() - 60))

        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._view = _ZoomableGraphicsView(self._scene, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self._view.reset_zoom()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """更新显示的图片并恢复 100% 居中。"""
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._view.reset_zoom()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt 签名
        """双击：重置为 100% 居中。"""
        self._view.reset_zoom()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 签名
        """Esc 关闭窗口。"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)
