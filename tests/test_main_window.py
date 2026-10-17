"""主窗口轻量测试：规划缓存、未框选画布路径、移动点进度标签。

使用 offscreen Qt，不弹窗、不依赖真实窗口/屏幕。
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import QRect

from chaji_autodraw.ui.main_window import MainWindow


def _make_window() -> tuple[object, MainWindow]:
    """创建 QApplication（如无）与带小图/画布状态的 MainWindow。"""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    image = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, 140, 100), outline=(0, 0, 0), width=4)
    window._gray = image.convert("L")
    window._rgb = image
    window._image_serial = 1
    window._canvas_rel = QRect(0, 0, 160, 120)
    return app, window


def _dispose(window: MainWindow) -> None:
    """停掉防抖定时器并关闭窗口，避免测试结束后仍有待触发事件。"""
    window._preview_timer.stop()
    window.close()
    window.deleteLater()


def test_plan_cache_reuses_unchanged_plan() -> None:
    """相同参数状态复用同一规划结果；改阈值后缓存失效并重算。"""
    app, window = _make_window()  # noqa: F841 - 保持 QApplication 存活
    try:
        first, target1 = window._compute_strokes()
        second, target2 = window._compute_strokes()
        assert second is first  # 直接命中缓存（不复制笔画列表）
        assert target2 == target1

        window._threshold_slider.setValue(200)
        changed, _ = window._compute_strokes()
        assert changed is not first
    finally:
        _dispose(window)


def test_preview_without_canvas_skips_plan() -> None:
    """未框选画布时只出二值化掩膜，不应进入完整笔画规划。"""
    app, window = _make_window()  # noqa: F841 - 保持 QApplication 存活
    try:
        window._canvas_rel = None
        calls: list[int] = []

        def fail() -> None:
            calls.append(1)
            raise AssertionError("未框选画布时不应规划笔画")

        window._compute_strokes = fail  # type: ignore[method-assign]
        window._update_binarize_preview()
        assert calls == []
    finally:
        _dispose(window)


def test_connect_gap_default_is_three_px() -> None:
    """「连接距离」控件默认值为 3px。"""
    app, window = _make_window()  # noqa: F841 - 保持 QApplication 存活
    try:
        assert window._connect_gap_spin.value() == 3.0
    finally:
        _dispose(window)


def test_progress_label_uses_moves_and_eta() -> None:
    """进度标签同时显示第几笔、移动点百分比与剩余时间。"""
    app, window = _make_window()  # noqa: F841 - 保持 QApplication 存活
    try:
        window._stroke_current = 3
        window._stroke_total = 10
        window._move_current = 50
        window._move_total = 100
        window._eta_seconds = 12
        window._update_progress_label()
        text = window._progress_label.text()
        assert "第 3/10 笔" in text
        assert "50%" in text
        assert "剩余约 12 秒" in text
    finally:
        _dispose(window)
