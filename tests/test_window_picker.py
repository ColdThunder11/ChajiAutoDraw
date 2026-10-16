"""window_picker 自动识别算法的单元测试（合成界面，不依赖显示器）。"""

from __future__ import annotations

import numpy as np

from chaji_autodraw.ui.window_picker import detect_white_canvas

_BG = 245  # 页面背景灰度
_EDGE = 251  # 窗口内缘高亮
_WHITE = 255  # 画布纯白


def _make_window(w: int = 372, h: int = 653) -> np.ndarray:
    """构造仿小程序界面：1px 深色边框、灰背景、纯白圆角画布、
    一列内缘高亮 251、底部整行贯通纯白工具栏。"""
    img = np.full((h, w, 3), _BG, dtype=np.uint8)
    img[:, 0] = 163  # 窗口边框
    img[:, -1] = 163
    img[0, :] = 170
    img[-1, :] = 163
    img[:, 1] = _EDGE  # 内缘高亮列（旧算法误识别来源之一）
    # 底部整行贯通纯白工具栏（旧算法误识别来源之二）
    img[h - 33 : h - 1, 1 : w - 1] = _WHITE
    # 画布：白色圆角矩形（角上留 6px 圆角近似）
    x0, y0, x1, y1 = 46, 111, 325, 524
    img[y0 : y1 + 1, x0 : x1 + 1] = _WHITE
    r = 6
    yy, xx = np.mgrid[y0 : y0 + r, x0 : x0 + r]
    corner = (yy - (y0 + r)) ** 2 + (xx - (x0 + r)) ** 2 > r * r
    img[y0 : y0 + r, x0 : x0 + r][corner] = _BG
    yy, xx = np.mgrid[y1 - r + 1 : y1 + 1, x1 - r + 1 : x1 + 1]
    corner = (yy - (y1 - r)) ** 2 + (xx - (x1 - r)) ** 2 > r * r
    img[y1 - r + 1 : y1 + 1, x1 - r + 1 : x1 + 1][corner] = _BG
    return img


def test_detect_finds_canvas_ignoring_toolbar_and_edge_highlight() -> None:
    """识别结果应为画布内部矩形，排除底部贯通白条与 251 高亮列。"""
    img = _make_window()
    x0, y0, x1, y1 = detect_white_canvas(img)  # type: ignore[arg-type]
    # 画布白色区域 46~325 / 111~524，检测含 2px 内缩；合成图为硬边，
    # 与真实截图的柔和描边（约 3px）不同，故左右比真实结果各宽 1px
    assert (x0, y0, x1, y1) == (48, 113, 323, 522)


def test_detect_returns_none_for_blank_background() -> None:
    """纯灰背景（无画布）时返回 None。"""
    img = np.full((200, 300, 3), _BG, dtype=np.uint8)
    assert detect_white_canvas(img) is None


def test_detect_returns_none_for_malformed_input() -> None:
    """非三通道输入返回 None，不抛异常。"""
    assert detect_white_canvas(np.zeros((10, 10), dtype=np.uint8)) is None


def test_detect_tolerates_canvas_with_content() -> None:
    """画布内有黑色笔画时仍能识别（白色占比仍高于下限）。"""
    img = _make_window()
    img[200:400, 100:250] = 0  # 画布中间涂黑
    x0, y0, x1, y1 = detect_white_canvas(img)  # type: ignore[arg-type]
    assert (x0, y0) == (48, 113)


def test_validate_centered_canvas_passes() -> None:
    """参考位置（宽 73%、高 62%、居中）应通过校验。"""
    from chaji_autodraw.ui.window_picker import validate_canvas_position

    # 真实标定：窗口 372×653，画布 (51,114)-(322,522)
    assert validate_canvas_position((51, 114, 322, 522), (372, 653)) is True


def test_validate_rejects_offcenter_or_small() -> None:
    """偏居角落或占比不足 50% 的都判定失败。"""
    from chaji_autodraw.ui.window_picker import validate_canvas_position

    # 占比不足：宽只占 40%
    assert validate_canvas_position((10, 100, 158, 500), (372, 653)) is False
    # 占比不足：高只占 46%
    assert validate_canvas_position((10, 200, 300, 500), (372, 653)) is False
