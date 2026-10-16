"""画布自动定位：在窗口截图中识别白色圆角画布并校验其位置先验。"""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
from PySide6.QtGui import QImage, QPixmap

#: 画布纯白判定下限。页面背景约 (245,245,245)、窗口内缘高亮约 251，
#: 画布为纯白 255，取 252 可把背景与内缘全部排除。
_WHITE_MIN: int = 252
#: 画布行的白色占比范围：画布宽约占窗口 74%，底部整行贯通纯白的
#: 工具栏/设置面板占比接近 1，空白背景行接近 0，均被排除
_ROW_FRAC_MIN: float = 0.20
_ROW_FRAC_MAX: float = 0.92
#: 画布列需贯穿行带高度的比例下限
_COL_FRAC_MIN: float = 0.80
#: 行带最少行数与画布最小宽高，低于此视为未找到
_MIN_BAND_ROWS: int = 20
_MIN_CANVAS_SIZE: int = 40
#: 检测结果向内收缩的像素数（避开圆角与柔和描边）
_CANVAS_INSET: int = 2


def detect_white_canvas(arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """在窗口截图 (h, w, 3) 中识别白色圆角画布。

    Args:
        arr: RGB 数组，形状 (h, w, 3)。

    Returns:
        画布矩形 (x0, y0, x1, y1) 闭区间坐标（相对截图）；找不到返回 None。
    """
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    white = (arr[:, :, :3] >= _WHITE_MIN).all(axis=2)
    row_frac = white.mean(axis=1)
    rows = (row_frac >= _ROW_FRAC_MIN) & (row_frac <= _ROW_FRAC_MAX)
    run = best_len = best_start = 0
    for i, ok in enumerate(rows):
        run = run + 1 if ok else 0
        if run > best_len:
            best_len, best_start = run, i - run + 1
    if best_len < _MIN_BAND_ROWS:
        return None
    top, bottom = best_start, best_start + best_len - 1
    col_frac = white[top : bottom + 1].mean(axis=0)
    cols = np.where(col_frac >= _COL_FRAC_MIN)[0]
    if cols.size == 0:
        return None
    left, right = int(cols.min()), int(cols.max())
    if (right - left + 1) < _MIN_CANVAS_SIZE or best_len < _MIN_CANVAS_SIZE:
        return None
    return (
        left + _CANVAS_INSET,
        top + _CANVAS_INSET,
        right - _CANVAS_INSET,
        bottom - _CANVAS_INSET,
    )


def pil_to_pixmap(img: PILImage.Image) -> QPixmap:
    """把 PIL 图片转换为 QPixmap（深拷贝像素缓冲，避免悬垂引用）。"""
    rgba = img.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(
        data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888
    )
    return QPixmap.fromImage(qimage.copy())


def validate_canvas_position(
    rect: tuple[int, int, int, int],
    window_size: tuple[int, int],
) -> bool:
    """校验检测到的画布是否符合「居中且占比 50% 以上」的先验位置。

    以参考截图标定：画布约占窗口宽 73%、高 62%，中心基本在窗口正中。
    同时满足「宽占比 ≥50%、高占比 ≥50%、中心位于窗口中部一半区域」
    才认为定位成功。

    Args:
        rect: 画布 (x0, y0, x1, y1)（窗口相对，闭区间）。
        window_size: 窗口尺寸 (w, h)。

    Returns:
        True 表示定位可信。
    """
    w, h = window_size
    x0, y0, x1, y1 = rect
    cw = x1 - x0 + 1
    ch = y1 - y0 + 1
    if w <= 0 or h <= 0 or cw <= 0 or ch <= 0:
        return False
    if cw < 0.5 * w or ch < 0.5 * h:
        return False
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    return 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75
