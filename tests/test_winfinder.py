"""winfinder 纯函数部分的单元测试（不触达真实窗口）。"""

from __future__ import annotations

from PIL import Image

from chaji_autodraw.core.winfinder import _crop_to_bounds


def _solid(width: int, height: int) -> Image.Image:
    """生成可辨识尺寸的测试图。"""
    return Image.new("RGB", (width, height), (200, 100, 50))


def test_crop_no_border_difference_identity() -> None:
    """DWM 边界与外框一致时不裁剪，尺寸不变。"""
    img = _solid(327, 555)
    rect = (10, 20, 337, 575)
    out = _crop_to_bounds(img, rect, rect)
    assert out.size == (327, 555)


def test_crop_removes_invisible_borders() -> None:
    """外框比 DWM 边界宽（左右下有不可见边框）时按偏移裁掉。"""
    img = _solid(100, 100)
    rect = (0, 0, 100, 100)
    # 可见区域从 (8, 0) 到 (92, 92)：左右各 8px、下 8px 边框
    bounds = (8, 0, 92, 92)
    out = _crop_to_bounds(img, rect, bounds)
    assert out.size == (84, 92)


def test_crop_clamps_out_of_range_offset() -> None:
    """偏移或尺寸越界时钳位到图片范围，不抛异常。"""
    img = _solid(50, 50)
    out = _crop_to_bounds(img, (0, 0, 50, 50), (100, 100, 200, 200))
    assert out.size == (1, 1)  # 钳位后保底 1×1


def test_crop_zero_size_bounds_stays_positive() -> None:
    """边界退化（宽或高为 0）时仍返回至少 1×1 的裁剪结果。"""
    img = _solid(40, 40)
    out = _crop_to_bounds(img, (0, 0, 40, 40), (10, 10, 10, 10))
    assert out.size == (1, 1)
