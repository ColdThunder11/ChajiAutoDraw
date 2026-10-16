"""imaging 模块单元测试。"""

import numpy as np
from PIL import Image

from chaji_autodraw.core.imaging import (
    BinarizeParams,
    binarize,
    load_grayscale,
    mask_to_preview,
)


def _make_image(
    w: int = 20,
    h: int = 20,
    box: tuple[int, int, int, int] = (5, 5, 15, 15),
) -> Image.Image:
    """构造白底黑方块灰度图（box 为 (left, top, right, bottom)，右/下开区间）。"""
    arr = np.full((h, w), 255, dtype=np.uint8)
    left, top, right, bottom = box
    arr[top:bottom, left:right] = 0
    return Image.fromarray(arr, mode="L")


def test_binarize_black_square_is_true() -> None:
    """默认参数：黑方块 mask 为 True，白底为 False。"""
    img = _make_image()
    mask, offset = binarize(img, BinarizeParams())

    assert offset == (0, 0)
    assert mask.dtype == np.bool_
    assert mask.shape == (20, 20)
    assert mask[5:15, 5:15].all()  # 黑方块全部需要绘制
    assert not mask[:5, :].any()
    assert not mask[15:, :].any()
    assert not mask[:, :5].any()
    assert not mask[:, 15:].any()


def test_binarize_invert_negates() -> None:
    """invert=True 时结果取反（绘制浅色像素）。"""
    img = _make_image()
    normal, offset_a = binarize(img, BinarizeParams())
    inverted, offset_b = binarize(img, BinarizeParams(invert=True))

    assert offset_a == offset_b == (0, 0)
    assert np.array_equal(inverted, ~normal)
    assert not inverted[5:15, 5:15].any()  # 黑方块反而不画
    assert inverted[0, 0]  # 白底要画


def test_binarize_target_letterbox() -> None:
    """非正方形图缩放到 100×100：形状正确、内容居中、letterbox 全 False。"""
    img = _make_image(w=30, h=20, box=(10, 5, 20, 15))
    target_w, target_h = 100, 100
    mask, (offset_x, offset_y) = binarize(
        img, BinarizeParams(), target_w=target_w, target_h=target_h
    )

    assert mask.shape == (target_h, target_w)

    # 复算等比缩放结果：min(100/30, 100/20) → (100, 67)
    scale = min(target_w / 30, target_h / 20)
    new_w = min(target_w, max(1, round(30 * scale)))
    new_h = min(target_h, max(1, round(20 * scale)))
    assert (new_w, new_h) == (100, 67)
    assert (offset_x, offset_y) == ((target_w - new_w) // 2, (target_h - new_h) // 2)

    # letterbox 区域必须全为 False
    assert not mask[:offset_y, :].any()
    assert not mask[offset_y + new_h :, :].any()
    assert not mask[:, :offset_x].any()
    assert not mask[:, offset_x + new_w :].any()

    # 内容区域仍保留了黑方块
    content = mask[offset_y : offset_y + new_h, offset_x : offset_x + new_w]
    assert content.any()


def test_binarize_requires_both_targets() -> None:
    """只给 target_w 或 target_h 之一应报错。"""
    import pytest

    img = _make_image(w=8, h=8)
    with pytest.raises(ValueError):
        binarize(img, BinarizeParams(), target_w=32)
    with pytest.raises(ValueError):
        binarize(img, BinarizeParams(), target_h=32)


def test_load_grayscale(tmp_path) -> None:
    """load_grayscale 读取任意图片并转为 L 模式。"""
    src = _make_image(w=8, h=8).convert("RGB")
    path = tmp_path / "img.png"
    src.save(path)

    loaded = load_grayscale(str(path))
    assert loaded.mode == "L"
    assert loaded.size == (8, 8)


def test_mask_to_preview_rgb() -> None:
    """预览图为 RGB 模式、尺寸与 mask 一致，黑点/白底颜色正确。"""
    mask = np.zeros((6, 10), dtype=bool)
    mask[2:4, 3:6] = True

    preview = mask_to_preview(mask)

    assert preview.mode == "RGB"
    assert preview.size == (10, 6)  # (宽, 高)
    arr = np.asarray(preview)
    assert arr.shape == (6, 10, 3)
    assert (arr[mask] == 0).all()  # 绘制像素为黑
    assert (arr[~mask] == 255).all()  # 背景为白


def test_extract_line_mask_finds_dark_line_on_white() -> None:
    """白底黑竖线：DoG 应在线条位置检出，远离线条处为 False。"""
    from PIL import Image, ImageDraw

    from chaji_autodraw.core.imaging import extract_line_mask

    img = Image.new("L", (100, 100), 255)
    d = ImageDraw.Draw(img)
    d.line([(50, 10), (50, 90)], fill=0, width=2)
    mask = extract_line_mask(img, sensitivity=10.0)
    assert mask[30:70, 50].mean() > 0.8  # 线条中心基本全部检出
    assert not mask[30:70, 10].any()  # 远离线条的白区不检出


def test_binarize_line_mode_shapes_and_content() -> None:
    """线条模式：按目标尺寸 letterbox，且线条被检出。"""
    from PIL import Image, ImageDraw

    from chaji_autodraw.core.imaging import BinarizeParams, binarize

    img = Image.new("L", (100, 100), 255)
    d = ImageDraw.Draw(img)
    d.line([(20, 10), (20, 90)], fill=0, width=2)
    params = BinarizeParams(line_mode=True, line_threshold=10.0)
    mask, offset = binarize(img, params, target_w=50, target_h=50)
    assert mask.shape == (50, 50)
    assert offset == (0, 0)  # 100x100 → 50x50 等比，无 letterbox
    assert mask[:, 10].mean() > 0.5  # 线条保留


def test_binarize_params_line_mode_defaults_off() -> None:
    """默认关闭线条模式，旧参数行为不变。"""
    from chaji_autodraw.core.imaging import BinarizeParams

    params = BinarizeParams()
    assert params.line_mode is False
    assert params.line_threshold == 12.0


def test_is_colorful_detects_colored_vs_grayscale() -> None:
    """大面积彩色图 → True；灰度图 → False。"""
    from PIL import Image

    from chaji_autodraw.core.imaging import is_colorful

    color_img = Image.new("RGB", (200, 200))
    for y in range(200):
        for x in range(100):
            color_img.putpixel((x, y), (200, 30, 30))    # 左半红色
        for x in range(100, 200):
            color_img.putpixel((x, y), (30, 60, 200))    # 右半蓝色
    assert is_colorful(color_img) is True

    gray_img = Image.new("RGB", (200, 200), (128, 128, 128))
    assert is_colorful(gray_img) is False

    # 黑白漫画式：黑色线条 + 白底 + 灰色网点
    manga = Image.new("RGB", (200, 200), (255, 255, 255))
    for x in range(0, 200, 4):
        for y in range(200):
            manga.putpixel((x, y), (0, 0, 0))
    assert is_colorful(manga) is False


def test_is_colorful_small_colored_region_fails_threshold() -> None:
    """只有零星彩色像素（占比 < 5%）不判定为大面积彩色。"""
    from PIL import Image

    from chaji_autodraw.core.imaging import is_colorful

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    for k in range(100):  # 100/40000 = 0.25% 彩色像素
        img.putpixel((k, 0), (255, 0, 0))
    assert is_colorful(img) is False
