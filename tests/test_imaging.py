"""imaging 模块单元测试。"""

import numpy as np
from PIL import Image, ImageDraw

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
    assert params.reject_shadow is True  # 阴影剔除默认开启
    assert params.shadow_protection == 40.0  # 默认强度 40


# ---- 线条模式阴影剔除（过滤光影分界线）----

def _line_and_shadow_image(
    base: tuple[int, int, int],
    shadow: tuple[int, int, int],
    line_color: int | tuple[int, int, int] = 20,
) -> tuple[Image.Image, Image.Image]:
    """构造「亮底 + 黑竖线 + 右侧阴影块」测试图。

    Returns:
        (rgb 图, 灰度图)，尺寸 200×140：竖线在 x=30（宽 5），阴影从
        x=110 起铺满右侧（光影分界线在 x≈110）。
    """
    img = Image.new("RGB", (200, 140), base)
    d = ImageDraw.Draw(img)
    d.line([(30, 8), (30, 132)], fill=line_color, width=5)
    d.rectangle([110, 0, 200, 140], fill=shadow)
    return img, img.convert("L")


#: 亮肤色底 → 同色调阴影（明暗落差约 65，低于对比度阈值）
_SKIN = (252, 226, 210)
_SKIN_SHADOW = (196, 158, 142)
#: 强同色调阴影（落差约 85：对比度规则放行，靠色调连续性规则滤除）
_SKIN_STRONG = (172, 138, 128)
#: 棕色头发（与肤色 hue 相近但饱和度约为其 3 倍，属不同材质）
_HAIR = (120, 80, 60)
#: 深海军蓝色块（对白底高落差，边界应保留）
_NAVY = (40, 45, 95)


def _line_region(mask: np.ndarray) -> int:
    """线条（x≈30）附近的检出像素数。"""
    return int(mask[20:120, 22:40].sum())


def _edge_region(mask: np.ndarray) -> int:
    """阴影/色块边界（x≈110）附近的检出像素数。"""
    return int(mask[20:120, 100:130].sum())


def test_line_mode_rejects_same_hue_shadow_edge() -> None:
    """同色调低对比阴影：分界线被 DoG 检出但被阴影剔除滤除，线条保留。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image(_SKIN, _SKIN_SHADOW)
    raw = extract_line_mask(gray, sensitivity=10.0, rgb=rgb, reject_shadow=False)
    assert _edge_region(raw) > 0  # 前提：边界本来会被当线条检出

    mask = extract_line_mask(gray, sensitivity=10.0, rgb=rgb)
    assert _line_region(mask) > 0  # 真实线条保留
    assert _edge_region(mask) == 0  # 光影分界线被滤除


def test_line_mode_rejects_strong_same_hue_shadow() -> None:
    """高落差同色调阴影（落差 85 > 对比度阈值）：靠色调连续性规则滤除。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 228, 214), _SKIN_STRONG)
    mask = extract_line_mask(gray, sensitivity=10.0, rgb=rgb)
    assert _line_region(mask) > 0
    assert _edge_region(mask) == 0


def test_line_mode_keeps_material_boundary_with_similar_hue() -> None:
    """棕发|肤色：hue 相近但饱和度差 3 倍（不同材质），边界保留。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 228, 214), _HAIR)
    mask = extract_line_mask(gray, sensitivity=10.0, rgb=rgb)
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0  # 发际线不是阴影，应保留


def test_line_mode_keeps_dark_region_boundary() -> None:
    """白底+深色块：高落差边界（角色轮廓线）保留。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 255, 255), _NAVY)
    mask = extract_line_mask(gray, sensitivity=10.0, rgb=rgb)
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0


def test_line_mode_without_rgb_keeps_high_contrast_edge() -> None:
    """无 RGB（灰度流程）：高落差边界仍保留，低对比阴影按亮度规则滤除。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    _, gray = _line_and_shadow_image((255, 255, 255), _NAVY)
    mask = extract_line_mask(gray, sensitivity=10.0, rgb=None)
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0

    _, gray2 = _line_and_shadow_image(_SKIN, _SKIN_SHADOW)
    mask2 = extract_line_mask(gray2, sensitivity=10.0, rgb=None)
    assert _edge_region(mask2) == 0  # 灰度下低对比分界同样被滤除


def test_line_mode_shadow_reject_can_be_disabled() -> None:
    """reject_shadow=False：回到旧行为，阴影边界照常检出。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image(_SKIN, _SKIN_SHADOW)
    mask = extract_line_mask(gray, sensitivity=10.0, rgb=rgb, reject_shadow=False)
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0


def test_line_mode_keeps_pale_region_boundary() -> None:
    """浅黄发色|白底：低对比（Δ≈45）但色度不同 → 材质边界保留。

    回归：旧版「低对比即阴影」规则会误删这类浅色材质边界
    （标准强度 50 下保留；默认强度 40 更激进）。
    """
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 255, 255), (230, 215, 150))
    mask = extract_line_mask(
        gray, sensitivity=10.0, rgb=rgb, shadow_protection=50.0
    )
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0


def test_line_mode_keeps_interior_detail_lines() -> None:
    """宽暗区内部的细节线（同色度更暗的发丝线）：几何是线条 → 保留。

    回归：旧版宽暗区判定会把整个中调区域划入阴影带，误删内部细节线。
    """
    from chaji_autodraw.core.imaging import extract_line_mask

    img = Image.new("RGB", (200, 140), (250, 220, 200))  # 外圈亮肤色
    d = ImageDraw.Draw(img)
    d.rectangle([100, 0, 200, 140], fill=(210, 180, 160))  # 中调肤色区域
    for x in range(108, 200, 10):  # 区域内部同色度发丝线
        d.line([(x, 5), (x, 135)], fill=(160, 130, 115), width=2)
    mask = extract_line_mask(img.convert("L"), sensitivity=10.0, rgb=img)
    assert mask[20:120, 102:200].sum() > 0  # 内部发丝线保留
    # 区域边界（x≈100-105，第一条发丝线在 x=108 之前）是同色度阶跃 → 滤除
    assert mask[20:120, 100:106].sum() == 0


def test_line_mode_keeps_dense_hatching() -> None:
    """密集排线（2px 线、6px 间距）：几何/宽度判据都不误删。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    img = Image.new("RGB", (200, 140), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x in range(20, 180, 6):
        d.line([(x, 5), (x, 135)], fill=0, width=2)
    mask = extract_line_mask(img.convert("L"), sensitivity=10.0, rgb=img)
    assert mask[10:130, 18:182].sum() > 0


def test_shadow_protection_strict_keeps_same_hue_shadow() -> None:
    """保护强度 100（最保守）：色度容差极小，同色调阴影不再被滤除。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 228, 214), _SKIN_STRONG)
    mask = extract_line_mask(
        gray, sensitivity=10.0, rgb=rgb, shadow_protection=100.0
    )
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0  # 高保护下同色阴影边界保留


def test_shadow_protection_loose_removes_tinted_shadow() -> None:
    """保护强度 0（最激进）：色度容差放宽，冷色调偏移的阴影也被滤除。

    白底 → (150,175,215) 蓝调阴影：色度距离约 0.055，默认强度下
    判为材质分界（保留），强度 0 下判为阴影（滤除）。
    """
    from chaji_autodraw.core.imaging import extract_line_mask

    rgb, gray = _line_and_shadow_image((255, 255, 255), (150, 175, 215))
    default = extract_line_mask(
        gray, sensitivity=10.0, rgb=rgb, shadow_protection=50.0
    )
    assert _edge_region(default) > 0  # 标准强度 50：色偏明显 → 保留

    loose = extract_line_mask(
        gray, sensitivity=10.0, rgb=rgb, shadow_protection=0.0
    )
    assert _line_region(loose) > 0
    assert _edge_region(loose) == 0  # 激进强度：被滤除


def test_shadow_protection_strict_keeps_gray_shadow_edge() -> None:
    """灰度流同样受保护强度影响：强度 100 时低对比分界不再被滤除。"""
    from chaji_autodraw.core.imaging import extract_line_mask

    _, gray = _line_and_shadow_image(_SKIN, _SKIN_SHADOW)
    mask = extract_line_mask(
        gray, sensitivity=10.0, rgb=None, shadow_protection=100.0
    )
    assert _line_region(mask) > 0
    assert _edge_region(mask) > 0  # Δ≈65 高于最保守的对比度阈值 40


def test_line_mode_full_res_detection_survives_downscale() -> None:
    """线条模式检测在原图分辨率执行：窄阴影带缩放后不再泄漏为线条。

    回归：旧版先缩放图像再做 DoG+阴影剔除，阴影剔除的几何判据绑定
    像素尺度——32px 宽的阴影带缩到 8px 后比结构元窄，边界被当成
    线条画出（全分辨率预览正常、框选画布后多出线条）。
    """
    from chaji_autodraw.core.imaging import BinarizeParams, binarize

    img = Image.new("RGB", (800, 560), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 0, 132, 560], fill=(200, 200, 200))  # 32px 同色度阴影带
    d.line([(400, 20), (400, 540)], fill=0, width=8)  # → 画布 2px 黑线
    params = BinarizeParams(line_mode=True, line_threshold=10.0)
    mask, offset = binarize(
        img.convert("L"), params, target_w=200, target_h=140, rgb=img
    )

    assert mask.shape == (140, 200)
    assert offset == (0, 0)  # 800x560 → 200x140 等比，无 letterbox
    assert mask[20:120, 96:105].sum() > 0  # 线条（x=400 → x=100）保留
    # 阴影带整体（画布 x≈25..33，含两条边界）不应出现任何线条
    assert mask[20:120, 18:42].sum() == 0


def test_binarize_line_mode_resizes_rgb_for_shadow_reject() -> None:
    """binarize 集成：rgb 随灰度同步缩放，缩放后阴影仍被滤除、线条保留。"""
    from chaji_autodraw.core.imaging import BinarizeParams, binarize

    img = Image.new("RGB", (400, 280), _SKIN)
    d = ImageDraw.Draw(img)
    d.line([(60, 16), (60, 264)], fill=20, width=10)
    d.rectangle([220, 0, 400, 280], fill=_SKIN_SHADOW)
    params = BinarizeParams(line_mode=True, line_threshold=10.0)
    mask, offset = binarize(
        img.convert("L"), params, target_w=200, target_h=140, rgb=img
    )

    assert mask.shape == (140, 200)
    assert offset == (0, 0)
    assert _line_region(mask) > 0
    assert _edge_region(mask) == 0


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
