"""图像二值化管线：灰度加载、可选高斯模糊、等比缩放、阈值二值化与预览生成。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class BinarizeParams:
    """二值化参数。

    Attributes:
        threshold: 灰度阈值（0-255），灰度值小于该值的像素视为深色。
        blur_radius: 高斯模糊半径（像素），0 表示不模糊。
        invert: False 时绘制深色像素，True 时绘制浅色像素。
        line_mode: 线稿提取模式（适合彩色二次元插画），开启后走 DoG
            线条检测而非阈值二值化，只画线条不涂色块。
        line_threshold: 线稿检测灵敏度（DoG 阈值），越小线条越多越密。
    """

    threshold: int = 128
    blur_radius: float = 0.0
    invert: bool = False
    line_mode: bool = False
    line_threshold: float = 12.0


def load_grayscale(path: str) -> Image.Image:
    """加载图片文件并转为灰度（L 模式）图像。"""
    with Image.open(path) as im:
        return im.convert("L")


def extract_line_mask(
    gray: Image.Image,
    sensitivity: float = 12.0,
    blur_radius: float = 1.0,
) -> np.ndarray:
    """线稿提取（DoG 差分高斯）：检测比周边暗的线条结构。

    适合彩色二次元插画——线条通常比色块暗且细。流程：灰度 → 内核
    高斯模糊 → 外核高斯模糊 → 外核 − 内核（线条处为正）→ 阈值化。
    平缓的色块渐变会被差分抵消，只留下线状结构。

    Args:
        gray: 灰度图（L 模式）。
        sensitivity: DoG 阈值，越小保留的线条越多越密，越大只保留最深的线。
        blur_radius: 内核模糊半径；外核为其 3 倍（决定线条的邻域尺度）。

    Returns:
        bool 掩码，True 表示线条像素（需要绘制）。
    """
    r_inner = max(0.5, blur_radius)
    r_outer = r_inner * 3.0
    inner = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(r_inner)), dtype=np.float32
    )
    outer = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(r_outer)), dtype=np.float32
    )
    dog = outer - inner  # 暗线条处 inner 更暗 → 差分为正
    return dog > float(sensitivity)


def is_colorful(
    image: Image.Image,
    saturation_threshold: int = 40,
    fraction_threshold: float = 0.05,
) -> bool:
    """判断图片是否为大面积彩色（如彩色二次元插画）。

    统计饱和度超过 saturation_threshold 的像素占比：超过
    fraction_threshold 即认为是大面积彩色。灰度图/黑白漫画饱和度
    接近 0，稳定返回 False。

    Args:
        image: 任意模式图片（内部转 RGB 并降采样加速）。
        saturation_threshold: 饱和度判定阈值（0~255，HSV 的 S 分量）。
        fraction_threshold: 彩色像素占比阈值（0~1）。

    Returns:
        True 表示大面积彩色，建议开启线条模式。
    """
    small = image.convert("RGB")
    small.thumbnail((256, 256))
    hsv = np.asarray(small.convert("HSV"), dtype=np.uint8)
    sat = hsv[:, :, 1]
    colored = int((sat > saturation_threshold).sum())
    return colored / float(sat.size) > fraction_threshold


def binarize(
    gray: Image.Image,
    params: BinarizeParams,
    target_w: int | None = None,
    target_h: int | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """把灰度图处理为待绘制的布尔掩码。

    流程：灰度图 → 可选高斯模糊 → 若给定 target 则按纵横比 LANCZOS 缩放到
    恰好放进 target_w×target_h（不变形）→ 阈值二值化 → 用 False 填充
    letterbox 到恰好 target_w×target_h。

    Args:
        gray: 灰度图（非 L 模式会自动转换）。
        params: 二值化参数。
        target_w: 目标画布宽度（像素），必须与 target_h 同时提供。
        target_h: 目标画布高度（像素）。

    Returns:
        (mask, (offset_x, offset_y))。mask 为 bool 二维数组，True 表示需要
        绘制的像素；offset 为缩放后内容在画布内的居中偏移（内容左上角坐标）。
        不给 target 时保持原图尺寸，offset 为 (0, 0)。

    Raises:
        ValueError: 参数非法（阈值越界、只给一个 target、尺寸非正等）。
    """
    if not 0 <= int(params.threshold) <= 255:
        raise ValueError("threshold 必须在 0-255 之间")
    if (target_w is None) != (target_h is None):
        raise ValueError("target_w 与 target_h 必须同时提供")
    if target_w is not None and target_w <= 0 or target_h is not None and target_h <= 0:
        raise ValueError("target 尺寸必须为正")

    work = gray if gray.mode == "L" else gray.convert("L")
    if params.blur_radius > 0:
        work = work.filter(ImageFilter.GaussianBlur(params.blur_radius))

    offset_x = 0
    offset_y = 0
    if target_w is not None and target_h is not None:
        w, h = work.size
        if w <= 0 or h <= 0:
            raise ValueError("源图尺寸必须为正")
        scale = min(target_w / w, target_h / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        # 防御浮点边界：缩放结果不应超出目标画布
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        if (new_w, new_h) != (w, h):
            work = work.resize((new_w, new_h), Image.LANCZOS)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2

    if params.line_mode:
        # 线稿模式：在画布分辨率上做 DoG 线条检测，threshold/invert 不参与
        mask = extract_line_mask(
            work,
            sensitivity=params.line_threshold,
            blur_radius=params.blur_radius if params.blur_radius > 0 else 1.0,
        )
    else:
        arr = np.asarray(work, dtype=np.uint8)
        threshold = int(params.threshold)
        if params.invert:
            mask: np.ndarray = arr >= threshold  # 绘制浅色像素
        else:
            mask = arr < threshold  # 绘制深色像素

    if target_w is not None and target_h is not None:
        canvas = np.zeros((target_h, target_w), dtype=bool)
        canvas[
            offset_y : offset_y + mask.shape[0],
            offset_x : offset_x + mask.shape[1],
        ] = mask
        mask = canvas

    return mask, (offset_x, offset_y)


def mask_to_preview(mask: np.ndarray) -> Image.Image:
    """把 bool 掩码转为白底黑点（绘制像素）的 RGB 预览图，供界面显示。"""
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    rgb = np.full((m.shape[0], m.shape[1], 3), 255, dtype=np.uint8)
    rgb[m] = (0, 0, 0)
    return Image.fromarray(rgb, mode="RGB")
