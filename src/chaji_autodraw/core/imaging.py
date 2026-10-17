"""图像二值化管线：灰度加载、可选高斯模糊、等比缩放、阈值二值化/线稿
提取（含光影分界线剔除）与预览生成。"""

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
        reject_shadow: 线条模式下过滤光影分界线（阴影边界），防止把
            阴影边缘当成线条画出。默认开启；仅 line_mode 下生效。
        shadow_protection: 阴影剔除保护强度（0~100，默认 40）：越大
            越保守——只有色度几乎一致且落差很小的分界才判为阴影，
            保留更多线条；越小滤除越激进。仅 reject_shadow 生效。
    """

    threshold: int = 128
    blur_radius: float = 0.0
    invert: bool = False
    line_mode: bool = False
    line_threshold: float = 12.0
    reject_shadow: bool = True
    shadow_protection: float = 40.0


# ---- 阴影剔除（线条模式过滤光影分界线）参数 ----
#: 比局部亮参考暗超过该值的像素计入「暗带」
_SHADOW_DARK_DELTA = 24
#: 双向亮度测试（Steger 线条模型）：两侧采样需比中心亮的最低量，
#: 取其与检测灵敏度一半的较大者
_SHADOW_SIDE_BRIGHT = 8
#: 保护强度 50（默认）时：两侧 rg 色度距离阈值、色度规则对比度
#: 上限、灰度低对比阈值
_SHADOW_CHROMA_MAX = 0.045
_SHADOW_CHROMA_CONTRAST_MAX = 150
_SHADOW_CONTRAST_MAX = 75
#: 保护强度 0（最激进）与 100（最保守）时上述三个阈值的端点值，
#: 实际阈值按保护强度在 (0, 50, 100) 三点间分段线性插值
_SHADOW_CHROMA_MAX_ENDS = (0.09, _SHADOW_CHROMA_MAX, 0.012)
_SHADOW_CHROMA_CONTRAST_ENDS = (200, _SHADOW_CHROMA_CONTRAST_MAX, 45)
_SHADOW_CONTRAST_ENDS = (110, _SHADOW_CONTRAST_MAX, 40)
#: 色度有效所需的两侧 RGB 通道和下限（过暗时色度噪声大，保守放行）
_SHADOW_SIGNAL_MIN = 60


def _box_extreme(arr: np.ndarray, radius: int, use_max: bool) -> np.ndarray:
    """盒形窗口极值滤波（最大/最小值滤波，即灰度膨胀/腐蚀）。

    用可分离的逐偏移聚合实现——每个轴做 2r+1 次向量化移位比较，
    避免逐像素扫描整个 (2r+1)² 窗口；600×600 图实测比 PIL 大窗口
    秩滤波快约 25 倍。边界按边缘像素复制填充。

    Args:
        arr: 被滤波数组（二维，或第三维为通道的三维）。
        radius: 窗口半径（窗口边长 2·radius+1），≥1。
        use_max: True 取窗口最大值（膨胀），False 取最小值（腐蚀）。

    Returns:
        与 arr 同形状同 dtype 的滤波结果。
    """
    op = np.maximum if use_max else np.minimum
    out = arr
    for axis in (0, 1):
        pad = [(0, 0)] * arr.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        out = padded[tuple(
            slice(0, padded.shape[i] - 2 * radius) if i == axis else slice(None)
            for i in range(padded.ndim)
        )]
        for offset in range(1, 2 * radius + 1):
            shifted = padded[tuple(
                slice(offset, padded.shape[i] - 2 * radius + offset)
                if i == axis else slice(None)
                for i in range(padded.ndim)
            )]
            out = op(out, shifted)
    return out


def _shift(arr: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    """整体平移二维数组 (dy, dx)，越界处补 fill。

    用于按方向偏移采样：把采样越界的像素填成极小值，等价于把该侧
    视为「不亮」，让该方向的双向测试判失败。
    """
    out = np.full_like(arr, fill)
    h, w = arr.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = arr[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def _is_line_like(inner: np.ndarray, halfwidth: int, t_side: int) -> np.ndarray:
    """双向亮度测试：判断候选像素更像「线条」还是「阶跃边缘」。

    线条（脊线）的剖面两侧都比中心亮（Steger 的线条模型：本质是
    一对相邻且相对的阶跃边）；阴影/色块边界是单侧阶跃，暗侧持续
    延伸。在两个采样距离（略过最粗可接受线条的半宽）、8 个方向上
    检查：只要存在一个方向两侧采样都比中心亮超过 ``t_side``，即判
    为线条。密集排线、区域内部的细节线因此都能保留。

    Args:
        inner: 内核高斯模糊后的灰度图（int16）。
        halfwidth: 可接受线条的最大半宽（采样距离由此推导）。
        t_side: 两侧采样需比中心亮的量。

    Returns:
        bool 掩码，True 表示几何上是线条（脊线）而非阶跃边缘。
    """
    bright = inner + t_side
    line_like = np.zeros(inner.shape, dtype=bool)
    for radius in (halfwidth + 2, halfwidth + 5):
        diag = max(1, int(round(radius * 0.7071)))
        for dy, dx in ((0, radius), (radius, 0), (diag, diag), (diag, -diag)):
            plus = _shift(inner, dy, dx, -32768)
            minus = _shift(inner, -dy, -dx, -32768)
            line_like |= (plus >= bright) & (minus >= bright)
    return line_like


def _shadow_edge_mask(
    inner: np.ndarray,
    rgb: Image.Image | None,
    r_inner: float,
    sensitivity: float,
    protection: float = 40.0,
) -> np.ndarray:
    """识别光影分界线（阴影边界），返回应从线条掩码中剔除的像素。

    DoG 对「窄暗线条」和「阶跃边缘」（阴影/色块边界）同样响应。
    一个候选像素必须同时通过三道判据才被认定为光影分界线：

    1. **几何是阶跃而非线条**（Steger 双向亮度测试）：8 个方向 ×
       两个距离采样，没有任何方向两侧都比中心亮——线条两侧总会
       恢复亮背景，阴影的暗侧持续延伸；
    2. **邻近宽暗区**：以局部最大值为亮参考找暗带，腐蚀掉窄线条
       后仍有宽暗区内核（排除密集排线这类暗带互相隔开的结构）；
    3. **颜色是光照变化而非材质变化**（Finlayson 色度不变性）：
       阴影只是把表面整体按比例调暗，两侧 rg 色度不变；材质分界
       （发色|肤色、浅黄发|白底、灰衣|白底）色度会改变。无彩色图
       时退化为低对比判据。

    三道判据缺一不删：浅色低对比的材质边界、宽暗区内部的细节线、
    紧贴阴影边界的线条都不会被误删。``protection``（0~100，默认
    40）只调节第 3 道判据的证据强度：越大越保守，色度容差与落差
    上限越小，需要越典型的阴影特征才会被滤除；几何与宽度判据不
    随之变化。

    Args:
        inner: 内核高斯模糊后的灰度图（int16 数组）。
        rgb: 与 inner 同内容的彩色图（尺寸不符会按 LANCZOS 缩放到
            一致），None 时只用亮度规则。
        r_inner: DoG 内核模糊半径，用于缩放结构元尺寸。
        sensitivity: DoG 检测灵敏度，用于推导双向测试的亮度阈值。
        protection: 保护强度（0~100），见上。

    Returns:
        bool 掩码，True 表示该像素处于光影分界线（不应绘制）。
    """
    halfwidth = max(3, int(round(2.5 * r_inner + 2)))
    bg_radius = max(8, 3 * halfwidth)
    margin = max(2, int(round(r_inner + 1)))

    bg = _box_extreme(inner, bg_radius, use_max=True)
    low = _box_extreme(inner, bg_radius, use_max=False)
    contrast = bg - low
    dark = (bg - inner) > _SHADOW_DARK_DELTA
    core = _box_extreme(dark.astype(np.int16), halfwidth, use_max=False) > 0
    zone = _box_extreme(
        core.astype(np.int16), halfwidth + margin, use_max=True
    ) > 0

    t_side = max(_SHADOW_SIDE_BRIGHT, int(round(0.5 * sensitivity)))
    line_like = _is_line_like(inner, halfwidth, t_side)

    # 保护强度 → 三个颜色/对比度阈值（(0, 50, 100) 三点分段线性插值）
    p = min(100.0, max(0.0, float(protection)))
    stops = (0.0, 50.0, 100.0)
    chroma_max = float(np.interp(p, stops, _SHADOW_CHROMA_MAX_ENDS))
    chroma_contrast_max = float(
        np.interp(p, stops, _SHADOW_CHROMA_CONTRAST_ENDS)
    )
    gray_contrast_max = float(np.interp(p, stops, _SHADOW_CONTRAST_ENDS))

    if rgb is not None:
        h, w = inner.shape
        if rgb.mode != "RGB":
            rgb = rgb.convert("RGB")
        if rgb.size != (w, h):
            rgb = rgb.resize((w, h), Image.LANCZOS)
        rgb_arr = np.asarray(rgb, dtype=np.int16)
        # 逐通道局部极值近似两侧颜色：最大 ≈ 亮侧，最小 ≈ 暗侧
        lit = _box_extreme(rgb_arr, bg_radius, use_max=True).astype(np.float32)
        drk = _box_extreme(rgb_arr, bg_radius, use_max=False).astype(np.float32)
        lit_sum = lit.sum(axis=2)
        drk_sum = drk.sum(axis=2)
        # rg 色度（RGB 归一化，天然只取前两维）：对按比例变暗不变
        with np.errstate(invalid="ignore", divide="ignore"):
            chroma_gap = np.abs(
                lit[:, :, :2] / lit_sum[:, :, None]
                - drk[:, :, :2] / drk_sum[:, :, None]
            ).max(axis=2)
        chroma_gap = np.nan_to_num(chroma_gap, nan=1.0)
        shadow_like = (
            (chroma_gap < chroma_max)
            & (lit_sum >= _SHADOW_SIGNAL_MIN)
            & (drk_sum >= _SHADOW_SIGNAL_MIN)
            & (contrast < chroma_contrast_max)
        )
    else:
        # 灰度图无法判色，退化为低对比判据
        shadow_like = contrast < gray_contrast_max

    return zone & ~line_like & shadow_like


def load_grayscale(path: str) -> Image.Image:
    """加载图片文件并转为灰度（L 模式）图像。"""
    with Image.open(path) as im:
        return im.convert("L")


def extract_line_mask(
    gray: Image.Image,
    sensitivity: float = 12.0,
    blur_radius: float = 1.0,
    rgb: Image.Image | None = None,
    reject_shadow: bool = True,
    shadow_protection: float = 40.0,
) -> np.ndarray:
    """线稿提取（DoG 差分高斯）：检测比周边暗的线条结构。

    适合彩色二次元插画——线条通常比色块暗且细。流程：灰度 → 内核
    高斯模糊 → 外核高斯模糊 → 外核 − 内核（线条处为正）→ 阈值化。
    平缓的色块渐变会被差分抵消，只留下线状结构。

    Args:
        gray: 灰度图（L 模式）。
        sensitivity: DoG 阈值，越小保留的线条越多越密，越大只保留最深的线。
        blur_radius: 内核模糊半径；外核为其 3 倍（决定线条的邻域尺度）。
        rgb: 与 gray 同内容的彩色图（可选）：reject_shadow 开启时用于
            区分「同色调变暗的阴影」与「不同材质的分界线」。
        reject_shadow: 过滤光影分界线（默认开启）。DoG 对阶跃边缘同样
            响应，阴影与亮部的分界会被当成线条画出；该选项按「几何
            是阶跃而非线条 + 邻近宽暗区 + 两侧色度一致（光照变化
            特征）」三道判据识别并剔除，真实线条（两侧恢复亮背景的
            窄暗带）不受影响。
        shadow_protection: 阴影剔除保护强度（0~100，默认 40）：越大
            越保守，需要越典型的阴影特征才会滤除。

    Returns:
        bool 掩码，True 表示线条像素（需要绘制）。
    """
    r_inner = max(0.5, blur_radius)
    r_outer = r_inner * 3.0
    inner_img = gray.filter(ImageFilter.GaussianBlur(r_inner))
    inner = np.asarray(inner_img, dtype=np.float32)
    outer = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(r_outer)), dtype=np.float32
    )
    dog = outer - inner  # 暗线条处 inner 更暗 → 差分为正
    mask = dog > float(sensitivity)
    if reject_shadow:
        mask &= ~_shadow_edge_mask(
            inner.astype(np.int16), rgb, r_inner, sensitivity, shadow_protection
        )
    return mask


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


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把全分辨率线条掩码缩放到目标尺寸：按面积覆盖率 ≥25% 保留。

    用 BOX（面积平均）缩放 0/255 掩码再按覆盖率阈值二值化：缩放 4 倍
    后仍有整像素宽的细线（覆盖率 0.25）得以保留；若用普通重采样+中值
    阈值，细线会因对比度被平均稀释而整体消失。

    Args:
        mask: bool 二维掩码（True 表示线条）。
        size: 目标尺寸 (宽, 高)。

    Returns:
        缩放后的 bool 掩码。
    """
    img = Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(size, Image.BOX)
    coverage = np.asarray(img, dtype=np.float32) / 255.0
    return coverage >= 0.25


def binarize(
    gray: Image.Image,
    params: BinarizeParams,
    target_w: int | None = None,
    target_h: int | None = None,
    rgb: Image.Image | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """把灰度图处理为待绘制的布尔掩码。

    流程：灰度图 → 可选高斯模糊（原图分辨率）→
    - 线条模式：DoG 线条检测 + 阴影剔除全部在**原图分辨率**执行（与
      未框选画布的预览一致——阴影剔除的几何判据绑定像素尺度，先缩放
      会让窄阴影带漏判），得到的掩码再按面积覆盖率缩放到 target；
    - 二值化模式：按纵横比 LANCZOS 缩放到恰好放进 target_w×target_h
      （不变形）后阈值二值化。
    最后用 False 填充 letterbox 到恰好 target_w×target_h。

    Args:
        gray: 灰度图（非 L 模式会自动转换）。
        params: 二值化参数。
        target_w: 目标画布宽度（像素），必须与 target_h 同时提供。
        target_h: 目标画布高度（像素）。
        rgb: 与 gray 同内容的彩色原图（可选）：线条模式开启阴影剔除时
            用于颜色判定（在原图分辨率上使用）。

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

    w, h = work.size
    if w <= 0 or h <= 0:
        raise ValueError("源图尺寸必须为正")
    offset_x = 0
    offset_y = 0
    new_w = new_h = None
    if target_w is not None and target_h is not None:
        scale = min(target_w / w, target_h / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        # 防御浮点边界：缩放结果不应超出目标画布
        new_w = min(new_w, target_w)
        new_h = min(new_h, target_h)
        offset_x = (target_w - new_w) // 2
        offset_y = (target_h - new_h) // 2

    if params.line_mode:
        # 线稿模式：DoG + 阴影剔除全部在原图分辨率执行，
        # threshold/invert 不参与；有画布时把掩码按覆盖率缩放
        mask = extract_line_mask(
            work,
            sensitivity=params.line_threshold,
            blur_radius=params.blur_radius if params.blur_radius > 0 else 1.0,
            rgb=rgb,
            reject_shadow=params.reject_shadow,
            shadow_protection=params.shadow_protection,
        )
        if new_w is not None and (new_w, new_h) != (w, h):
            mask = _resize_mask(mask, (new_w, new_h))
    else:
        work_bin = work
        if new_w is not None and (new_w, new_h) != (w, h):
            work_bin = work.resize((new_w, new_h), Image.LANCZOS)
        arr = np.asarray(work_bin, dtype=np.uint8)
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
