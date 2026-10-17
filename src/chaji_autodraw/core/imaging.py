"""图像二值化管线：灰度加载、可选高斯模糊、等比缩放、阈值二值化/线稿
提取（含光影分界线剔除）与预览生成。"""

from __future__ import annotations

import os
import weakref
from concurrent.futures import ThreadPoolExecutor
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
            阴影边缘当成线条画出。默认关闭；仅 line_mode 下生效。
        shadow_protection: 阴影剔除保护强度（0~100，默认 40）：越大
            越保守——只有色度几乎一致且落差很小的分界才判为阴影，
            保留更多线条；越小滤除越激进。仅 reject_shadow 生效。
        full_res_detect: 线条模式下检测（DoG 与阴影剔除的全部滤波）
            在原图全分辨率执行（默认）：框选画布前后检测效果一致，
            掩码按面积覆盖率缩放、细线不丢失；False 时回到旧行为——
            先把图片缩放到画布分辨率再检测，超大图上更快。
    """

    threshold: int = 128
    blur_radius: float = 0.0
    invert: bool = False
    line_mode: bool = False
    line_threshold: float = 12.0
    reject_shadow: bool = False
    shadow_protection: float = 40.0
    full_res_detect: bool = True


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
#: 稀疏 gather 的支撑集密度上限：dog>1 的像素占比不超过该值时，
#: 双向亮度测试与色度除法只在支撑集上计算（实测 2.8MP：占比 10%
#: 提速约 2 倍、占比 0.6% 提速 16~25 倍；占比 40% 时 gather 反而慢
#: 2~3 倍）。两条路径结果逐位一致，仅为性能取舍。
_SPARSE_SUPPORT_MAX = 0.15
#: 线程池并行滤波的最小像素数：低于该值串行——复用池的提交开销
#: 虽不足 1ms，但小图滤波本身只省几毫秒，且避免在画布尺寸的小图
#: 上引入线程噪声（实测 ≥512² 各尺寸均有 1.35~1.75x 加速）
_PARALLEL_MIN_PIXELS = 512 * 512
#: 模块级复用的滤波线程池（惰性创建；进程存活期间复用）。
#: 池上限 min(4, CPU 数)，且每轮最多只提交 2 个滤波任务（加主线程
#: 并发共 3 线程），不会占满多核机器。
_FILTER_POOL: ThreadPoolExecutor | None = None


def _filter_pool(size: int) -> ThreadPoolExecutor | None:
    """大图返回复用线程池做并行滤波，小图返回 None 走串行。

    numpy 的元素级运算与 ufunc.accumulate 基本释放 GIL，实测把两个
    独立的 RGB 极值滤波交给线程池、主线程同时做模糊与灰度滤波，
    整体墙钟时间约 1.35~1.75x 加速；池按模块级复用，避免每次重建
    线程的开销。滤镜是纯函数，并行与串行结果逐位一致。
    """
    global _FILTER_POOL
    if size < _PARALLEL_MIN_PIXELS:
        return None
    workers = min(4, os.cpu_count() or 1)
    if workers < 2:
        return None
    if _FILTER_POOL is None:
        _FILTER_POOL = ThreadPoolExecutor(
            workers, thread_name_prefix="imgfilter"
        )
    return _FILTER_POOL


def _box_extreme(arr: np.ndarray, radius: int, use_max: bool) -> np.ndarray:
    """盒形窗口极值滤波（最大/最小值滤波，即灰度膨胀/腐蚀）。

    用 van Herk–Gil–Werman 块分解实现：序列按窗口长度分块，块内
    前缀/后缀极值各线性扫描一遍；窗口长度恰等于块长、至多跨两块，
    由「起点到其块尾的后缀 ∪ 终点所在块首到终点的前缀」精确覆盖。
    每轴只需常数次线性扫描，代价与窗口半径无关（旧逐偏移聚合实现
    每轴要做 2r+1 次全图比较，r=12 时实测慢约 2~2.6 倍）。支持
    int/uint/bool（bool 的 min/max 即按位与/或，掩码形态学可直接
    使用，无需转 int16）。边界按边缘像素复制填充。

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
        res = _extreme_along_last(np.moveaxis(out, axis, -1), radius, op)
        out = np.moveaxis(res, -1, axis)
    return out


def _extreme_along_last(
    a: np.ndarray, radius: int, op: np.ufunc
) -> np.ndarray:
    """对最后一维做窗口 [i−r, i+r] 极值滤波（van Herk 块分解）。"""
    n = a.shape[-1]
    block = 2 * radius + 1
    # 一次补齐：左侧 r、右侧补到块长整数倍（均为边缘复制）
    total = ((n + 2 * radius + block - 1) // block) * block
    right = total - n - radius
    padded = np.pad(
        a, [(0, 0)] * (a.ndim - 1) + [(radius, right)], mode="edge"
    )
    blocks = padded.reshape(a.shape[:-1] + (total // block, block))
    prefix = op.accumulate(blocks, axis=-1)
    suffix = op.accumulate(blocks[..., ::-1], axis=-1)[..., ::-1]
    # 原坐标窗口 [i−r, i+r] → 填充坐标 [i, i+2r]，两段恰好拼成窗口
    return op(
        suffix.reshape(a.shape[:-1] + (total,))[..., :n],
        prefix.reshape(a.shape[:-1] + (total,))[
            ..., 2 * radius : 2 * radius + n
        ],
    )


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


def _is_line_like(
    inner: np.ndarray, halfwidth: int, t_side: int,
    radii: tuple[int, ...] | None = None,
) -> np.ndarray:
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
        radii: 参与的采样距离子集（默认两个都用）。并行时按距离拆成
            两半各算一半、结果按位或——与全量计算逐位一致。

    Returns:
        bool 掩码，True 表示几何上是线条（脊线）而非阶跃边缘。
    """
    if radii is None:
        radii = (halfwidth + 2, halfwidth + 5)
    bright = inner + t_side
    line_like = np.zeros(inner.shape, dtype=bool)
    for radius in radii:
        diag = max(1, int(round(radius * 0.7071)))
        for dy, dx in ((0, radius), (radius, 0), (diag, diag), (diag, -diag)):
            plus = _shift(inner, dy, dx, -32768)
            minus = _shift(inner, -dy, -dx, -32768)
            line_like |= (plus >= bright) & (minus >= bright)
    return line_like


def _is_line_like_at(
    inner: np.ndarray,
    halfwidth: int,
    t_side: int,
    ys: np.ndarray,
    xs: np.ndarray,
    radii: tuple[int, ...] | None = None,
) -> np.ndarray:
    """_is_line_like 的稀疏版本：只在给定坐标上做同样的双向测试。

    与全量版在这些坐标处逐位一致——采样同样取 ``inner[y±dy, x±dx]``、
    越界同样填 -32768（视为「不亮」，该方向判失败）。用一维扁平索引
    + ``np.take`` 做采样，坐标稀疏时（如线稿图的候选像素只占几个
    百分点）比 16 次全图移位快一个数量级。radii 含义同 _is_line_like。
    """
    if radii is None:
        radii = (halfwidth + 2, halfwidth + 5)
    pad = halfwidth + 5
    padded = np.pad(inner, pad, mode="constant", constant_values=-32768)
    stride = padded.shape[1]
    flat = (ys + pad) * stride + (xs + pad)
    buf = padded.reshape(-1)
    bright = inner[ys, xs] + t_side
    out = np.zeros(ys.shape, dtype=bool)
    for radius in radii:
        diag = max(1, int(round(radius * 0.7071)))
        for dy, dx in ((0, radius), (radius, 0), (diag, diag), (diag, -diag)):
            plus = buf.take(flat + dy * stride + dx)
            minus = buf.take(flat - dy * stride - dx)
            out |= (plus >= bright) & (minus >= bright)
    return out


class _LineCache:
    """线条模式检测的单条目中间结果缓存。

    同一张图、同一内核半径与采样阈值下，DoG 与阴影剔除的全部滤波只
    计算一次；「线条阈值」（只影响 DoG 阈值比较）与「保护强度」（只
    影响最后的阈值比较）变化时仅重做布尔组合，毫秒级完成。以对源图
    的弱引用判定身份：换图（或源图被回收后 id 复用）即失效重建，
    不比较像素内容。仅 UI 线程调用，无需加锁。
    """

    def __init__(
        self,
        gray: Image.Image,
        rgb: Image.Image | None,
        r_inner: float,
        t_side: int,
        need_shadow: bool,
        min_sensitivity: float = 1.0,
    ) -> None:
        """计算并缓存全部中间结果；need_shadow=False 时只算 DoG 部分。

        min_sensitivity 为本次调用的检测灵敏度：稀疏路径的支撑集取
        dog > min(1, min_sensitivity)，只允许灵敏度不低于该阈值的后续
        调用复用（matches 校验候选集必然被支撑集覆盖）。
        """
        self._gray_ref = weakref.ref(gray)
        self._rgb_ref = weakref.ref(rgb) if rgb is not None else None
        self._r_inner = r_inner
        self._t_side = t_side
        self._need_shadow = need_shadow

        r_outer = r_inner * 3.0
        halfwidth = max(3, int(round(2.5 * r_inner + 2)))
        bg_radius = max(8, 3 * halfwidth)
        margin = max(2, int(round(r_inner + 1)))
        radii = (halfwidth + 2, halfwidth + 5)

        rgb_arr: np.ndarray | None = None
        if need_shadow and rgb is not None:
            if rgb.mode != "RGB" or rgb.size != gray.size:
                if rgb.mode != "RGB":
                    rgb = rgb.convert("RGB")
                if rgb.size != gray.size:
                    rgb = rgb.resize(gray.size, Image.LANCZOS)
            rgb_arr = np.asarray(rgb, dtype=np.int16)

        # 两阶段并行（大图，结果与串行逐位一致；小图串行避免线程开销）：
        # 阶段一：内核/外核模糊与两个 RGB 极值滤波互不依赖，4 个任务
        #         全部提交线程池，主线程等待；
        # 阶段二：灰度两个极值滤波 + 双向亮度测试按采样距离拆两半，
        #         共 4 个任务提交；主线程拿到 bg 后立即做暗带→腐蚀→
        #         膨胀（zone），与其余任务重叠。
        pool = _filter_pool(gray.size[0] * gray.size[1])
        if pool is not None:
            f_blur_in = pool.submit(gray.filter, ImageFilter.GaussianBlur(r_inner))
            f_blur_out = pool.submit(gray.filter, ImageFilter.GaussianBlur(r_outer))
            f_lit = f_drk = None
            if rgb_arr is not None:
                f_lit = pool.submit(_box_extreme, rgb_arr, bg_radius, True)
                f_drk = pool.submit(_box_extreme, rgb_arr, bg_radius, False)
            inner_img = f_blur_in.result()
            outer = np.asarray(f_blur_out.result(), dtype=np.int16)
        else:
            f_lit = f_drk = None
            inner_img = gray.filter(ImageFilter.GaussianBlur(r_inner))
            outer = np.asarray(
                gray.filter(ImageFilter.GaussianBlur(r_outer)), dtype=np.int16
            )
        self.inner = np.asarray(inner_img, dtype=np.int16)
        self.dog = outer - self.inner

        self.zone: np.ndarray | None = None
        self.line_like: np.ndarray | None = None
        self.contrast: np.ndarray | None = None
        self.chroma_gap: np.ndarray | None = None
        self.chroma_ok: np.ndarray | None = None
        #: 本缓存结果可复用的最低灵敏度（稀疏支撑集阈值；全量路径为 0）
        self.support_level = 0.0
        if not need_shadow:
            return

        # 支撑集密度低（典型线稿图）→ 双向测试与色度只在支撑集上算，
        # 结果散回全图（逐位等价）；密度高（噪声图）→ 全量更划算
        support_thresh = min(1.0, float(min_sensitivity))
        support = self.dog > support_thresh
        sparse = support.mean() <= _SPARSE_SUPPORT_MAX
        ys = xs = None
        if sparse:
            self.support_level = support_thresh
            ys, xs = np.nonzero(support)

        if pool is not None:
            f_bg = pool.submit(_box_extreme, self.inner, bg_radius, True)
            f_low = pool.submit(_box_extreme, self.inner, bg_radius, False)
            if sparse:
                f_ll1 = pool.submit(
                    _is_line_like_at, self.inner, halfwidth, t_side, ys, xs,
                    (radii[0],),
                )
                f_ll2 = pool.submit(
                    _is_line_like_at, self.inner, halfwidth, t_side, ys, xs,
                    (radii[1],),
                )
            else:
                f_ll1 = pool.submit(
                    _is_line_like, self.inner, halfwidth, t_side, (radii[0],)
                )
                f_ll2 = pool.submit(
                    _is_line_like, self.inner, halfwidth, t_side, (radii[1],)
                )
            bg = f_bg.result()
            # zone 只依赖 bg：主线程现在算，与 low/双向测试任务重叠
            dark = (bg - self.inner) > _SHADOW_DARK_DELTA
            # 掩码形态学直接用 bool（min/max 即按位与/或）
            core = _box_extreme(dark, halfwidth, use_max=False)
            self.zone = _box_extreme(core, halfwidth + margin, use_max=True)
            low = f_low.result()
            line_like_half = f_ll1.result() | f_ll2.result()
        else:
            bg = _box_extreme(self.inner, bg_radius, use_max=True)
            low = _box_extreme(self.inner, bg_radius, use_max=False)
            dark = (bg - self.inner) > _SHADOW_DARK_DELTA
            core = _box_extreme(dark, halfwidth, use_max=False)
            self.zone = _box_extreme(core, halfwidth + margin, use_max=True)
            if sparse:
                line_like_half = _is_line_like_at(
                    self.inner, halfwidth, t_side, ys, xs
                )
            else:
                line_like_half = _is_line_like(self.inner, halfwidth, t_side)
        self.contrast = bg - low
        if sparse:
            self.line_like = np.zeros(self.inner.shape, dtype=bool)
            self.line_like[ys, xs] = line_like_half
        else:
            self.line_like = line_like_half

        if rgb_arr is not None:
            # 逐通道局部极值近似两侧颜色：最大 ≈ 亮侧，最小 ≈ 暗侧
            if f_lit is not None:
                lit = f_lit.result().astype(np.float32)
                drk = f_drk.result().astype(np.float32)
            else:
                lit = _box_extreme(rgb_arr, bg_radius, use_max=True).astype(
                    np.float32
                )
                drk = _box_extreme(
                    rgb_arr, bg_radius, use_max=False
                ).astype(np.float32)
            # rg 色度（RGB 归一化，天然只取前两维）：对按比例变暗不变
            if ys is not None:
                lit_rows = lit[ys, xs]
                drk_rows = drk[ys, xs]
                lit_sum = lit_rows.sum(axis=1)
                drk_sum = drk_rows.sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    gap_rows = np.abs(
                        lit_rows[:, :2] / lit_sum[:, None]
                        - drk_rows[:, :2] / drk_sum[:, None]
                    ).max(axis=1)
                self.chroma_gap = np.zeros(self.inner.shape, dtype=np.float32)
                self.chroma_gap[ys, xs] = np.nan_to_num(gap_rows, nan=1.0)
                self.chroma_ok = np.zeros(self.inner.shape, dtype=bool)
                self.chroma_ok[ys, xs] = (lit_sum >= _SHADOW_SIGNAL_MIN) & (
                    drk_sum >= _SHADOW_SIGNAL_MIN
                )
            else:
                lit_sum = lit.sum(axis=2)
                drk_sum = drk.sum(axis=2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    gap = np.abs(
                        lit[:, :, :2] / lit_sum[:, :, None]
                        - drk[:, :, :2] / drk_sum[:, :, None]
                    ).max(axis=2)
                self.chroma_gap = np.nan_to_num(gap, nan=1.0)
                self.chroma_ok = (lit_sum >= _SHADOW_SIGNAL_MIN) & (
                    drk_sum >= _SHADOW_SIGNAL_MIN
                )

    def matches(
        self,
        gray: Image.Image,
        rgb: Image.Image | None,
        r_inner: float,
        t_side: int,
        need_shadow: bool,
        sensitivity: float = 1.0,
    ) -> bool:
        """缓存是否可复用：源图弱引用身份 + 参数键一致 + 候选集被支撑。"""
        if (
            self._gray_ref() is not gray
            or self._r_inner != r_inner
            or self._t_side != t_side
            or self._need_shadow != need_shadow
            or sensitivity < self.support_level
        ):
            return False
        if rgb is None:
            return self._rgb_ref is None
        return self._rgb_ref is not None and self._rgb_ref() is rgb

    def shadow_edge_mask(self, protection: float) -> np.ndarray:
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
        紧贴阴影边界的线条都不会被误删。``protection``（0~100）只
        调节第 3 道判据的证据强度：越大越保守，色度容差与落差上限
        越小，需要越典型的阴影特征才会被滤除；几何与宽度判据不随之
        变化。本方法只做阈值比较与布尔组合，滤波结果全部来自缓存。
        """
        assert self.zone is not None and self.line_like is not None
        assert self.contrast is not None

        # 保护强度 → 三个颜色/对比度阈值（(0, 50, 100) 三点分段线性插值）
        p = min(100.0, max(0.0, float(protection)))
        stops = (0.0, 50.0, 100.0)
        chroma_max = float(np.interp(p, stops, _SHADOW_CHROMA_MAX_ENDS))
        chroma_contrast_max = float(
            np.interp(p, stops, _SHADOW_CHROMA_CONTRAST_ENDS)
        )
        gray_contrast_max = float(np.interp(p, stops, _SHADOW_CONTRAST_ENDS))

        if self.chroma_gap is not None:
            assert self.chroma_ok is not None
            shadow_like = (
                (self.chroma_gap < chroma_max)
                & self.chroma_ok
                & (self.contrast < chroma_contrast_max)
            )
        else:
            # 灰度图无法判色，退化为低对比判据
            shadow_like = self.contrast < gray_contrast_max

        return self.zone & ~self.line_like & shadow_like


#: 线条模式检测的单条目缓存（换图或参数键变化即整体重建）
_LINE_CACHE: _LineCache | None = None


def load_grayscale(path: str) -> Image.Image:
    """加载图片文件并转为灰度（L 模式）图像。"""
    with Image.open(path) as im:
        return im.convert("L")


def extract_line_mask(
    gray: Image.Image,
    sensitivity: float = 12.0,
    blur_radius: float = 1.0,
    rgb: Image.Image | None = None,
    reject_shadow: bool = False,
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
        reject_shadow: 过滤光影分界线（默认关闭）。DoG 对阶跃边缘同样
            响应，阴影与亮部的分界会被当成线条画出；该选项按「几何
            是阶跃而非线条 + 邻近宽暗区 + 两侧色度一致（光照变化
            特征）」三道判据识别并剔除，真实线条（两侧恢复亮背景的
            窄暗带）不受影响。
        shadow_protection: 阴影剔除保护强度（0~100，默认 40）：越大
            越保守，需要越典型的阴影特征才会滤除。

    Returns:
        bool 掩码，True 表示线条像素（需要绘制）。

    Notes:
        内部维护单条目中间结果缓存：同一张图、同一模糊半径下重复
        调用（如仅改 sensitivity / shadow_protection）不再重算滤波，
        只重做阈值比较，毫秒级完成。
    """
    r_inner = max(0.5, blur_radius)
    t_side = max(_SHADOW_SIDE_BRIGHT, int(round(0.5 * sensitivity)))
    global _LINE_CACHE
    if _LINE_CACHE is None or not _LINE_CACHE.matches(
        gray, rgb, r_inner, t_side, reject_shadow, float(sensitivity)
    ):
        _LINE_CACHE = _LineCache(
            gray, rgb, r_inner, t_side, reject_shadow, float(sensitivity)
        )
    mask = _LINE_CACHE.dog > float(sensitivity)
    if reject_shadow:
        mask &= ~_LINE_CACHE.shadow_edge_mask(shadow_protection)
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


#: binarize 可选高斯模糊的单条目缓存（同一源图与半径直接复用）
_BLUR_CACHE: tuple[weakref.ref, float, Image.Image] | None = None


def _blurred(gray: Image.Image, radius: float) -> Image.Image:
    """缓存全分辨率高斯模糊结果。

    阈值/线条阈值等参数变化时不再重付整张原图的模糊成本（大图上
    一次数十毫秒）；以源图弱引用判定身份，换图即失效。
    """
    global _BLUR_CACHE
    if (
        _BLUR_CACHE is not None
        and _BLUR_CACHE[0]() is gray
        and _BLUR_CACHE[1] == radius
    ):
        return _BLUR_CACHE[2]
    out = gray.filter(ImageFilter.GaussianBlur(radius))
    _BLUR_CACHE = (weakref.ref(gray), radius, out)
    return out


def binarize(
    gray: Image.Image,
    params: BinarizeParams,
    target_w: int | None = None,
    target_h: int | None = None,
    rgb: Image.Image | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """把灰度图处理为待绘制的布尔掩码。

    流程：灰度图 → 可选高斯模糊（原图分辨率）→
    - 线条模式（full_res_detect=True，默认）：DoG 线条检测 + 阴影
      剔除全部在**原图分辨率**执行（与未框选画布的预览一致——阴影
      剔除的几何判据绑定像素尺度，先缩放会让窄阴影带漏判），得到的
      掩码再按面积覆盖率缩放到 target；False 时先缩放再检测（旧行为）；
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
        work = _blurred(work, params.blur_radius)

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
        # 线稿模式：threshold/invert 不参与
        if params.full_res_detect:
            # 全分辨率（默认）：DoG + 阴影剔除全部滤波在原图分辨率执行
            # （与未框选画布的预览一致，避免缩放改变几何判定），掩码
            # 再按面积覆盖率缩放到画布
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
            # 缩放后分辨率（旧行为）：先把灰度与彩色图缩放到画布分辨率
            # 再检测，超大图上更快，但缩放可能改变阴影剔除的几何判定
            work_line = work
            rgb_line = rgb
            if new_w is not None and (new_w, new_h) != (w, h):
                work_line = work.resize((new_w, new_h), Image.LANCZOS)
                if rgb is not None:
                    rgb_line = rgb.resize((new_w, new_h), Image.LANCZOS)
            mask = extract_line_mask(
                work_line,
                sensitivity=params.line_threshold,
                blur_radius=params.blur_radius if params.blur_radius > 0 else 1.0,
                rgb=rgb_line,
                reject_shadow=params.reject_shadow,
                shadow_protection=params.shadow_protection,
            )
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
