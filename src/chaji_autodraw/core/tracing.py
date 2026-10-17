"""二值图案 → 连贯笔画路径追踪（8 邻域贪心 + 距离抽稀）。

纯 Python/numpy 实现，面向几百×几百像素规模：整体复杂度 O(W*H + N)，
不使用全局最近邻等 O(n²) 结构。

另含手绘模拟（trace_hand_drawn_paths）：先把线条区域骨架化为中心线，
再用「掩膜内桥接」把骨架追踪留下的弧线碎片接成连续长笔，最后叠加
多频手抖、轻微弓形与起收笔过冲；粗色块改为沿主轴的斜向连续排线。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

# 8 邻域偏移 (dx, dy)，固定扫描顺序（保证结果确定性）
_NEIGHBORS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def trace_paths(mask: np.ndarray, min_step: float = 2.0) -> list[list[tuple[int, int]]]:
    """把二值矩阵分解为笔画列表。

    算法：按行优先扫描找到未访问的 True 像素作为笔画起点，从当前像素看
    8 个邻居：优先走未访问的邻居，并优先延续上一步的方向（直线 > 转向），
    直到无未访问邻居则结束该笔；走过的像素立即标记为已访问。
    之后做距离抽稀：与上一保留点距离小于 min_step 的中间点丢弃
    （笔画首尾点必须保留）。

    Args:
        mask: 二维 bool 数组，True 表示需要绘制的像素。
        min_step: 抽稀最小步长（像素），<=0 表示不抽稀。

    Returns:
        笔画列表，每笔为 [(x, y), ...]（屏幕坐标系，x 在前）。
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask 必须是二维数组")

    height, width = m.shape
    visited = np.zeros((height, width), dtype=bool)
    strokes: list[list[tuple[int, int]]] = []

    for y in range(height):
        for x in np.flatnonzero(m[y]):
            xi = int(x)
            if visited[y, xi]:
                continue
            stroke = _trace_one(m, visited, xi, y)
            strokes.append(_simplify(stroke, min_step))
    return strokes


def count_points(strokes: Iterable[Sequence[tuple[int, int]]]) -> int:
    """统计笔画列表的总点数，供进度统计使用。"""
    return sum(len(stroke) for stroke in strokes)


def _trace_one(
    mask: np.ndarray,
    visited: np.ndarray,
    start_x: int,
    start_y: int,
) -> list[tuple[int, int]]:
    """从 (start_x, start_y) 出发做一次 8 邻域贪心追踪，返回完整点列。"""
    height, width = mask.shape
    points: list[tuple[int, int]] = [(start_x, start_y)]
    visited[start_y, start_x] = True

    cx, cy = start_x, start_y
    last_dir: tuple[int, int] | None = None
    while True:
        # ① 优先直线延续：上一步方向上的邻居未访问则直接走（直线 > 转向）
        if last_dir is not None:
            sx = cx + last_dir[0]
            sy = cy + last_dir[1]
            if (
                0 <= sx < width
                and 0 <= sy < height
                and mask[sy, sx]
                and not visited[sy, sx]
            ):
                cx, cy = sx, sy
                visited[cy, cx] = True
                points.append((cx, cy))
                continue

        # ② 无直线可走时，在剩余邻居里选转向最小的（与 last_dir 点积越大越优先），
        #    同分时保持固定邻域顺序（首个最大者胜出，保证确定性）
        best: tuple[int, int] | None = None
        best_score = 0
        for dx, dy in _NEIGHBORS:
            nx = cx + dx
            ny = cy + dy
            if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                score = dx * last_dir[0] + dy * last_dir[1] if last_dir is not None else 0
                if best is None or score > best_score:
                    best = (dx, dy)
                    best_score = score
        if best is None:
            break
        dx, dy = best
        cx += dx
        cy += dy
        visited[cy, cx] = True
        points.append((cx, cy))
        last_dir = (dx, dy)
    return points


def _simplify(
    points: list[tuple[int, int]],
    min_step: float,
) -> list[tuple[int, int]]:
    """按最小步长抽稀点列：直线段内丢弃距上一保留点 < min_step 的点。

    **方向变化处（拐点）强制保留**：抽稀后用直线弦连接保留点，若切掉
    拐点，弦会偏离原路径，在实心区域的连续转弯处留下漏画缝隙（表现为
    黑色区域内的白色细线）。直线段内弦与路径重合，可安全抽稀。

    笔画首尾点必须保留；min_step <= 0 或点数 <= 2 时原样返回。
    """
    if min_step <= 0 or len(points) <= 2:
        return list(points)
    limit_sq = min_step * min_step
    kept: list[tuple[int, int]] = [points[0]]
    prev_dir: tuple[int, int] | None = None
    for prev, cur in zip(points, points[1:]):
        dx = cur[0] - prev[0]
        dy = cur[1] - prev[1]
        if prev_dir is None:
            # 首步仅建立方向，points[1] 交给距离判断
            prev_dir = (dx, dy)
        elif (dx, dy) != prev_dir:
            # 方向变化：保留转角顶点 prev（旧方向的最后一个点），
            # 保证弦线完全贴合原路径、不切角
            if prev != kept[-1]:
                kept.append(prev)
            prev_dir = (dx, dy)
        lx, ly = kept[-1]
        sx = cur[0] - lx
        sy = cur[1] - ly
        if sx * sx + sy * sy >= limit_sq:
            kept.append(cur)
    if points[-1] != kept[-1]:
        kept.append(points[-1])
    return kept


def path_length(stroke: Sequence[tuple[int, int]]) -> float:
    """笔画路径总长度（相邻点线段距离之和），单点笔画为 0。"""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def stroke_reach(stroke: Sequence[tuple[int, int]]) -> float:
    """笔画的有效拖拽距离：按下点（首点）到路径上最远点的距离。

    小程序画布按「按下点到最远拖拽点」判定笔画是否有效（约需
    3~4px）。该值可能远小于路径总长：例如往返、绕圈或回折的笔画
    路径很长，但最远点始终离按下点很近，画布仍会判为无效。短笔
    判定必须使用本函数而不是 ``path_length``。
    """
    if not stroke:
        return 0.0
    x0, y0 = stroke[0]
    return max(math.hypot(x - x0, y - y0) for x, y in stroke)


def filter_strokes(
    strokes: list[list[tuple[int, int]]],
    min_length: float,
) -> list[list[tuple[int, int]]]:
    """过滤路径长度小于 min_length 的碎笔画（缩放产生的孤立点/微段）。

    min_length <= 0 时原样返回（不过滤）。单点笔画长度为 0，
    min_length > 0 时即被去除。
    """
    if min_length <= 0:
        return strokes
    return [s for s in strokes if path_length(s) >= min_length]

def trace_hand_drawn_paths(
    mask: np.ndarray,
    min_step: float = 2.0,
    jitter: float = 0.75,
    overshoot: float = 1.2,
    seed: int = 2024,
    pen_width: int = 1,
    fill_sensitivity: float = 65.0,
) -> list[list[tuple[int, int]]]:
    """手绘模拟：沿线条走向的中心线绘制 + 粗色块排线填涂。

    流程：按距离核 + 邻域上下文把掩膜分成「线条区域」（宽度 < 约
    9px）与「粗色块区域」——

    - 线条区域：骨架化为中心线，用贪心追踪器沿走向运笔。骨架在
      陡坡/缓坡圆弧上会留下大量 2~3px 的「梯格碎片」（追踪器只走
      了梯格的一侧）：先剔除被明显更长笔画覆盖的重复碎片，其余按
      原掩膜做端点桥接——只要两点之间的直线段大部分落在待画黑色
      区域内就合并成一笔，桥接段画在已涂黑处不留痕迹，弧线因此
      保持连续、孤立短笔画大幅减少；
    - 粗色块区域：按笔宽间距生成排线（牛耕式往返折线，像人涂阴影
      一样尽量不抬笔）。排线方向按色块主轴自适应：细长色块顺着
      长轴排线，团块随机取 32°~55° 斜向；行距与端点带轻微随机，
      避免机械平铺。色块外围约 4px 的边界环留在线条区域里，骨架化
      后自然成为轮廓线（先描边再涂黑，符合人类习惯）。

    两类笔画统一做弧长重采样 + 滑动平均平滑 + 人类化笔迹（低频
    漂移、中频摆动、高频轻颤、轻微弓形、起笔提前与收笔过冲）+
    距离抽稀。随机效果用固定种子驱动：同一掩膜每次结果完全一致
    （预览即实绘）。

    Args:
        mask: 二维 bool 数组，True 表示需要绘制的像素。
        min_step: 抽稀最小步长（像素），<=0 表示不抽稀。
        jitter: 手抖总体幅度（像素）：由低/中/高三档平滑噪声叠加，
            粗线更耐抖、短线自动减弱。
        overshoot: 端点过冲上限（像素）：进笔提前约 0~0.6 倍、
            收笔过冲约 0.4~1.0 倍（短线按长度衰减）。
        seed: 随机种子（固定值保证结果可复现）。
        pen_width: 画笔线宽（像素）：排线间距据此设定（留少量重叠
            保证涂满）。
        fill_sensitivity: 涂色灵敏度（0~100）：越大越多的粗笔画区域
            被按色块排线涂色（如眯眼式的粗黑眼睛笔画），越小越严格
            （只有大面积实心色块才涂色，线条交角/重叠一律按线条画）。

    Returns:
        笔画列表，每笔为 [(x, y), ...]（屏幕坐标系，x 在前）。
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask 必须是二维数组")
    if not m.any():
        return []
    rng = np.random.default_rng(seed)
    strokes: list[list[tuple[int, int]]] = []

    def _finalize(
        pts: list[tuple[float, float]],
        jitter_scale: float = 1.0,
        overshoot_scale: float = 1.0,
        window: int | None = None,
    ) -> None:
        pts = _resample_polyline(pts, spacing=1.0)
        # 长笔画用更宽的窗口消除像素梯格；短线用窄窗口保留尖端细节
        if window is None:
            window = 7 if len(pts) >= 12 else 5
        passes = 2 if window >= 5 else 1
        pts = _smooth_polyline(pts, window=window, passes=passes)
        pts = _humanize(
            pts,
            rng,
            jitter * jitter_scale,
            overshoot * overshoot_scale,
        )
        pts = _thin_polyline(pts, min_step)
        strokes.append([(int(round(x)), int(round(y))) for x, y in pts])

    # ① 线条区域：骨架中心线沿走向运笔，再按原掩膜把碎片接成长笔
    blob = _blob_region(
        m, _fill_ratio_from_sensitivity(fill_sensitivity)
    )
    line_region = m & ~blob
    if line_region.any():
        skel = _skeletonize(line_region)
        if skel.any():
            segs = _connect_strokes_through_mask(
                trace_paths(skel, min_step=0.0), line_region
            )
            for seg in segs:
                _finalize(seg)

    # ② 粗色块区域：人类化排线填涂（先描边后涂黑，符合人类习惯）
    if blob.any():
        step = max(1, int(round(pen_width - 0.3)))
        for chain in _hatch_strokes(blob, step, rng):
            # 排线已很密，手抖收窄、平滑窗口放小，以免行间抖动/转折
            # 圆角把涂黑区域推出白缝
            _finalize(
                chain,
                jitter_scale=0.3,
                overshoot_scale=0.3,
                window=3,
            )

    return [s for s in strokes if s]


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """串行简单点细化：把线条区域收缩为 1~2px 中心线。

    按 3-4 chamfer 距离（到背景）升序逐像素删除「简单点」——8 邻域
    环的 Rutovitz 交叉数为 1（删除不改变局部连通性）。两条保护规则：

    - 端点保留：存活邻居数 < 2 的像素不删（线条不缩短消失）；
    - 中轴保护：不存在距离更深的存活邻居时不删（等距像素互相拆解
      的「合法塌缩」在数学上被阻断——斜线/圆弧只剥到等距梯格为止）。

    相比经典并行算法（Zhang-Suen 会把完美 45° 斜线整体腐蚀掉），
    串行 + 距离序对任意角度稳健；偶数宽度线条留下 2px 梯格，由
    _segment_skeleton 的真交叉判定按路径处理。
    """
    m = mask.copy()
    if not m.any():
        return m
    h, w = m.shape
    d = _chamfer_distance(m)

    ys, xs = np.nonzero(m)
    order = np.lexsort((ys, xs, d[ys, xs]))
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if not m[y, x]:
            continue
        ring: list[int] = []
        deeper = False
        for dy, dx in ((-1, 0), (-1, 1), (0, 1), (1, 1),
                       (1, 0), (1, -1), (0, -1), (-1, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and m[ny, nx]:
                ring.append(1)
                if d[ny, nx] > d[y, x]:
                    deeper = True
            else:
                ring.append(0)
        if sum(ring) < 2 or not deeper:
            continue
        cross = sum(
            1 for i in range(8) if ring[i] == 0 and ring[(i + 1) % 8] == 1
        )
        if cross == 1:
            m[y, x] = False
    return m




def _chamfer_distance(mask: np.ndarray) -> np.ndarray:
    """3-4 chamfer 距离变换：每个前景像素到背景的距离（背景为 0）。

    轴向步代价 3、对角步代价 4，约为欧氏距离的 3.5 倍。采用标准两遍
    扫描：正向（从上到下 + 行内从左到右）与反向（从下到上 + 行内从
    右到左）各一次，每行用 ``np.minimum.accumulate`` 做一维 chamfer
    传播。总代价 O(W·H) 且与色块厚度无关；旧实现逐轮向 8 邻域传播
    1px，厚色块要迭代上百轮，是手绘模式的主要瓶颈。两遍扫描的结果
    与旧实现逐位一致（越界按「非背景」处理，整块前景保持 inf）。
    """
    h, w = mask.shape
    inf = 1 << 20
    d = np.where(mask, inf, 0).astype(np.int32)
    idx = np.arange(w, dtype=np.int32)
    cost_x = 3 * idx  # 行内一维传播的左/右代价

    # ① 正向：先合并上一行的三个邻域，再在行内向左传播
    for y in range(h):
        row = d[y]
        if y > 0:
            up = d[y - 1]
            row = np.minimum(row, up + 3)
            left = np.empty_like(up)
            left[0] = inf
            left[1:] = up[:-1]
            right = np.empty_like(up)
            right[-1] = inf
            right[:-1] = up[1:]
            row = np.minimum(row, left + 4)
            row = np.minimum(row, right + 4)
        d[y] = np.minimum.accumulate(row - cost_x) + cost_x

    # ② 反向：先合并下一行的三个邻域，再在行内向右传播
    for y in range(h - 1, -1, -1):
        row = d[y]
        if y < h - 1:
            down = d[y + 1]
            row = np.minimum(row, down + 3)
            left = np.empty_like(down)
            left[0] = inf
            left[1:] = down[:-1]
            right = np.empty_like(down)
            right[-1] = inf
            right[:-1] = down[1:]
            row = np.minimum(row, left + 4)
            row = np.minimum(row, right + 4)
        vals = row + cost_x
        d[y] = np.minimum.accumulate(vals[::-1])[::-1] - cost_x
    return d


#: 色块判定：chamfer 距离达到该值的像素构成「色块核」（约对应
#: ≥9px 宽；更细的结构一律按线条骨架运笔）
_FILL_MIN_CHAMFER = 15
#: 色块核膨胀回原尺寸的迭代轮数（8 邻域，每轮 1px）
_FILL_DILATE_ROUNDS = 4
#: 色块核连通分量的最小面积（过滤发尖角这类零星加厚）
_FILL_MIN_CORE_AREA = 16
#: 上下文判定的包围盒外扩半径：真色块的核心在其邻域掩膜中占主导；
#: 线条交角/双线重叠处的加厚核心周围线条像素占比高
_FILL_CONTEXT_RADIUS = 16
#: 涂色灵敏度（0~100，默认 65）→ 邻域核心占比阈值的映射：
#: 阈值 = 0.5 − 0.45·灵敏度/100，即 0 → 0.5（最严格，只有大面积
#: 色块才涂）、100 → 0.05（最宽松，宽度达标即涂）。实测「眯眼式」
#: 粗笔画弧占比约 0.20~0.25、双线重叠约 0.19，两者接近——无法用
#: 固定阈值同时满足，故开放为用户可调。
_FILL_SENS_DEFAULT = 65.0


def _fill_ratio_from_sensitivity(sensitivity: float) -> float:
    """涂色灵敏度（0~100）映射为邻域核心占比阈值（0.05~0.5，反比）。"""
    s = min(100.0, max(0.0, float(sensitivity)))
    return 0.5 - 0.45 * s / 100.0


def _blob_region(
    mask: np.ndarray, ratio_threshold: float | None = None
) -> np.ndarray:
    """提取「粗色块」区域（需要排线填涂的部分）。

    距离达到 _FILL_MIN_CHAMFER 的像素构成色块核（= 按半径腐蚀，
    细线条整条消失）。对每个核连通分量再做两道上下文判定，剔除
    「局部加厚」的假色块——真色块的核心在其邻域掩膜中占主导面积，
    而发尖角内侧、两条线短距离重叠等加厚处的核心面积小且周围
    线条像素占绝大多数，应按线条运笔：

    - 核面积 ≥ _FILL_MIN_CORE_AREA；
    - 邻域包围盒内（核面积 / 掩膜像素数）≥ _FILL_CONTEXT_RATIO
      （占比按包围盒内全部核像素计，避免大色块的碎核被误杀）。

    通过判定的核膨胀回原尺寸后与原掩膜相交；返回区域与线条区域
    （mask 减本区域）互补。
    """
    d = _chamfer_distance(mask)
    core = d >= _FILL_MIN_CHAMFER
    if not core.any():
        return np.zeros_like(mask)

    if ratio_threshold is None:
        ratio_threshold = _fill_ratio_from_sensitivity(_FILL_SENS_DEFAULT)
    h, w = mask.shape
    kept = np.zeros((h, w), dtype=bool)
    for ys, xs in _core_components(core):
        area = len(xs)
        if area < _FILL_MIN_CORE_AREA:
            continue
        y0 = max(0, int(ys.min()) - _FILL_CONTEXT_RADIUS)
        y1 = min(h, int(ys.max()) + 1 + _FILL_CONTEXT_RADIUS)
        x0 = max(0, int(xs.min()) - _FILL_CONTEXT_RADIUS)
        x1 = min(w, int(xs.max()) + 1 + _FILL_CONTEXT_RADIUS)
        box_mask = int(mask[y0:y1, x0:x1].sum())
        if box_mask <= 0:
            continue
        box_core = int(core[y0:y1, x0:x1].sum())
        if box_core / box_mask < ratio_threshold:
            continue  # 邻域内线条占主导：交角/重叠的局部加厚，按线条画
        kept[ys, xs] = True
    if not kept.any():
        return np.zeros_like(mask)

    grown = kept
    for _ in range(_FILL_DILATE_ROUNDS):
        nxt = grown.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx or dy:
                    shifted = np.zeros((h, w), dtype=bool)
                    ys0, ys1 = max(0, -dy), min(h, h - dy)
                    xs0, xs1 = max(0, -dx), min(w, w - dx)
                    shifted[ys0:ys1, xs0:xs1] = grown[
                        ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx
                    ]
                    nxt |= shifted
        grown = nxt
    return mask & grown


def _core_components(core: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """8 连通分量标记（返回每组的 (ys, xs) 坐标数组）。

    把每行的连续前景像素视为一个「行程」节点；相邻两行的行程在 x 方向
    交叠、或恰好相距 1px（8 连通）时用并查集合并。最后按行程首次出现
    的顺序输出分量。节点数约为「行数 × 每行段数」而非像素数，实心色块
    上比逐像素 BFS 少一个数量级；分组结果与 BFS 完全一致。
    """
    m = np.asarray(core, dtype=bool)
    h, w = m.shape
    if h == 0 or w == 0 or not m.any():
        return []

    parent: list[int] = []
    rank_size: list[int] = []
    run_row: list[int] = []
    run_s: list[int] = []
    run_e: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # 路径减半
            i = parent[i]
        return i

    prev: list[int] = []  # 上一行的行程 id（按起点升序）
    for y in range(h):
        row = m[y]
        if not row.any():
            prev = []
            continue
        padded = np.zeros(w + 2, dtype=bool)
        padded[1:-1] = row
        starts = np.flatnonzero(~padded[:-1] & padded[1:]).tolist()
        ends = (np.flatnonzero(padded[:-1] & ~padded[1:]) - 1).tolist()
        cur: list[int] = []
        j = 0
        for s, e in zip(starts, ends):
            rid = len(parent)
            parent.append(rid)
            rank_size.append(1)
            run_row.append(y)
            run_s.append(s)
            run_e.append(e)
            cur.append(rid)
            # 上一行中与 [s-1, e+1] 相交的行程均与该行程 8 连通
            while j < len(prev) and run_e[prev[j]] < s - 1:
                j += 1
            k = j
            root = find(rid)
            while k < len(prev) and run_s[prev[k]] <= e + 1:
                other = find(prev[k])
                if root != other:
                    if rank_size[root] < rank_size[other]:
                        root, other = other, root
                    parent[other] = root
                    rank_size[root] += rank_size[other]
                k += 1
        prev = cur

    # 行程 id 按行优先递增，dict 插入顺序即分量的首次出现顺序
    groups: dict[int, list[int]] = {}
    for rid in range(len(parent)):
        groups.setdefault(find(rid), []).append(rid)

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for run_ids in groups.values():
        ys: list[np.ndarray] = []
        xs: list[np.ndarray] = []
        for rid in run_ids:
            y = run_row[rid]
            s = run_s[rid]
            e = run_e[rid]
            ys.append(np.full(e - s + 1, y, dtype=int))
            xs.append(np.arange(s, e + 1, dtype=int))
        out.append((np.concatenate(ys), np.concatenate(xs)))
    return out


def _outward_dir(
    stroke: Sequence[tuple[float, float]],
    at_start: bool,
    depth: int = 4,
) -> tuple[float, float] | None:
    """笔画端点处指向笔迹外侧的单位方向（短笔/单点返回 None）。"""
    if len(stroke) < 2:
        return None
    if at_start:
        k = min(depth, len(stroke) - 1)
        dx = stroke[0][0] - stroke[k][0]
        dy = stroke[0][1] - stroke[k][1]
    else:
        k = max(0, len(stroke) - 1 - depth)
        dx = stroke[-1][0] - stroke[k][0]
        dy = stroke[-1][1] - stroke[k][1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return None
    return (dx / norm, dy / norm)


def _connect_strokes_through_mask(
    strokes: list[list[tuple[int, int]]],
    mask: np.ndarray,
    max_gap: float = 6.0,
    min_inside: float = 0.65,
) -> list[list[tuple[int, int]]]:
    """按原掩膜把相邻笔画的端点桥接成长笔（减少弧线碎笔）。

    8 邻域追踪器在陡坡/缓坡圆弧上会只走骨架梯格的一侧，留下大量
    2~3px 的独立短笔画。这些像素仍是原线条的一部分，不能直接丢弃：
    本函数在端点间寻找不超过 max_gap 的配对，仅当两点连线大部分落在
    ``mask``（待画黑色区域）内时才连接——桥接段画在已涂黑处，视觉上
    不可见，却能保证弧线不断开。背景上的缺口不会被桥接，因此相邻的
    平行线/独立线条不会被误连。

    单点笔画作为零长度端点参与桥接（骨架梯格残留的孤立像素会
    被吸收进相邻长笔，不再单独成笔）。
    返回笔画列表，每笔为 [(x, y), ...]。
    """
    if max_gap <= 0:
        return [list(s) for s in strokes]
    pool = [list(s) for s in strokes if s]
    if len(pool) <= 1:
        return pool
    lengths = [path_length(s) for s in pool]

    # 先剔除「贴在明显更长的笔画旁边」的重复碎片：它们是骨架梯格的
    # 另一侧，本就会被长笔的笔宽覆盖；若参与桥接，追踪会沿这条并行
    # 碎片链反向走一遍，白白重描整条线。判定很保守：点必须落在长笔
    # 中心线 1.5px 内，且长笔至少是它的两倍长，避免误删真正的细节。
    anchor_idx = [i for i, n in enumerate(lengths) if n >= 12.0]
    if anchor_idx:
        near_cell = 1.5
        point_grid: dict[
            tuple[int, int], list[tuple[int, float, float]]
        ] = {}
        for i in anchor_idx:
            for x, y in pool[i]:
                point_grid.setdefault(
                    (int(x // near_cell), int(y // near_cell)), []
                ).append((i, float(x), float(y)))
        kept: list[list[tuple[int, int]]] = []
        max_len = max(lengths)
        for i, s in enumerate(pool):
            if lengths[i] >= max_len:
                kept.append(s)  # 最长笔不会被任何笔覆盖
                continue
            covered = True
            for x, y in s:
                gx, gy = int(x // near_cell), int(y // near_cell)
                best2 = float("inf")
                for ax in (gx - 1, gx, gx + 1):
                    for ay in (gy - 1, gy, gy + 1):
                        for j, px, py in point_grid.get((ax, ay), ()):
                            if lengths[j] < max(12.0, 2.0 * lengths[i]):
                                continue  # 只允许被明显更长的笔画覆盖
                            d2 = (px - x) ** 2 + (py - y) ** 2
                            if d2 < best2:
                                best2 = d2
                if best2 > near_cell * near_cell:
                    covered = False
                    break
            if not covered:
                kept.append(s)
        pool = kept
        lengths = [path_length(s) for s in pool]
        if len(pool) <= 1:
            return pool

    consumed = [False] * len(pool)
    cell = max(1.0, float(max_gap))
    index: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for i in range(len(pool)):
        for at_start, p in ((True, pool[i][0]), (False, pool[i][-1])):
            index.setdefault(
                (int(p[0] // cell), int(p[1] // cell)), []
            ).append((i, at_start))

    height, width = mask.shape

    def _inside_ratio(
        a: tuple[float, float], b: tuple[float, float]
    ) -> float:
        """线段 ab 落在 mask 内的采样比例（沿路径 1.5px 采样）。"""
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(math.ceil(dist * 1.5)))
        hit = 0
        for k in range(steps + 1):
            t = k / steps
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                hit += 1
        return hit / (steps + 1)

    def _bridge(
        a: tuple[float, float], b: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """a→b 的 1px 插值点（不含端点，供平滑使用）。"""
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(math.ceil(dist)))
        return [
            (a[0] + (b[0] - a[0]) * k / steps,
             a[1] + (b[1] - a[1]) * k / steps)
            for k in range(1, steps + 1)
        ]

    def _find(end: tuple[float, float]) -> tuple[int, bool] | None:
        """在 end 附近找最近的、可桥接的未消费笔画端点。"""
        gx, gy = int(end[0] // cell), int(end[1] // cell)
        best: tuple[int, bool] | None = None
        best_d = float("inf")
        for ax in (gx - 1, gx, gx + 1):
            for ay in (gy - 1, gy, gy + 1):
                for idx, at_start in index.get((ax, ay), ()):
                    if consumed[idx]:
                        continue
                    p = pool[idx][0] if at_start else pool[idx][-1]
                    d = math.hypot(p[0] - end[0], p[1] - end[1])
                    if d > max_gap or d >= best_d:
                        continue
                    if d > 0.0 and _inside_ratio(end, p) < min_inside:
                        continue  # 连线穿过背景：这是两条不同的线
                    if d > 0.6 and lengths[idx] >= 8.0:
                        # 长笔段的走向检查：若对方从连接点向回折（端点
                        # 指纹朝连接方向），接上会沿着已画线重复描一遍，
                        # 拒绝；垂直或顺向的连接允许。
                        out = _outward_dir(pool[idx], at_start)
                        if out is not None:
                            ux, uy = (p[0] - end[0]) / d, (p[1] - end[1]) / d
                            if out[0] * ux + out[1] * uy > 0.2:
                                continue
                    best = (idx, at_start)
                    best_d = d
        return best

    out: list[list[tuple[float, float]]] = []
    for i0 in range(len(pool)):
        if consumed[i0]:
            continue
        consumed[i0] = True
        chain: list[tuple[float, float]] = [
            (float(x), float(y)) for x, y in pool[i0]
        ]
        while True:  # 向链尾延伸
            hit = _find(chain[-1])
            if hit is None:
                break
            idx, at_start = hit
            consumed[idx] = True
            piece = pool[idx] if at_start else list(reversed(pool[idx]))
            piece = [(float(x), float(y)) for x, y in piece]
            chain.extend(_bridge(chain[-1], piece[0]))
            chain.extend(piece)
        while True:  # 向链首延伸
            hit = _find(chain[0])
            if hit is None:
                break
            idx, at_start = hit
            consumed[idx] = True
            # 近端是对方起点时反转，使其终点与链首相接
            piece = pool[idx] if not at_start else list(reversed(pool[idx]))
            piece = [(float(x), float(y)) for x, y in piece]
            chain = piece + _bridge(piece[-1], chain[0]) + chain
        out.append(chain)

    return [[(int(round(x)), int(round(y))) for x, y in s] for s in out]


def _hatch_strokes(
    region: np.ndarray,
    step: int,
    rng: np.random.Generator | None = None,
) -> list[list[tuple[float, float]]]:
    """把色块区域转为人类化排线（连续牛耕往返 + 斜向笔触）。

    与机械的逐行水平扫描不同：

    - 排线方向按色块主轴自适应：细长色块的笔触顺着长轴走，团块
      随机取 32°~55° 的斜向（左右方向由种子决定）；
    - 行距、端点位置带轻微随机，避免完美平行的「打印机」质感；
    - 相邻行首尾相接成一笔，接不上的地方才抬笔另起——像人涂阴影
      一样尽量不抬笔，返回的折线在接笔处为斜线，画在已涂区域上
      不留痕迹。

    每个 8 连通分量独立排线，避免不同色块之间连出多余长线。
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    spacing = max(1.0, float(step))
    chains: list[list[tuple[float, float]]] = []
    for ys, xs in _core_components(region):
        if len(xs) < 4:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(centered)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        minor = max(float(np.min(evals)), 1e-9)
        aspect = math.sqrt(max(float(np.max(evals)), 1e-9) / minor)
        if aspect >= 2.4:
            angle = math.atan2(major[1], major[0])  # 顺着长轴排线
        else:
            angle = math.radians(
                rng.choice([-1.0, 1.0]) * rng.uniform(32.0, 55.0)
            )
        ca, sa = math.cos(angle), math.sin(angle)
        # t 为行进方向（垂直于笔触），s 为笔触方向
        t = -sa * xs + ca * ys
        s = ca * xs + sa * ys
        t_min, t_max = float(t.min()), float(t.max())

        centers: list[float] = []
        tcur = t_min + rng.uniform(0.15, 0.6) * spacing
        while tcur <= t_max + 0.5 * spacing:
            centers.append(tcur)
            # 行距压到笔宽的 0.55~0.75：留出足够重叠，手抖也不会出白缝
            tcur += max(1.0, spacing * rng.uniform(0.55, 0.75))
        if not centers:
            centers = [float(t.mean())]
        centers_arr = np.asarray(centers)
        # 按最近行心分组，O(n log rows)；不可用 n×rows 广播（大色块会爆内存）
        idx = (
            np.searchsorted((centers_arr[:-1] + centers_arr[1:]) / 2.0, t)
            if len(centers_arr) > 1
            else np.zeros(len(t), dtype=int)
        )

        def to_xy(ss: float, tt: float) -> tuple[float, float]:
            return (ca * ss - sa * tt, sa * ss + ca * tt)

        comp_rows: list[tuple[float, list[tuple[float, float]]]] = []
        for ri, tc in enumerate(centers):
            sv = np.sort(s[idx == ri])
            if len(sv) == 0:
                continue
            gaps = np.nonzero(np.diff(sv) > max(1.5, spacing * 0.9))[0]
            starts = np.concatenate(([0], gaps + 1))
            ends = np.concatenate((gaps, [len(sv) - 1]))
            runs = [
                (float(sv[a]), float(sv[b]))
                for a, b in zip(starts, ends)
                if sv[b] - sv[a] >= 0.5
            ]
            if runs:
                comp_rows.append((tc, runs))

        if not comp_rows:
            continue
        current: list[tuple[float, float]] = []
        last_end: tuple[float, float] | None = None
        for k, (tc, runs) in enumerate(comp_rows):
            if k % 2 == 1:
                runs.reverse()  # 奇数行反向，形成连续往返
            for s0, s1 in runs:
                # 端点轻微外扩：模拟收笔不出边界又不过分齐整
                s0 -= rng.uniform(0.2, 0.7)
                s1 += rng.uniform(0.2, 0.7)
                if k % 2 == 0:
                    a, b = to_xy(s0, tc), to_xy(s1, tc)
                else:
                    a, b = to_xy(s1, tc), to_xy(s0, tc)
                if last_end is None or math.hypot(
                    a[0] - last_end[0], a[1] - last_end[1]
                ) > 2.4 * spacing + 3.0:
                    if current:
                        chains.append(current)
                    current = [a, b]
                else:
                    current.extend([a, b])
                last_end = b
        if current:
            chains.append(current)
    return chains


def _resample_polyline(
    points: Sequence[tuple[float, float]],
    spacing: float,
) -> list[tuple[float, float]]:
    """沿折线按弧长以 spacing 间距均匀重采样（首尾点保留）。"""
    if len(points) < 2:
        return [(float(p[0]), float(p[1])) for p in points]
    out = [(float(points[0][0]), float(points[0][1]))]
    carry = 0.0  # 段起点距上一个已输出采样点的弧长
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        d = spacing - carry
        while d <= seg:
            out.append((x0 + ux * d, y0 + uy * d))
            d += spacing
        carry = (carry + seg) % spacing
    out.append((float(points[-1][0]), float(points[-1][1])))
    return out


def _smooth_polyline(
    points: list[tuple[float, float]],
    window: int = 5,
    passes: int = 2,
) -> list[tuple[float, float]]:
    """滑动平均平滑（端点保持不动），去掉像素级阶梯锯齿。"""
    if len(points) < 3 or window < 2:
        return list(points)
    half = window // 2
    out = list(points)
    for _ in range(passes):
        arr = np.asarray(out, dtype=np.float64)
        padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
        cum = np.cumsum(padded, axis=0)
        sums = cum[window - 1 :] - np.vstack(
            [np.zeros((1, 2)), cum[: -window]]
        )
        smoothed = sums / window
        smoothed[0] = arr[0]
        smoothed[-1] = arr[-1]
        out = [tuple(p) for p in smoothed]
    return out


def _smooth_noise(
    n: int,
    rng: np.random.Generator,
    wavelength: float,
    amplitude: float,
) -> np.ndarray:
    """一维低频平滑噪声：控制点随机 + 三次平滑插值。

    控制点间距由 ``wavelength``（采样点数）决定，三次 smoothstep
    插值保证一阶导连续，叠加后不会出现折线感。
    """
    if n <= 1 or amplitude == 0.0:
        return np.zeros(n)
    count = max(2, int(round((n - 1) / max(1.0, wavelength))) + 1)
    ctrl = rng.normal(0.0, 1.0, count)
    pos = np.linspace(0.0, count - 1.0, n)
    i = np.clip(np.floor(pos).astype(int), 0, count - 2)
    t = pos - i
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    return (ctrl[i] * (1.0 - t) + ctrl[i + 1] * t) * amplitude


def _humanize(
    points: list[tuple[float, float]],
    rng: np.random.Generator,
    jitter: float,
    overshoot: float,
) -> list[tuple[float, float]]:
    """给平滑后的中心线加人类笔迹特征。

    手抖不是单一白噪声，而是三档尺度叠加的平滑偏移：

    - 低频漂移（约 70px 波长）：整条线缓慢偏离中心；
    - 中频摆动（约 22px）：手腕调整造成的起伏；
    - 高频轻颤（约 7px）：笔尖的细微不稳。

    另加与笔迹长度成正比的轻微弓形（人习惯把长线画成微弧），以及
    起笔提前 / 收笔过冲。短线按长度衰减扰动，避免把 1~2px 的细节
    抖成噪点。
    """
    if len(points) < 3:
        # 点太少：只给孤立短线段做端点延伸
        if len(points) == 2 and overshoot > 0:
            arr = np.asarray(points, dtype=np.float64)
            t = arr[-1] - arr[0]
            nrm = math.hypot(*t)
            if nrm > 1e-9:
                scale = min(1.0, nrm / 10.0)
                arr[0] = arr[0] - t / nrm * overshoot * 0.5 * scale
                arr[-1] = arr[-1] + t / nrm * overshoot * 0.8 * scale
                return [tuple(p) for p in arr]
        return list(points)
    if jitter <= 0 and overshoot <= 0:
        return list(points)
    arr = np.asarray(points, dtype=np.float64)
    n = len(arr)
    seg = np.diff(arr, axis=0)
    length = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
    scale = min(1.0, length / 20.0)
    amp = max(0.0, jitter) * scale

    if amp > 0:
        tangents = np.gradient(arr, axis=0)
        norm = np.hypot(tangents[:, 0], tangents[:, 1])
        norm[norm < 1e-9] = 1.0
        normals = np.stack(
            [-tangents[:, 1] / norm, tangents[:, 0] / norm], axis=1
        )
        offsets = (
            0.55 * _smooth_noise(n, rng, 70.0, amp)
            + 0.35 * _smooth_noise(n, rng, 22.0, amp)
            + 0.15 * _smooth_noise(n, rng, 7.0, amp)
        )
        # 轻微弓形：两端为 0、中间最大，符号随机
        u = np.linspace(0.0, 1.0, n)
        offsets += (
            rng.uniform(-1.0, 1.0) * 0.35 * scale * 4.0 * u * (1.0 - u)
        )
        # 限制总偏移，避免长线的低频噪声累积成明显走样
        offsets = np.clip(offsets, -1.0, 1.0)
        arr = arr + normals * offsets[:, None]

    if overshoot > 0 and len(arr) >= 2:
        # 进笔提前（0~0.6 倍）与收笔过冲（0.4~1.0 倍），长线更明显
        t0 = arr[0] - arr[1]
        n0 = math.hypot(*t0)
        t1 = arr[-1] - arr[-2]
        n1 = math.hypot(*t1)
        if n0 > 1e-9:
            pre = t0 / n0 * overshoot * 0.6 * scale * rng.uniform(0.0, 1.0)
            arr[0] = arr[0] + pre
        if n1 > 1e-9:
            post = t1 / n1 * overshoot * 0.8 * scale * rng.uniform(0.4, 1.0)
            arr[-1] = arr[-1] + post
    return [tuple(p) for p in arr]


def _thin_polyline(
    points: list[tuple[float, float]],
    min_step: float,
) -> list[tuple[float, float]]:
    """按最小步长抽稀（首尾点保留；平滑后的曲线无需保留拐点）。"""
    if min_step <= 0 or len(points) <= 2:
        return list(points)
    limit_sq = min_step * min_step
    kept = [points[0]]
    for p in points[1:-1]:
        lx, ly = kept[-1]
        if (p[0] - lx) ** 2 + (p[1] - ly) ** 2 >= limit_sq:
            kept.append(p)
    if points[-1] != kept[-1]:
        kept.append(points[-1])
    return kept


def _band_offsets(radius: float) -> list[tuple[int, int]]:
    """笔刷覆盖带相对中心点的像素偏移（圆形，半径 radius）。"""
    r = int(math.ceil(radius))
    sq = (radius + 0.25) ** 2
    return [
        (dx, dy)
        for dx in range(-r, r + 1)
        for dy in range(-r, r + 1)
        if dx * dx + dy * dy <= sq
    ]


def _walk_points(stroke: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """沿笔画相邻点 1px 步进插值，返回全部路径像素。"""
    points: list[tuple[int, int]] = []
    for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
        steps = max(1, int(math.hypot(x1 - x0, y1 - y0)))
        for k in range(steps):
            t = k / steps
            points.append(
                (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
            )
    if stroke:
        points.append((int(stroke[-1][0]), int(stroke[-1][1])))
    return points


def deduplicate_strokes(
    strokes: list[list[tuple[int, int]]],
    pen_width: int,
    coverage_threshold: float = 0.8,
) -> list[list[tuple[int, int]]]:
    """按画笔粗细合并冗余笔画：中心线已被先前笔画的笔刷带覆盖的直接跳过。

    粗画笔会同时盖住邻近的细线（如密集排线区域），逐笔规划时把它们去掉
    以减少笔画数。保持原有笔画顺序（先出现者保留）。pen_width <= 1 时
    原样返回。

    Args:
        strokes: 笔画列表，每笔为 [(x, y), ...]。
        pen_width: 规划的画笔线宽（像素），>1 才启用去重。
        coverage_threshold: 笔画中心线被既有笔刷带覆盖的比例达到该值
            即判定为冗余并丢弃。

    Returns:
        去重后的笔画列表（可能是原列表本身或其子集）。
    """
    if pen_width <= 1 or not strokes:
        return strokes
    radius = (pen_width - 1) / 2.0
    band = _band_offsets(radius)
    margin = int(math.ceil(radius))
    min_x = min(x for s in strokes for x, _ in s) - margin
    min_y = min(y for s in strokes for _, y in s) - margin
    max_x = max(x for s in strokes for x, _ in s) + margin
    max_y = max(y for s in strokes for _, y in s) + margin
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    grid = np.zeros((height, width), dtype=bool)
    kept: list[list[tuple[int, int]]] = []
    for stroke in strokes:
        path_pts = _walk_points(stroke)
        total = len(path_pts)
        covered = sum(
            1
            for px, py in path_pts
            if grid[py - min_y, px - min_x]
        )
        if total > 0 and covered / total >= coverage_threshold:
            continue  # 中心线大半已被先前笔画盖住，冗余
        kept.append(stroke)
        for px, py in path_pts:
            for dx, dy in band:
                gx, gy = px + dx - min_x, py + dy - min_y
                if 0 <= gx < width and 0 <= gy < height:
                    grid[gy, gx] = True
    return kept


def _overshoot_score(
    start: tuple[int, int],
    dxy: tuple[float, float],
    min_length: float,
    mask: np.ndarray,
) -> float:
    """从 start 沿 dxy 到 min_length 的线段落在 mask 黑色内的比例。"""
    height, width = mask.shape
    steps = max(1, int(math.ceil(min_length)))
    hit = total = 0
    for k in range(1, steps + 1):
        t = k / steps
        x = int(round(start[0] + dxy[0] * min_length * t))
        y = int(round(start[1] + dxy[1] * min_length * t))
        total += 1
        if 0 <= x < width and 0 <= y < height and mask[y, x]:
            hit += 1
    return hit / total if total else 0.0


def _black_centroid_direction(
    start: tuple[int, int],
    mask: np.ndarray,
    radius: int,
) -> tuple[float, float] | None:
    """从 start 指向就近黑色区域质心的单位方向（窗口内无黑色则 None）。"""
    height, width = mask.shape
    x0 = max(0, int(start[0]) - radius)
    x1 = min(width, int(start[0]) + radius + 1)
    y0 = max(0, int(start[1]) - radius)
    y1 = min(height, int(start[1]) + radius + 1)
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    if len(xs) == 0:
        return None
    cx = float(xs.mean()) + x0 - start[0]
    cy = float(ys.mean()) + y0 - start[1]
    n = math.hypot(cx, cy)
    if n < 1e-6:
        return None
    return (cx / n, cy / n)


def extend_short_strokes(
    strokes: list[list[tuple[int, int]]],
    min_length: float,
    mask: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    """把有效拖拽距离不足的笔画延伸为足量位移，避免画布不落笔。

    小程序画布按「按下点到最远拖拽点」判定笔画是否有效（约需
    3~4px）。注意这不是路径总长：原路往返、绕圈、回折的笔画可能很
    长，但最远点始终离按下点很近，仍会丢笔。因此用 ``stroke_reach``
    判断，并对不足者沿最优方向**延伸**，使最远点距按下点达到
    ``min_length``。

    方向在「笔画正向/反向/垂直/就近黑色区域质心」中按延伸路径落入
    黑色区域的比例挑选——优先延伸进已涂黑区域，溢出尽量小。即使
    所有方向都在白色区域（孤立点/孤立碎线），也仍然延伸：回退成
    原地往返不会增加最远距离，等于放弃绘制；此时选择黑色占比最高
    的方向，可能留下很短的可视尾迹，可把「最短笔画」调大以过滤。

    Args:
        strokes: 笔画列表，每笔为 [(x, y), ...]。
        min_length: 按下点到最远拖拽点的最小距离（像素），<=0 原样返回。
        mask: 画布尺寸二值掩膜（True=待画的黑色像素），用于挑选延伸
            方向；None 时按笔画正方向直接延伸。

    Returns:
        新的笔画列表（已达标的笔画原样保留，不足者被替换为延伸路径）。
    """
    if min_length <= 0:
        return strokes
    out: list[list[tuple[int, int]]] = []
    for s in strokes:
        if stroke_reach(s) >= min_length:
            out.append(s)
            continue
        start = s[0]
        dirs: list[tuple[float, float]] = [
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        ]
        if len(s) >= 2:
            fx = s[-1][0] - start[0]
            fy = s[-1][1] - start[1]
            n = math.hypot(fx, fy)
            if n > 0:
                fx, fy = fx / n, fy / n
                dirs = [(fx, fy), (-fx, -fy), (-fy, fx), (fy, -fx)] + dirs
        if mask is not None:
            cd = _black_centroid_direction(start, mask, radius=40)
            if cd is not None:
                dirs.append(cd)

        end = s[-1]
        best: tuple[tuple[float, float], float] | None = None
        for dxy in dirs:
            score = (
                _overshoot_score(start, dxy, min_length, mask)
                if mask is not None
                else 1.0
            )
            if best is None or score > best[1]:
                best = (dxy, score)
        # 即使黑色占比为 0 也必须延伸；否则画布不会落笔。
        dxy, _score = best
        # 原路径照画，再从终点沿最优方向延伸到距起点 min_length 的目标点
        target = (start[0] + dxy[0] * min_length, start[1] + dxy[1] * min_length)
        pts = list(s)
        span = math.hypot(target[0] - end[0], target[1] - end[1])
        steps = max(1, int(math.ceil(span)))
        for k in range(1, steps + 1):
            t = k / steps
            pts.append(
                (
                    int(round(end[0] + (target[0] - end[0]) * t)),
                    int(round(end[1] + (target[1] - end[1]) * t)),
                )
            )
        # 斜向目标取整后可能比 min_length 略短（最多约 1px）；补一个
        # 沿同方向的点，保证最终有效拖拽距离真正达标
        if stroke_reach(pts) < min_length:
            extra = (
                int(round(start[0] + dxy[0] * (min_length + 1.0))),
                int(round(start[1] + dxy[1] * (min_length + 1.0))),
            )
            if extra != pts[-1]:
                pts.append(extra)
        out.append(pts)
    return out


def connect_nearby_strokes(
    strokes: list[list[tuple[int, int]]],
    max_gap: float = 3.0,
) -> list[list[tuple[int, int]]]:
    """把端点相距不超过 max_gap 的相邻笔画首尾相连成一笔。

    追踪器因贪心走不出去而拆开的笔画，其断点通常只差 1px；相连后可
    显著减少落笔（mouseDown/Up）次数。贪心链式合并：从任一笔画出发，
    反复把端点邻近的其他笔画接到链尾/链首（必要时反转对方），直到
    无法再接。max_gap <= 0 或笔画数 <= 1 时原样返回。

    Args:
        strokes: 笔画列表，每笔为 [(x, y), ...]。
        max_gap: 允许连接的最大端点距离（像素，默认 3）。

    Returns:
        合并后的笔画列表（可能少于输入数量，覆盖内容不变）。
    """
    if max_gap <= 0 or len(strokes) <= 1:
        return list(strokes)
    gap_sq = max_gap * max_gap
    pool = [list(s) for s in strokes if s]
    consumed = [False] * len(pool)
    cell = max(1.0, max_gap)
    index: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for i, s in enumerate(pool):
        for at_start, p in ((True, s[0]), (False, s[-1])):
            index.setdefault((int(p[0] // cell), int(p[1] // cell)), []).append(
                (i, at_start)
            )

    def _find(end: tuple[int, int]) -> tuple[int, bool] | None:
        """在 end 附近找最近的未消费笔画端点。"""
        cx, cy = int(end[0] // cell), int(end[1] // cell)
        best: tuple[int, bool] | None = None
        best_d = 0.0
        for gx in (cx - 1, cx, cx + 1):
            for gy in (cy - 1, cy, cy + 1):
                for idx, at_start in index.get((gx, gy), ()):
                    if consumed[idx]:
                        continue
                    p = pool[idx][0] if at_start else pool[idx][-1]
                    d = (p[0] - end[0]) ** 2 + (p[1] - end[1]) ** 2
                    if d <= gap_sq and (best is None or d < best_d):
                        best = (idx, at_start)
                        best_d = d
        return best

    out: list[list[tuple[int, int]]] = []
    for i0 in range(len(pool)):
        if consumed[i0]:
            continue
        consumed[i0] = True
        chain = pool[i0]
        while True:  # 向链尾延伸
            hit = _find(chain[-1])
            if hit is None:
                break
            idx, at_start = hit
            consumed[idx] = True
            other = pool[idx]
            chain.extend(other if at_start else list(reversed(other)))
        while True:  # 向链首延伸
            hit = _find(chain[0])
            if hit is None:
                break
            idx, at_start = hit
            consumed[idx] = True
            other = pool[idx]
            # 对方的近端点要与链首相接：近端是起点则反转变为其终点
            prefix = list(reversed(other)) if at_start else list(other)
            chain[:0] = prefix
        out.append(chain)
    return out
