"""二值图案 → 连贯笔画路径追踪（8 邻域贪心 + 距离抽稀）。

纯 Python/numpy 实现，面向几百×几百像素规模：整体复杂度 O(W*H + N)，
不使用全局最近邻等 O(n²) 结构。
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


def _back_and_forth(s: list[tuple[int, int]], min_length: float) -> list[tuple[int, int]]:
    """原地往返拖拽（附近无可延伸黑色区域时的回退方案）。"""
    length = path_length(s)
    if len(s) < 2 or length <= 0:
        x, y = s[0]
        half = min_length / 2.0
        return [(int(round(x - half)), int(y)), (int(round(x + half)), int(y))]
    legs = max(2, math.ceil(min_length / length))
    result = list(s)
    for i in range(legs - 1):
        result.extend(reversed(s[:-1]) if i % 2 == 0 else s[1:])
    return result


def extend_short_strokes(
    strokes: list[list[tuple[int, int]]],
    min_length: float,
    mask: np.ndarray | None = None,
) -> list[list[tuple[int, int]]]:
    """（实验性）把短笔画延伸为足量位移的拖拽，避免画布不落笔。

    小程序画布按「按下点到最远拖拽点的距离」判定笔画是否有效（约需
    3~4px），往返拖拽无法提高最远距离。因此短笔画沿最优方向**延伸**，
    使最远点距按下点达到 min_length：方向在「笔画正向/反向/垂直/就近
    黑色区域质心」中按延伸路径落入黑色区域的比例挑选——溢出部分涂在
    已涂黑区域不留痕迹；附近没有黑色区域时回退为原地往返拖拽。

    Args:
        strokes: 笔画列表，每笔为 [(x, y), ...]。
        min_length: 按下点到最远拖拽点的最小距离（像素），<=0 原样返回。
        mask: 画布尺寸二值掩膜（True=待画的黑色像素），用于挑选延伸
            方向；None 时按笔画正方向直接延伸。

    Returns:
        新的笔画列表（长笔画原样保留，短笔画被替换为延伸路径）。
    """
    if min_length <= 0:
        return strokes
    out: list[list[tuple[int, int]]] = []
    for s in strokes:
        if path_length(s) >= min_length:
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
        dxy, score = best
        if mask is not None and score < 0.6:
            # 附近没有可延伸的黑色区域：回退原地往返
            out.append(_back_and_forth(s, min_length))
            continue
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
        out.append(pts)
    return out


def connect_nearby_strokes(
    strokes: list[list[tuple[int, int]]],
    max_gap: float = 1.5,
) -> list[list[tuple[int, int]]]:
    """（实验性）把端点相距不超过 max_gap 的相邻笔画首尾相连成一笔。

    追踪器因贪心走不出去而拆开的笔画，其断点通常只差 1px；相连后可
    显著减少落笔（mouseDown/Up）次数。贪心链式合并：从任一笔画出发，
    反复把端点邻近的其他笔画接到链尾/链首（必要时反转对方），直到
    无法再接。max_gap <= 0 或笔画数 <= 1 时原样返回。

    Args:
        strokes: 笔画列表，每笔为 [(x, y), ...]。
        max_gap: 允许连接的最大端点距离（像素）。

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
