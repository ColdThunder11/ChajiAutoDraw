"""tracing 模块单元测试。"""

import math

import numpy as np

from chaji_autodraw.core.tracing import (
    count_points,
    deduplicate_strokes,
    filter_strokes,
    path_length,
    trace_paths,
)


def _cross_mask(size: int = 9) -> np.ndarray:
    """构造十字形 mask（中间行 + 中间列全 True）。"""
    m = np.zeros((size, size), dtype=bool)
    c = size // 2
    m[c, :] = True
    m[:, c] = True
    return m


def _covered_points(strokes: list[list[tuple[int, int]]]) -> dict[tuple[int, int], int]:
    """统计每条笔画覆盖到的像素及次数。"""
    covered: dict[tuple[int, int], int] = {}
    for stroke in strokes:
        for x, y in stroke:
            covered[(x, y)] = covered.get((x, y), 0) + 1
    return covered


def test_cross_full_coverage() -> None:
    """十字形：所有 True 像素都出现在某条笔画中，且至少覆盖一次。"""
    mask = _cross_mask(9)
    true_pixels = {(int(x), int(y)) for y, x in zip(*np.nonzero(mask), strict=True)}
    assert len(true_pixels) == 17  # 9 + 9 - 1 中心重复

    strokes = trace_paths(mask, min_step=0.0)
    covered = _covered_points(strokes)

    assert set(covered) == true_pixels  # 覆盖完整性
    assert all(n >= 1 for n in covered.values())  # 至少一次
    # 不抽稀时每个像素恰好被访问一次
    assert count_points(strokes) == int(mask.sum()) == 17


def test_cross_greedy_strokes_cover_and_connected() -> None:
    """十字形：贪心不回溯，中心像素可能需要重新起笔，合法产出 1~4 笔。

    硬性要求：覆盖完整性（所有 True 像素至少出现在一条笔画中），
    且每笔相邻点在 8 邻域下连续。
    """
    mask = _cross_mask(9)
    true_pixels = {(int(x), int(y)) for y, x in zip(*np.nonzero(mask), strict=True)}

    strokes = trace_paths(mask, min_step=0.0)

    assert 1 <= len(strokes) <= 4  # 贪心不回溯，每条手臂可能需从端点重新起笔
    covered = _covered_points(strokes)
    assert set(covered) == true_pixels  # 覆盖完整性
    assert all(n >= 1 for n in covered.values())
    for stroke in strokes:
        for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
            assert max(abs(x1 - x0), abs(y1 - y0)) == 1  # 相邻点 8 邻域连续


def test_diagonal_single_stroke() -> None:
    """单条对角线：8 邻域连通，只产生 1 笔，且每个像素各到一次。"""
    mask = np.zeros((10, 10), dtype=bool)
    for i in range(10):
        mask[i, i] = True

    strokes = trace_paths(mask, min_step=0.0)

    assert len(strokes) == 1
    stroke = strokes[0]
    assert len(stroke) == 10
    assert stroke[0] == (0, 0)
    assert stroke[-1] == (9, 9)
    assert {p for p in stroke} == {(i, i) for i in range(10)}


def test_xy_order_is_screen_coords() -> None:
    """点使用 (x, y) 顺序：3×5 mask 最后一行横线，y 恒为 2。"""
    mask = np.zeros((3, 5), dtype=bool)  # shape = (高, 宽)
    mask[2, :] = True  # 最后一行

    strokes = trace_paths(mask, min_step=0.0)

    assert len(strokes) == 1
    stroke = strokes[0]
    assert [p[0] for p in stroke] == [0, 1, 2, 3, 4]  # x 从左到右
    assert all(p[1] == 2 for p in stroke)  # y 恒为 2（行号）


def test_min_step_simplify_keeps_endpoints() -> None:
    """min_step 抽稀：点数变少但首尾点保留。"""
    mask = np.zeros((1, 21), dtype=bool)
    mask[0, :] = True

    full = trace_paths(mask, min_step=0.0)
    assert len(full) == 1
    assert len(full[0]) == 21

    thinned = trace_paths(mask, min_step=5.0)
    assert len(thinned) == 1
    points = thinned[0]
    assert 2 <= len(points) <= 21
    assert points[0] == (0, 0)  # 首点保留
    assert points[-1] == (20, 0)  # 尾点保留
    # 抽稀后相邻保留点距离 >= min_step（首尾点除外时仍成立）
    for (x0, _), (x1, _) in zip(points, points[1:]):
        assert abs(x1 - x0) >= 5.0


def test_count_points() -> None:
    """count_points 统计所有笔画的总点数。"""
    strokes = [
        [(0, 0), (1, 0), (2, 0)],
        [(5, 5)],
    ]
    assert count_points(strokes) == 4


def test_empty_mask_returns_empty() -> None:
    """空 mask 返回空列表。"""
    strokes = trace_paths(np.zeros((5, 5), dtype=bool))
    assert strokes == []
    assert count_points(strokes) == 0


def test_path_length() -> None:
    """路径长度：3-4-5 直角三角形段长为 5；单点为 0。"""
    assert path_length([(0, 0), (3, 4)]) == 5.0
    assert path_length([(7, 7)]) == 0.0


def test_filter_strokes_drops_fragments() -> None:
    """过滤碎笔画：孤立点(0)与 1px 微段被去掉，长笔画保留。"""
    strokes = [
        [(5, 5)],                # 单点，长度 0
        [(0, 0), (1, 0)],        # 长度 1 的微段
        [(0, 0), (0, 10)],       # 长度 10 的正常笔画
    ]
    kept = filter_strokes(strokes, 2.0)
    assert kept == [[(0, 0), (0, 10)]]


def test_filter_strokes_zero_threshold_keeps_all() -> None:
    """min_length <= 0 时不过滤。"""
    strokes = [[(1, 1)], [(0, 0), (1, 0)]]
    assert filter_strokes(strokes, 0.0) == strokes


def test_deduplicate_merges_pen_covered_neighbor() -> None:
    """粗笔去重：相距 1px 的平行线被 3px 笔合并，相距 3px 的保留。"""
    a = [(0, 0), (10, 0)]
    b = [(0, 1), (10, 1)]
    c = [(0, 3), (10, 3)]
    kept = deduplicate_strokes([a, b, c], pen_width=3)
    assert kept == [a, c]


def test_deduplicate_noop_for_thin_pen() -> None:
    """pen_width <= 1 时原样返回。"""
    strokes = [[(0, 0), (5, 0)], [(0, 1), (5, 1)]]
    assert deduplicate_strokes(strokes, pen_width=1) is strokes
    assert deduplicate_strokes([], pen_width=5) == []


def test_deduplicate_partial_overlap_kept() -> None:
    """仅部分重叠（低于阈值）的笔画保留。"""
    a = [(0, 0), (20, 0)]
    b = [(10, 1), (30, 1)]  # 只有左半与 a 相邻
    kept = deduplicate_strokes([a, b], pen_width=3)
    assert kept == [a, b]


def test_simplify_keeps_corner_vertex() -> None:
    """转角顶点必须保留：L 形路径两段都短于 min_step 时不得弦切角。"""
    from chaji_autodraw.core.tracing import _simplify

    path = [(0, 0), (4, 0), (4, 4)]  # 两段长 4，min_step=5 均不达抽稀距离
    kept = _simplify(path, 5.0)
    assert (4, 0) in kept  # 转角点被保留（旧实现会丢失，弦切角留白）
    assert kept[0] == (0, 0) and kept[-1] == (4, 4)


def test_simplify_straight_run_still_thinned() -> None:
    """直线段内抽稀行为不变：保留点间距不低于 min_step。"""
    from chaji_autodraw.core.tracing import _simplify

    kept = _simplify([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], 2.0)
    for (x0, _), (x1, _) in zip(kept, kept[1:]):
        assert x1 - x0 >= 2.0


def test_extend_directional_into_black_mask() -> None:
    """纯黑掩膜下：3px 短笔画延伸后最远点距按下点 >= 6px，且覆盖原路径。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes

    mask = np.ones((100, 100), dtype=bool)
    out = extend_short_strokes([[(10, 10), (13, 10)]], 6.0, mask)
    assert len(out) == 1
    stroke = out[0]
    assert stroke[0] == (10, 10)  # 按下点不变
    farthest = max(math.hypot(x - 10, y - 10) for x, y in stroke)
    assert farthest >= 6.0
    assert (13, 10) in stroke  # 原笔画像素仍在路径上


def test_extend_single_point_with_mask() -> None:
    """纯黑掩膜下：单点延伸为距按下点 6px 的直线。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes

    mask = np.ones((100, 100), dtype=bool)
    out = extend_short_strokes([[(10, 10)]], 6.0, mask)
    assert len(out[0]) >= 2
    farthest = max(math.hypot(x - 10, y - 10) for x, y in out[0])
    assert 5.5 <= farthest <= 6.5


def test_extend_without_mask_follows_stroke_direction() -> None:
    """无掩膜（None）：沿笔画正方向延伸，不做黑色区域评估。"""
    from chaji_autodraw.core.tracing import extend_short_strokes

    out = extend_short_strokes([[(0, 0), (3, 0)]], 6.0, None)
    assert out == [[(0, 0), (3, 0), (4, 0), (5, 0), (6, 0)]]
    assert path_length(out[0]) == 6.0


def test_extend_fallback_when_no_black_nearby() -> None:
    """周围全白（无可延伸黑色区域）：回退为原地往返拖拽。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes

    mask = np.zeros((100, 100), dtype=bool)
    out = extend_short_strokes([[(50, 50), (53, 50)]], 6.0, mask)
    assert out == [[(50, 50), (53, 50), (50, 50)]]


def test_extend_short_strokes_zero_threshold_identity() -> None:
    """min_length <= 0 原样返回。"""
    from chaji_autodraw.core.tracing import extend_short_strokes

    strokes = [[(0, 0), (3, 0)], [(5, 5)]]
    assert extend_short_strokes(strokes, 0.0, None) is strokes


def test_connect_nearby_merges_1px_gap() -> None:
    """端点相距 1px 的两笔连接成一笔，顺序保持。"""
    from chaji_autodraw.core.tracing import connect_nearby_strokes

    out = connect_nearby_strokes([[(0, 0), (5, 0)], [(6, 0), (10, 0)]], 1.5)
    assert out == [[(0, 0), (5, 0), (6, 0), (10, 0)]]


def test_connect_nearby_reverses_second_stroke() -> None:
    """近端是对方终点时，反转对方后连接。"""
    from chaji_autodraw.core.tracing import connect_nearby_strokes

    out = connect_nearby_strokes([[(0, 0), (5, 0)], [(7, 1), (6, 1)]], 1.5)
    assert out == [[(0, 0), (5, 0), (6, 1), (7, 1)]]


def test_connect_nearby_gap_too_far() -> None:
    """端点距离超过阈值时不连接。"""
    from chaji_autodraw.core.tracing import connect_nearby_strokes

    strokes = [[(0, 0), (5, 0)], [(9, 0), (12, 0)]]
    assert connect_nearby_strokes(strokes, 1.5) == strokes


def test_connect_nearby_chain_of_three() -> None:
    """三条断笔链式合并为一笔。"""
    from chaji_autodraw.core.tracing import connect_nearby_strokes

    out = connect_nearby_strokes(
        [[(5, 0), (10, 0)], [(0, 0), (1, 0)], [(2, 0), (4, 0)]], 1.5
    )
    assert len(out) == 1
    assert out[0][0] == (0, 0) and out[0][-1] == (10, 0)
