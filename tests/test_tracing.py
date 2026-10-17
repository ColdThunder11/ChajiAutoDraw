"""tracing 模块单元测试。"""

import math

import numpy as np
from PIL import Image, ImageDraw

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


def test_stroke_reach_uses_farthest_point_from_press() -> None:
    """有效拖拽距离 = 首点到最远点；往返/绕圈不能用路径总长代替。"""
    from chaji_autodraw.core.tracing import path_length, stroke_reach

    assert stroke_reach([(0, 0)]) == 0.0
    assert stroke_reach([(0, 0), (3, 4)]) == 5.0
    back = [(0, 0), (3, 0), (0, 0)]
    assert path_length(back) == 6.0
    assert stroke_reach(back) == 3.0  # 路径够长但最远点不够


def test_extend_uses_reach_not_path_length() -> None:
    """路径总长已达标、但有效拖拽距离不足的笔画仍必须延伸。"""
    import numpy as np

    from chaji_autodraw.core.tracing import (
        extend_short_strokes,
        path_length,
        stroke_reach,
    )

    stroke = [(10, 10), (13, 10), (10, 10)]  # 路径 6px，最远仅 3px
    assert path_length(stroke) >= 6.0
    out = extend_short_strokes(
        [stroke], 6.0, np.zeros((100, 100), dtype=bool)
    )
    assert out[0][0] == (10, 10)  # 按下点不变
    assert stroke_reach(out[0]) >= 5.5  # 最远点被补到目标距离
    for point in stroke:
        assert point in out[0]  # 原路径仍保留


def test_extend_no_black_still_reaches_target() -> None:
    """周围全白时也必须延伸到达标位移，不能回退成原地往返。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes, stroke_reach

    mask = np.zeros((100, 100), dtype=bool)
    out = extend_short_strokes([[(50, 50), (53, 50)]], 6.0, mask)
    stroke = out[0]
    assert stroke[0] == (50, 50)
    assert stroke[-1] != (50, 50)  # 不是原地往返
    assert stroke_reach(stroke) >= 6.0


def test_extend_diagonal_rounding_still_reaches_target() -> None:
    """斜向端点取整后若略短，必须补点保证有效距离达到 min_length。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes, stroke_reach

    mask = np.zeros((30, 30), dtype=bool)
    out = extend_short_strokes([[(10, 10), (11, 11)]], 6.0, mask)
    assert stroke_reach(out[0]) >= 6.0


def test_extend_pipeline_all_strokes_reach_target() -> None:
    """混合短笔/单点/往返笔：延伸后全部达到最小有效拖拽距离。"""
    import numpy as np

    from chaji_autodraw.core.tracing import extend_short_strokes, stroke_reach

    strokes = [
        [(0, 0), (3, 0), (0, 0)],   # 往返
        [(10, 10)],                  # 单点
        [(20, 20), (23, 20)],        # 普通短笔
    ]
    out = extend_short_strokes(
        strokes, 6.0, np.zeros((60, 60), dtype=bool)
    )
    assert len(out) == len(strokes)
    assert all(stroke_reach(s) >= 5.5 for s in out)


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


def test_connect_nearby_default_gap_is_three() -> None:
    """函数默认连接距离为 3px：2px 间隙不传参也应合并。"""
    from chaji_autodraw.core.tracing import connect_nearby_strokes

    out = connect_nearby_strokes([[(0, 0), (5, 0)], [(7, 0), (10, 0)]])
    assert out == [[(0, 0), (5, 0), (7, 0), (10, 0)]]


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


# ---- 手绘模拟（骨架中心线 + 人类化笔迹）----

def test_hand_drawn_follows_line_direction() -> None:
    """粗竖条：笔画沿竖直中心线（x 聚焦在条中心 ±1.5px），而非描边。"""
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    mask = np.zeros((120, 120), dtype=bool)
    mask[10:110, 58:66] = True  # 8px 宽竖条，中心 x=62
    strokes = trace_hand_drawn_paths(mask, min_step=2.0)
    assert 1 <= len(strokes) <= 4
    xs = [p[0] for s in strokes for p in s]
    assert 58 <= min(xs) and max(xs) <= 65  # 聚焦中心线
    for s in strokes:
        ys = [p[1] for p in s]
        assert max(ys) - min(ys) > 30  # 沿竖直方向运笔


def test_hand_drawn_is_deterministic() -> None:
    """固定随机种子：同一掩膜两次调用结果完全一致（预览即实绘）。"""
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 38:42] = True
    mask[30:34, 10:70] = True
    a = trace_hand_drawn_paths(mask, min_step=2.0)
    b = trace_hand_drawn_paths(mask, min_step=2.0)
    assert a == b
    assert a  # 非空


def test_hand_drawn_ring_stays_walkable() -> None:
    """厚圆环：骨架保持完整可走（按应用链路=过滤+连接后笔画数有限）。"""
    from chaji_autodraw.core.tracing import (
        connect_nearby_strokes,
        filter_strokes,
        trace_hand_drawn_paths,
    )

    yy, xx = np.mgrid[:120, :120]
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    mask = (r2 > 36**2) & (r2 < 44**2)
    strokes = trace_hand_drawn_paths(mask, min_step=2.0)
    assert max(len(s) for s in strokes) >= 40  # 至少一段长弧
    # 应用实际绘制链路：碎片段被端点连接合并
    chained = connect_nearby_strokes(filter_strokes(strokes, 2.0), 1.5)
    assert len(chained) <= 25


def test_hand_drawn_blob_filled_with_hatch() -> None:
    """实心大块：连续牛耕排线铺满，笔画数很少且覆盖率足够。

    新排线按主轴/斜向连续往返（尽量不抬笔），不再是多条水平线；
    因此以「笔画数少 + 光栅化覆盖率」判断，而不是排线方向。
    """
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    mask = np.zeros((120, 120), dtype=bool)
    mask[20:100, 20:100] = True  # 80x80 实心色块
    strokes = trace_hand_drawn_paths(mask, min_step=2.0, pen_width=3)
    assert 1 <= len(strokes) <= 5  # 连续排线，不再拆成大量碎笔
    assert max(len(s) for s in strokes) >= 100

    img = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(img)
    for s in strokes:
        draw.line(
            [(float(x), float(y)) for x, y in s],
            fill=(0, 0, 0),
            width=3,
            joint="curve",
        )
    painted = np.asarray(img.convert("L")) < 128
    covered = (painted & mask).sum() / mask.sum()
    assert covered >= 0.95  # 排线重叠足够，涂黑基本没有白缝


def test_hand_drawn_empty_mask() -> None:
    """空掩膜返回空列表。"""
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    assert trace_hand_drawn_paths(np.zeros((10, 10), dtype=bool)) == []


def test_hand_drawn_diagonal_survives() -> None:
    """回归：经典并行细化（Zhang-Suen）会把 45° 斜线整体腐蚀掉；
    串行+中轴保护的实现必须保留斜线走向。"""
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    mask = np.zeros((100, 100), dtype=bool)
    for k in range(90):
        mask[k, k : k + 3] = True
    strokes = trace_hand_drawn_paths(mask, min_step=2.0)
    pts = [p for s in strokes for p in s]
    assert len(pts) > 40  # 斜线未塌缩


def test_hand_drawn_arc_is_continuous() -> None:
    """重点回归：陡/缓弧线都应画成连续长笔，不产生孤立短笔画。

    骨架在斜向圆弧上会留下大量 2~3px 梯格碎片；新流程用掩膜内
    桥接把这些碎片接回主笔。每种朝向下都要求：至少一笔超过 200px，
    少于 4px 的孤立碎笔为 0，整条弧最多 2 笔（骨架偶有并行双轨）。
    """
    from chaji_autodraw.core.tracing import trace_hand_drawn_paths

    for start, end in ((0, 120), (60, 180), (120, 240), (210, 330)):
        im = Image.new("L", (300, 300), 255)
        ImageDraw.Draw(im).arc(
            (40, 40, 260, 260), start, end, fill=0, width=5
        )
        mask = np.asarray(im) < 128
        strokes = trace_hand_drawn_paths(
            mask, min_step=2.0, pen_width=1
        )
        assert max(path_length(s) for s in strokes) > 200, (
            f"弧 {start}-{end} 未形成连续长笔"
        )
        assert not [
            s for s in strokes if path_length(s) < 4.0
        ], f"弧 {start}-{end} 仍有孤立短笔画"
        assert len(strokes) <= 2, f"弧 {start}-{end} 被拆成了多笔"


def test_connect_through_mask_respects_background() -> None:
    """端点桥接只走掩膜内部：背景缺口不连，补上黑色连接带后才连。"""
    from chaji_autodraw.core.tracing import _connect_strokes_through_mask

    strokes = [[(5, 17), (34, 17)], [(45, 17), (74, 17)]]
    mask = np.zeros((40, 80), dtype=bool)
    mask[15:20, 5:35] = True
    mask[15:20, 45:75] = True
    assert len(_connect_strokes_through_mask(strokes, mask, max_gap=12)) == 2
    mask[15:20, 35:45] = True  # 两段之间是原图黑色：可以桥接
    assert len(_connect_strokes_through_mask(strokes, mask, max_gap=12)) == 1


def test_blob_region_rejects_line_junction_thickening() -> None:
    """回归：发尖角/双线短重叠的局部加厚不再被误判为色块。

    旧判定只看局部宽度（≥9px），线条交角内侧与重叠段会被错误排线
    涂色；新判定加「核心面积 + 邻域占比」上下文——真色块的核心在
    邻域掩膜中占主导，线条结构周围线条像素占绝大多数。
    """
    from chaji_autodraw.core.tracing import _blob_region

    # 发尖角：两条 5px 线呈 V 形相交，内侧局部宽度超阈值
    corner = np.zeros((120, 120), dtype=bool)
    for k in range(90):
        y = 10 + k
        corner[y, 10 + k : 15 + k] = True
        corner[y, 105 - k : 110 - k] = True
    assert _blob_region(corner).sum() == 0

    # 双线短距离重叠：6px 竖线中段与另一条并拢成 12px 宽
    overlap = np.zeros((140, 140), dtype=bool)
    overlap[10:130, 30:36] = True
    overlap[50:90, 36:42] = True
    assert _blob_region(overlap).sum() == 0

    # 10px 宽长粗笔画：按线条运笔而非色块
    thick = np.zeros((140, 140), dtype=bool)
    thick[10:130, 60:70] = True
    assert _blob_region(thick).sum() == 0


def test_blob_region_still_detects_real_blobs() -> None:
    """真色块（含附着细线的场景）仍被正确检出且不卷入细线。"""
    from chaji_autodraw.core.tracing import _blob_region

    blob = np.zeros((120, 120), dtype=bool)
    blob[20:100, 20:100] = True
    assert _blob_region(blob).sum() > 5000

    mix = np.zeros((140, 140), dtype=bool)
    mix[40:100, 40:100] = True  # 60x60 色块
    mix[10:40, 66:74] = True    # 上方 8px 细线
    region = _blob_region(mix)
    assert region.sum() > 3000
    # 细线上段（远离交汇处）不被卷入；交汇处局部真实变宽，
    # 允许 ~5px 的过渡尾巴
    assert region[:30, :].sum() == 0
    assert region[:40, :].sum() < 100


def test_fill_sensitivity_controls_eye_stroke() -> None:
    """涂色灵敏度：默认 65 时「眯眼式」粗黑弧笔画被涂色（用户场景），
    双线重叠仍按线条；灵敏度拉满后重叠也涂、归零后眼睛也按线条。"""
    from chaji_autodraw.core.tracing import (
        _blob_region,
        _fill_ratio_from_sensitivity,
    )

    def disk(m, cx, cy, r):
        yy, xx = np.mgrid[: m.shape[0], : m.shape[1]]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = True

    eye = np.zeros((100, 100), dtype=bool)
    for k in range(40):
        a = np.pi * k / 39
        disk(eye, int(round(50 - 20 * np.cos(a))), int(round(50 - 8 * np.sin(a))), 6)
    overlap = np.zeros((140, 140), dtype=bool)
    overlap[10:130, 30:36] = True
    overlap[50:90, 36:42] = True

    th = _fill_ratio_from_sensitivity(65.0)
    assert _blob_region(eye, th).sum() > 0  # 默认：眯眼弧涂色
    assert _blob_region(overlap, th).sum() == 0  # 双线重叠按线条

    th_hi = _fill_ratio_from_sensitivity(100.0)
    assert _blob_region(overlap, th_hi).sum() > 0  # 拉满：重叠也涂

    th_lo = _fill_ratio_from_sensitivity(0.0)
    assert _blob_region(eye, th_lo).sum() == 0  # 归零：眼睛也按线条


# ---- 3-4 chamfer 距离变换（两遍扫描 vs 迭代参考实现）----

def _reference_chamfer(mask: np.ndarray) -> np.ndarray:
    """同步迭代的 3-4 chamfer（小掩膜专用，作为两遍扫描的对照）。"""
    h, w = mask.shape
    inf = 1 << 20
    d = np.where(mask, inf, 0).astype(np.int32)
    for _ in range(h * w + 2):
        prev = d.copy()
        md = d.copy()
        for dy, dx, cost in (
            (0, -1, 3), (0, 1, 3), (-1, 0, 3), (1, 0, 3),
            (-1, -1, 4), (-1, 1, 4), (1, -1, 4), (1, 1, 4),
        ):
            shifted = np.full_like(d, inf)
            ys0, ys1 = max(0, -dy), min(h, h - dy)
            xs0, xs1 = max(0, -dx), min(w, w - dx)
            shifted[ys0:ys1, xs0:xs1] = np.minimum(
                inf, d[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx] + cost
            )
            md = np.minimum(md, shifted)
        d = md
        if np.array_equal(d, prev):
            break
    return d


def test_chamfer_distance_matches_iterative_reference() -> None:
    """两遍扫描必须与旧迭代实现逐位一致（含高厚色块/边界/随机掩膜）。"""
    from chaji_autodraw.core.tracing import _chamfer_distance

    masks = [np.zeros((7, 9), dtype=bool), np.ones((7, 9), dtype=bool)]
    masks.append(np.eye(6, dtype=bool))
    single = np.zeros((9, 9), dtype=bool)
    single[4, 4] = True
    masks.append(single)
    block = np.zeros((9, 9), dtype=bool)
    block[2:7, 2:7] = True
    masks.append(block)
    border = np.zeros((9, 9), dtype=bool)
    border[:, 4] = True
    masks.append(border)

    rng = np.random.default_rng(2026)
    for _ in range(40):
        h = int(rng.integers(1, 25))
        w = int(rng.integers(1, 25))
        masks.append(rng.random((h, w)) < rng.uniform(0.05, 0.95))

    for mask in masks:
        got = _chamfer_distance(mask)
        want = _reference_chamfer(mask)
        assert got.dtype == np.int32
        assert got.shape == mask.shape
        assert np.array_equal(got, want)


def test_chamfer_distance_single_pixel_distance() -> None:
    """单个前景像素到最近背景的 3-4 chamfer 距离为 3。"""
    from chaji_autodraw.core.tracing import _chamfer_distance

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    distances = _chamfer_distance(mask)
    assert int(distances[2, 2]) == 3


# ---- 8 连通分量（行程 + 并查集 vs 逐像素 BFS）----

def _reference_components(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """逐像素 BFS 的 8 连通分量（小掩膜对照实现）。"""
    h, w = mask.shape
    remaining = mask.copy()
    out: list[tuple[np.ndarray, np.ndarray]] = []
    ys, xs = np.nonzero(remaining)
    for y0, x0 in zip(ys, xs):
        if not remaining[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        remaining[y0, x0] = False
        comp_ys: list[int] = []
        comp_xs: list[int] = []
        while stack:
            cy, cx = stack.pop()
            comp_ys.append(cy)
            comp_xs.append(cx)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and remaining[ny, nx]:
                        remaining[ny, nx] = False
                        stack.append((ny, nx))
        out.append((np.asarray(comp_ys), np.asarray(comp_xs)))
    return out


def test_components_match_bfs_reference() -> None:
    """行程并查集必须与逐像素 BFS 得到完全相同的 8 连通分组。"""
    from chaji_autodraw.core.tracing import _core_components

    masks = [np.zeros((6, 7), dtype=bool), np.ones((6, 7), dtype=bool)]
    single = np.zeros((5, 5), dtype=bool)
    single[2, 2] = True
    masks.append(single)
    diagonal = np.zeros((4, 4), dtype=bool)  # 仅对角相邻：8 连通算一组
    diagonal[0, 0] = True
    diagonal[1, 1] = True
    masks.append(diagonal)
    separated = np.zeros((8, 8), dtype=bool)
    separated[1:3, 1:3] = True
    separated[5:7, 5:7] = True
    masks.append(separated)
    border = np.zeros((7, 7), dtype=bool)
    border[:, 0] = True
    border[3, :] = True
    masks.append(border)

    rng = np.random.default_rng(2026)
    for _ in range(50):
        h = int(rng.integers(1, 28))
        w = int(rng.integers(1, 28))
        masks.append(rng.random((h, w)) < rng.uniform(0.05, 0.95))

    for mask in masks:
        got = _core_components(mask)
        want = _reference_components(mask)
        assert len(got) == len(want)
        for (gy, gx), (wy, wx) in zip(got, want):
            assert set(zip(gy.tolist(), gx.tolist())) == set(
                zip(wy.tolist(), wx.tolist())
            )
        if got:
            # 分量顺序保持扫描顺序：第一组包含最上（同行为最左）的像素
            fy, fx = np.argwhere(mask)[0]
            assert (int(fy), int(fx)) in set(
                zip(got[0][0].tolist(), got[0][1].tolist())
            )
