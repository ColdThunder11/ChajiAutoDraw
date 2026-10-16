"""drawer 单元测试：插值函数与 Esc 急停逻辑。

回归背景：多点笔画曾被 zip(strict=True) 必然抛错；Esc 急停为
GetAsyncKeyState 轮询，测试中以 monkeypatch 模拟按键状态。
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from chaji_autodraw.core import drawer
from chaji_autodraw.core.drawer import DrawWorker, _resample


@pytest.fixture(scope="module")
def qapp() -> object:
    """提供模块级 QApplication（offscreen），供 QThread 对象创建。"""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _FakeBackend:
    """记录动作序列的假输入后端。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []

    def resolve(self) -> str:
        return "fake"

    def prepare(self) -> None:
        self.calls.append(("prepare", 0.0, 0.0))

    def activate(self) -> None:
        self.calls.append(("activate", 0.0, 0.0))

    def move(self, x: float, y: float, *, pressed: bool) -> None:
        self.calls.append(("move" if pressed else "hover", x, y))

    def press(self, x: float, y: float) -> None:
        self.calls.append(("press", x, y))

    def release(self, x: float, y: float) -> None:
        self.calls.append(("release", x, y))


def _run_worker(
    monkeypatch: pytest.MonkeyPatch,
    qapp: object,
    esc: bool,
    strokes: list[list[tuple[int, int]]],
) -> tuple[bool, str, _FakeBackend]:
    """同步执行 DrawWorker.run()（不开线程），返回 (成功?, 消息, 后端)。"""
    monkeypatch.setattr(drawer, "_esc_pressed", lambda: esc)
    backend = _FakeBackend()
    worker = DrawWorker(
        strokes, 0, 0, backend, step_px=5.0, delay_s=0.0, countdown_s=0.0
    )
    results: list[tuple[bool, str]] = []
    worker.done.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()
    assert results, "done 信号未发出"
    return results[-1][0], results[-1][1], backend


def test_resample_single_point_stroke() -> None:
    """孤立像素形成的 1 点笔画：原样返回且为浮点。"""
    assert _resample([(7, 9)], 3.0) == [(7.0, 9.0)]


def test_resample_two_point_segment_spacing() -> None:
    """两点线段按步长插值：首尾必达，相邻点距不超过步长。"""
    out = _resample([(0, 0), (9, 0)], 3.0)
    assert out[0] == (0.0, 0.0)
    assert out[-1] == (9.0, 0.0)
    for (xa, _), (xb, _) in zip(out, out[1:]):
        assert xb - xa <= 3.0 + 1e-9


def test_resample_keeps_all_vertices() -> None:
    """折线所有原始拐点都必须出现在插值结果中。"""
    stroke = [(0, 0), (10, 0), (10, 10), (0, 10)]
    out = _resample(stroke, 4.0)
    for vertex in stroke:
        assert (float(vertex[0]), float(vertex[1])) in out


def test_resample_short_segments_no_extra_points() -> None:
    """段长不足步长时不额外插点，拐点原样保留。"""
    out = _resample([(0, 0), (1, 0), (2, 0)], 5.0)
    assert out == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def test_resample_diagonal_spacing_bounded() -> None:
    """斜线段插值后相邻点欧氏距离不超过步长（容差 1px）。"""
    out = _resample([(0, 0), (30, 40)], 3.0)  # 段长 50
    assert len(out) == 1 + math.ceil(50 / 3.0)
    for (xa, ya), (xb, yb) in zip(out, out[1:]):
        assert math.hypot(xb - xa, yb - ya) <= 3.0 + 1e-9


def test_single_point_stroke_nudges_to_paint(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """孤立单点：按下后补一个 1px 按下移动，否则画布对纯点击不落笔。"""
    ok, _msg, backend = _run_worker(
        monkeypatch, qapp, esc=False, strokes=[[(5, 5)]]
    )
    assert ok is True
    seq = [(c[0], c[1], c[2]) for c in backend.calls]
    assert seq == [
        ("prepare", 0.0, 0.0),
        ("activate", 0.0, 0.0),
        ("hover", 5.0, 5.0),   # 未按下移动到落点
        ("press", 5.0, 5.0),
        ("move", 6.0, 5.0),    # 1px 按下移动（补笔点）
        ("release", 6.0, 5.0),
    ]


def test_normal_completion_without_esc(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """Esc 未按下：两笔正常画完，done(True)。"""
    strokes = [[(0, 0), (10, 0)], [(0, 5), (10, 5)]]
    ok, msg, backend = _run_worker(monkeypatch, qapp, esc=False, strokes=strokes)
    assert ok is True
    assert "2" in msg  # 共 2 笔
    names = [c[0] for c in backend.calls]
    assert names.count("press") == 2 and names.count("release") == 2
    assert names[-1] == "release"


def test_esc_aborts_during_drawing(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """Esc 在笔画中途按下：立即中止并报「Esc」消息；finally 保证抬起。"""
    # 轮询时序：第1次=笔画循环头，第2次起=每个移动点；第4次起 Esc 按下
    state = {"n": 0}

    def esc_mid_stroke() -> bool:
        state["n"] += 1
        return state["n"] >= 4

    monkeypatch.setattr(drawer, "_esc_pressed", esc_mid_stroke)
    backend = _FakeBackend()
    worker = DrawWorker(
        [[(0, 0), (20, 0)]], 0, 0, backend, step_px=5.0, delay_s=0.0, countdown_s=0.0
    )
    results: list[tuple[bool, str]] = []
    worker.done.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()
    ok, msg = results[-1]
    assert ok is False
    assert "Esc" in msg
    names = [c[0] for c in backend.calls]
    # 中途按下：press 已发生，抬起由 finally 兜底，不能落下按着的笔
    assert names.count("press") == 1
    assert names.count("release") == 1
    assert names[-1] == "release"
    assert names.index("press") < names.index("release")


def test_countdown_esc_aborts_before_any_action(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """倒计时阶段按 Esc：一个绘制动作都不发生。"""
    monkeypatch.setattr(drawer, "_esc_pressed", lambda: True)
    backend = _FakeBackend()
    worker = DrawWorker(
        [[(0, 0), (10, 0)]], 0, 0, backend, countdown_s=5.0, delay_s=0.0
    )
    results: list[tuple[bool, str]] = []
    worker.done.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()
    assert results[-1][0] is False
    assert "Esc" in results[-1][1]
    assert backend.calls == []  # 未执行任何绘制动作


def test_stroke_gap_sleeps_between_strokes(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """笔间停顿：两笔之间出现一次 0.05s 停顿，最后一笔后不停。"""
    sleeps: list[float] = []
    monkeypatch.setattr(drawer.time, "sleep", lambda s: sleeps.append(round(s, 4)))
    monkeypatch.setattr(drawer, "_esc_pressed", lambda: False)
    backend = _FakeBackend()
    worker = DrawWorker(
        [[(0, 0), (10, 0)], [(0, 5), (10, 5)]],
        0,
        0,
        backend,
        step_px=10.0,  # 每笔一个移动点，不产生额外延时
        delay_s=0.0,
        stroke_gap_s=0.05,
        countdown_s=0.0,
    )
    results: list[tuple[bool, str]] = []
    worker.done.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()
    assert results[-1][0] is True
    # 停顿 0.05s 以 0.02s 步进拆分（[0.02, 0.02, 0.01]），且只在两笔之间
    assert sum(sleeps) == pytest.approx(0.05)
    assert max(sleeps) <= 0.02


def test_eta_signal_and_elapsed_message(
    monkeypatch: pytest.MonkeyPatch, qapp: object
) -> None:
    """剩余时间信号按 0.5s 节流发出，完成消息带总用时。"""
    clock = {"t": 0.0}

    def fake_monotonic() -> float:
        clock["t"] += 0.05  # 每次调用推进 50ms
        return clock["t"]

    monkeypatch.setattr(drawer.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(drawer, "_esc_pressed", lambda: False)
    backend = _FakeBackend()
    # 单笔 40 点、步长 1 → 39 次移动
    stroke = [(i, 0) for i in range(40)]
    worker = drawer.DrawWorker(
        [stroke], 0, 0, backend, step_px=1.0, delay_s=0.0, countdown_s=0.0
    )
    remaining: list[int] = []
    results: list[tuple[bool, str]] = []
    worker.remaining.connect(lambda sec: remaining.append(sec))
    worker.done.connect(lambda ok, msg: results.append((ok, msg)))
    worker.run()

    assert results[-1][0] is True
    assert "用时" in results[-1][1]
    assert len(remaining) >= 1  # 时钟持续推进，0.5s 节流后至少发一次
    assert all(isinstance(v, int) and v >= 0 for v in remaining)
    # 移动全部完成后剩余应为 0
    assert remaining[-1] == 0
