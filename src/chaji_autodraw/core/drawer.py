"""鼠标绘制执行器：QThread 后台线程按笔画调用输入后端在画布上绘图。

本模块不直接依赖 pyautogui / win32：具体输入动作由构造时传入的后端
（core.input_backends.ForegroundMouseBackend / PostMessageBackend）完成，
后端方法内部自行延迟 import，保证 core 包在无显示器/无 QApplication
环境下可安全导入。
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QThread, Signal

_VK_ESCAPE = 0x1B
#: 每个移动点除显式延时外的标称开销（约 3ms），用于初始 ETA 与预览预估
_STEP_OVERHEAD_S = 0.003


def _esc_pressed() -> bool:
    """系统级检测 Esc 键是否正被按下（GetAsyncKeyState，与焦点无关）。

    绘制线程在每个移动步进上轮询本函数（间隔约 delay_s），因此前台
    鼠标被接管、点不到停止按钮时，按 Esc 即可急停。
    """
    if sys.platform != "win32":
        return False
    import ctypes

    return bool(ctypes.windll.user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)

if TYPE_CHECKING:
    from typing import Protocol

    class DrawBackend(Protocol):
        """绘制输入后端协议（见 core.input_backends 两个实现，不做运行时校验）。"""

        def resolve(self) -> str: ...

        def prepare(self) -> None: ...

        def activate(self) -> None: ...

        def move(self, x: float, y: float, *, pressed: bool) -> None: ...

        def press(self, x: float, y: float) -> None: ...

        def release(self, x: float, y: float) -> None: ...


class DrawWorker(QThread):
    """后台绘制线程：把笔画列表转换为输入后端动作在画布上绘图。

    Signals:
        progress: (当前第几笔，从 1 开始, 总笔数)。
        move_progress: (已完成移动数, 总移动数)。长笔画（尤其是整块
            连续排线）可能一笔画几十秒，按笔画更新的进度条会长时间
            不动；该信号每 0.5s 及每笔结束时发出，供进度条平滑推进。
        remaining: 预估剩余时间（整数秒），每 0.5s 至多更新一次。
        done: (是否成功完成, 消息)。QThread 自带 finished 信号，
            故完成信号命名为 done，禁止覆盖 finished。
    """

    progress = Signal(int, int)
    move_progress = Signal(int, int)
    remaining = Signal(int)
    done = Signal(bool, str)

    def __init__(
        self,
        strokes: list[list[tuple[int, int]]],
        origin_x: int,
        origin_y: int,
        backend: DrawBackend,
        step_px: float = 3.0,
        delay_s: float = 0.008,
        stroke_gap_s: float = 0.0,
        countdown_s: float = 3.0,
        parent: QObject | None = None,
    ) -> None:
        """创建绘制线程（调用 start() 前不会执行任何绘制动作）。

        Args:
            strokes: 笔画列表，每笔为 [(x, y), ...]（画布内坐标）。
            origin_x: 画布左上角在屏幕上的物理像素 x 坐标。
            origin_y: 画布左上角在屏幕上的物理像素 y 坐标。
            backend: 输入后端（前台鼠标 / 后台消息），提供
                prepare/activate/move/press/release 方法。
            step_px: 路径插值步长（像素），相邻移动点的间距。
            delay_s: 每次移动后的停顿秒数。
            stroke_gap_s: 每笔画完后的停顿秒数（给窗口处理消息的时间，
                目标窗口响应不过来/丢笔时调大）。
            countdown_s: 开始绘制前的倒计时秒数。
            parent: Qt 父对象。
        """
        super().__init__(parent)
        if step_px <= 0:
            raise ValueError("step_px 必须为正")
        self._strokes = strokes
        self._origin_x = int(origin_x)
        self._origin_y = int(origin_y)
        self._backend = backend
        self._step_px = float(step_px)
        self._delay_s = max(0.0, float(delay_s))
        self._stroke_gap_s = max(0.0, float(stroke_gap_s))
        self._countdown_s = max(0.0, float(countdown_s))
        #: 按步长插值后的总移动数（构造时算一次；run 与 UI 共用）
        self._total_moves = estimate_moves(strokes, self._step_px)
        self._stop_event = threading.Event()
        self._stop_reason = "已手动停止"

    @property
    def total_moves(self) -> int:
        """预计执行的鼠标移动次数（含孤立单点的补笔移动）。"""
        return self._total_moves

    def stop(self) -> None:
        """线程安全地请求停止（置停止标志，run 中各步进点轮询响应）。"""
        self._stop_event.set()

    @property
    def _stopped(self) -> bool:
        return self._stop_event.is_set()

    def _check_stop(self) -> bool:
        """检查停止标志或 Esc 急停；触发时记录停止原因。

        Returns:
            True 表示已请求停止（按钮或 Esc）。
        """
        if not self._stop_event.is_set() and _esc_pressed():
            self._stop_reason = "已按 Esc 停止绘制"
            self._stop_event.set()
        return self._stop_event.is_set()

    def _wait_countdown(self) -> bool:
        """倒计时 countdown_s 秒，以 0.1s 步进轮询停止标志/Esc（不发信号）。

        Returns:
            True 表示倒计时正常结束；False 表示中途被请求停止。
        """
        remaining = self._countdown_s
        while remaining > 0:
            if self._check_stop():
                return False
            time.sleep(min(0.1, remaining))
            remaining -= 0.1
        return not self._check_stop()

    def _wait_gap(self) -> bool:
        """笔间停顿 stroke_gap_s 秒，以 0.02s 步进轮询停止标志/Esc。

        Returns:
            True 表示停顿正常结束；False 表示中途被请求停止。
        """
        remaining = self._stroke_gap_s
        while remaining > 0:
            if self._check_stop():
                return False
            time.sleep(min(0.02, remaining))
            remaining -= 0.02
        return not self._check_stop()

    def run(self) -> None:
        """线程主函数：倒计时 → 后端准备/激活 → 逐笔按下/拖动/抬起完成绘制。"""
        # ① 倒计时（0.1s 步进检查停止标志/Esc；倒计时阶段不发信号）
        if not self._wait_countdown():
            self.done.emit(False, self._stop_reason)
            return

        total = len(self._strokes)
        # 预期总移动数（构造时已按 resample 后的点数统计），用于进度与剩余时间
        total_moves = self._total_moves
        pressed = False  # 当前是否处于按下状态（finally 只在按下时补一次抬起）
        last_x = 0
        last_y = 0
        done_moves = 0
        t0 = time.monotonic()
        last_eta_at = t0
        # 发车即按标称值给初始估算（每步延时 + ~3ms 单步开销 + 笔间停顿），
        # 之后由实测速率持续修正，保证剩余时间从头到尾始终显示
        nominal = estimate_duration(
            total_moves, total, self._delay_s, self._stroke_gap_s
        )
        self.remaining.emit(max(0, int(round(nominal))))
        try:
            # ②③ 后端准备与激活（准备画布窗口接收输入）
            self._backend.prepare()
            self._backend.activate()
            self.move_progress.emit(0, total_moves)
            # ④ 逐笔绘制
            for i, stroke in enumerate(self._strokes, start=1):
                if self._check_stop():
                    self.done.emit(False, self._stop_reason)
                    return
                if not stroke:
                    continue
                self.progress.emit(i, total)

                points = _resample(stroke, self._step_px)
                x0 = int(round(self._origin_x + points[0][0]))
                y0 = int(round(self._origin_y + points[0][1]))
                self._backend.move(x0, y0, pressed=False)
                self._backend.press(x0, y0)
                pressed = True
                last_x, last_y = x0, y0
                if len(points) == 1:
                    # 画布对「无移动的点击」不落笔（touchstart 不产生笔迹），
                    # 孤立单点补一个 1px 的按下移动，确保笔点真的画上去
                    last_x, last_y = x0 + 1, y0
                    self._backend.move(last_x, last_y, pressed=True)
                    done_moves += 1
                for x, y in points[1:]:
                    if self._check_stop():
                        self.done.emit(False, self._stop_reason)
                        return  # finally 保证抬起
                    last_x = int(round(self._origin_x + x))
                    last_y = int(round(self._origin_y + y))
                    self._backend.move(last_x, last_y, pressed=True)
                    done_moves += 1
                    now = time.monotonic()
                    if now - last_eta_at >= 0.5:
                        # 实测速率估算剩余时间（含剩余笔画的笔间停顿）
                        per_move = (now - t0) / done_moves
                        eta = (total_moves - done_moves) * per_move + (
                            total - i
                        ) * self._stroke_gap_s
                        self.remaining.emit(max(0, int(round(eta))))
                        self.move_progress.emit(done_moves, total_moves)
                        last_eta_at = now
                    if self._delay_s > 0:
                        time.sleep(self._delay_s)
                self._backend.release(last_x, last_y)
                pressed = False
                self.move_progress.emit(done_moves, total_moves)
                # 笔间停顿（最后一笔后不等，避免画完还干等）
                if self._stroke_gap_s > 0 and i < total:
                    if not self._wait_gap():
                        self.done.emit(False, self._stop_reason)
                        return

            elapsed = time.monotonic() - t0
            self.move_progress.emit(total_moves, total_moves)
            self.done.emit(True, f"绘制完成，共 {total} 笔，用时 {self._fmt_duration(elapsed)}")
        except RuntimeError as e:
            # 后端业务失败（含 UIPI 拒绝、前台急停），消息已是中文可直接展示
            self.done.emit(False, str(e))
        except Exception as e:  # noqa: BLE001 - 后台线程需兜底一切异常
            self.done.emit(False, f"绘制出错: {e}")
        finally:
            if pressed:
                try:
                    self._backend.release(last_x, last_y)
                except Exception:  # noqa: BLE001 - 兜底抬起失败只能忽略
                    pass

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """把秒数格式化为「X分Y秒」或「Y秒」。"""
        total = max(0, int(round(seconds)))
        if total < 60:
            return f"{total} 秒"
        return f"{total // 60} 分 {total % 60} 秒"


def _resample(
    stroke: list[tuple[int, int]],
    step: float,
) -> list[tuple[float, float]]:
    """沿笔画折线以 step 像素间距线性插值。

    原始拐点全部保留（每段终点必达），段长不足 step 时只保留拐点。
    """
    if len(stroke) == 1:
        return [(float(stroke[0][0]), float(stroke[0][1]))]
    out: list[tuple[float, float]] = [(float(stroke[0][0]), float(stroke[0][1]))]
    for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
        dist = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, math.ceil(dist / step))
        for k in range(1, steps + 1):
            t = k / steps
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def estimate_moves(
    strokes: list[list[tuple[int, int]]],
    step_px: float,
) -> int:
    """估算实际鼠标移动次数（与 DrawWorker.run 的移动序列一致）。

    单点笔画在 run 中会补一次 1px 的按下移动，其余笔画按插值点数减一
    计；空笔画跳过。构造 DrawWorker 与 UI 预估共用本函数，避免两套
    计数口径不一致。
    """
    total = 0
    for stroke in strokes:
        if not stroke:
            continue
        points = _resample(stroke, step_px)
        total += 1 if len(points) == 1 else len(points) - 1
    return total


def estimate_duration(
    total_moves: int,
    stroke_count: int,
    delay_s: float,
    stroke_gap_s: float,
) -> float:
    """标称绘制耗时（秒）：移动数 ×（步延时 + 单步开销）+ 笔间停顿。"""
    return total_moves * (
        max(0.0, float(delay_s)) + _STEP_OVERHEAD_S
    ) + max(0, int(stroke_count) - 1) * max(0.0, float(stroke_gap_s))
