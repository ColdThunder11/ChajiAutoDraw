"""input_backends 模块单元测试（全部使用假模块注入，不做任何真实输入操作）。

手法与 test_prep.py 一致：monkeypatch.setitem(sys.modules, ...) 注入
MagicMock 假模块。win32con 是纯常量库，直接用真模块。
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import win32con

from chaji_autodraw.core.input_backends import PostMessageBackend


def _makelong(low: int, high: int) -> int:
    """与 win32api.MAKELONG 等价的纯 Python 实现（低字在前）。"""
    return ((int(high) & 0xFFFF) << 16) | (int(low) & 0xFFFF)


# --------------------------------------------------------------- 子窗口解析


def _inject_win32gui_children(
    monkeypatch: pytest.MonkeyPatch,
    class_by_hwnd: dict[int, str],
) -> mock.MagicMock:
    """注入假 win32gui：EnumChildWindows 同步回调所有子窗口，GetClassName 查表。"""
    fake = mock.MagicMock(name="win32gui")
    children = [h for h in class_by_hwnd if h != 0]

    def _enum(_parent: int, cb, _param: object) -> None:
        for child in children:
            cb(child, None)

    fake.EnumChildWindows.side_effect = _enum
    fake.GetClassName.side_effect = lambda hwnd: class_by_hwnd[hwnd]
    monkeypatch.setitem(sys.modules, "win32gui", fake)
    return fake


def test_resolve_prefers_render_widget_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """存在 Chrome_RenderWidgetHostHWND 子窗口时优先选它。"""
    top, render, other = 0x1000, 0x1001, 0x1002
    _inject_win32gui_children(
        monkeypatch,
        {
            top: "WeChatAppExWnd",
            render: "Chrome_RenderWidgetHostHWND",
            other: "Chrome_WidgetWin_1",
        },
    )
    backend = PostMessageBackend(top)

    desc = backend.resolve()

    assert backend.target_hwnd == render
    assert backend.target_class == "Chrome_RenderWidgetHostHWND"
    assert "Chrome_RenderWidgetHostHWND" in desc
    assert f"0x{render:x}" in desc


def test_resolve_falls_back_to_chrome_widgetwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """只有 Chrome_WidgetWin_0 子窗口时回退选它。"""
    top, widget = 0x2000, 0x2001
    _inject_win32gui_children(
        monkeypatch,
        {top: "WeChatAppExWnd", widget: "Chrome_WidgetWin_0"},
    )
    backend = PostMessageBackend(top)

    backend.resolve()

    assert backend.target_hwnd == widget
    assert backend.target_class == "Chrome_WidgetWin_0"


def test_resolve_falls_back_to_top_level_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有任何匹配子窗口时用顶级窗口本身，类名描述为「顶级窗口」。"""
    top = 0x3000
    _inject_win32gui_children(
        monkeypatch,
        {top: "WeChatAppExWnd", 0x3001: "Button", 0x3002: "Static"},
    )
    backend = PostMessageBackend(top)

    desc = backend.resolve()

    assert backend.target_hwnd == top
    assert backend.target_class == "顶级窗口"
    assert "顶级窗口" in desc


def test_prepare_auto_resolves_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """未显式 resolve 时，prepare() 自动解析出实际投递子窗口。"""
    top, render = 0x4000, 0x4001
    _inject_win32gui_children(
        monkeypatch,
        {top: "WeChatAppExWnd", render: "Chrome_RenderWidgetHostHWND"},
    )
    fake_api = mock.MagicMock(name="win32api")
    monkeypatch.setitem(sys.modules, "win32api", fake_api)

    backend = PostMessageBackend(top)
    backend.prepare()

    assert backend.target_hwnd == render
    assert backend.target_class == "Chrome_RenderWidgetHostHWND"


# ------------------------------------------------------- 消息序列与坐标换算


@pytest.fixture
def posted_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, int, int, int]]:
    """注入假 win32gui/win32api：记录全部 PostMessage 消息四元组。

    假 ScreenToClient 返回 (屏幕x - 100, 屏幕y - 200)，模拟子窗口位于屏幕
    (100, 200) 处（客户区原点在屏幕 (100, 200)）。
    """
    fake_gui = mock.MagicMock(name="win32gui")
    fake_api = mock.MagicMock(name="win32api")
    posted: list[tuple[int, int, int, int]] = []

    fake_gui.EnumChildWindows.side_effect = lambda _h, _cb, _p: None  # 无子窗口
    fake_gui.ScreenToClient.side_effect = lambda _hwnd, pt: (pt[0] - 100, pt[1] - 200)
    fake_gui.PostMessage.side_effect = (
        lambda hwnd, msg, wparam, lparam: posted.append((hwnd, msg, wparam, lparam))
    )
    fake_api.MAKELONG.side_effect = lambda low, high: _makelong(low, high)
    monkeypatch.setitem(sys.modules, "win32gui", fake_gui)
    monkeypatch.setitem(sys.modules, "win32api", fake_api)
    return posted


def test_message_sequence_and_client_coords(
    posted_messages: list[tuple[int, int, int, int]],
) -> None:
    """activate/move/press/release 的消息类型、wParam 与客户区坐标换算。"""
    backend = PostMessageBackend(0x500)
    backend.prepare()
    backend.activate()
    backend.move(350, 450, pressed=True)  # 客户区 (250, 250)
    backend.move(300, 400, pressed=False)  # 客户区 (200, 200)
    backend.press(310, 420)  # 客户区 (210, 220)
    backend.release(320, 430)  # 客户区 (220, 230)

    assert len(posted_messages) == 5
    # 全部消息都投给解析后的目标（无子窗口 → 顶级窗口本身）
    assert all(msg[0] == 0x500 for msg in posted_messages)

    # WM_ACTIVATE：wParam=WA_ACTIVE，lParam=0
    activate = posted_messages[0]
    assert activate[1] == win32con.WM_ACTIVATE
    assert activate[2] == win32con.WA_ACTIVE
    assert activate[3] == 0

    # WM_MOUSEMOVE：lParam=MAKELONG(客户区坐标)，按下 wParam=MK_LBUTTON，否则 0
    move_pressed = posted_messages[1]
    assert move_pressed[1] == win32con.WM_MOUSEMOVE
    assert move_pressed[3] == _makelong(250, 250)
    assert move_pressed[2] == win32con.MK_LBUTTON
    move_free = posted_messages[2]
    assert move_free[1] == win32con.WM_MOUSEMOVE
    assert move_free[3] == _makelong(200, 200)
    assert move_free[2] == 0

    # WM_LBUTTONDOWN：wParam=MK_LBUTTON
    down = posted_messages[3]
    assert down[1] == win32con.WM_LBUTTONDOWN
    assert down[2] == win32con.MK_LBUTTON
    assert down[3] == _makelong(210, 220)

    # WM_LBUTTONUP：wParam=0
    up = posted_messages[4]
    assert up[1] == win32con.WM_LBUTTONUP
    assert up[2] == 0
    assert up[3] == _makelong(220, 230)


# ------------------------------------------------------------ UIPI 拒绝检测


def _inject_refusing_postmessage(
    monkeypatch: pytest.MonkeyPatch,
    post_result: object,
) -> None:
    """注入返回假值（或抛异常）的假 PostMessage，模拟 UIPI 静默拒绝。"""
    fake_gui = mock.MagicMock(name="win32gui")
    fake_api = mock.MagicMock(name="win32api")
    fake_gui.EnumChildWindows.side_effect = lambda _h, _cb, _p: None
    fake_gui.ScreenToClient.side_effect = lambda _hwnd, pt: (pt[0], pt[1])
    fake_gui.PostMessage.return_value = post_result
    fake_api.MAKELONG.side_effect = lambda low, high: _makelong(low, high)
    monkeypatch.setitem(sys.modules, "win32gui", fake_gui)
    monkeypatch.setitem(sys.modules, "win32api", fake_api)


def test_uipi_rejected_post_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostMessage 返回 0（UIPI 静默拒绝）→ 各动作抛含「管理员/UIPI」的 RuntimeError。"""
    _inject_refusing_postmessage(monkeypatch, 0)
    backend = PostMessageBackend(0x900)
    backend.prepare()  # prepare 本身不投递消息，不应抛错

    with pytest.raises(RuntimeError, match="管理员|UIPI"):
        backend.activate()
    with pytest.raises(RuntimeError, match="管理员|UIPI"):
        backend.move(100, 100, pressed=False)
    with pytest.raises(RuntimeError, match="管理员|UIPI"):
        backend.press(100, 100)
    with pytest.raises(RuntimeError, match="管理员|UIPI"):
        backend.release(100, 100)


def test_uipi_rejected_post_raises_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostMessage 抛异常同样转成含中文提示的 RuntimeError。"""
    fake_gui = mock.MagicMock(name="win32gui")
    fake_api = mock.MagicMock(name="win32api")
    fake_gui.EnumChildWindows.side_effect = lambda _h, _cb, _p: None
    fake_gui.PostMessage.side_effect = OSError("access denied")
    fake_api.MAKELONG.side_effect = lambda low, high: _makelong(low, high)
    monkeypatch.setitem(sys.modules, "win32gui", fake_gui)
    monkeypatch.setitem(sys.modules, "win32api", fake_api)

    backend = PostMessageBackend(0x901)
    backend.prepare()

    with pytest.raises(RuntimeError, match="管理员|UIPI"):
        backend.activate()


# ------------------------------------------------------ 前台鼠标后端


@pytest.fixture
def fake_pyautogui(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """向 sys.modules 注入假 pyautogui（不做真实鼠标操作）。"""
    fake = mock.MagicMock(name="pyautogui")
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    return fake


class _FakeWintypes:
    """ctypes.wintypes 轻量替身：仅提供 SendInput 用到的类型。"""

    class LONG:
        pass

    class DWORD:
        pass

    class ULONG:
        pass


class _FakeCtypes:
    """ctypes 替身：windll.user32 注入假实现，结构体退化为普通类。"""

    def __init__(self, user32) -> None:
        self.windll = types.SimpleNamespace(user32=user32)
        self.wintypes = _FakeWintypes()
        self.POINTER = lambda tp: tp
        self.sizeof = lambda obj: 40
        self.byref = lambda obj: obj

    class Structure:
        _fields_ = []

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Union:
        _fields_ = []

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


def test_foreground_move_uses_sendinput(monkeypatch: pytest.MonkeyPatch) -> None:
    """前台后端移动走 SendInput（NOCOALESCE 防合并），不走 pyautogui。"""
    send_calls: list[int] = []
    def fake_send_input(n, p, size):
        send_calls.append(n)
        return 1

    fake_user32 = types.SimpleNamespace(
        GetSystemMetrics=lambda i: {76: 0, 77: 0, 78: 1920, 79: 1080}[i],
        SendInput=fake_send_input,
    )
    monkeypatch.setitem(sys.modules, "ctypes", _FakeCtypes(fake_user32))
    monkeypatch.setitem(sys.modules, "ctypes.wintypes", _FakeWintypes())
    fake_pg = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    from chaji_autodraw.core.input_backends import ForegroundMouseBackend

    backend = ForegroundMouseBackend()
    backend.move(960, 540, pressed=True)
    assert len(send_calls) == 1  # SendInput 注入一次
    assert fake_pg.moveTo.call_args_list == []  # 不再走 pyautogui


def test_foreground_press_release_still_pyautogui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """按下/抬起仍由 pyautogui 完成（移动已被 SendInput 接管）。"""
    fake_pg = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    from chaji_autodraw.core.input_backends import ForegroundMouseBackend

    backend = ForegroundMouseBackend()
    backend.press(10, 20)
    backend.release(30, 40)
    assert fake_pg.mouseDown.call_count == 1
    assert fake_pg.mouseUp.call_count == 1
