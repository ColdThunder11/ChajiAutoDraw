"""绘制输入后端：前台真实鼠标（pyautogui）与后台消息（PostMessage）。

两种后端实现同一组方法，供 core.drawer.DrawWorker 在 QThread 里调用：
    resolve() -> str                  # 解析/校验目标，返回描述文本（供状态栏显示）
    activate() -> None                # 开始绘制前的激活/准备动作
    move(x, y, *, pressed) -> None    # 屏幕绝对坐标
    press(x, y) -> None
    release(x, y) -> None
所有方法允许抛 RuntimeError（带中文信息），由 DrawWorker 捕获后上报。
pyautogui / win32 全部在方法内部延迟 import（头模块导入保持无副作用）。
"""

from __future__ import annotations

#: PostMessage 失败（返回假值或抛异常）时的统一中文错误提示
_POST_REFUSED_MSG = (
    "后台消息被系统拒绝（PostMessage 失败）。常见原因：目标窗口以管理员权限运行"
    "而本程序不是（UIPI 限制），请以相同权限重试。"
)


class ForegroundMouseBackend:
    """前台真实鼠标后端：驱动 pyautogui 移动真实光标绘制（要求窗口在前台）。

    移动通过 SendInput + MOUSEEVENTF_MOVE_NOCOALESCE 注入：阻止系统
    合并连续移动事件——快速绘制时每个中间点都会作为独立消息送达目标
    窗口，避免丢点；多显示器按虚拟桌面坐标换算。按下/抬起仍由
    pyautogui 完成。注意：移动不再触发 pyautogui 左上角 FAILSAFE，
    急停以 Esc 为准。
    """

    _MOUSEEVENTF_MOVE = 0x0001
    _MOUSEEVENTF_ABSOLUTE = 0x8000
    _MOUSEEVENTF_VIRTUALDESK = 0x4000
    _MOUSEEVENTF_MOVE_NOCOALESCE = 0x0800
    _INPUT_MOUSE = 0
    _SM_XVIRTUALSCREEN = 76
    _SM_YVIRTUALSCREEN = 77
    _SM_CXVIRTUALSCREEN = 78
    _SM_CYVIRTUALSCREEN = 79

    def resolve(self) -> str:
        """返回后端描述文本（供状态栏显示）。"""
        return "前台鼠标（移动真实光标）"

    def prepare(self) -> None:
        """延迟 import pyautogui 并关闭自动停顿、保持急停可用。"""
        import pyautogui

        pyautogui.PAUSE = 0  # moveTo 不再自动停顿，节奏由 delay_s 控制
        pyautogui.FAILSAFE = True  # 鼠标甩到屏幕左上角立即中止

    def activate(self) -> None:
        """前台鼠标无需额外激活动作。"""

    def move(self, x: float, y: float, *, pressed: bool) -> None:
        """移动真实光标到屏幕绝对坐标（pressed 仅是协议参数，不影响动作）。"""
        self._sendinput_move(x, y)

    def press(self, x: float, y: float) -> None:
        """移动到屏幕绝对坐标并按下左键。"""
        import pyautogui

        pyautogui.moveTo(x, y)
        pyautogui.mouseDown()

    def release(self, x: float, y: float) -> None:
        """移动到屏幕绝对坐标并抬起左键。"""
        import pyautogui

        pyautogui.moveTo(x, y)
        pyautogui.mouseUp()

    def _sendinput_move(self, x: float, y: float) -> None:
        """用 SendInput(MOVE|ABSOLUTE|VIRTUALDESK|NOCOALESCE) 注入移动。"""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        vx = user32.GetSystemMetrics(self._SM_XVIRTUALSCREEN)
        vy = user32.GetSystemMetrics(self._SM_YVIRTUALSCREEN)
        vw = user32.GetSystemMetrics(self._SM_CXVIRTUALSCREEN)
        vh = user32.GetSystemMetrics(self._SM_CYVIRTUALSCREEN)
        if vw <= 1 or vh <= 1:
            raise RuntimeError("无法获取虚拟桌面尺寸，NOCOALESCE 移动失败")

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)),
            ]

        class _INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

        nx = int(round((x - vx) * 65535 / (vw - 1)))
        ny = int(round((y - vy) * 65535 / (vh - 1)))
        nx = max(0, min(65535, nx))
        ny = max(0, min(65535, ny))
        inp = INPUT(
            type=self._INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=nx,
                dy=ny,
                mouseData=0,
                dwFlags=(
                    self._MOUSEEVENTF_MOVE
                    | self._MOUSEEVENTF_ABSOLUTE
                    | self._MOUSEEVENTF_VIRTUALDESK
                    | self._MOUSEEVENTF_MOVE_NOCOALESCE
                ),
                time=0,
                dwExtraInfo=None,
            ),
        )
        sent = user32.SendInput(
            1, ctypes.byref(inp), ctypes.sizeof(INPUT)
        )
        if sent != 1:
            raise RuntimeError("SendInput 注入移动失败（NOCOALESCE 实验模式）")

class PostMessageBackend:
    """后台消息后端：向目标窗口的 Chromium 渲染子窗口投递鼠标消息。

    微信小程序窗口（WeChatAppEx.exe，CEF/Chromium 架构）由渲染子窗口处理
    鼠标输入，因此优先把消息投给 Chrome_RenderWidgetHostHWND，找不到再回退
    Chrome_WidgetWin_0，最后回退顶级窗口本身。目标窗口可被遮挡甚至不在前台，
    但不能最小化；目标以管理员运行时 PostMessage 会被系统静默拒绝（UIPI）。
    """

    #: 渲染子窗口类名优先级（索引越小越优先）
    _CHILD_CLASS_PRIORITY: tuple[str, ...] = (
        "Chrome_RenderWidgetHostHWND",
        "Chrome_WidgetWin_0",
    )

    def __init__(self, target_hwnd: int) -> None:
        """以顶级窗口句柄创建后端（resolve() 后 target_hwnd 变为实际投递对象）。"""
        self.target_hwnd = int(target_hwnd)
        self.target_class: str = ""  # resolve() 后为类名，顶级窗口时为 "顶级窗口"
        self._resolved = False

    def resolve(self) -> str:
        """递归枚举子窗口并按优先级选定实际投递对象。

        Returns:
            形如 "后台消息 → Chrome_RenderWidgetHostHWND (0x1a2b)" 的描述文本。
        """
        import win32gui

        top_hwnd = self.target_hwnd
        children: list[int] = []

        def _on_child(hwnd: int, _param: object) -> None:
            children.append(int(hwnd))

        try:
            win32gui.EnumChildWindows(top_hwnd, _on_child, None)
        except Exception:  # noqa: BLE001 - 无子窗口/枚举中断时用已收集结果继续
            pass

        chosen: int | None = None
        chosen_class = ""
        for cls_name in self._CHILD_CLASS_PRIORITY:
            for child in children:
                try:
                    cls = win32gui.GetClassName(child)
                except Exception:  # noqa: BLE001 - 单窗口异常跳过
                    continue
                if cls == cls_name:
                    chosen = child
                    chosen_class = cls
                    break
            if chosen is not None:
                break

        if chosen is None:  # 没有任何匹配子窗口：回退顶级窗口本身
            chosen = top_hwnd
            chosen_class = "顶级窗口"

        self.target_hwnd = chosen
        self.target_class = chosen_class
        self._resolved = True
        return f"后台消息 → {chosen_class} (0x{chosen:x})"

    def prepare(self) -> None:
        """确认 win32 可用，且实际投递对象已解析（未解析则自动 resolve 一次）。"""
        import win32api
        import win32con
        import win32gui

        del win32api, win32con, win32gui  # 延迟 import 本身即探测模块可用性

        if not self._resolved:
            self.resolve()

    def activate(self) -> None:
        """伪激活目标窗口（WM_ACTIVATE），让 Chromium 认为窗口处于活动态。"""
        import win32con

        self._post(self.target_hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)

    def move(self, x: float, y: float, *, pressed: bool) -> None:
        """投递 WM_MOUSEMOVE（屏幕绝对坐标换算为目标窗口客户区坐标）。"""
        import win32api
        import win32con
        import win32gui

        cx, cy = win32gui.ScreenToClient(self.target_hwnd, (int(x), int(y)))
        lparam = win32api.MAKELONG(int(cx), int(cy))
        wparam = win32con.MK_LBUTTON if pressed else 0
        self._post(self.target_hwnd, win32con.WM_MOUSEMOVE, wparam, lparam)

    def press(self, x: float, y: float) -> None:
        """投递 WM_LBUTTONDOWN（wParam=MK_LBUTTON）。"""
        import win32api
        import win32con
        import win32gui

        cx, cy = win32gui.ScreenToClient(self.target_hwnd, (int(x), int(y)))
        lparam = win32api.MAKELONG(int(cx), int(cy))
        self._post(self.target_hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)

    def release(self, x: float, y: float) -> None:
        """投递 WM_LBUTTONUP（wParam=0）。"""
        import win32api
        import win32con
        import win32gui

        cx, cy = win32gui.ScreenToClient(self.target_hwnd, (int(x), int(y)))
        lparam = win32api.MAKELONG(int(cx), int(cy))
        self._post(self.target_hwnd, win32con.WM_LBUTTONUP, 0, lparam)

    def _post(self, hwnd: int, msg: int, wparam: int, lparam: int) -> None:
        """统一投递入口：异常或返回假值（UIPI 拒绝等）都转成中文 RuntimeError。

        pywin32 的 PostMessage 成功时可能返回 None，None 不算失败。
        """
        import win32gui

        try:
            result = win32gui.PostMessage(hwnd, msg, wparam, lparam)
        except Exception as exc:
            raise RuntimeError(_POST_REFUSED_MSG) from exc
        if result is not None and not result:
            raise RuntimeError(_POST_REFUSED_MSG)
