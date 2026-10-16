"""Windows 窗口检测：枚举匹配窗口、取精确边界、截取窗口画面。

本模块为 Windows 专用（依赖 pywin32 / ctypes-dwmapi / ImageGrab），
全部 Windows API 调用均带容错：个别窗口拒绝访问时直接跳过，不抛出。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from PIL import Image

_DWMWA_EXTENDED_FRAME_BOUNDS = 9  # DWMWA_EXTENDED_FRAME_BOUNDS
_WS_EX_TOOLWINDOW = 0x00000080
#: PrintWindow 强制渲染全部内容（含 DirectX/Chromium 硬件加速窗口）的标志
_PW_RENDERFULLCONTENT = 0x00000002


@dataclass
class WindowInfo:
    """一个可见顶层窗口的信息。

    Attributes:
        hwnd: 窗口句柄。
        title: 窗口标题。
        pid: 所属进程 ID。
        process_name: 进程 exe 文件名（basename），如 "WeChatAppEx.exe"。
        rect: (left, top, right, bottom)，屏幕物理像素坐标。
    """

    hwnd: int
    title: str
    pid: int
    process_name: str
    rect: tuple[int, int, int, int]


class _RECT(ctypes.Structure):
    """Win32 RECT 结构，用于 DWM 扩展边界查询。"""

    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


def _get_process_name(pid: int) -> str:
    """取进程 exe 文件名（只取 basename），任何失败返回空串。

    优先 OpenProcess + win32process.GetModuleFileNameEx，失败回退
    win32api.GetModuleFileNameEx。
    """
    if pid <= 0:
        return ""
    try:
        import win32api
        import win32con
        import win32process
    except ImportError:
        return ""

    handle = None
    try:
        try:
            access = win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ
            handle = win32api.OpenProcess(access, False, pid)
        except Exception:
            access = win32con.PROCESS_QUERY_LIMITED_INFORMATION
            handle = win32api.OpenProcess(access, False, pid)
        if not handle:
            return ""
        try:
            path = win32process.GetModuleFileNameEx(handle, None)
        except Exception:
            path = win32api.GetModuleFileNameEx(handle, 0)
        if not path:
            return ""
        return path.replace("/", "\\").rsplit("\\", 1)[-1]
    except Exception:
        return ""
    finally:
        if handle is not None:
            try:
                win32api.CloseHandle(handle)
            except Exception:
                pass


def list_matches(
    keyword: str = "霸王茶姬",
    process_hint: str = "WeChatAppEx.exe",
) -> list[WindowInfo]:
    """枚举标题包含 keyword 的可见顶层窗口。

    过滤条件：可见（IsWindowVisible）、有标题、非工具窗口
    （ExStyle 不含 WS_EX_TOOLWINDOW）且标题包含 keyword。
    排序：进程名匹配 process_hint 的排前面，其余按标题排序。

    Args:
        keyword: 窗口标题包含的关键字，默认 "霸王茶姬"。
        process_hint: 进程名提示，匹配者排前面，默认 "WeChatAppEx.exe"。

    Returns:
        匹配的窗口信息列表（找不到时为空列表）。
    """
    import win32con
    import win32gui
    import win32process

    results: list[WindowInfo] = []

    def _on_window(hwnd: int, _param: object) -> None:
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if not title or keyword not in title:
                return
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if ex_style & _WS_EX_TOOLWINDOW:
                return
            pid = 0
            try:
                pid = int(win32process.GetWindowThreadProcessId(hwnd)[1])
            except Exception:
                pid = 0
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            results.append(
                WindowInfo(
                    hwnd=int(hwnd),
                    title=title,
                    pid=pid,
                    process_name=_get_process_name(pid),
                    rect=(int(left), int(top), int(right), int(bottom)),
                )
            )
        except Exception:
            return  # 个别窗口拒绝访问时跳过

    win32gui.EnumWindows(_on_window, None)

    hint = process_hint.lower()
    results.sort(key=lambda info: (0 if info.process_name.lower() == hint else 1, info.title))
    return results


def get_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """取窗口精确可见边界（物理像素）。

    优先用 DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS) 取不含
    不可见缩放边框的精确边界；失败回退 win32gui.GetWindowRect。
    """
    rect = _RECT()
    try:
        result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(_DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            wintypes.DWORD(ctypes.sizeof(rect)),
        )
        if result == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))
    except Exception:
        pass

    import win32gui

    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    return (int(left), int(top), int(right), int(bottom))


class _BITMAPINFOHEADER(ctypes.Structure):
    """Win32 BITMAPINFOHEADER 结构，用于 GetDIBits 读取位图像素。"""

    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def _print_window_image(hwnd: int, width: int, height: int) -> Image.Image | None:
    """用 PrintWindow(PW_RENDERFULLCONTENT) 让窗口把内容渲染到内存位图。

    窗口被其他窗口遮挡时也能取到内容（最小化不行）。任何一步失败
    （句柄无效、被系统拒绝、读回行数不符）返回 None，由调用方回退
    屏幕截图方案。
    """
    if sys.platform != "win32" or width <= 0 or height <= 0:
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    # 显式声明签名，避免 64 位下句柄被截断为 32 位 int
    user32.GetWindowDC.argtypes = [wintypes.HWND]
    user32.GetWindowDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = wintypes.BOOL
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.PrintWindow.restype = wintypes.BOOL
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL

    hdc_window = user32.GetWindowDC(hwnd)
    if not hdc_window:
        return None
    hdc_mem = None
    bitmap = None
    try:
        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        bitmap = gdi32.CreateCompatibleBitmap(hdc_window, width, height)
        if not hdc_mem or not bitmap:
            return None
        old = gdi32.SelectObject(hdc_mem, bitmap)
        try:
            if not user32.PrintWindow(hwnd, hdc_mem, _PW_RENDERFULLCONTENT):
                return None
        finally:
            # GetDIBits 要求位图未被选入 DC：先恢复旧位图再读像素
            gdi32.SelectObject(hdc_mem, old)

        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # 负值 = 自上而下的行序
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0  # BI_RGB
        buf = ctypes.create_string_buffer(width * height * 4)
        lines = gdi32.GetDIBits(hdc_mem, bitmap, 0, height, buf, ctypes.byref(bmi), 0)
        if lines != height:
            return None
        # frombytes 会复制数据，与底层缓冲区解耦
        return Image.frombytes("RGB", (width, height), buf, "raw", "BGRX", 0, 1)
    finally:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if hdc_mem:
            gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_window)


def _crop_to_bounds(
    img: Image.Image,
    window_rect: tuple[int, int, int, int],
    bounds: tuple[int, int, int, int],
) -> Image.Image:
    """把 PrintWindow 输出（GetWindowRect 尺寸）裁剪到 DWM 可见边界。

    PrintWindow 渲染的是含不可见缩放边框的外框，而坐标换算统一基于
    get_bounds（DWM 扩展边界），因此按二者偏移裁掉边框；越界时钳位。
    """
    ox = max(0, bounds[0] - window_rect[0])
    oy = max(0, bounds[1] - window_rect[1])
    width = max(1, bounds[2] - bounds[0])
    height = max(1, bounds[3] - bounds[1])
    ox = min(ox, max(0, img.width - 1))
    oy = min(oy, max(0, img.height - 1))
    width = min(width, img.width - ox)
    height = min(height, img.height - oy)
    return img.crop((ox, oy, ox + width, oy + height))


def capture(hwnd: int) -> Image.Image:
    """截取窗口当前画面（物理像素，对齐 get_bounds 边界）。

    优先 PrintWindow：内容由窗口自行渲染到内存位图，窗口被其他窗口
    遮挡、不在最前时也能取到（最小化不行）。PrintWindow 失败时回退
    ImageGrab 屏幕区域截图（需要窗口未被遮挡）。
    """
    left, top, right, bottom = get_bounds(hwnd)
    bounds = (left, top, right, bottom)
    if right <= left or bottom <= top:
        raise ValueError(f"窗口 {hwnd} 尺寸无效，无法截图: {bounds}")

    import win32gui

    try:
        window_rect = tuple(int(v) for v in win32gui.GetWindowRect(hwnd))
    except Exception:
        window_rect = bounds

    img = _print_window_image(
        hwnd, window_rect[2] - window_rect[0], window_rect[3] - window_rect[1]
    )
    if img is not None:
        return _crop_to_bounds(img, window_rect, bounds)

    from PIL import ImageGrab

    return ImageGrab.grab(bbox=bounds, all_screens=True)
