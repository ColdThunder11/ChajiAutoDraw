"""程序入口：启动 PySide6 主窗口。"""

from __future__ import annotations

import os
import sys


def _icon_path() -> str | None:
    """定位应用图标（assets/icon.ico），兼容源码运行与 PyInstaller 打包。

    打包（onefile）时资源在 sys._MEIPASS/assets；源码运行时在仓库根
    目录 assets（main.py 位于 src/chaji_autodraw/ 下，向上两级）。
    """
    candidates: list[str] = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, "assets", "icon.ico"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "assets", "icon.ico")))
    candidates.append(os.path.normpath(os.path.join(here, "assets", "icon.ico")))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def run() -> None:
    """程序入口：创建 QApplication 与主窗口并进入事件循环。

    DPI 感知无需手动设置：Qt 6 在创建 QApplication 时默认启用
    PER_MONITOR_AWARE_V2，进程即处于 DPI 感知状态，win32 API、
    ImageGrab 与 pyautogui 拿到的都是物理像素坐标。若在 QApplication
    之前手动 SetProcessDpiAwareness，反而会令 Qt 设置默认上下文时
    报「拒绝访问」警告。
    """
    from PySide6.QtGui import QIcon

    from PySide6.QtWidgets import QApplication

    from chaji_autodraw.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("霸王茶姬杯贴自动绘制")
    app.setOrganizationName("ChajiAutoDraw")
    icon_path = _icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))  # 窗口与任务栏图标
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
