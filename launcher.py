"""PyInstaller 打包入口：仅转发到包入口，保持业务代码全部在包内。"""

from chaji_autodraw.main import run

if __name__ == "__main__":
    run()
