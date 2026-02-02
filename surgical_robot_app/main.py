import os
# 在导入任何 PyTorch 相关模块之前设置 MPS 回退环境变量
# 这对于 Mac M1/M2 芯片很重要，因为 MPS 不支持某些操作（如 upsample_bicubic2d）
# 设置此环境变量后，PyTorch 会在遇到不支持的操作时自动回退到 CPU
if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import time
from PyQt5.QtWidgets import QApplication

# 初始化日志系统（在导入其他模块之前）
try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.main")

# 测量导入时间
import_start = time.time()
try:
    from .gui.main_window import MainWindow
except Exception:
    try:
        from surgical_robot_app.gui.main_window import MainWindow
    except Exception:
        from gui.main_window import MainWindow
import_time = time.time() - import_start
logger.info(f"模块导入耗时: {import_time:.2f} 秒")


def main():
    app_start = time.time()
    app = QApplication([])
    app_time = time.time() - app_start
    logger.info(f"QApplication 创建耗时: {app_time:.2f} 秒")
    
    window_start = time.time()
    w = MainWindow()
    window_time = time.time() - window_start
    logger.info(f"MainWindow 初始化耗时: {window_time:.2f} 秒")
    
    show_start = time.time()
    w.show()
    show_time = time.time() - show_start
    logger.info(f"窗口显示耗时: {show_time:.2f} 秒")
    
    total_time = time.time() - app_start
    logger.info(f"总启动时间: {total_time:.2f} 秒")
    logger.info("=" * 50)
    
    app.exec_()


if __name__ == "__main__":
    main()