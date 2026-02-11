"""
Threading Utils: 提供通用的异步任务处理工具
"""

import logging
import traceback
from PyQt5.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)

class Worker(QObject):
    """
    通用 Worker 类，用于在后台线程执行任务
    """
    finished = pyqtSignal(object)  # 任务完成信号，传递结果
    error = pyqtSignal(str)       # 错误信号
    progress = pyqtSignal(int)     # 进度信号
    tree_update = pyqtSignal(object)  # RRT 树实时可视化信号

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """执行任务"""
        try:
            # 自动注入回调参数
            var_names = self.fn.__code__.co_varnames
            if 'progress_callback' in var_names:
                self.kwargs['progress_callback'] = self.progress.emit
            if 'tree_callback' in var_names:
                self.kwargs['tree_callback'] = self.tree_update.emit
            
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            error_msg = f"Worker Error: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error.emit(str(e))

def run_in_thread(parent, fn, on_finished=None, on_error=None, on_progress=None, on_tree_update=None, *args, **kwargs):
    """
    便捷函数：在独立线程中运行函数
    
    Args:
        parent: 父对象 (通常是 Controller)
        fn: 要执行的函数
        on_finished: 完成回调 (接收结果)
        on_error: 错误回调 (接收错误消息)
        on_progress: 进度回调 (接收 0-100 的整数)
        on_tree_update: RRT 树实时可视化回调 (接收边列表)
        *args, **kwargs: 传递给 fn 的参数
    """
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    
    # 存储引用防止被回收
    if not hasattr(parent, '_active_threads'):
        parent._active_threads = []
    parent._active_threads.append((thread, worker))
    
    # 连接信号
    thread.started.connect(worker.run)
    
    def cleanup():
        if (thread, worker) in parent._active_threads:
            parent._active_threads.remove((thread, worker))
        thread.deleteLater()
        worker.deleteLater()

    if on_finished:
        worker.finished.connect(on_finished)
    if on_error:
        worker.error.connect(on_error)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_tree_update:
        worker.tree_update.connect(on_tree_update)
        
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(cleanup)
    
    thread.start()
    return thread, worker

