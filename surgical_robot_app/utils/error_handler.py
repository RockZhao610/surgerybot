"""
统一的错误处理模块

提供统一的错误处理装饰器和工具函数，用于：
1. 自动记录错误日志
2. 统一显示错误消息框
3. 减少重复的错误处理代码
"""

from functools import wraps
from typing import Optional, Callable, Any, TypeVar
import logging

# 尝试导入 PyQt5，如果不可用则使用 None
try:
    from PyQt5.QtWidgets import QMessageBox
    QMessageBox_AVAILABLE = True
except ImportError:
    QMessageBox_AVAILABLE = False
    QMessageBox = None

# 获取日志记录器
logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


def handle_errors(
    parent_widget: Optional[Any] = None,
    show_message: bool = True,
    message_title: str = "Error",
    log_level: int = logging.ERROR,
    return_on_error: Any = None,
    context: Optional[str] = None
):
    """
    统一的错误处理装饰器
    
    自动捕获异常，记录日志，并可选地显示错误消息框。
    
    Args:
        parent_widget: 用于显示消息框的父窗口（QWidget 实例）
        show_message: 是否显示错误消息框（默认 True）
        message_title: 消息框标题（默认 "Error"）
        log_level: 日志级别（默认 logging.ERROR）
        return_on_error: 发生错误时返回的值（默认 None）
        context: 错误上下文描述（如果为 None，则使用函数名）
    
    Returns:
        装饰器函数
    
    Example:
        ```python
        @handle_errors(parent_widget=self, show_message=True)
        def on_reconstruct_3d(self):
            # 自动处理所有异常
            volume = self.data_manager.get_volume()
            # ...
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            # 确定错误上下文
            error_context = context if context is not None else func.__name__
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 记录日志（包含完整的堆栈信息）
                logger.log(
                    log_level,
                    f"Error in {error_context}: {e}",
                    exc_info=True
                )
                
                # 显示错误消息框
                if show_message and QMessageBox_AVAILABLE:
                    # 尝试从 args 中获取 parent_widget（如果是方法，第一个参数是 self）
                    widget = parent_widget
                    if widget is None and len(args) > 0:
                        potential_widget = args[0]
                        # 检查是否具有 parent_widget 属性（通用模式）
                        if hasattr(potential_widget, 'parent_widget'):
                            widget = potential_widget.parent_widget
                        else:
                            widget = potential_widget
                    
                    if widget is not None:
                        try:
                            # 确保 widget 确实是 QWidget 类型（或者为 None）
                            # 如果显示失败，catch 块会记录警告
                            QMessageBox.critical(
                                widget if hasattr(widget, 'winId') else None,
                                message_title,
                                f"An error occurred in {error_context}:\n{str(e)}"
                            )
                        except Exception as msg_error:
                            # 如果显示消息框失败，只记录日志
                            logger.warning(
                                f"Failed to show error message box: {msg_error}"
                            )
                
                return return_on_error
        return wrapper
    return decorator


def safe_execute(
    func: Callable[[], T],
    context: str = "",
    parent_widget: Optional[Any] = None,
    show_message: bool = True,
    default: Optional[T] = None,
    log_level: int = logging.ERROR
) -> Optional[T]:
    """
    安全执行函数，自动处理异常
    
    这是一个函数式接口，用于在不使用装饰器的情况下安全执行代码。
    
    Args:
        func: 要执行的函数（无参数）
        context: 错误上下文描述
        parent_widget: 用于显示消息框的父窗口
        show_message: 是否显示错误消息框
        default: 发生错误时返回的默认值
        log_level: 日志级别
    
    Returns:
        函数返回值，如果发生错误则返回 default
    
    Example:
        ```python
        result = safe_execute(
            lambda: risky_operation(),
            context="process_data",
            parent_widget=self,
            default=None
        )
        ```
    """
    try:
        return func()
    except Exception as e:
        # 记录日志
        logger.log(
            log_level,
            f"Error in {context}: {e}",
            exc_info=True
        )
        
        # 显示消息框
        if show_message and QMessageBox_AVAILABLE and parent_widget is not None:
            try:
                QMessageBox.critical(
                    parent_widget,
                    "Error",
                    f"An error occurred in {context}:\n{str(e)}"
                )
            except Exception as msg_error:
                logger.warning(
                    f"Failed to show error message box: {msg_error}"
                )
        
        return default


def handle_specific_errors(
    *exception_types: type,
    parent_widget: Optional[Any] = None,
    show_message: bool = True,
    message_title: str = "Error",
    log_level: int = logging.ERROR,
    return_on_error: Any = None,
    context: Optional[str] = None
):
    """
    处理特定类型的异常（只捕获指定的异常类型）
    
    Args:
        *exception_types: 要捕获的异常类型（如 ValueError, FileNotFoundError）
        parent_widget: 用于显示消息框的父窗口
        show_message: 是否显示错误消息框
        message_title: 消息框标题
        log_level: 日志级别
        return_on_error: 发生错误时返回的值
        context: 错误上下文描述
    
    Returns:
        装饰器函数
    
    Example:
        ```python
        @handle_specific_errors(
            ValueError, FileNotFoundError,
            parent_widget=self,
            show_message=True
        )
        def load_file(self, path):
            # 只捕获 ValueError 和 FileNotFoundError
            # 其他异常会正常抛出
            # ...
        ```
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            error_context = context if context is not None else func.__name__
            
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                # 记录日志
                logger.log(
                    log_level,
                    f"Error in {error_context}: {e}",
                    exc_info=True
                )
                
                # 显示错误消息框
                if show_message and QMessageBox_AVAILABLE:
                    widget = parent_widget
                    if widget is None and len(args) > 0:
                        widget = args[0]
                    
                    if widget is not None:
                        try:
                            QMessageBox.critical(
                                widget,
                                message_title,
                                f"An error occurred in {error_context}:\n{str(e)}"
                            )
                        except Exception as msg_error:
                            logger.warning(
                                f"Failed to show error message box: {msg_error}"
                            )
                
                return return_on_error
            # 其他异常正常抛出
        return wrapper
    return decorator

