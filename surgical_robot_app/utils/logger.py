"""
统一的日志系统

提供统一的日志配置和管理，替换所有 print 语句。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# 日志文件路径
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "surgical_robot.log"

# 确保日志目录存在
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file: Optional[Path] = None,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
) -> None:
    """
    配置全局日志系统
    
    Args:
        level: 全局日志级别 (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: 是否输出到文件
        log_to_console: 是否输出到控制台
        log_file: 日志文件路径，如果为 None 则使用默认路径
        console_level: 控制台日志级别（默认使用 WARNING 以简化终端输出）
        file_level: 文件日志级别（默认使用全局 level）
    """
    if log_file is None:
        log_file = LOG_FILE
    
    # 处理默认级别
    if console_level is None:
        console_level = logging.WARNING
    if file_level is None:
        file_level = level

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        # 添加过滤器，只显示本项目的日志
        console_handler.addFilter(AppLogFilter())
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器（文件中保留所有日志）
    if log_to_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 抑制第三方库的INFO日志（只在控制台显示WARNING及以上）
    _suppress_third_party_loggers()


class AppLogFilter(logging.Filter):
    """只允许本项目的日志通过到控制台，过滤掉第三方库的噪声"""
    
    # 允许显示的日志名称前缀
    ALLOWED_PREFIXES = (
        'surgical_robot_app',
        '__main__',
    )
    
    # 需要过滤的消息关键词（第三方库的噪声日志）
    BLOCKED_MESSAGES = (
        'For numpy array image',
        'Computing image embeddings',
        'Image embeddings computed',
        'assume (HxWxC) format',
    )
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 始终显示WARNING及以上级别
        if record.levelno >= logging.WARNING:
            return True
        
        # 检查是否是需要过滤的消息
        msg = record.getMessage()
        for blocked in self.BLOCKED_MESSAGES:
            if blocked in msg:
                return False
        
        # INFO和DEBUG只显示本项目的日志
        if record.name.startswith(self.ALLOWED_PREFIXES):
            return True
        
        # root logger - 只显示本项目相关的消息
        if record.name == 'root':
            return False  # 默认不显示 root logger 的 INFO 日志
        
        return False


def _suppress_third_party_loggers():
    """抑制第三方库的详细日志"""
    # 这些库的INFO日志通常不需要显示在控制台
    noisy_loggers = [
        'PIL',
        'matplotlib',
        'urllib3',
        'sam2',
        'torch',
        'transformers',
        'huggingface_hub',
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称（通常是模块名）
    
    Returns:
        日志记录器实例
    """
    # 如果日志系统尚未初始化，则初始化它
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        setup_logging()
    
    return logging.getLogger(name)


# 在模块导入时自动初始化日志系统
# 控制台默认 WARNING 级别，文件保留 INFO 级别
setup_logging(
    level=logging.INFO,
    log_to_file=True,
    log_to_console=True,
    console_level=logging.WARNING,
    file_level=logging.INFO
)
