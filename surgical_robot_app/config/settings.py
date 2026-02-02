"""
应用配置管理

集中管理应用中的所有配置值，避免硬编码。
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
import os

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.config.settings")


@dataclass
class PathPlanningConfig:
    """路径规划配置"""
    grid_size: Tuple[int, int, int] = (100, 100, 105)
    obstacle_expansion: int = 4
    default_grid_size: Tuple[int, int, int] = (100, 100, 100)  # A* 算法的默认网格大小


@dataclass
class SegmentationConfig:
    """分割相关配置"""
    brush_size: int = 10  # 手动分割画笔大小
    radius_ratio: float = 0.02  # 3D点标记半径比例
    default_coord: float = 50.0  # 默认坐标值
    default_radius: float = 2.0  # 默认半径


@dataclass
class View3DConfig:
    """3D视图配置"""
    marker_radius_ratio: float = 0.02  # 标记点半径比例
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)  # VTK背景色
    coordinate_system_size_ratio: float = 0.2  # 坐标系大小比例（相对于模型最大尺寸，0.2表示20%）
    axes_actor_scale_factor: float = 0.6  # XYZ轴指示器缩放因子（1.0为默认大小，>1.0更大，<1.0更小）
    # 3D重建填充参数
    morph_kernel_size: int = 3  # 形态学操作核大小（3x3更保守，5x5更激进）
    morph_iterations: int = 1  # 形态学闭运算迭代次数（1次更保守，3次更激进）
    vtk_hole_size: float = 1000.0  # VTK填充过滤器的孔洞大小（1000更保守，10000更激进）
    enable_contour_filling: bool = True  # 是否启用轮廓填充（True会填充所有轮廓内部）


@dataclass
class HSVThresholdConfig:
    """HSV阈值配置"""
    h1_low: int = 0
    h1_high: int = 10
    h2_low: int = 160
    h2_high: int = 180
    s_low: int = 50
    s_high: int = 255
    v_low: int = 50
    v_high: int = 255
    
    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式（用于兼容现有代码）"""
        return {
            "h1_low": self.h1_low,
            "h1_high": self.h1_high,
            "h2_low": self.h2_low,
            "h2_high": self.h2_high,
            "s_low": self.s_low,
            "s_high": self.s_high,
            "v_low": self.v_low,
            "v_high": self.v_high,
        }


@dataclass
class UIConfig:
    """UI相关配置"""
    vtk_widget_min_height: int = 300
    image_label_min_height: int = 400
    slice_label_min_height: int = 150
    toggle_button_max_width: int = 80


@dataclass
class AppConfig:
    """应用主配置"""
    path_planning: PathPlanningConfig = field(default_factory=PathPlanningConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    view3d: View3DConfig = field(default_factory=View3DConfig)
    hsv_threshold: HSVThresholdConfig = field(default_factory=HSVThresholdConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    def to_dict(self) -> dict:
        """转换为字典（用于序列化）"""
        return {
            'path_planning': asdict(self.path_planning),
            'segmentation': asdict(self.segmentation),
            'view3d': asdict(self.view3d),
            'hsv_threshold': asdict(self.hsv_threshold),
            'ui': asdict(self.ui),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        """从字典创建配置"""
        return cls(
            path_planning=PathPlanningConfig(**data.get('path_planning', {})),
            segmentation=SegmentationConfig(**data.get('segmentation', {})),
            view3d=View3DConfig(**data.get('view3d', {})),
            hsv_threshold=HSVThresholdConfig(**data.get('hsv_threshold', {})),
            ui=UIConfig(**data.get('ui', {})),
        )


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    获取全局配置实例（单例模式）
    
    Returns:
        AppConfig: 应用配置实例
    """
    global _config
    if _config is None:
        _config = AppConfig()
        logger.info("使用默认配置")
    return _config


def set_config(config: AppConfig):
    """
    设置全局配置实例
    
    Args:
        config: 应用配置实例
    """
    global _config
    _config = config
    logger.info("配置已更新")


def load_config_from_file(config_path: Optional[Path] = None) -> AppConfig:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径，如果为 None，则使用默认路径
        
    Returns:
        AppConfig: 加载的配置实例
    """
    if config_path is None:
        # 默认配置文件路径
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config.json"
    
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return AppConfig()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = AppConfig.from_dict(data)
        set_config(config)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}，使用默认配置")
        return AppConfig()


def save_config_to_file(config: Optional[AppConfig] = None, config_path: Optional[Path] = None):
    """
    保存配置到文件
    
    Args:
        config: 要保存的配置，如果为 None，则保存当前全局配置
        config_path: 配置文件路径，如果为 None，则使用默认路径
    """
    if config is None:
        config = get_config()
    
    if config_path is None:
        # 默认配置文件路径
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config.json"
    
    try:
        # 确保目录存在
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"成功保存配置文件: {config_path}")
    except Exception as e:
        logger.error(f"保存配置文件失败: {e}")


def load_config_from_env() -> AppConfig:
    """
    从环境变量加载配置（可选功能）
    
    环境变量格式：
    - SURGICAL_ROBOT_GRID_SIZE: "100,100,105"
    - SURGICAL_ROBOT_OBSTACLE_EXPANSION: "4"
    - SURGICAL_ROBOT_BRUSH_SIZE: "10"
    
    Returns:
        AppConfig: 配置实例
    """
    config = AppConfig()
    
    # 路径规划配置
    grid_size_str = os.getenv('SURGICAL_ROBOT_GRID_SIZE')
    if grid_size_str:
        try:
            parts = [int(x.strip()) for x in grid_size_str.split(',')]
            if len(parts) == 3:
                config.path_planning.grid_size = tuple(parts)
        except ValueError:
            logger.warning(f"无效的网格大小环境变量: {grid_size_str}")
    
    obstacle_expansion_str = os.getenv('SURGICAL_ROBOT_OBSTACLE_EXPANSION')
    if obstacle_expansion_str:
        try:
            config.path_planning.obstacle_expansion = int(obstacle_expansion_str)
        except ValueError:
            logger.warning(f"无效的障碍物扩展环境变量: {obstacle_expansion_str}")
    
    # 分割配置
    brush_size_str = os.getenv('SURGICAL_ROBOT_BRUSH_SIZE')
    if brush_size_str:
        try:
            config.segmentation.brush_size = int(brush_size_str)
        except ValueError:
            logger.warning(f"无效的画笔大小环境变量: {brush_size_str}")
    
    set_config(config)
    logger.info("从环境变量加载配置")
    return config

