"""
坐标系可视化工具

在3D视图中心显示标准坐标系（轴向/冠状/矢状）
"""

from typing import Optional, Tuple
import logging

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__) if get_logger else None

try:
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkRenderingCore import vtkRenderer
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    vtkAxesActor = None
    vtkRenderer = None
    logger.error("VTK coordinate system modules not available") if logger else None


class CoordinateSystemVisualizer:
    """标准坐标系可视化器"""
    
    def __init__(self, renderer: Optional[vtkRenderer] = None):
        """
        初始化坐标系可视化器
        
        Args:
            renderer: VTK渲染器
        """
        self.renderer = renderer
        self.axes_actor: Optional[vtkAxesActor] = None
        self.visible = False
        self.center = (0.0, 0.0, 0.0)
        self.size = 5.0
    
    def show_coordinate_system(
        self,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: Optional[float] = None,
        scale_factor: Optional[float] = None
    ) -> bool:
        """
        在指定位置显示坐标系
        
        Args:
            center: 坐标系中心位置（世界坐标）
            size: 坐标轴长度（如果为None，则使用self.size）
            scale_factor: 缩放因子（如果为None，则尝试从配置中读取，否则使用1.0）
        
        Returns:
            bool: 是否成功显示
        """
        if not VTK_AVAILABLE or self.renderer is None:
            return False
        
        try:
            # 如果已存在，先移除
            if self.axes_actor is not None:
                self.hide_coordinate_system()
            
            # 创建坐标轴
            self.axes_actor = vtkAxesActor()
            
            # 设置坐标轴标签
            self.axes_actor.SetXAxisLabelText("x")
            self.axes_actor.SetYAxisLabelText("y")
            self.axes_actor.SetZAxisLabelText("z")
            
            # 设置坐标轴长度
            axis_size = size if size is not None else self.size
            
            # 应用缩放因子
            if scale_factor is None:
                # 尝试从配置中读取
                try:
                    from surgical_robot_app.config.settings import get_config
                    config = get_config()
                    scale_factor = config.view3d.axes_actor_scale_factor
                except Exception:
                    scale_factor = 1.0
            
            final_size = axis_size * scale_factor
            self.axes_actor.SetTotalLength(final_size, final_size, final_size)
            
            # 设置坐标系位置
            self.axes_actor.SetOrigin(center[0], center[1], center[2])
            
            # 设置坐标轴颜色（默认：X=红，Y=绿，Z=蓝）
            # VTK默认颜色已经是正确的，不需要修改
            
            # 添加到渲染器
            self.renderer.AddActor(self.axes_actor)
            
            self.center = center
            if size is not None:
                self.size = size
            self.visible = True
            
            logger.info(f"坐标系已显示: center={center}, size={final_size} (base={axis_size}, scale={scale_factor})") if logger else None
            return True
            
        except Exception as e:
            logger.error(f"显示坐标系时出错: {e}") if logger else None
            return False
    
    def hide_coordinate_system(self) -> bool:
        """
        隐藏坐标系
        
        Returns:
            bool: 是否成功隐藏
        """
        if self.renderer is None or self.axes_actor is None:
            return False
        
        try:
            self.renderer.RemoveActor(self.axes_actor)
            self.axes_actor = None
            self.visible = False
            logger.info("坐标系已隐藏") if logger else None
            return True
        except Exception as e:
            logger.error(f"隐藏坐标系时出错: {e}") if logger else None
            return False
    
    def set_size(self, size: float, scale_factor: Optional[float] = None) -> bool:
        """
        设置坐标系大小
        
        Args:
            size: 坐标轴长度
            scale_factor: 缩放因子（如果为None，则尝试从配置中读取）
        
        Returns:
            bool: 是否成功设置
        """
        if size <= 0:
            return False
        
        self.size = size
        
        # 如果坐标系已显示，重新显示以应用新大小
        if self.visible and self.center:
            return self.show_coordinate_system(self.center, size, scale_factor=scale_factor)
        
        return True
    
    def set_center(self, center: Tuple[float, float, float]) -> bool:
        """
        设置坐标系中心位置
        
        Args:
            center: 中心位置（世界坐标）
        
        Returns:
            bool: 是否成功设置
        """
        self.center = center
        
        # 如果坐标系已显示，重新显示以应用新位置
        if self.visible:
            return self.show_coordinate_system(center, self.size)
        
        return True
    
    def auto_size_from_bounds(
        self, 
        bounds: Optional[Tuple[float, float, float, float, float, float]],
        size_ratio: Optional[float] = None
    ) -> bool:
        """
        根据模型边界自动设置坐标系大小
        
        Args:
            bounds: 模型边界 (xmin, xmax, ymin, ymax, zmin, zmax)
            size_ratio: 坐标系大小比例（相对于模型最大尺寸），如果为None则使用默认值0.2（20%）
        
        Returns:
            bool: 是否成功设置
        """
        if bounds is None:
            return False
        
        try:
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            size_x = abs(x_max - x_min)
            size_y = abs(y_max - y_min)
            size_z = abs(z_max - z_min)
            
            # 使用最大尺寸的比例作为坐标轴长度
            max_size = max(size_x, size_y, size_z)
            ratio = size_ratio if size_ratio is not None else 0.2
            auto_size = max_size * ratio
            
            # 确保最小尺寸
            auto_size = max(auto_size, 1.0)
            
            return self.set_size(auto_size)
        except Exception as e:
            logger.error(f"自动设置坐标系大小时出错: {e}") if logger else None
            return False
    
    def auto_center_and_size_from_bounds(
        self, 
        bounds: Optional[Tuple[float, float, float, float, float, float]],
        size_ratio: Optional[float] = None
    ) -> bool:
        """
        根据模型边界自动设置坐标系中心位置和大小
        
        Args:
            bounds: 模型边界 (xmin, xmax, ymin, ymax, zmin, zmax)
            size_ratio: 坐标系大小比例（相对于模型最大尺寸），如果为None则使用默认值0.2（20%）
        
        Returns:
            bool: 是否成功设置
        """
        if bounds is None:
            return False
        
        try:
            x_min, x_max, y_min, y_max, z_min, z_max = bounds
            
            # 计算模型中心
            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            center_z = (z_min + z_max) / 2.0
            center = (center_x, center_y, center_z)
            
            # 计算大小
            size_x = abs(x_max - x_min)
            size_y = abs(y_max - y_min)
            size_z = abs(z_max - z_min)
            
            # 使用最大尺寸的比例作为坐标轴长度
            max_size = max(size_x, size_y, size_z)
            ratio = size_ratio if size_ratio is not None else 0.2
            auto_size = max_size * ratio
            
            # 确保最小尺寸
            auto_size = max(auto_size, 1.0)
            
            # 同时设置中心位置和大小
            self.center = center
            self.size = auto_size
            
            # 获取缩放因子
            scale_factor = None
            try:
                from surgical_robot_app.config.settings import get_config
                config = get_config()
                scale_factor = config.view3d.axes_actor_scale_factor
            except Exception:
                pass
            
            # 显示坐标系（无论之前是否显示过）
            return self.show_coordinate_system(center, auto_size, scale_factor=scale_factor)
        except Exception as e:
            logger.error(f"自动设置坐标系位置和大小时出错: {e}") if logger else None
            return False

