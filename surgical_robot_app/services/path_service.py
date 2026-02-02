"""
PathService: 路径规划业务逻辑服务

职责：
- 协调障碍物数据的准备（从 Volume 或 STL）
- 调用 PathController 进行路径规划
- 处理路径数据的转换与格式化
- (未来) 处理机器人指令封装
"""

import logging
from typing import Tuple, Optional, List, Callable
import numpy as np
try:
    from vtkmodules.vtkRenderingCore import vtkRenderer
except ImportError:
    vtkRenderer = None

from surgical_robot_app.path_planning.path_controller import PathController
from surgical_robot_app.utils.logger import get_logger

logger = get_logger("surgical_robot_app.services.path_service")

class PathService:
    def __init__(self, path_controller: PathController):
        self.path_controller = path_controller

    def prepare_obstacles(self, data_manager, vtk_renderer: vtkRenderer) -> bool:
        """
        准备障碍物数据。优先从分割结果获取，其次从 STL 模型获取。
        
        Returns:
            bool: 是否成功设置了障碍物
        """
        obstacle_set = False
        
        # 1. 尝试从分割体数据设置
        if data_manager:
            try:
                # 检查 seg_mask_volume 是否存在且有效
                if hasattr(data_manager, 'seg_mask_volume') and data_manager.seg_mask_volume is not None:
                    seg_mask_volume = data_manager.seg_mask_volume
                    # 获取体素间距
                    spacing = None
                    metadata = data_manager.get_metadata()
                    if isinstance(metadata, dict) and "Spacing" in metadata:
                        s = metadata["Spacing"]
                        if isinstance(s, (list, tuple)) and len(s) == 3:
                            spacing = (float(s[0]), float(s[1]), float(s[2]))
                    
                    self.path_controller.set_obstacle_from_volume(seg_mask_volume, spacing=spacing)
                    obstacle_set = True
                    logger.info("Obstacles set from segmentation volume")
            except Exception as e:
                logger.warning(f"Failed to set obstacles from volume: {e}")
        
        # 2. 尝试从 STL 模型设置
        if not obstacle_set and vtk_renderer:
            try:
                actors = vtk_renderer.GetActors()
                actors.InitTraversal()
                while True:
                    actor = actors.GetNextItem()
                    if actor is None: break
                    
                    mapper = actor.GetMapper()
                    if mapper:
                        input_data = mapper.GetInput()
                        # 简单的启发式判断是否是解剖模型（点数较多）
                        if input_data and hasattr(input_data, 'GetNumberOfPoints') and input_data.GetNumberOfPoints() > 100:
                            result = self.path_controller.set_obstacle_from_mesh(input_data)
                            if result is not None:
                                obstacle_set = True
                                logger.info(f"Obstacles set from STL model ({input_data.GetNumberOfPoints()} points)")
                                break
            except Exception as e:
                logger.warning(f"Failed to set obstacles from STL: {e}")
                
        return obstacle_set

    def plan_path(self, smooth: bool = True, progress_callback: Optional[Callable[[int], None]] = None) -> List[Tuple[float, float, float]]:
        """调用控制器规划路径"""
        try:
            return self.path_controller.generate_path(smooth=smooth, progress_callback=progress_callback)
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            raise e

    def plan_local_segment(self, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> Optional[List[Tuple[float, float, float]]]:
        """
        为特定段规划局部避障路径
        """
        if not self.path_controller.rrt_planner:
            return None
            
        try:
            # 统一坐标系：转换到规划空间（物理坐标优先）
            start_planner = self.path_controller._space_to_physical(start)
            end_planner = self.path_controller._space_to_physical(end)

            # 使用较小的迭代次数进行快速局部规划
            # 暂时修改 planner 的参数以适应局部规划需求
            old_iter = self.path_controller.rrt_planner.max_iterations
            self.path_controller.rrt_planner.max_iterations = 2000 
            
            segment = self.path_controller.rrt_planner.plan(start_planner, end_planner)
            
            # 恢复参数
            self.path_controller.rrt_planner.max_iterations = old_iter
            if segment is None:
                return None
            # 转回 UI 使用的归一化空间坐标
            return self.path_controller._convert_points_from_planner_space(segment)
        except Exception as e:
            logger.warning(f"局部避障规划失败: {e}")
            return None

    def generate_simple_path(self) -> List[Tuple[float, float, float]]:
        """调用控制器规划直线路径"""
        return self.path_controller.generate_simple_path()

