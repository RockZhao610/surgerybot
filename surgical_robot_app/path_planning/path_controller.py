"""
PathController: 路径规划控制器

职责：
- 管理路径规划相关的数据（起点/终点/中间点、最终路径点）
- 支持 A* 和 RRT 两种路径规划算法
- 维护障碍物数据（栅格、点云或 SDF）
- 支持 SDF (Signed Distance Field) 碰撞检测，解决点云内部空洞问题

注意：
- 不依赖 Qt，仅依赖路径规划器和 numpy，方便单元测试与复用。
- 可以通过 config.json 配置所有参数
"""

from typing import List, Tuple, Optional, Callable, Union, TYPE_CHECKING

import numpy as np
import logging

try:
    from surgical_robot_app.path_planning.a_star_planner import (
        AStarPlanner,
        create_obstacle_grid,
    )
    from surgical_robot_app.path_planning.rrt_planner import RRTPlanner
    from surgical_robot_app.path_planning.point_cloud_utils import (
        PointCloudCollisionChecker,
        volume_to_point_cloud,
        mesh_to_point_cloud,
    )
    from surgical_robot_app.path_planning.sdf_utils import (
        SDFCollisionChecker,
        compute_sdf_from_mask,
        create_sdf_collision_checker,
    )
except ImportError:
    from path_planning.a_star_planner import (
        AStarPlanner,
        create_obstacle_grid,
    )
    from path_planning.rrt_planner import RRTPlanner
    from path_planning.point_cloud_utils import (
        PointCloudCollisionChecker,
        volume_to_point_cloud,
        mesh_to_point_cloud,
    )
    from path_planning.sdf_utils import (
        SDFCollisionChecker,
        compute_sdf_from_mask,
        create_sdf_collision_checker,
    )

# 类型检查时导入配置类型
if TYPE_CHECKING:
    from surgical_robot_app.config.settings import PathPlanningConfig

# 碰撞检测器类型（兼容点云和 SDF）
CollisionCheckerType = Union[PointCloudCollisionChecker, SDFCollisionChecker, None]

logger = logging.getLogger(__name__)

SpacePoint = Tuple[float, float, float]  # (x, y, z) in [0, 100]


class PathController:
    """
    路径规划控制器
    
    支持两种创建方式：
    1. 直接传参: PathController(use_rrt=True, ...)
    2. 从配置创建: PathController.from_config() 或 PathController.from_config(config)
    """
    
    def __init__(
        self,
        planner: Optional[AStarPlanner] = None,
        grid_size: Tuple[int, int, int] = (100, 100, 105),
        obstacle_expansion: int = 4,
        use_rrt: bool = True,  # 默认使用RRT
        rrt_step_size: float = 2.0,
        rrt_safety_radius: float = 5.0,  # 增加默认安全半径从3.0到5.0
        use_sdf: bool = True,  # 默认使用 SDF 碰撞检测（解决内部空洞问题）
    ):
        """
        初始化路径规划控制器

        Args:
            planner: AStarPlanner 实例（如果使用A*）；如果为 None，则内部创建一个
            grid_size: A* 使用的栅格大小（如果使用A*）
            obstacle_expansion: 障碍物在 X/Y 方向的膨胀大小（Z 不膨胀，仅用于A*）
            use_rrt: 是否使用RRT算法（True）还是A*算法（False）
            rrt_step_size: RRT步长
            rrt_safety_radius: RRT安全半径
            use_sdf: 是否使用 SDF（有符号距离场）进行碰撞检测
                     - True: 使用 SDF，完美区分内外，解决点云内部空洞问题
                     - False: 使用传统点云方式
        """
        self.use_rrt = use_rrt
        self.use_sdf = use_sdf
        self.grid_size = grid_size
        self.obstacle_expansion = obstacle_expansion
        
        # 根据算法类型初始化规划器
        if use_rrt:
            self.rrt_planner: Optional[RRTPlanner] = None
            self.a_star_planner: Optional[AStarPlanner] = None
        else:
            self.a_star_planner: AStarPlanner = planner or AStarPlanner(grid_size=grid_size)
            self.rrt_planner: Optional[RRTPlanner] = None
        
        self.rrt_step_size = rrt_step_size
        self.rrt_safety_radius = rrt_safety_radius
        
        # 路径简化参数
        self.path_simplify_enabled: bool = True  # 是否启用路径简化
        self.path_max_points: Optional[int] = 50  # 最大路径点数（None 表示不限制）
        self.path_simplify_tolerance: float = 1.0  # 简化容差（点到直线的最大偏离）

        self.start_point: Optional[SpacePoint] = None
        self.end_point: Optional[SpacePoint] = None
        self.waypoints: List[SpacePoint] = []
        self._path_points: List[SpacePoint] = []
        
        # 撤销/重做栈
        self.undo_stack: List[dict] = []
        self.redo_stack: List[dict] = []

        # 障碍物数据（根据算法类型使用不同的格式）
        self._obstacle_grid: Optional[np.ndarray] = None  # A*使用
        self._point_cloud: Optional[np.ndarray] = None  # RRT点云使用
        self._sdf: Optional[np.ndarray] = None  # SDF 数据
        self._sdf_metadata: Optional[dict] = None  # SDF 元数据
        self._collision_checker: CollisionCheckerType = None  # 碰撞检测器（点云或SDF）
        
        # 物理坐标相关（用于体数据：单位mm）
        self._use_physical_coords: bool = False
        self._volume_shape: Optional[Tuple[int, int, int]] = None
        self._volume_spacing: Optional[Tuple[float, float, float]] = None
        self._space_bounds_physical: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
        self._physical_path_points: List[SpacePoint] = []
        
        # 保存初始状态
        self._save_state()
    
    @classmethod
    def from_config(cls, config: Optional['PathPlanningConfig'] = None) -> 'PathController':
        """
        从配置文件创建 PathController 实例
        
        Args:
            config: PathPlanningConfig 实例，如果为 None 则从全局配置获取
        
        Returns:
            PathController 实例
        
        示例:
            # 使用全局配置（从 config.json 加载）
            controller = PathController.from_config()
            
            # 使用自定义配置
            from surgical_robot_app.config.settings import PathPlanningConfig
            config = PathPlanningConfig(use_rrt=True, safety_radius=8.0)
            controller = PathController.from_config(config)
        """
        if config is None:
            # 从全局配置获取
            try:
                from surgical_robot_app.config.settings import get_config
                app_config = get_config()
                config = app_config.path_planning
            except ImportError:
                from config.settings import get_config
                app_config = get_config()
                config = app_config.path_planning
        
        # 创建实例
        instance = cls(
            grid_size=config.grid_size,
            obstacle_expansion=config.obstacle_expansion,
            use_rrt=config.use_rrt,
            rrt_step_size=config.rrt_step_size,
            rrt_safety_radius=config.safety_radius,
            use_sdf=config.use_sdf,
        )
        
        # 设置路径简化参数
        instance.path_simplify_enabled = config.simplify_enabled
        instance.path_max_points = config.get_max_points()
        instance.path_simplify_tolerance = config.simplify_tolerance
        
        logger.info(f"从配置创建 PathController: "
                    f"use_rrt={config.use_rrt}, use_sdf={config.use_sdf}, "
                    f"safety_radius={config.safety_radius}, "
                    f"simplify_max_points={config.simplify_max_points}")
        
        return instance

    # -------------------- 路径点管理 --------------------

    def _save_state(self) -> None:
        """保存当前路径点状态到撤销栈"""
        state = {
            'start_point': self.start_point,
            'end_point': self.end_point,
            'waypoints': list(self.waypoints),
            'path_points': list(self._path_points)
        }
        
        # 如果新状态与栈顶状态相同，则不保存（深比较）
        if self.undo_stack and self.undo_stack[-1] == state:
            return
            
        self.undo_stack.append(state)
        # 限制栈大小
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        
        # 每次手动保存状态时，清空重做栈
        # 注意：undo/redo 过程中不调用这个方法
        self.redo_stack.clear()
        logger.debug(f"Path state saved. Undo stack size: {len(self.undo_stack)}")

    def undo(self) -> bool:
        """执行撤销操作"""
        if len(self.undo_stack) <= 1:
            logger.info("Nothing to undo")
            return False
            
        # 将当前状态保存到重做栈
        current_state = {
            'start_point': self.start_point,
            'end_point': self.end_point,
            'waypoints': list(self.waypoints),
            'path_points': list(self._path_points)
        }
        self.redo_stack.append(current_state)
        
        # 弹出当前状态（栈顶是当前状态）
        self.undo_stack.pop()
        
        # 应用上一个状态
        prev_state = self.undo_stack[-1]
        self._apply_state(prev_state)
        
        logger.info(f"Undo successful. Undo stack: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)}")
        return True

    def redo(self) -> bool:
        """执行重做操作"""
        if not self.redo_stack:
            logger.info("Nothing to redo")
            return False
            
        # 将当前状态保存到撤销栈
        current_state = {
            'start_point': self.start_point,
            'end_point': self.end_point,
            'waypoints': list(self.waypoints),
            'path_points': list(self._path_points)
        }
        self.undo_stack.append(current_state)
        
        # 弹出重做状态
        next_state = self.redo_stack.pop()
        
        # 应用新状态
        self._apply_state(next_state)
        
        logger.info(f"Redo successful. Undo stack: {len(self.undo_stack)}, Redo stack: {len(self.redo_stack)}")
        return True

    @property
    def path_points(self) -> List[SpacePoint]:
        return self._path_points

    @path_points.setter
    def path_points(self, points: List[SpacePoint]) -> None:
        self._path_points = list(points)
        self._save_state()

    def _apply_state(self, state: dict) -> None:
        """应用保存的状态"""
        self.start_point = state['start_point']
        self.end_point = state['end_point']
        self.waypoints = list(state['waypoints'])
        self._path_points = list(state['path_points'])  # 直接赋值，不触发 save_state

    def set_start(self, coord: SpacePoint) -> None:
        self.start_point = coord
        self._save_state()

    def add_waypoint(self, coord: SpacePoint) -> None:
        self.waypoints.append(coord)
        self._save_state()

    def set_end(self, coord: SpacePoint) -> None:
        self.end_point = coord
        self._save_state()

    def clear_path(self) -> None:
        """清除起点/终点/中间点和已生成路径"""
        self.start_point = None
        self.end_point = None
        self.waypoints = []
        self.path_points = []
        self._save_state()
    
    def set_path_simplification(
        self,
        enabled: bool = True,
        max_points: Optional[int] = 50,
        tolerance: float = 1.0
    ) -> None:
        """
        配置路径简化参数
        
        Args:
            enabled: 是否启用路径简化
            max_points: 最大路径点数（None 表示不限制）
                        较小的值 = 更少的点，但可能丢失细节
            tolerance: RDP 简化容差（点到直线的最大偏离距离）
                       较大的值 = 更激进的简化
        
        示例:
            # 只保留 20 个点
            controller.set_path_simplification(max_points=20)
            
            # 禁用简化（保留所有点）
            controller.set_path_simplification(enabled=False)
            
            # 激进简化
            controller.set_path_simplification(max_points=10, tolerance=3.0)
        """
        self.path_simplify_enabled = enabled
        self.path_max_points = max_points
        self.path_simplify_tolerance = tolerance
        logger.info(f"路径简化配置: enabled={enabled}, max_points={max_points}, tolerance={tolerance}")

    # -------------------- 障碍物数据管理 --------------------

    def set_obstacle_from_volume(
        self,
        seg_mask_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        从 3D 分割体数据生成障碍物数据（SDF、点云或栅格）

        Args:
            seg_mask_volume: 3D 掩码体数据 (Z, H, W)
            spacing: 体素间距（可选，用于物理坐标转换）

        Returns:
            如果使用 SDF/RRT，返回 SDF 或点云数组；如果使用 A*，返回障碍物栅格
        """
        # 记录体数据物理信息（用于坐标转换）
        if spacing is not None and seg_mask_volume is not None:
            self._volume_shape = seg_mask_volume.shape[:3]
            self._volume_spacing = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
            self._use_physical_coords = True
            self._space_bounds_physical = self._build_physical_bounds(self._volume_shape, self._volume_spacing)
            if not self.use_rrt and self.a_star_planner:
                self.a_star_planner.space_bounds = self._space_bounds_physical
        else:
            self._use_physical_coords = False
            self._space_bounds_physical = None
            if not self.use_rrt and self.a_star_planner:
                self.a_star_planner.space_bounds = None

        if self.use_rrt:
            # 根据 use_sdf 选择碰撞检测方式
            if self.use_sdf:
                return self._set_obstacle_from_volume_sdf(seg_mask_volume, spacing)
            else:
                return self._set_obstacle_from_volume_pointcloud(seg_mask_volume, spacing)
        else:
            # 使用A*：生成栅格
            obstacle_grid = create_obstacle_grid(
                seg_mask_volume,
                grid_size=self.grid_size,
                expansion=self.obstacle_expansion,
            )
            self._obstacle_grid = obstacle_grid
            self.a_star_planner.set_obstacle_grid(obstacle_grid)
            logger.info(f"A*障碍物栅格设置完成: {obstacle_grid.shape}")
            return obstacle_grid
    
    def _set_obstacle_from_volume_sdf(
        self,
        seg_mask_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        使用 SDF（有符号距离场）设置障碍物
        
        SDF 的优势：
        1. 完美区分内部/外部，不存在内部空洞问题
        2. 直接获取精确距离，无需最近邻搜索
        3. 支持三线性插值，检测更平滑
        
        Args:
            seg_mask_volume: 3D 掩码体数据 (Z, H, W)
            spacing: 体素间距
        
        Returns:
            SDF 数组
        """
        effective_safety_radius = max(self.rrt_safety_radius, 5.0)
        
        # 计算 SDF
        sdf, metadata = compute_sdf_from_mask(
            seg_mask_volume,
            spacing=spacing,
            normalize_to_100=(spacing is None)
        )
        
        self._sdf = sdf
        self._sdf_metadata = metadata
        
        # 创建 SDF 碰撞检测器
        self._collision_checker = SDFCollisionChecker(
            sdf=sdf,
            metadata=metadata,
            safety_radius=effective_safety_radius,
            use_interpolation=True  # 使用三线性插值提高精度
        )
        
        # 创建 RRT 规划器
        self.rrt_planner = RRTPlanner(
            self._collision_checker,
            step_size=self.rrt_step_size,
            goal_bias=0.1,
            max_iterations=5000,
            goal_threshold=2.0,
            bounds=self._space_bounds_physical if self._use_physical_coords else None,
        )
        
        logger.info(f"SDF 障碍物设置完成: shape={sdf.shape}, "
                    f"range=[{metadata['sdf_range'][0]:.2f}, {metadata['sdf_range'][1]:.2f}], "
                    f"safety_radius={effective_safety_radius}")
        
        return sdf
    
    def _set_obstacle_from_volume_pointcloud(
        self,
        seg_mask_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> np.ndarray:
        """
        使用传统点云方式设置障碍物（原有实现）
        
        Args:
            seg_mask_volume: 3D 掩码体数据 (Z, H, W)
            spacing: 体素间距
        
        Returns:
            点云数组
        """
        # 生成点云
        point_cloud = volume_to_point_cloud(
            seg_mask_volume,
            threshold=128,
            downsample_factor=1,
            spacing=spacing
        )
        self._point_cloud = point_cloud
        
        # 创建碰撞检测器
        effective_safety_radius = max(self.rrt_safety_radius, 5.0)
        
        self._collision_checker = PointCloudCollisionChecker(
            point_cloud,
            safety_radius=effective_safety_radius
        )
        
        # 创建 RRT 规划器
        self.rrt_planner = RRTPlanner(
            self._collision_checker,
            step_size=self.rrt_step_size,
            goal_bias=0.1,
            max_iterations=5000,
            goal_threshold=2.0,
            bounds=self._space_bounds_physical if self._use_physical_coords else None,
        )
        
        logger.info(f"点云障碍物设置完成: {len(point_cloud)} 个点")
        return point_cloud
    
    def _create_sdf_from_mesh(self, poly_data) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        从 VTK 网格创建 SDF（通过体素化）
        
        Args:
            poly_data: VTK PolyData 对象
        
        Returns:
            (sdf, metadata) 元组
        """
        try:
            from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
        except ImportError:
            logger.warning("VTK vtkImplicitPolyDataDistance 不可用")
            return None, None
        
        # 获取网格边界
        bounds = poly_data.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # 计算网格尺寸（归一化到 0-100 空间）
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]
        max_range = max(x_range, y_range, z_range)
        
        # 体素化分辨率（每个维度的体素数）
        resolution = 100
        
        # 创建隐式距离函数
        implicit_distance = vtkImplicitPolyDataDistance()
        implicit_distance.SetInput(poly_data)
        
        # 创建 SDF 数组
        sdf = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        
        # 计算每个体素的 SDF 值
        for k in range(resolution):
            z = bounds[4] + (k / (resolution - 1)) * z_range
            for j in range(resolution):
                y = bounds[2] + (j / (resolution - 1)) * y_range
                for i in range(resolution):
                    x = bounds[0] + (i / (resolution - 1)) * x_range
                    # VTK 的隐式距离：负值在内部，正值在外部
                    sdf[k, j, i] = implicit_distance.EvaluateFunction(x, y, z)
        
        # 不需要归一化 SDF 值，保持原始物理距离
        # sdf 值已经是在原始坐标空间中的距离
        
        # 创建元数据（使用原始世界坐标，与 VTK 场景对齐）
        metadata = {
            'bounds': {
                'x': (bounds[0], bounds[1]),  # 原始 X 边界
                'y': (bounds[2], bounds[3]),  # 原始 Y 边界
                'z': (bounds[4], bounds[5]),  # 原始 Z 边界
            },
            'spacing': (x_range / resolution, y_range / resolution, z_range / resolution),
            'sdf_range': (float(sdf.min()), float(sdf.max())),
            'original_bounds': bounds,
        }
        
        logger.info(f"从网格创建 SDF: shape={sdf.shape}, "
                    f"bounds=({bounds[0]:.1f}-{bounds[1]:.1f}, {bounds[2]:.1f}-{bounds[3]:.1f}, {bounds[4]:.1f}-{bounds[5]:.1f}), "
                    f"range=[{sdf.min():.2f}, {sdf.max():.2f}]")
        return sdf, metadata

    def set_obstacle_from_mesh(self, poly_data) -> Optional[np.ndarray]:
        """
        从网格（VTK PolyData）生成障碍物数据（支持 SDF 和点云）

        Args:
            poly_data: VTK PolyData 对象

        Returns:
            点云数组或 SDF 数据
        """
        if not self.use_rrt:
            logger.warning("set_obstacle_from_mesh 仅支持RRT算法")
            return None

        # STL模型保持使用归一化空间坐标
        self._use_physical_coords = False
        self._space_bounds_physical = None
        
        # 从网格生成点云（增加点数以提高碰撞检测精度）
        point_cloud = mesh_to_point_cloud(poly_data, num_points=5000, downsample_factor=1)
        if point_cloud is None:
            return None
        
        self._point_cloud = point_cloud
        
        # STL模型使用更大的安全半径
        mesh_safety_radius = self.rrt_safety_radius
        
        # 如果使用 SDF，尝试从网格创建 SDF
        if self.use_sdf:
            try:
                sdf, metadata = self._create_sdf_from_mesh(poly_data)
                if sdf is not None:
                    self._sdf = sdf
                    self._sdf_metadata = metadata
                    self._collision_checker = SDFCollisionChecker(
                        sdf, metadata, safety_radius=mesh_safety_radius
                    )
                    logger.info(f"从 STL 网格创建 SDF 碰撞检测器: shape={sdf.shape}")
                    
                    # 初始化 RRT 规划器
                    self.rrt_planner = RRTPlanner(
                        self._collision_checker,
                        step_size=3.0,
                        goal_bias=0.2,
                        max_iterations=15000,
                        goal_threshold=self.rrt_step_size
                    )
                    return sdf
            except Exception as e:
                logger.warning(f"从网格创建 SDF 失败，回退到点云: {e}")
        
        # 回退到点云碰撞检测
        self._collision_checker = PointCloudCollisionChecker(
            point_cloud,
            safety_radius=mesh_safety_radius
        )
        
        # STL模型使用更激进的RRT参数
        self.rrt_planner = RRTPlanner(
            self._collision_checker,
            step_size=3.0,  # 更大的步长
            goal_bias=0.2,  # 更高的目标偏向
            max_iterations=15000,  # 更多迭代
            goal_threshold=5.0  # 更大的目标阈值
        )
        
        logger.info(f"从网格生成RRT障碍物: {len(point_cloud)} 个点, 安全半径={mesh_safety_radius}")
        return point_cloud
    
    def generate_simple_path(self) -> List[SpacePoint]:
        """
        生成简单直线路径（当RRT失败时的备选方案）
        会经过所有已设置的 waypoints
        
        Returns:
            路径点列表
        """
        if self.start_point is None or self.end_point is None:
            return []
        
        # 构建完整的路径点列表：起点 -> 中间点 -> 终点
        key_points = self.build_waypoint_list()
        
        if len(key_points) < 2:
            return []
        
        # 在每对关键点之间插入中间点
        path = []
        # 每段之间插入的点数，根据距离动态计算
        for i in range(len(key_points) - 1):
            p1 = key_points[i]
            p2 = key_points[i + 1]
            
            # 计算两点间距离
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            # 每隔 2.0 单位插入一个点
            num_segments = max(1, int(dist / 2.0))
            
            # 添加起始点（如果是第一段，则包含起始点；否则起始点就是上一段的终点，已添加）
            if i == 0:
                path.append(p1)
            
            # 插入中间插值点
            for j in range(1, num_segments + 1):
                t = j / num_segments
                x = p1[0] * (1 - t) + p2[0] * t
                y = p1[1] * (1 - t) + p2[1] * t
                z = p1[2] * (1 - t) + p2[2] * t
                path.append((x, y, z))
        
        self.path_points = path
        logger.info(f"生成简单直线路径: {len(path)} 个点 (经过 {len(key_points)} 个关键点)")
        return path

    def has_obstacle_data(self) -> bool:
        """检查是否已设置障碍物数据"""
        if self.use_rrt:
            if self.use_sdf:
                # 检查 SDF 数据
                return self._sdf is not None and self._sdf.size > 0
            else:
                # 检查点云数据
                return self._point_cloud is not None and len(self._point_cloud) > 0
        else:
            return self._obstacle_grid is not None
    
    def get_collision_checker_type(self) -> str:
        """获取当前使用的碰撞检测器类型"""
        if self._collision_checker is None:
            return "none"
        elif isinstance(self._collision_checker, SDFCollisionChecker):
            return "sdf"
        elif isinstance(self._collision_checker, PointCloudCollisionChecker):
            return "pointcloud"
        else:
            return "unknown"
    
    def check_instrument_collision(
        self,
        instrument_radius: float = 2.0
    ) -> Tuple[bool, List[int], float]:
        """
        检查当前路径是否对给定半径的器械安全
        
        考虑器械的实际体积，而不仅仅是路径线
        
        Args:
            instrument_radius: 器械半径
        
        Returns:
            (is_safe, collision_segments, max_safe_radius):
            - is_safe: 路径是否安全
            - collision_segments: 碰撞发生的路径段索引
            - max_safe_radius: 该路径允许的最大器械半径
        """
        if not self._path_points or len(self._path_points) < 2:
            return True, [], 0.0
        
        # 使用规划空间的坐标
        path = self.get_planner_path_points()
        if not path:
            path = self._path_points
        
        # 如果是 SDF 检测器，使用器械碰撞检测
        if isinstance(self._collision_checker, SDFCollisionChecker):
            is_safe, collision_indices = self._collision_checker.is_path_safe_for_instrument(
                path, instrument_radius
            )
            max_radius = self._collision_checker.get_min_safe_radius(path)
            return is_safe, collision_indices, max_radius
        
        # 点云检测器：简单检查每个点
        collision_indices = []
        min_distance = float('inf')
        
        for i, point in enumerate(path):
            if self._collision_checker is not None:
                dist = self._collision_checker.get_distance_to_obstacle(point)
                min_distance = min(min_distance, dist)
                if dist < instrument_radius:
                    collision_indices.append(i)
        
        return len(collision_indices) == 0, collision_indices, max(0, min_distance)

    # -------------------- 路径规划 --------------------

    def can_generate_path(self) -> bool:
        """检查是否具备生成路径的必要条件"""
        return self.start_point is not None and self.end_point is not None

    def build_waypoint_list(self) -> List[SpacePoint]:
        """按顺序组合起点-中间点-终点"""
        if self.start_point is None or self.end_point is None:
            return []
        waypoints: List[SpacePoint] = [self.start_point]
        waypoints.extend(self.waypoints)
        waypoints.append(self.end_point)
        logger.info(f"build_waypoint_list: start={self.start_point}, waypoints={self.waypoints}, end={self.end_point}")
        logger.info(f"build_waypoint_list 返回 {len(waypoints)} 个关键点: {waypoints}")
        return waypoints

    def generate_path(
        self,
        smooth: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
        tree_callback: Optional[Callable[[dict], None]] = None
    ) -> List[SpacePoint]:
        """
        生成多段路径（起点 -> 中间点... -> 终点），并保存到 self.path_points

        Raises:
            RuntimeError: 若起点/终点未设置或未设置障碍物数据
        """
        if not self.can_generate_path():
            raise RuntimeError("Start or end point not set")
        
        if not self.has_obstacle_data():
            if self.use_rrt:
                raise RuntimeError("Point cloud not set; call set_obstacle_from_volume or set_obstacle_from_mesh first")
            else:
                raise RuntimeError("Obstacle grid not set; call set_obstacle_from_volume first")

        waypoints = self.build_waypoint_list()
        if len(waypoints) < 2:
            raise RuntimeError("At least start and end points are required")

        # 如果使用物理坐标，先转换
        planner_waypoints = self._convert_points_to_planner_space(waypoints)

        # 根据算法类型选择规划器
        if self.use_rrt:
            if self.rrt_planner is None:
                raise RuntimeError("RRT planner not initialized")
            path = self.rrt_planner.plan_multi_segment(
                planner_waypoints, smooth=smooth,
                progress_callback=progress_callback,
                tree_callback=tree_callback
            )
            if path is None or len(path) == 0:
                raise RuntimeError("RRT planner returned empty path")
        else:
            if self.a_star_planner is None:
                raise RuntimeError("A* planner not initialized")
            # A* 目前不支持进度回调，但运行很快
            path = self.a_star_planner.plan_multi_segment(planner_waypoints, smooth=smooth)
            if path is None or len(path) == 0:
                raise RuntimeError("A* planner returned empty path")

        # 统一后处理：重采样 -> 平滑 -> 碰撞复验（在规划空间内执行）
        path = self._postprocess_path(path)
        self._physical_path_points = path if self._use_physical_coords else []

        # 规划完成后，转换回 UI 使用的空间坐标
        path_space = self._convert_points_from_planner_space(path)
        self.path_points = path_space
        return path_space

    def get_planner_path_points(self) -> List[SpacePoint]:
        """获取用于碰撞评估的路径坐标（物理坐标优先）"""
        if self._use_physical_coords and self._physical_path_points:
            return self._physical_path_points
        return self.path_points

    # -------------------- 路径后处理 --------------------

    def _postprocess_path(self, path: List[SpacePoint]) -> List[SpacePoint]:
        """
        对路径执行统一后处理：
        1) 路径简化（去除不必要的中间点）
        2) 可选的重采样
        3) 平滑并做碰撞复验
        4) 保持起点/中间点/终点不被移动
        """
        if not path or len(path) < 2:
            return path
        
        logger.debug(f"后处理开始: 原始路径 {len(path)} 个点")

        # 以关键点（起点/中间点/终点）为锚点分段处理，确保关键点不漂移
        anchors = self.build_waypoint_list()
        if len(anchors) < 2:
            return path

        # 将锚点转换到规划空间（如果使用物理坐标）
        anchors = self._convert_points_to_planner_space(anchors)

        # 按顺序寻找每个 anchor 在 path 中的最近索引
        path_np = np.array(path)
        anchor_indices: List[int] = []
        last_idx = 0
        for anchor in anchors:
            if last_idx >= len(path):
                anchor_indices.append(len(path) - 1)
                continue
            anchor_np = np.array(anchor)
            remaining = path_np[last_idx:]
            dists = np.linalg.norm(remaining - anchor_np, axis=1)
            nearest_offset = int(np.argmin(dists))
            nearest_idx = last_idx + nearest_offset
            anchor_indices.append(nearest_idx)
            last_idx = nearest_idx

        # 去重并确保索引单调递增
        cleaned_indices: List[int] = []
        for idx in anchor_indices:
            if not cleaned_indices or idx > cleaned_indices[-1]:
                cleaned_indices.append(idx)
        if len(cleaned_indices) < 2:
            return path

        # 计算每段允许的最大点数
        num_segments = len(cleaned_indices) - 1
        max_points_per_segment = None
        if self.path_max_points is not None:
            max_points_per_segment = max(3, self.path_max_points // num_segments)

        # 分段处理
        processed: List[SpacePoint] = []
        for i in range(num_segments):
            s_idx = cleaned_indices[i]
            e_idx = cleaned_indices[i + 1]
            segment = path[s_idx:e_idx + 1]
            if len(segment) < 2:
                continue

            # 1) 路径简化（如果启用）
            if self.path_simplify_enabled and len(segment) > 3:
                # 先用碰撞检测简化（保证安全）
                if self._collision_checker is not None:
                    segment = self._simplify_path_by_collision(segment)
                # 再用 RDP 算法简化（控制点数）
                if len(segment) > max_points_per_segment if max_points_per_segment else 10:
                    segment = self._simplify_path_rdp(segment, tolerance=self.path_simplify_tolerance)

            # 2) 如果简化后点数仍然太多，进行重采样
            if max_points_per_segment and len(segment) > max_points_per_segment:
                segment = self._resample_path(segment, step_size=self.rrt_step_size, max_points=max_points_per_segment)

            # 3) 平滑 + 碰撞复验（仅在有碰撞检测器时）
            smoothed = segment
            if self._collision_checker is not None and len(segment) > 2:
                smoothed = self._smooth_path_moving_average(segment, window_size=3)
                if not self._is_path_collision_free(smoothed, self._collision_checker):
                    smoothed = segment

            # 4) 拼接（避免重复点）
            if processed:
                processed.extend(smoothed[1:])
            else:
                processed.extend(smoothed)

        logger.info(f"后处理完成: {len(path)} -> {len(processed)} 个点")
        return processed if processed else path

    def _resample_path(self, path: List[SpacePoint], step_size: float = 2.0, max_points: Optional[int] = None) -> List[SpacePoint]:
        """
        沿路径弧长进行等距重采样，保持首尾不变。
        
        Args:
            path: 原始路径点列表
            step_size: 采样步长
            max_points: 最大点数限制（可选），如果设置会自动调整步长
        """
        if len(path) < 2:
            return path

        pts = np.array(path, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        total_len = float(np.sum(seg_lens))
        if total_len <= 1e-6:
            return path

        # 计算目标采样点数
        num_samples = max(2, int(total_len / max(step_size, 1e-6)) + 1)
        
        # 如果设置了最大点数限制，调整采样数
        if max_points is not None and num_samples > max_points:
            num_samples = max(2, max_points)
            logger.debug(f"路径重采样: 限制点数从 {int(total_len / step_size) + 1} 到 {num_samples}")
        
        target_dists = np.linspace(0.0, total_len, num_samples)

        # 累计弧长
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lens)])
        resampled = []
        seg_idx = 0
        for d in target_dists:
            while seg_idx < len(seg_lens) - 1 and d > cumulative[seg_idx + 1]:
                seg_idx += 1
            seg_start = pts[seg_idx]
            seg_end = pts[seg_idx + 1]
            seg_len = seg_lens[seg_idx]
            if seg_len <= 1e-6:
                resampled.append(tuple(seg_start))
                continue
            t = (d - cumulative[seg_idx]) / seg_len
            p = seg_start * (1 - t) + seg_end * t
            resampled.append((float(p[0]), float(p[1]), float(p[2])))

        # 保证首尾严格一致
        resampled[0] = path[0]
        resampled[-1] = path[-1]
        return resampled

    def _smooth_path_moving_average(self, path: List[SpacePoint], window_size: int = 3) -> List[SpacePoint]:
        """移动平均平滑，保持首尾不变。"""
        if len(path) <= 2 or window_size <= 1:
            return path

        half = window_size // 2
        smoothed = [path[0]]
        for i in range(1, len(path) - 1):
            start = max(0, i - half)
            end = min(len(path), i + half + 1)
            window = np.array(path[start:end], dtype=float)
            avg = np.mean(window, axis=0)
            smoothed.append((float(avg[0]), float(avg[1]), float(avg[2])))
        smoothed.append(path[-1])
        return smoothed
    
    def _simplify_path_rdp(self, path: List[SpacePoint], tolerance: float = 1.0) -> List[SpacePoint]:
        """
        使用 Ramer-Douglas-Peucker 算法简化路径
        
        该算法会保留路径的整体形状，同时去除不必要的中间点。
        
        Args:
            path: 原始路径点列表
            tolerance: 简化容差，点到直线的最大允许偏离距离
                       较大的值 = 更少的点，但可能偏离原路径更多
        
        Returns:
            简化后的路径
        """
        if len(path) <= 2:
            return path
        
        def point_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
            """计算点到线段的距离"""
            line_vec = line_end - line_start
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-10:
                return np.linalg.norm(point - line_start)
            
            # 投影参数 t
            t = max(0, min(1, np.dot(point - line_start, line_vec) / (line_len ** 2)))
            projection = line_start + t * line_vec
            return np.linalg.norm(point - projection)
        
        def rdp_recursive(points: np.ndarray, start: int, end: int, tolerance: float) -> List[int]:
            """递归实现 RDP 算法"""
            if end <= start + 1:
                return [start]
            
            # 找到距离首尾连线最远的点
            max_dist = 0.0
            max_idx = start
            line_start = points[start]
            line_end = points[end]
            
            for i in range(start + 1, end):
                dist = point_line_distance(points[i], line_start, line_end)
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            # 如果最大距离超过容差，递归处理两段
            if max_dist > tolerance:
                left = rdp_recursive(points, start, max_idx, tolerance)
                right = rdp_recursive(points, max_idx, end, tolerance)
                return left + right
            else:
                return [start]
        
        points_array = np.array(path)
        indices = rdp_recursive(points_array, 0, len(path) - 1, tolerance)
        indices.append(len(path) - 1)  # 确保包含终点
        
        simplified = [path[i] for i in indices]
        logger.debug(f"RDP 路径简化: {len(path)} -> {len(simplified)} 点 (容差={tolerance})")
        return simplified
    
    def _simplify_path_by_collision(self, path: List[SpacePoint]) -> List[SpacePoint]:
        """
        基于碰撞检测的路径简化
        
        尝试跳过中间点，只保留必要的转折点。
        这种方法保证简化后的路径仍然是安全的。
        
        Args:
            path: 原始路径点列表
        
        Returns:
            简化后的路径
        """
        if len(path) <= 2 or self._collision_checker is None:
            return path
        
        simplified = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 尝试直接连接到尽可能远的点
            best_next = current_idx + 1
            
            # 从末尾向前搜索，找到可以直接连接的最远点
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                if self._collision_checker.is_path_collision_free(path[current_idx], path[next_idx]):
                    best_next = next_idx
                    break
            
            simplified.append(path[best_next])
            current_idx = best_next
        
        logger.debug(f"碰撞检测路径简化: {len(path)} -> {len(simplified)} 点")
        return simplified

    def _is_path_collision_free(self, path: List[SpacePoint], collision_checker: CollisionCheckerType) -> bool:
        """路径全段碰撞复验（支持点云和 SDF 碰撞检测器）。"""
        if len(path) < 2 or collision_checker is None:
            return True
        for i in range(len(path) - 1):
            if not collision_checker.is_path_collision_free(path[i], path[i + 1]):
                return False
        return True

    # -------------------- 坐标转换（空间<->物理） --------------------

    def _build_physical_bounds(
        self,
        volume_shape: Tuple[int, int, int],
        spacing: Tuple[float, float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """根据体素形状与间距构建物理坐标边界（mm）。"""
        depth, height, width = volume_shape
        sx, sy, sz = spacing
        x_max = max(0.0, (width - 1) * sx)
        y_max = max(0.0, (height - 1) * sy)
        z_max = max(0.0, (depth - 1) * sz)
        return (0.0, x_max), (0.0, y_max), (0.0, z_max)

    def _space_to_physical(self, coord: SpacePoint) -> SpacePoint:
        """将归一化空间坐标 [0,100] 转为物理坐标（mm）。"""
        if not self._use_physical_coords or self._space_bounds_physical is None:
            return coord
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._space_bounds_physical
        x = xmin + (coord[0] / 100.0) * (xmax - xmin)
        y = ymin + (coord[1] / 100.0) * (ymax - ymin)
        z = zmin + (coord[2] / 100.0) * (zmax - zmin)
        return (float(x), float(y), float(z))

    def _physical_to_space(self, coord: SpacePoint) -> SpacePoint:
        """将物理坐标（mm）转为归一化空间坐标 [0,100]。"""
        if not self._use_physical_coords or self._space_bounds_physical is None:
            return coord
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = self._space_bounds_physical
        x_range = max(xmax - xmin, 1e-6)
        y_range = max(ymax - ymin, 1e-6)
        z_range = max(zmax - zmin, 1e-6)
        x = ((coord[0] - xmin) / x_range) * 100.0
        y = ((coord[1] - ymin) / y_range) * 100.0
        z = ((coord[2] - zmin) / z_range) * 100.0
        return (float(x), float(y), float(z))

    def _convert_points_to_planner_space(self, points: List[SpacePoint]) -> List[SpacePoint]:
        if not self._use_physical_coords:
            return list(points)
        return [self._space_to_physical(p) for p in points]

    def _convert_points_from_planner_space(self, points: List[SpacePoint]) -> List[SpacePoint]:
        if not self._use_physical_coords:
            return list(points)
        return [self._physical_to_space(p) for p in points]


