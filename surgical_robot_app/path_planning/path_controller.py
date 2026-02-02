"""
PathController: 路径规划控制器

职责：
- 管理路径规划相关的数据（起点/终点/中间点、最终路径点）
- 支持 A* 和 RRT 两种路径规划算法
- 维护障碍物数据（栅格或点云）

注意：
- 不依赖 Qt，仅依赖路径规划器和 numpy，方便单元测试与复用。
"""

from typing import List, Tuple, Optional, Callable

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

logger = logging.getLogger(__name__)

SpacePoint = Tuple[float, float, float]  # (x, y, z) in [0, 100]


class PathController:
    def __init__(
        self,
        planner: Optional[AStarPlanner] = None,
        grid_size: Tuple[int, int, int] = (100, 100, 105),
        obstacle_expansion: int = 4,
        use_rrt: bool = True,  # 默认使用RRT
        rrt_step_size: float = 2.0,
        rrt_safety_radius: float = 5.0,  # 增加默认安全半径从3.0到5.0
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
        """
        self.use_rrt = use_rrt
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

        self.start_point: Optional[SpacePoint] = None
        self.end_point: Optional[SpacePoint] = None
        self.waypoints: List[SpacePoint] = []
        self._path_points: List[SpacePoint] = []
        
        # 撤销/重做栈
        self.undo_stack: List[dict] = []
        self.redo_stack: List[dict] = []
        
        # 保存初始状态
        self._save_state()

        # 障碍物数据（根据算法类型使用不同的格式）
        self._obstacle_grid: Optional[np.ndarray] = None  # A*使用
        self._point_cloud: Optional[np.ndarray] = None  # RRT使用
        self._collision_checker: Optional[PointCloudCollisionChecker] = None  # RRT使用
        
        # 物理坐标相关（用于体数据：单位mm）
        self._use_physical_coords: bool = False
        self._volume_shape: Optional[Tuple[int, int, int]] = None
        self._volume_spacing: Optional[Tuple[float, float, float]] = None
        self._space_bounds_physical: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
        self._physical_path_points: List[SpacePoint] = []

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

    # -------------------- 障碍物数据管理 --------------------

    def set_obstacle_from_volume(
        self,
        seg_mask_volume: np.ndarray,
        spacing: Optional[Tuple[float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """
        从 3D 分割体数据生成障碍物数据（点云或栅格）

        Args:
            seg_mask_volume: 3D 掩码体数据 (Z, H, W)
            spacing: 体素间距（可选，用于点云生成）

        Returns:
            如果使用RRT，返回点云数组；如果使用A*，返回障碍物栅格
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
            # 使用RRT：生成点云
            point_cloud = volume_to_point_cloud(
                seg_mask_volume,
                threshold=128,
                downsample_factor=1,  # 减少下采样，增加点云密度以提高碰撞检测精度
                spacing=spacing
            )
            self._point_cloud = point_cloud
            
            # 创建碰撞检测器，使用更大的安全半径
            # 确保路径不会穿过物体表面
            effective_safety_radius = max(self.rrt_safety_radius, 5.0)  # 至少5.0的安全半径
            
            self._collision_checker = PointCloudCollisionChecker(
                point_cloud,
                safety_radius=effective_safety_radius
            )
            
            # 创建RRT规划器
            self.rrt_planner = RRTPlanner(
                self._collision_checker,
                step_size=self.rrt_step_size,
                goal_bias=0.1,
                max_iterations=5000,
                goal_threshold=2.0,
                bounds=self._space_bounds_physical if self._use_physical_coords else None,
            )
            
            logger.info(f"RRT障碍物设置完成: {len(point_cloud)} 个点")
            return point_cloud
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

    def set_obstacle_from_mesh(self, poly_data) -> Optional[np.ndarray]:
        """
        从网格（VTK PolyData）生成点云障碍物（仅用于RRT）

        Args:
            poly_data: VTK PolyData 对象

        Returns:
            点云数组
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
        
        # STL模型使用更大的安全半径，确保路径不会穿过物体
        # 安全半径应该足够大，以考虑路径点的体积和误差
        # 增加安全半径到5.0，确保路径有足够的安全距离
        mesh_safety_radius = 5.0  # 从3.0增加到5.0，确保有足够的安全距离
        
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
            return self._point_cloud is not None and len(self._point_cloud) > 0
        else:
            return self._obstacle_grid is not None

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

    def generate_path(self, smooth: bool = True, progress_callback: Optional[Callable[[int], None]] = None) -> List[SpacePoint]:
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
            path = self.rrt_planner.plan_multi_segment(planner_waypoints, smooth=smooth, progress_callback=progress_callback)
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
        1) 重采样到固定步长
        2) 在可用时进行平滑并做碰撞复验
        3) 保持起点/中间点/终点不被移动
        """
        if not path or len(path) < 2:
            return path

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

        # 分段处理
        processed: List[SpacePoint] = []
        for i in range(len(cleaned_indices) - 1):
            s_idx = cleaned_indices[i]
            e_idx = cleaned_indices[i + 1]
            segment = path[s_idx:e_idx + 1]
            if len(segment) < 2:
                continue

            # 1) 重采样（沿路径弧长等距插值）
            step = self.rrt_step_size if self.use_rrt else 2.0
            resampled = self._resample_path(segment, step_size=step)

            # 2) 平滑 + 碰撞复验（仅在有碰撞检测器时）
            smoothed = resampled
            if self._collision_checker is not None:
                smoothed = self._smooth_path_moving_average(resampled, window_size=3)
                if not self._is_path_collision_free(smoothed, self._collision_checker):
                    smoothed = resampled

            # 3) 拼接（避免重复点）
            if processed:
                processed.extend(smoothed[1:])
            else:
                processed.extend(smoothed)

        return processed if processed else path

    def _resample_path(self, path: List[SpacePoint], step_size: float = 2.0) -> List[SpacePoint]:
        """沿路径弧长进行等距重采样，保持首尾不变。"""
        if len(path) < 2:
            return path

        pts = np.array(path, dtype=float)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        total_len = float(np.sum(seg_lens))
        if total_len <= 1e-6:
            return path

        # 目标采样点数
        num_samples = max(2, int(total_len / max(step_size, 1e-6)) + 1)
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

    def _is_path_collision_free(self, path: List[SpacePoint], collision_checker: PointCloudCollisionChecker) -> bool:
        """路径全段碰撞复验。"""
        if len(path) < 2:
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


