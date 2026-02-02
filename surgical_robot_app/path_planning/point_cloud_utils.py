"""
点云工具模块

职责：
- 从体数据或网格生成点云
- 点云的预处理和优化
- 点云格式转换
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    cKDTree = None

try:
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.util.numpy_support import vtk_to_numpy
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    vtkPolyData = None
    vtk_to_numpy = None

logger = logging.getLogger(__name__)


def volume_to_point_cloud(
    volume: np.ndarray,
    threshold: int = 128,
    downsample_factor: int = 1,
    spacing: Optional[Tuple[float, float, float]] = None
) -> np.ndarray:
    """
    从3D体数据生成点云
    
    Args:
        volume: 3D体数据，形状 (Z, H, W)
        threshold: 阈值，大于此值的体素会被包含在点云中
        downsample_factor: 下采样因子（1表示不采样，2表示每2个体素采样1个）
        spacing: 体素间距 (sx, sy, sz)，用于转换为真实坐标
    
    Returns:
        point_cloud: Nx3 numpy数组，每行是一个点的 (x, y, z) 坐标
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.ndim}D")
    
    z, h, w = volume.shape
    
    # 找到所有非零体素的坐标
    mask = volume > threshold
    z_coords, y_coords, x_coords = np.where(mask)
    
    # 转换为空间坐标
    if spacing is not None:
        # 如果有体素间距，直接转换为真实物理坐标（mm）
        sx, sy, sz = spacing
        x_space = x_coords * float(sx)
        y_space = y_coords * float(sy)
        z_space = z_coords * float(sz)
    else:
        # 无间距信息则归一化到 [0, 100]
        x_space = (x_coords / w) * 100.0
        y_space = (y_coords / h) * 100.0
        z_space = (z_coords / z) * 100.0
    
    # 组合成点云
    point_cloud = np.column_stack([x_space, y_space, z_space])
    
    # 下采样
    if downsample_factor > 1:
        point_cloud = point_cloud[::downsample_factor]
    
    logger.info(f"生成点云: {len(point_cloud)} 个点 (阈值={threshold}, 下采样={downsample_factor})")
    return point_cloud


def mesh_to_point_cloud(
    poly_data: vtkPolyData,
    num_points: Optional[int] = None,
    downsample_factor: int = 1
) -> Optional[np.ndarray]:
    """
    从VTK网格（PolyData）生成点云
    
    Args:
        poly_data: VTK PolyData 对象
        num_points: 目标点数（如果为None，使用所有顶点）
        downsample_factor: 下采样因子
    
    Returns:
        point_cloud: Nx3 numpy数组，每行是一个点的 (x, y, z) 坐标
    """
    if not VTK_AVAILABLE or poly_data is None:
        return None
    
    try:
        # 获取顶点
        points = poly_data.GetPoints()
        if points is None:
            return None
        
        # 转换为numpy数组
        vtk_points = vtk_to_numpy(points.GetData())
        
        # 转换为空间坐标 [0, 100]
        # 假设网格坐标需要归一化
        if vtk_points.size > 0:
            # 获取边界
            bounds = poly_data.GetBounds()
            x_min, x_max = bounds[0], bounds[1]
            y_min, y_max = bounds[2], bounds[3]
            z_min, z_max = bounds[4], bounds[5]
            
            # 归一化到 [0, 100]
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            z_range = z_max - z_min if z_max > z_min else 1.0
            
            x_norm = ((vtk_points[:, 0] - x_min) / x_range) * 100.0
            y_norm = ((vtk_points[:, 1] - y_min) / y_range) * 100.0
            z_norm = ((vtk_points[:, 2] - z_min) / z_range) * 100.0
            
            point_cloud = np.column_stack([x_norm, y_norm, z_norm])
            
            # 下采样
            if downsample_factor > 1:
                point_cloud = point_cloud[::downsample_factor]
            
            # 如果指定了目标点数，进行采样
            if num_points is not None and len(point_cloud) > num_points:
                indices = np.random.choice(len(point_cloud), num_points, replace=False)
                point_cloud = point_cloud[indices]
            
            logger.info(f"从网格生成点云: {len(point_cloud)} 个点")
            return point_cloud
    except Exception as e:
        logger.error(f"从网格生成点云失败: {e}")
        return None
    
    return None


def expand_point_cloud(
    point_cloud: np.ndarray,
    expansion_radius: float,
    num_samples: int = 10
) -> np.ndarray:
    """
    对点云进行膨胀，在障碍物周围生成额外的点
    
    Args:
        point_cloud: 原始点云，Nx3
        expansion_radius: 膨胀半径（空间单位）
        num_samples: 每个点周围的采样点数
    
    Returns:
        expanded_point_cloud: 膨胀后的点云
    """
    if len(point_cloud) == 0:
        return point_cloud
    
    expanded_points = [point_cloud]
    
    # 为每个点生成周围的采样点
    for point in point_cloud:
        # 在球面上均匀采样
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        for angle in angles:
            # 在球面上采样
            phi = np.random.uniform(0, np.pi)
            x_offset = expansion_radius * np.sin(phi) * np.cos(angle)
            y_offset = expansion_radius * np.sin(phi) * np.sin(angle)
            z_offset = expansion_radius * np.cos(phi)
            
            new_point = point + np.array([x_offset, y_offset, z_offset])
            # 限制在 [0, 100] 范围内
            new_point = np.clip(new_point, 0, 100)
            expanded_points.append(new_point.reshape(1, -1))
    
    expanded_cloud = np.vstack(expanded_points)
    logger.info(f"点云膨胀: {len(point_cloud)} -> {len(expanded_cloud)} 个点")
    return expanded_cloud


class PointCloudCollisionChecker:
    """基于点云的碰撞检测器（使用KD-Tree）"""
    
    def __init__(self, point_cloud: np.ndarray, safety_radius: float = 2.0):
        """
        初始化碰撞检测器
        
        Args:
            point_cloud: 点云，Nx3 numpy数组
            safety_radius: 安全半径（空间单位），路径点与障碍物的最小距离
        """
        self.point_cloud = point_cloud
        self.safety_radius = safety_radius
        
        # 构建KD-Tree用于快速最近邻查询
        if SCIPY_AVAILABLE and len(point_cloud) > 0:
            self.kdtree = cKDTree(point_cloud)
        else:
            self.kdtree = None
            if not SCIPY_AVAILABLE:
                logger.warning("scipy不可用，使用简单的线性搜索进行碰撞检测")
    
    def is_collision(self, point: Tuple[float, float, float]) -> bool:
        """
        检查点是否与障碍物碰撞
        
        Args:
            point: 3D点坐标 (x, y, z)
        
        Returns:
            True 如果碰撞，False 如果安全
        """
        if len(self.point_cloud) == 0:
            return False
        
        if self.kdtree is not None:
            # 使用KD-Tree快速查询
            distance, _ = self.kdtree.query(point, k=1)
            return distance < self.safety_radius
        else:
            # 简单的线性搜索（较慢）
            distances = np.linalg.norm(self.point_cloud - np.array(point), axis=1)
            min_distance = np.min(distances)
            return min_distance < self.safety_radius
    
    def is_path_collision_free(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        num_check_points: int = 10
    ) -> bool:
        """
        检查路径段是否无碰撞
        
        Args:
            start: 起点
            end: 终点
            num_check_points: 路径上的检查点数
        
        Returns:
            True 如果路径无碰撞，False 如果有碰撞
        """
        # 计算路径长度，根据长度动态调整检查点数量
        path_length = np.sqrt(
            (end[0] - start[0]) ** 2 +
            (end[1] - start[1]) ** 2 +
            (end[2] - start[2]) ** 2
        )
        
        # 确保检查点之间的最大距离不超过安全半径的一半
        # 这样可以确保不会漏掉任何可能穿过障碍物的路径段
        max_check_distance = self.safety_radius * 0.5
        required_check_points = max(num_check_points, int(path_length / max_check_distance) + 1)
        
        # 在路径上均匀采样检查点
        for i in range(required_check_points + 1):
            t = i / required_check_points if required_check_points > 0 else 0
            check_point = (
                start[0] * (1 - t) + end[0] * t,
                start[1] * (1 - t) + end[1] * t,
                start[2] * (1 - t) + end[2] * t
            )
            if self.is_collision(check_point):
                return False
        return True
    
    def get_distance_to_obstacle(self, point: Tuple[float, float, float]) -> float:
        """
        获取点到最近障碍物的距离
        
        Args:
            point: 3D点坐标
        
        Returns:
            到最近障碍物的距离
        """
        if len(self.point_cloud) == 0:
            return float('inf')
        
        if self.kdtree is not None:
            distance, _ = self.kdtree.query(point, k=1)
            return float(distance)
        else:
            distances = np.linalg.norm(self.point_cloud - np.array(point), axis=1)
            return float(np.min(distances))
    
    def is_inside_obstacle(self, point: Tuple[float, float, float], strict_threshold: float = 0.5) -> bool:
        """
        检查点是否真的在障碍物内部（使用更严格的阈值，只检查是否真的在物体内部）
        
        Args:
            point: 3D点坐标
            strict_threshold: 严格阈值，只有距离小于此值才认为在障碍物内（默认0.5）
        
        Returns:
            True 如果点在障碍物内部，False 如果点在外部
        """
        distance = self.get_distance_to_obstacle(point)
        return distance < strict_threshold
    
    def adjust_point_away_from_obstacle(
        self, 
        point: Tuple[float, float, float], 
        max_attempts: int = 20,
        step_size: float = 1.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        尝试将点从障碍物附近调整到安全位置
        
        Args:
            point: 原始点坐标
            max_attempts: 最大尝试次数
            step_size: 每次调整的步长
        
        Returns:
            调整后的点坐标，如果无法调整则返回None
        """
        distance = self.get_distance_to_obstacle(point)
        
        # 如果点已经在安全距离外，不需要调整
        if distance >= self.safety_radius:
            return point
        
        # 如果点真的在障碍物内部（距离<0.5），尝试向外推
        if distance < 0.5:
            # 找到最近的障碍物点
            if self.kdtree is not None:
                dist, idx = self.kdtree.query(point, k=1)
                nearest_obstacle = self.point_cloud[idx]
            else:
                distances = np.linalg.norm(self.point_cloud - np.array(point), axis=1)
                idx = np.argmin(distances)
                nearest_obstacle = self.point_cloud[idx]
            
            # 计算从障碍物指向点的方向
            direction = np.array(point) - np.array(nearest_obstacle)
            if np.linalg.norm(direction) < 1e-6:
                # 如果方向太小，随机选择一个方向
                direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # 尝试向外推
            for attempt in range(max_attempts):
                new_point = np.array(point) + direction * step_size * (attempt + 1)
                new_point = np.clip(new_point, [0, 0, 0], [100, 100, 100])
                new_distance = self.get_distance_to_obstacle(tuple(new_point))
                
                if new_distance >= self.safety_radius:
                    return tuple(new_point)
        
        # 如果点只是太靠近障碍物（但不在内部），尝试向远离障碍物的方向移动
        if distance < self.safety_radius:
            # 找到最近的障碍物点
            if self.kdtree is not None:
                dist, idx = self.kdtree.query(point, k=1)
                nearest_obstacle = self.point_cloud[idx]
            else:
                distances = np.linalg.norm(self.point_cloud - np.array(point), axis=1)
                idx = np.argmin(distances)
                nearest_obstacle = self.point_cloud[idx]
            
            # 计算从障碍物指向点的方向
            direction = np.array(point) - np.array(nearest_obstacle)
            if np.linalg.norm(direction) < 1e-6:
                # 如果方向太小，随机选择一个方向
                direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction)
            
            # 计算需要移动的距离
            required_distance = self.safety_radius - distance + 0.5  # 额外0.5的安全余量
            new_point = np.array(point) + direction * required_distance
            new_point = np.clip(new_point, [0, 0, 0], [100, 100, 100])
            
            if self.get_distance_to_obstacle(tuple(new_point)) >= self.safety_radius:
                return tuple(new_point)
        
        return None

