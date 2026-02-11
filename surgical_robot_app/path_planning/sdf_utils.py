"""
有符号距离场 (Signed Distance Field) 工具模块

职责：
- 从分割掩码生成 SDF
- 基于 SDF 的高效碰撞检测
- 解决点云方案中的内部空洞问题

SDF 的优势：
1. 内部点有负值，外部点有正值，完美区分内外
2. 可以直接获取到最近表面的精确距离
3. 碰撞检测只需要简单的数值比较，非常高效

原理：
- SDF(p) < 0  → 点 p 在物体内部
- SDF(p) = 0  → 点 p 在物体表面
- SDF(p) > 0  → 点 p 在物体外部
- |SDF(p)| = 点 p 到最近表面的距离
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

try:
    from scipy import ndimage
    from scipy.interpolate import RegularGridInterpolator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None
    RegularGridInterpolator = None

logger = logging.getLogger(__name__)


def compute_sdf_from_mask(
    mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    normalize_to_100: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    从二值分割掩码计算有符号距离场 (SDF)
    
    Args:
        mask: 3D 二值掩码，形状 (Z, H, W)，非零值表示物体内部
        spacing: 体素间距 (sx, sy, sz)，用于计算真实物理距离
        normalize_to_100: 是否将坐标归一化到 [0, 100] 空间
    
    Returns:
        sdf: 3D 有符号距离场，负值表示内部，正值表示外部
        metadata: 元数据字典，包含坐标映射信息
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for SDF computation. Install with: pip install scipy")
    
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    
    # 确保是二值掩码
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 填充内部空洞（确保没有内部空腔）
    # 这一步很重要：确保物体内部完全被填充
    filled_mask = ndimage.binary_fill_holes(binary_mask).astype(np.uint8)
    
    logger.info(f"Mask shape: {filled_mask.shape}, "
                f"Original voxels: {np.sum(binary_mask)}, "
                f"Filled voxels: {np.sum(filled_mask)}")
    
    # 计算到物体边界的欧氏距离变换
    # distance_transform_edt 计算每个点到最近的 False(0) 点的距离
    
    # 外部距离：从物体外部到表面的距离
    exterior_dist = ndimage.distance_transform_edt(
        ~filled_mask.astype(bool),  # 反转：物体内部变为 False
        sampling=spacing if spacing else (1.0, 1.0, 1.0)
    )
    
    # 内部距离：从物体内部到表面的距离
    interior_dist = ndimage.distance_transform_edt(
        filled_mask.astype(bool),  # 物体内部为 True
        sampling=spacing if spacing else (1.0, 1.0, 1.0)
    )
    
    # SDF = 外部距离 - 内部距离
    # 外部点：exterior_dist > 0, interior_dist = 0 → SDF > 0
    # 内部点：exterior_dist = 0, interior_dist > 0 → SDF < 0
    # 表面点：两者都接近 0 → SDF ≈ 0
    sdf = exterior_dist - interior_dist
    
    # 构建元数据
    depth, height, width = mask.shape
    
    if spacing is not None:
        # 使用物理坐标
        x_max = (width - 1) * spacing[0]
        y_max = (height - 1) * spacing[1]
        z_max = (depth - 1) * spacing[2]
        coord_system = "physical"
    elif normalize_to_100:
        # 归一化到 [0, 100]
        x_max = 100.0
        y_max = 100.0
        z_max = 100.0
        coord_system = "normalized"
    else:
        # 使用体素坐标
        x_max = float(width - 1)
        y_max = float(height - 1)
        z_max = float(depth - 1)
        coord_system = "voxel"
    
    metadata = {
        "shape": mask.shape,
        "spacing": spacing,
        "coord_system": coord_system,
        "bounds": {
            "x": (0.0, x_max),
            "y": (0.0, y_max),
            "z": (0.0, z_max)
        },
        "sdf_range": (float(np.min(sdf)), float(np.max(sdf)))
    }
    
    logger.info(f"SDF computed: shape={sdf.shape}, "
                f"range=[{metadata['sdf_range'][0]:.2f}, {metadata['sdf_range'][1]:.2f}], "
                f"coord_system={coord_system}")
    
    return sdf.astype(np.float32), metadata


class SDFCollisionChecker:
    """
    基于有符号距离场 (SDF) 的碰撞检测器
    
    相比点云方案的优势：
    1. 完美区分内部/外部，不存在内部空洞问题
    2. 直接获取精确距离，无需最近邻搜索
    3. 支持三线性插值，检测更平滑
    4. 内存效率更高（存储一个 3D 数组而非大量点）
    """
    
    def __init__(
        self,
        sdf: np.ndarray,
        metadata: dict,
        safety_radius: float = 5.0,
        use_interpolation: bool = True
    ):
        """
        初始化 SDF 碰撞检测器
        
        Args:
            sdf: 3D 有符号距离场
            metadata: SDF 元数据（包含坐标边界等信息）
            safety_radius: 安全半径，路径点与障碍物的最小距离
            use_interpolation: 是否使用三线性插值（更精确但稍慢）
        """
        self.sdf = sdf
        self.metadata = metadata
        self.safety_radius = safety_radius
        self.use_interpolation = use_interpolation
        
        # 提取边界信息
        bounds = metadata["bounds"]
        self.x_min, self.x_max = bounds["x"]
        self.y_min, self.y_max = bounds["y"]
        self.z_min, self.z_max = bounds["z"]
        
        # SDF 数组维度
        self.depth, self.height, self.width = sdf.shape
        
        # 创建插值器（用于连续坐标查询）
        if use_interpolation and SCIPY_AVAILABLE:
            # 创建坐标网格
            z_coords = np.linspace(self.z_min, self.z_max, self.depth)
            y_coords = np.linspace(self.y_min, self.y_max, self.height)
            x_coords = np.linspace(self.x_min, self.x_max, self.width)
            
            # 注意：RegularGridInterpolator 期望的顺序是 (z, y, x)
            self.interpolator = RegularGridInterpolator(
                (z_coords, y_coords, x_coords),
                sdf,
                method='linear',
                bounds_error=False,
                fill_value=float(np.max(sdf))  # 边界外返回最大值（安全）
            )
        else:
            self.interpolator = None
        
        logger.info(f"SDFCollisionChecker initialized: "
                    f"shape={sdf.shape}, safety_radius={safety_radius}, "
                    f"interpolation={use_interpolation}")
    
    def _space_to_voxel(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        将空间坐标转换为体素索引
        
        Args:
            point: (x, y, z) 空间坐标
        
        Returns:
            (z_idx, y_idx, x_idx) 体素索引
        """
        x, y, z = point
        
        # 归一化到 [0, 1]
        x_norm = (x - self.x_min) / max(self.x_max - self.x_min, 1e-6)
        y_norm = (y - self.y_min) / max(self.y_max - self.y_min, 1e-6)
        z_norm = (z - self.z_min) / max(self.z_max - self.z_min, 1e-6)
        
        # 转换为体素索引
        x_idx = int(x_norm * (self.width - 1))
        y_idx = int(y_norm * (self.height - 1))
        z_idx = int(z_norm * (self.depth - 1))
        
        # 限制在有效范围内
        x_idx = max(0, min(x_idx, self.width - 1))
        y_idx = max(0, min(y_idx, self.height - 1))
        z_idx = max(0, min(z_idx, self.depth - 1))
        
        return (z_idx, y_idx, x_idx)
    
    def get_distance(self, point: Tuple[float, float, float]) -> float:
        """
        获取点到最近表面的有符号距离
        
        Args:
            point: (x, y, z) 空间坐标
        
        Returns:
            有符号距离：
            - 负值 = 在物体内部
            - 正值 = 在物体外部
            - 绝对值 = 到最近表面的距离
        """
        if self.interpolator is not None:
            # 使用三线性插值（更精确）
            x, y, z = point
            # 注意：插值器期望 (z, y, x) 顺序
            return float(self.interpolator([[z, y, x]])[0])
        else:
            # 使用最近体素（更快）
            z_idx, y_idx, x_idx = self._space_to_voxel(point)
            return float(self.sdf[z_idx, y_idx, x_idx])
    
    def is_collision(self, point: Tuple[float, float, float]) -> bool:
        """
        检查点是否与障碍物碰撞（在物体内部或太靠近表面）
        
        Args:
            point: (x, y, z) 空间坐标
        
        Returns:
            True = 碰撞（不安全），False = 安全
        """
        distance = self.get_distance(point)
        # 距离 < 安全半径 → 不安全（包括内部的负值）
        return distance < self.safety_radius
    
    def is_inside_obstacle(self, point: Tuple[float, float, float]) -> bool:
        """
        检查点是否在障碍物内部
        
        Args:
            point: (x, y, z) 空间坐标
        
        Returns:
            True = 在物体内部
        """
        return self.get_distance(point) < 0
    
    def is_path_collision_free(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        num_check_points: int = 10
    ) -> bool:
        """
        检查路径段是否无碰撞
        
        Args:
            start: 起点 (x, y, z)
            end: 终点 (x, y, z)
            num_check_points: 基础检查点数
        
        Returns:
            True = 路径安全，False = 存在碰撞
        """
        # 计算路径长度
        start_arr = np.array(start)
        end_arr = np.array(end)
        path_length = np.linalg.norm(end_arr - start_arr)
        
        # 动态调整检查点数量
        # 确保检查点间距不超过安全半径的一半
        max_check_distance = self.safety_radius * 0.5
        required_check_points = max(num_check_points, int(path_length / max_check_distance) + 1)
        
        # 沿路径均匀采样检查
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
        获取点到最近障碍物表面的距离（用于路径评估）
        
        注意：这个方法返回的是绝对距离，用于兼容现有的 PathEvaluator
        
        Args:
            point: (x, y, z) 空间坐标
        
        Returns:
            到最近表面的距离（始终为正值或零）
        """
        sdf_value = self.get_distance(point)
        # 如果在内部（负值），返回 0（已碰撞）
        # 如果在外部（正值），返回该距离
        return max(0.0, sdf_value)
    
    def adjust_point_away_from_obstacle(
        self,
        point: Tuple[float, float, float],
        max_attempts: int = 20,
        step_size: float = 1.0
    ) -> Optional[Tuple[float, float, float]]:
        """
        尝试将点从障碍物内部/附近调整到安全位置
        
        使用 SDF 梯度来找到最快远离障碍物的方向
        
        Args:
            point: 原始点坐标
            max_attempts: 最大尝试次数
            step_size: 每次调整的步长
        
        Returns:
            调整后的安全点，如果无法调整则返回 None
        """
        distance = self.get_distance(point)
        
        # 如果已经安全，不需要调整
        if distance >= self.safety_radius:
            return point
        
        # 计算 SDF 梯度（指向外部的方向）
        gradient = self._compute_gradient(point)
        
        if gradient is None or np.linalg.norm(gradient) < 1e-6:
            # 梯度为零，尝试随机方向
            gradient = np.random.randn(3)
        
        # 归一化梯度
        gradient = gradient / np.linalg.norm(gradient)
        
        # 沿梯度方向移动
        current_point = np.array(point)
        for attempt in range(max_attempts):
            current_point = current_point + gradient * step_size
            
            # 限制在边界内
            current_point[0] = np.clip(current_point[0], self.x_min, self.x_max)
            current_point[1] = np.clip(current_point[1], self.y_min, self.y_max)
            current_point[2] = np.clip(current_point[2], self.z_min, self.z_max)
            
            new_distance = self.get_distance(tuple(current_point))
            if new_distance >= self.safety_radius:
                logger.info(f"Point adjusted: {point} -> {tuple(current_point)}")
                return tuple(current_point)
        
        logger.warning(f"Failed to adjust point {point} to safe location")
        return None
    
    def _compute_gradient(self, point: Tuple[float, float, float], epsilon: float = 0.5) -> Optional[np.ndarray]:
        """
        计算 SDF 在给定点的梯度（使用中心差分）
        
        梯度指向 SDF 增加的方向（即远离障碍物的方向）
        
        Args:
            point: (x, y, z) 空间坐标
            epsilon: 差分步长
        
        Returns:
            梯度向量 (gx, gy, gz)
        """
        x, y, z = point
        
        # 中心差分计算梯度
        dx = (self.get_distance((x + epsilon, y, z)) - 
              self.get_distance((x - epsilon, y, z))) / (2 * epsilon)
        dy = (self.get_distance((x, y + epsilon, z)) - 
              self.get_distance((x, y - epsilon, z))) / (2 * epsilon)
        dz = (self.get_distance((x, y, z + epsilon)) - 
              self.get_distance((x, y, z - epsilon))) / (2 * epsilon)
        
        return np.array([dx, dy, dz])
    
    def visualize_slice(self, z_index: int = None, axis: str = 'z') -> np.ndarray:
        """
        获取 SDF 的 2D 切片用于可视化
        
        Args:
            z_index: 切片索引（如果为 None，使用中间切片）
            axis: 切片轴 ('x', 'y', 'z')
        
        Returns:
            2D SDF 切片
        """
        if axis == 'z':
            idx = z_index if z_index is not None else self.depth // 2
            return self.sdf[idx, :, :]
        elif axis == 'y':
            idx = z_index if z_index is not None else self.height // 2
            return self.sdf[:, idx, :]
        elif axis == 'x':
            idx = z_index if z_index is not None else self.width // 2
            return self.sdf[:, :, idx]
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    # ==================== 器械体积碰撞检测 ====================
    
    def is_capsule_collision(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        instrument_radius: float = 1.0
    ) -> bool:
        """
        检测胶囊体（手术器械）是否与障碍物碰撞
        
        胶囊体 = 两端半球 + 中间圆柱，模拟真实器械形状
        
        Args:
            start: 器械起点 (x, y, z)
            end: 器械终点 (x, y, z)
            instrument_radius: 器械半径
        
        Returns:
            True = 碰撞，False = 安全
        """
        # 计算路径长度和采样点数
        start_arr = np.array(start)
        end_arr = np.array(end)
        path_length = np.linalg.norm(end_arr - start_arr)
        
        # 沿路径采样：确保采样间距小于器械半径
        sample_distance = min(instrument_radius * 0.5, self.safety_radius * 0.3)
        num_samples = max(2, int(path_length / sample_distance) + 1)
        
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            point = start_arr * (1 - t) + end_arr * t
            
            # 检查该点处的 SDF 值
            # 如果 SDF < instrument_radius，说明器械会碰到障碍物
            sdf_value = self.get_distance(tuple(point))
            
            if sdf_value < instrument_radius:
                return True  # 碰撞
        
        return False  # 安全
    
    def is_path_safe_for_instrument(
        self,
        path: List[Tuple[float, float, float]],
        instrument_radius: float = 1.0
    ) -> Tuple[bool, List[int]]:
        """
        检查整条路径是否对给定半径的器械安全
        
        Args:
            path: 路径点列表
            instrument_radius: 器械半径
        
        Returns:
            (is_safe, collision_indices): 
            - is_safe: 整条路径是否安全
            - collision_indices: 碰撞发生的路径段索引列表
        """
        if len(path) < 2:
            return True, []
        
        collision_indices = []
        
        for i in range(len(path) - 1):
            if self.is_capsule_collision(path[i], path[i + 1], instrument_radius):
                collision_indices.append(i)
        
        return len(collision_indices) == 0, collision_indices
    
    def get_min_safe_radius(
        self,
        path: List[Tuple[float, float, float]]
    ) -> float:
        """
        计算路径上允许的最大器械半径
        
        Args:
            path: 路径点列表
        
        Returns:
            最大安全器械半径（路径上最小的 SDF 值）
        """
        if not path:
            return 0.0
        
        min_distance = float('inf')
        
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])
            path_length = np.linalg.norm(end - start)
            
            num_samples = max(2, int(path_length / 0.5) + 1)
            
            for j in range(num_samples):
                t = j / (num_samples - 1) if num_samples > 1 else 0
                point = start * (1 - t) + end * t
                sdf_value = self.get_distance(tuple(point))
                min_distance = min(min_distance, sdf_value)
        
        return max(0.0, min_distance)


def create_sdf_collision_checker(
    seg_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    safety_radius: float = 5.0,
    use_interpolation: bool = True
) -> SDFCollisionChecker:
    """
    便捷函数：从分割掩码创建 SDF 碰撞检测器
    
    Args:
        seg_mask: 3D 分割掩码
        spacing: 体素间距
        safety_radius: 安全半径
        use_interpolation: 是否使用插值
    
    Returns:
        SDFCollisionChecker 实例
    """
    sdf, metadata = compute_sdf_from_mask(
        seg_mask,
        spacing=spacing,
        normalize_to_100=(spacing is None)
    )
    
    return SDFCollisionChecker(
        sdf=sdf,
        metadata=metadata,
        safety_radius=safety_radius,
        use_interpolation=use_interpolation
    )
