import numpy as np
from typing import List, Tuple, Optional
import heapq

# 初始化日志系统
try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.path_planning.a_star_planner")


def pixel_to_space(pixel_coord: Tuple[int, int, int], data_shape: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    将像素坐标（z,y,x）归一化到[0,100]物理空间坐标
    
    Args:
        pixel_coord: (z, y, x) 像素坐标
        data_shape: (depth, height, width) 数据形状
    
    Returns:
        (x, y, z) 空间坐标，范围[0, 100]
    """
    z, y, x = pixel_coord
    depth, height, width = data_shape
    
    # 归一化到[0, 100]
    x_space = (x / width) * 100.0 if width > 0 else 0.0
    y_space = (y / height) * 100.0 if height > 0 else 0.0
    z_space = (z / depth) * 100.0 if depth > 0 else 0.0
    
    return (x_space, y_space, z_space)


def space_to_pixel(space_coord: Tuple[float, float, float], data_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    将空间坐标转换回像素坐标
    
    Args:
        space_coord: (x, y, z) 空间坐标，范围[0, 100]
        data_shape: (depth, height, width) 数据形状
    
    Returns:
        (z, y, x) 像素坐标
    """
    x_space, y_space, z_space = space_coord
    depth, height, width = data_shape
    
    x = int((x_space / 100.0) * width) if width > 0 else 0
    y = int((y_space / 100.0) * height) if height > 0 else 0
    z = int((z_space / 100.0) * depth) if depth > 0 else 0
    
    # 限制在有效范围内
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    z = max(0, min(z, depth - 1))
    
    return (z, y, x)


def space_to_grid(
    space_coord: Tuple[float, float, float],
    grid_size: Tuple[int, int, int] = (100, 100, 100),
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[int, int, int]:
    """
    将空间坐标映射到3D栅格
    
    Args:
        space_coord: (x, y, z) 空间坐标，范围[0, 100]
        grid_size: (width, height, depth) 栅格大小，默认100×100×100
    
    Returns:
        (x, y, z) 栅格坐标
    """
    x_space, y_space, z_space = space_coord
    grid_w, grid_h, grid_d = grid_size

    if bounds is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        x_range = max(xmax - xmin, 1e-6)
        y_range = max(ymax - ymin, 1e-6)
        z_range = max(zmax - zmin, 1e-6)
        x_grid = int(((x_space - xmin) / x_range) * grid_w)
        y_grid = int(((y_space - ymin) / y_range) * grid_h)
        z_grid = int(((z_space - zmin) / z_range) * grid_d)
    else:
        x_grid = int((x_space / 100.0) * grid_w)
        y_grid = int((y_space / 100.0) * grid_h)
        z_grid = int((z_space / 100.0) * grid_d)
    
    # 限制在有效范围内
    x_grid = max(0, min(x_grid, grid_w - 1))
    y_grid = max(0, min(y_grid, grid_h - 1))
    z_grid = max(0, min(z_grid, grid_d - 1))
    
    return (x_grid, y_grid, z_grid)


def grid_to_space(
    grid_coord: Tuple[int, int, int],
    grid_size: Tuple[int, int, int] = (100, 100, 100),
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[float, float, float]:
    """
    将栅格坐标转换回空间坐标
    
    Args:
        grid_coord: (x, y, z) 栅格坐标
        grid_size: (width, height, depth) 栅格大小
    
    Returns:
        (x, y, z) 空间坐标，范围[0, 100]
    """
    x_grid, y_grid, z_grid = grid_coord
    grid_w, grid_h, grid_d = grid_size

    if bounds is not None:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        x_space = xmin + (x_grid / grid_w) * (xmax - xmin) if grid_w > 0 else xmin
        y_space = ymin + (y_grid / grid_h) * (ymax - ymin) if grid_h > 0 else ymin
        z_space = zmin + (z_grid / grid_d) * (zmax - zmin) if grid_d > 0 else zmin
    else:
        x_space = (x_grid / grid_w) * 100.0 if grid_w > 0 else 0.0
        y_space = (y_grid / grid_h) * 100.0 if grid_h > 0 else 0.0
        z_space = (z_grid / grid_d) * 100.0 if grid_d > 0 else 0.0
    
    return (x_space, y_space, z_space)


def create_obstacle_grid(mask_array: np.ndarray, grid_size: Tuple[int, int, int] = (100, 100, 100), expansion: int = 1) -> np.ndarray:
    """
    根据mask数组生成障碍物栅格图，3D模型作为障碍物
    
    Args:
        mask_array: 3D mask数组，形状为(depth, height, width)，非零值表示障碍物（3D模型）
        grid_size: (width, height, depth) 栅格大小
        expansion: 障碍物膨胀半径（栅格单位），确保路径避开模型表面附近
    
    Returns:
        3D布尔数组，True表示障碍物，False表示可通行
    """
    grid_w, grid_h, grid_d = grid_size
    obstacle_grid = np.zeros((grid_d, grid_h, grid_w), dtype=bool)
    
    if mask_array is None or mask_array.size == 0:
        return obstacle_grid
    
    mask_depth, mask_height, mask_width = mask_array.shape
    
    # 将mask下采样到栅格大小
    for z_grid in range(grid_d):
        for y_grid in range(grid_h):
            for x_grid in range(grid_w):
                # 计算对应的mask坐标
                z_mask = int((z_grid / grid_d) * mask_depth)
                y_mask = int((y_grid / grid_h) * mask_height)
                x_mask = int((x_grid / grid_w) * mask_width)
                
                # 限制范围
                z_mask = max(0, min(z_mask, mask_depth - 1))
                y_mask = max(0, min(y_mask, mask_height - 1))
                x_mask = max(0, min(x_mask, mask_width - 1))
                
                # 如果mask中该位置有值（非零），则为障碍物（3D模型内部）
                if mask_array[z_mask, y_mask, x_mask] > 0:
                    obstacle_grid[z_grid, y_grid, x_grid] = True
    
    # 对障碍物进行膨胀操作，确保路径避开模型表面附近
    # 只在x和y方向膨胀，z方向不膨胀，允许路径从上下（z方向）穿过障碍物
    if expansion > 0:
        try:
            from scipy import ndimage
            # 使用非对称膨胀结构：z方向不膨胀（只有1），x和y方向正常膨胀
            # 这样路径可以从上下（z方向）穿过障碍物，但会避开左右和前后
            structure = np.ones((1, 2*expansion+1, 2*expansion+1))  # z方向只有1，不膨胀
            obstacle_grid = ndimage.binary_dilation(
                obstacle_grid, 
                structure=structure,
                iterations=1
            )
        except ImportError:
            # 如果scipy不可用，使用简单的邻域检查
            logger.warning("scipy not available, using simple expansion")
            expanded_grid = obstacle_grid.copy()
            for z in range(grid_d):
                for y in range(grid_h):
                    for x in range(grid_w):
                        if obstacle_grid[z, y, x]:
                            # 标记邻域为障碍物，只在x和y方向膨胀，z方向不膨胀
                            for dz in [0]:  # z方向不膨胀，只检查当前层
                                for dy in range(-expansion, expansion + 1):
                                    for dx in range(-expansion, expansion + 1):
                                        nz, ny, nx = z + dz, y + dy, x + dx
                                        if (0 <= nz < grid_d and 0 <= ny < grid_h and 0 <= nx < grid_w):
                                            expanded_grid[nz, ny, nx] = True
            obstacle_grid = expanded_grid
    
    return obstacle_grid


def euclidean_distance(coord1: Tuple[int, int, int], coord2: Tuple[int, int, int]) -> float:
    """计算两点之间的欧氏距离"""
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return np.sqrt(dx*dx + dy*dy + dz*dz)


def get_neighbors(coord: Tuple[int, int, int], grid_size: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """
    获取26邻域（包括对角线）的邻居节点
    
    Args:
        coord: (x, y, z) 当前节点坐标
        grid_size: (width, height, depth) 栅格大小
    
    Returns:
        邻居节点列表
    """
    x, y, z = coord
    grid_w, grid_h, grid_d = grid_size
    neighbors = []
    
    # 26邻域
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < grid_w and 0 <= ny < grid_h and 0 <= nz < grid_d:
                    neighbors.append((nx, ny, nz))
    
    return neighbors


def a_star_search(
    start_grid: Tuple[int, int, int],
    end_grid: Tuple[int, int, int],
    obstacle_grid: np.ndarray,
    grid_size: Tuple[int, int, int] = (100, 100, 100)
) -> Optional[List[Tuple[int, int, int]]]:
    """
    A*算法搜索路径
    
    Args:
        start_grid: (x, y, z) 起点栅格坐标
        end_grid: (x, y, z) 终点栅格坐标
        obstacle_grid: 障碍物栅格图
        grid_size: (width, height, depth) 栅格大小
    
    Returns:
        路径栅格坐标列表，如果无路径则返回None
    """
    grid_w, grid_h, grid_d = grid_size
    
    # 检查起点和终点是否在障碍物内
    # start_grid和end_grid是(x, y, z)格式，obstacle_grid是(z, y, x)格式
    sx, sy, sz = start_grid
    ex, ey, ez = end_grid
    
    # 确保坐标在有效范围内
    sx = max(0, min(sx, grid_w - 1))
    sy = max(0, min(sy, grid_h - 1))
    sz = max(0, min(sz, grid_d - 1))
    ex = max(0, min(ex, grid_w - 1))
    ey = max(0, min(ey, grid_h - 1))
    ez = max(0, min(ez, grid_d - 1))
    
    if obstacle_grid[sz, sy, sx]:
        logger.warning(f"Start point ({sx}, {sy}, {sz}) is in obstacle")
        return None
    
    if obstacle_grid[ez, ey, ex]:
        logger.warning(f"End point ({ex}, {ey}, {ez}) is in obstacle")
        return None
    
    # 初始化
    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    
    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: euclidean_distance(start_grid, end_grid)}
    
    visited = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # 到达终点
        if current == end_grid:
            # 重构路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        # 检查邻居
        # neighbor是(x, y, z)格式，obstacle_grid是(z, y, x)格式
        for neighbor in get_neighbors(current, grid_size):
            nx, ny, nz = neighbor
            
            # 确保坐标在有效范围内
            if not (0 <= nx < grid_w and 0 <= ny < grid_h and 0 <= nz < grid_d):
                continue
            
            # 跳过障碍物（注意：obstacle_grid索引顺序是[z, y, x]）
            if obstacle_grid[nz, ny, nx]:
                continue
            
            # 计算代价
            tentative_g = g_score[current] + euclidean_distance(current, neighbor)
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h_score = euclidean_distance(neighbor, end_grid)
                f_score[neighbor] = tentative_g + h_score
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # 无路径
    return None


def smooth_path(path: List[Tuple[float, float, float]], window_size: int = 3) -> List[Tuple[float, float, float]]:
    """
    使用移动平均滤波平滑路径
    
    Args:
        path: 路径点列表，每个点为(x, y, z)
        window_size: 移动平均窗口大小
    
    Returns:
        平滑后的路径
    """
    if len(path) <= window_size:
        return path
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(path)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(path), i + half_window + 1)
        
        window = path[start_idx:end_idx]
        avg_x = sum(p[0] for p in window) / len(window)
        avg_y = sum(p[1] for p in window) / len(window)
        avg_z = sum(p[2] for p in window) / len(window)
        
        smoothed.append((avg_x, avg_y, avg_z))
    
    return smoothed


class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, grid_size: Tuple[int, int, int] = (100, 100, 100)):
        self.grid_size = grid_size
        self.obstacle_grid = None
        self.space_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
    
    def set_obstacle_grid(self, obstacle_grid: np.ndarray):
        """设置障碍物栅格"""
        self.obstacle_grid = obstacle_grid
    
    def plan(
        self,
        start_space: Tuple[float, float, float],
        end_space: Tuple[float, float, float],
        smooth: bool = True
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        规划路径
        
        Args:
            start_space: (x, y, z) 起点空间坐标
            end_space: (x, y, z) 终点空间坐标
            smooth: 是否平滑路径
        
        Returns:
            路径点列表，每个点为(x, y, z)空间坐标
        """
        if self.obstacle_grid is None:
            logger.warning("Obstacle grid not set")
            return None
        
        # 转换为栅格坐标
        start_grid = space_to_grid(start_space, self.grid_size, bounds=self.space_bounds)
        end_grid = space_to_grid(end_space, self.grid_size, bounds=self.space_bounds)
        
        # A*搜索
        path_grid = a_star_search(start_grid, end_grid, self.obstacle_grid, self.grid_size)
        
        if path_grid is None:
            return None
        
        # 转换回空间坐标
        path_space = [grid_to_space(p, self.grid_size, bounds=self.space_bounds) for p in path_grid]
        
        # 平滑
        if smooth and len(path_space) > 3:
            path_space = smooth_path(path_space)
        
        return path_space
    
    def plan_multi_segment(
        self,
        waypoints: List[Tuple[float, float, float]],
        smooth: bool = True
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        规划多段路径（起点→中间点1→中间点2→...→终点）
        
        Args:
            waypoints: 路径点列表，包括起点、中间点和终点
            smooth: 是否平滑路径
        
        Returns:
            完整路径点列表
        """
        if len(waypoints) < 2:
            return None
        
        full_path = []
        
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            segment = self.plan(start, end, smooth=smooth)
            if segment is None:
                logger.warning(f"Cannot find path from waypoint {i} to {i+1}")
                return None
            
            # 添加段路径（避免重复点）
            if i == 0:
                full_path.extend(segment)
            else:
                full_path.extend(segment[1:])  # 跳过第一个点（与上一段的终点重复）
        
        return full_path
