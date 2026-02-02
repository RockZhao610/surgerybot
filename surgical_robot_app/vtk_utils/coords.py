"""
VTK 坐标相关工具函数

职责：
- 从 renderer 获取模型边界
- 在世界坐标 <-> 归一化空间坐标 ([0, 100]) 之间进行转换
- 根据模型大小估算合适的可视化半径
"""

from typing import Tuple, Optional


def get_model_bounds(renderer) -> Optional[Tuple[float, float, float, float, float, float]]:
    """安全获取可见几何的边界 (xmin, xmax, ymin, ymax, zmin, zmax)"""
    if renderer is None:
        return None
    try:
        bounds = renderer.ComputeVisiblePropBounds()
        if bounds and len(bounds) >= 6:
            return (
                float(bounds[0]),
                float(bounds[1]),
                float(bounds[2]),
                float(bounds[3]),
                float(bounds[4]),
                float(bounds[5]),
            )
    except Exception:
        return None
    return None


def world_to_space(
    bounds: Optional[Tuple[float, float, float, float, float, float]],
    world_coord: Tuple[float, float, float],
    default: float = 50.0,
) -> Tuple[float, float, float]:
    """
    将世界坐标转换为 [0, 100] 的归一化空间坐标
    若 bounds 不可用，则假设模型在 [-50, 50] 范围内
    """
    x_world, y_world, z_world = world_coord

    if bounds is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        if x_max > x_min:
            space_x = ((x_world - x_min) / (x_max - x_min)) * 100.0
        else:
            space_x = default

        if y_max > y_min:
            space_y = ((y_world - y_min) / (y_max - y_min)) * 100.0
        else:
            space_y = default

        if z_max > z_min:
            space_z = ((z_world - z_min) / (z_max - z_min)) * 100.0
        else:
            space_z = default
    else:
        # 回退：假设世界坐标在 [-50, 50]，线性平移到 [0, 100]
        space_x = max(0.0, min(100.0, x_world + 50.0))
        space_y = max(0.0, min(100.0, y_world + 50.0))
        space_z = max(0.0, min(100.0, z_world + 50.0))

    return float(space_x), float(space_y), float(space_z)


def space_to_world(
    bounds: Optional[Tuple[float, float, float, float, float, float]],
    space_coord: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """
    将 [0, 100] 归一化空间坐标转换回世界坐标
    若 bounds 不可用，则简单平移到 [-50, 50]
    """
    x, y, z = space_coord

    if bounds is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        world_x = x_min + (x / 100.0) * (x_max - x_min)
        world_y = y_min + (y / 100.0) * (y_max - y_min)
        world_z = z_min + (z / 100.0) * (z_max - z_min)
    else:
        world_x = x - 50.0
        world_y = y - 50.0
        world_z = z - 50.0

    return float(world_x), float(world_y), float(world_z)


def estimate_radius(
    bounds: Optional[Tuple[float, float, float, float, float, float]],
    ratio: float = 0.02,
    default: float = 2.0,
) -> float:
    """
    根据模型尺寸估算可视化球体半径

    Args:
        bounds: 模型边界
        ratio: 使用模型最大边长的百分比作为半径
        default: 当 bounds 不可用时使用的默认半径
    """
    if bounds is None:
        return float(default)

    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    size = max(x_max - x_min, y_max - y_min, z_max - z_min)
    if size <= 0:
        return float(default)
    return float(size * ratio)


