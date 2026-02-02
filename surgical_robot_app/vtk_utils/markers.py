"""
VTK 标记（marker）相关工具函数

职责：
- 根据归一化空间坐标 [0, 100] 在 3D 场景中创建球体标记 actor
"""

from typing import Tuple

try:
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
    from vtkmodules.vtkFiltersSources import vtkSphereSource
except Exception:  # 允许在无 VTK 环境下导入模块但不使用
    vtkActor = None  # type: ignore
    vtkPolyDataMapper = None  # type: ignore
    vtkSphereSource = None  # type: ignore

from .coords import get_model_bounds, space_to_world, estimate_radius


def create_sphere_marker(
    renderer,
    space_coord: Tuple[float, float, float],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    radius_ratio: float = 0.012,
):
    """
    在给定 renderer 中，根据空间坐标创建一个球体标记 actor 并添加到场景中。

    Args:
        renderer: VTK renderer 实例
        space_coord: 归一化空间坐标 (x, y, z)，范围 [0, 100]
        color: 球体颜色 (r, g, b)，范围 [0, 1]
        radius_ratio: 球体半径占模型最大尺寸的比例

    Returns:
        创建好的 vtkActor；若 VTK 不可用则返回 None
    """
    if vtkSphereSource is None or vtkActor is None or vtkPolyDataMapper is None:
        return None
    if renderer is None:
        return None

    # 获取模型边界，并将空间坐标转换为世界坐标
    bounds = get_model_bounds(renderer)
    world_x, world_y, world_z = space_to_world(bounds, space_coord)
    radius = estimate_radius(bounds, ratio=radius_ratio)

    # 创建球体几何
    sphere = vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetCenter(world_x, world_y, world_z)
    sphere.SetThetaResolution(20)
    sphere.SetPhiResolution(20)
    sphere.Update()

    # 创建 mapper 和 actor
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(sphere.GetOutput())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(float(color[0]), float(color[1]), float(color[2]))
    actor.GetProperty().SetSpecular(0.5)
    actor.GetProperty().SetSpecularPower(20.0)

    # 添加到渲染器
    renderer.AddActor(actor)
    return actor


