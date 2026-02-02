"""
VTK 路径（折线）可视化工具

职责：
- 根据归一化空间坐标 [0, 100] 序列，在 3D 场景中创建路径折线 actor
"""

from typing import List, Tuple

try:
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
except Exception:  # 允许在无 VTK 环境下导入模块但不使用
    vtkPoints = None  # type: ignore
    vtkCellArray = None  # type: ignore
    vtkPolyData = None  # type: ignore
    vtkActor = None  # type: ignore
    vtkPolyDataMapper = None  # type: ignore

from .coords import get_model_bounds, space_to_world


def create_polyline_actor_from_space_points(
    renderer,
    path_points: List[Tuple[float, float, float]],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    line_width: float = 3.0,
):
    """
    根据一系列空间坐标 [0, 100] 创建路径折线 actor，并添加到 renderer 中。

    Args:
        renderer: VTK renderer 实例
        path_points: 空间坐标点列表 [(x, y, z), ...]，范围 [0, 100]
        color: 线条颜色 (r, g, b)
        line_width: 线宽

    Returns:
        创建好的 vtkActor；若 VTK 不可用或参数无效则返回 None
    """
    if vtkPoints is None or vtkCellArray is None or vtkPolyData is None or vtkActor is None or vtkPolyDataMapper is None:
        return None
    if renderer is None or len(path_points) < 2:
        return None

    # 获取模型边界，用于空间坐标 -> 世界坐标转换
    bounds = get_model_bounds(renderer)
    if bounds and len(bounds) >= 6:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
    else:
        # 退化情况下的默认范围
        x_min, x_max = -50.0, 50.0
        y_min, y_max = -50.0, 50.0
        z_min, z_max = -50.0, 50.0

    # 构建点集
    points = vtkPoints()
    for x, y, z in path_points:
        world_x, world_y, world_z = space_to_world(
            (x_min, x_max, y_min, y_max, z_min, z_max),
            (x, y, z),
        )
        points.InsertNextPoint(world_x, world_y, world_z)

    # 构建线单元
    lines = vtkCellArray()
    for i in range(len(path_points) - 1):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint(i + 1)

    poly_data = vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetLines(lines)

    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly_data)
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(float(color[0]), float(color[1]), float(color[2]))
    actor.GetProperty().SetLineWidth(float(line_width))

    renderer.AddActor(actor)
    return actor


