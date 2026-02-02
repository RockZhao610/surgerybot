"""
3D模型表面选点工具

实现基于3D模型表面的两次点击选点功能：
- 第一次点击：在主视角点击模型表面，确定投影坐标
- 第二次点击：切换到正交视角，再次点击，通过几何计算确定3D坐标
- 显示深度参考线辅助用户
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import logging

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__) if get_logger else None

try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkCellPicker, vtkActor
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    vtkRenderer = None
    vtkCellPicker = None
    vtkActor = None
    vtkPolyData = None
    vtkPolyDataNormals = None


def detect_camera_view_type(camera) -> Optional[str]:
    """
    检测当前相机视角类型
    
    Args:
        camera: VTK相机对象
    
    Returns:
        'axial' | 'coronal' | 'sagittal' | None
    """
    if camera is None:
        return None
    
    try:
        view_up = camera.GetViewUp()
        position = camera.GetPosition()
        focal = camera.GetFocalPoint()
        
        # 计算视线方向
        view_dir = np.array([
            focal[0] - position[0],
            focal[1] - position[1],
            focal[2] - position[2]
        ])
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
        
        # 计算view_up方向
        up_dir = np.array([view_up[0], view_up[1], view_up[2]])
        up_dir = up_dir / (np.linalg.norm(up_dir) + 1e-8)
        
        # 计算右方向（view_dir × up_dir）
        right_dir = np.cross(view_dir, up_dir)
        right_dir = right_dir / (np.linalg.norm(right_dir) + 1e-8)
        
        # 判断主要方向
        # 轴向：主要看向Z方向（view_dir接近(0,0,1)或(0,0,-1)）
        # 冠状：主要看向Y方向（view_dir接近(0,1,0)或(0,-1,0)）
        # 矢状：主要看向X方向（view_dir接近(1,0,0)或(-1,0,0)）
        
        abs_view_dir = np.abs(view_dir)
        max_idx = np.argmax(abs_view_dir)
        
        if max_idx == 2:  # Z方向最大
            return 'axial'
        elif max_idx == 1:  # Y方向最大
            return 'coronal'
        elif max_idx == 0:  # X方向最大
            return 'sagittal'
        else:
            return None
    except Exception as e:
        logger.error(f"Error detecting camera view type: {e}") if logger else None
        return None


def get_surface_normal(picker, renderer) -> Optional[Tuple[float, float, float]]:
    """
    获取点击位置处的表面法向量
    
    Args:
        picker: VTK CellPicker对象
        renderer: VTK渲染器
    
    Returns:
        法向量 (nx, ny, nz) 或 None
    """
    if picker is None or renderer is None:
        return None
    
    try:
        cell_id = picker.GetCellId()
        if cell_id < 0:
            return None
        
        # 获取被点击的actor
        actor = picker.GetActor()
        if actor is None:
            return None
        
        # 获取mapper和poly data
        mapper = actor.GetMapper()
        if mapper is None:
            return None
        
        poly_data = mapper.GetInput()
        if poly_data is None:
            return None
        
        # 获取cell的法向量
        # 方法1：使用cell的中心点和相邻点计算法向量
        cell = poly_data.GetCell(cell_id)
        if cell is None:
            return None
        
        # 获取cell的点
        num_points = cell.GetNumberOfPoints()
        if num_points < 3:
            return None
        
        # 计算cell的法向量（使用前三个点）
        p0 = cell.GetPoints().GetPoint(0)
        p1 = cell.GetPoints().GetPoint(1)
        p2 = cell.GetPoints().GetPoint(2)
        
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]])
        v2 = np.array([p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]])
        
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
            return tuple(normal.astype(float))
        
        return None
    except Exception as e:
        logger.error(f"Error getting surface normal: {e}") if logger else None
        return None


def project_to_view_plane(
    world_pos: Tuple[float, float, float],
    view_type: str,
    camera
) -> Tuple[float, float]:
    """
    将世界坐标投影到当前视角的2D平面
    
    Args:
        world_pos: 世界坐标 (x, y, z)
        view_type: 视角类型 ('axial', 'coronal', 'sagittal')
        camera: VTK相机对象
    
    Returns:
        投影坐标 (u, v)，归一化到 [0, 100]
    """
    x, y, z = world_pos
    
    if view_type == 'axial':
        # 轴向：投影到XY平面，Z为深度
        return (x, y)
    elif view_type == 'coronal':
        # 冠状：投影到XZ平面，Y为深度
        return (x, z)
    elif view_type == 'sagittal':
        # 矢状：投影到YZ平面，X为深度
        return (y, z)
    else:
        # 默认：使用XY投影
        return (x, y)


def compute_3d_from_two_projections(
    click1: Dict,
    click2: Dict,
    bounds: Optional[Tuple[float, float, float, float, float, float]]
) -> Optional[Tuple[float, float, float]]:
    """
    从两次正交视角的投影计算3D坐标
    
    Args:
        click1: 第一次点击数据 {'world_pos': (x,y,z), 'view_type': str, 'projection_2d': (u,v)}
        click2: 第二次点击数据
        bounds: 模型边界
    
    Returns:
        3D空间坐标 (x, y, z)，归一化到 [0, 100]
    """
    view1 = click1.get('view_type')
    view2 = click2.get('view_type')
    
    if view1 == view2:
        logger.warning("两次点击必须在不同视角") if logger else None
        return None
    
    world_pos1 = click1.get('world_pos')
    world_pos2 = click2.get('world_pos')
    
    if world_pos1 is None or world_pos2 is None:
        return None
    
    # 根据视角组合计算3D坐标
    # 情况1：轴向 + 冠状
    if (view1 == 'axial' and view2 == 'coronal') or (view1 == 'coronal' and view2 == 'axial'):
        if view1 == 'axial':
            # 轴向提供 X, Y；冠状提供 X, Z
            x = world_pos1[0]  # 从轴向
            y = world_pos1[1]  # 从轴向
            z = world_pos2[2]  # 从冠状
        else:
            x = world_pos2[0]  # 从冠状
            y = world_pos1[1]  # 从轴向
            z = world_pos1[2]  # 从冠状
    
    # 情况2：轴向 + 矢状
    elif (view1 == 'axial' and view2 == 'sagittal') or (view1 == 'sagittal' and view2 == 'axial'):
        if view1 == 'axial':
            # 轴向提供 X, Y；矢状提供 Y, Z
            x = world_pos1[0]  # 从轴向
            y = world_pos2[1]  # 从矢状
            z = world_pos2[2]  # 从矢状
        else:
            x = world_pos1[0]  # 从矢状
            y = world_pos2[1]  # 从轴向
            z = world_pos1[2]  # 从矢状
    
    # 情况3：冠状 + 矢状
    elif (view1 == 'coronal' and view2 == 'sagittal') or (view1 == 'sagittal' and view2 == 'coronal'):
        if view1 == 'coronal':
            # 冠状提供 X, Z；矢状提供 Y, Z
            x = world_pos1[0]  # 从冠状
            y = world_pos2[1]  # 从矢状
            z = (world_pos1[2] + world_pos2[2]) / 2.0  # 取平均值
        else:
            x = world_pos2[0]  # 从冠状
            y = world_pos1[1]  # 从矢状
            z = (world_pos1[2] + world_pos2[2]) / 2.0  # 取平均值
    
    else:
        logger.warning(f"不支持的视角组合: {view1} + {view2}") if logger else None
        return None
    
    # 转换为空间坐标 [0, 100]
    try:
        from surgical_robot_app.vtk_utils.coords import world_to_space
    except ImportError:
        try:
            from vtk_utils.coords import world_to_space
        except ImportError:
            logger.error("world_to_space not available") if logger else None
            return None
    
    space_coord = world_to_space(bounds, (x, y, z))
    return space_coord


def create_depth_reference_line(
    renderer,
    start_pos: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    length: float = 10.0,
    color: Tuple[float, float, float] = (1.0, 1.0, 0.0),  # 黄色
    opacity: float = 0.6
) -> Optional[vtkActor]:
    """
    创建深度参考线（垂直于当前视角的虚线）
    
    Args:
        renderer: VTK渲染器
        start_pos: 起点（模型表面点击位置）
        normal: 表面法向量
        length: 线长度
        color: 线颜色
        opacity: 透明度
    
    Returns:
        VTK Actor对象
    """
    if not VTK_AVAILABLE or renderer is None:
        return None
    
    try:
        from vtkmodules.vtkCommonDataModel import vtkPoints, vtkCellArray, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
        from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
        
        # 计算终点
        end_pos = (
            start_pos[0] + normal[0] * length,
            start_pos[1] + normal[1] * length,
            start_pos[2] + normal[2] * length
        )
        
        # 创建点
        points = vtkPoints()
        points.InsertNextPoint(start_pos[0], start_pos[1], start_pos[2])
        points.InsertNextPoint(end_pos[0], end_pos[1], end_pos[2])
        
        # 创建线
        lines = vtkCellArray()
        lines.InsertNextCell(2)
        lines.InsertCellPoint(0)
        lines.InsertCellPoint(1)
        
        # 创建PolyData
        poly_data = vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        # 创建Mapper
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        # 创建Actor
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color[0], color[1], color[2])
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(2.0)
        actor.GetProperty().SetLineStipplePattern(0xf0f0)  # 虚线模式
        
        return actor
    except Exception as e:
        logger.error(f"Error creating depth reference line: {e}") if logger else None
        return None

