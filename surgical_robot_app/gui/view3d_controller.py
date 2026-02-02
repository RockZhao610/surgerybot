"""
View3DController: 3D 视图控制器

职责：
- 控制 VTK 渲染器与渲染窗口的基本操作：
  - 重置相机
  - 加载 STL 模型
  - 清空 3D 视图

注意：
- 不依赖 Qt Widget，只接收 renderer / render_window 等对象，便于复用。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    from vtkmodules.vtkRenderingCore import vtkActor
except Exception:
    vtkActor = None

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.gui.view3d_controller")

try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper
    from vtkmodules.vtkIOGeometry import vtkSTLWriter, vtkSTLReader
except Exception:  # 允许在无 VTK 环境下导入模块但不使用
    vtkRenderer = None  # type: ignore
    vtkActor = None  # type: ignore
    vtkPolyDataMapper = None  # type: ignore
    vtkSTLWriter = None  # type: ignore
    vtkSTLReader = None  # type: ignore


@dataclass
class CameraInfo:
    position: Tuple[float, float, float]
    focal: Tuple[float, float, float]
    view_up: Tuple[float, float, float]


class View3DController:
    def __init__(self, renderer, vtk_widget):
        self.renderer = renderer
        self.vtk_widget = vtk_widget
        self.current_model_actor: Optional[vtkActor] = None  # 保存当前模型Actor

    # -------------------- 相机相关 --------------------

    def get_camera_info(self) -> Optional[CameraInfo]:
        """获取当前相机位置信息"""
        if self.renderer is None or self.vtk_widget is None:
            return None
        try:
            camera = self.renderer.GetActiveCamera()
            if camera:
                position = camera.GetPosition()
                focal = camera.GetFocalPoint()
                view_up = camera.GetViewUp()
                return CameraInfo(
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    focal=(float(focal[0]), float(focal[1]), float(focal[2])),
                    view_up=(float(view_up[0]), float(view_up[1]), float(view_up[2])),
                )
        except Exception as e:
            logger.error(f"Error getting camera info: {e}")
        return None

    def reset_camera(self) -> Optional[CameraInfo]:
        """重置相机并返回最新的相机信息"""
        if self.renderer is None or self.vtk_widget is None:
            return None
        try:
            self.renderer.ResetCamera()
            self.renderer.ResetCameraClippingRange()
            rw = self.vtk_widget.GetRenderWindow() if hasattr(self.vtk_widget, "GetRenderWindow") else None
            if rw:
                rw.Render()
                self.vtk_widget.update()
            return self.get_camera_info()
        except Exception as e:
            logger.error(f"Error resetting camera: {e}")
            return None

    # -------------------- 模型加载与视图控制 --------------------

    def clear_view(self) -> None:
        """清空当前 3D 视图中的所有可见对象"""
        if self.renderer is None or self.vtk_widget is None:
            return
        try:
            self.renderer.RemoveAllViewProps()
            rw = self.vtk_widget.GetRenderWindow() if hasattr(self.vtk_widget, "GetRenderWindow") else None
            if rw:
                rw.Render()
                self.vtk_widget.update()
        except Exception as e:
            logger.warning(f"Failed to clear 3D view in View3DController: {e}")

    def load_stl_model(self, file_path: str) -> Optional[Tuple[int, int]]:
        """
        加载 STL 模型并渲染到 3D 视图中。

        Returns:
            (points, cells) 若加载成功，否则 None
        """
        if self.renderer is None or self.vtk_widget is None:
            return None
        if vtkSTLReader is None or vtkActor is None or vtkPolyDataMapper is None:
            logger.warning("VTK STL reader/actor not available.")
            return None

        reader = vtkSTLReader()
        reader.SetFileName(file_path)
        try:
            reader.Update()
        except Exception as e:
            logger.error(f"Failed to read STL file: {e}")
            return None

        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            logger.warning("Empty STL model.")
            return None

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        mapper.ScalarVisibilityOff()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.3, 0.6, 1.0)
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(15.0)

        self.renderer.RemoveAllViewProps()
        self.renderer.AddActor(actor)
        self.current_model_actor = actor  # 保存模型Actor引用
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()

        try:
            rw = self.vtk_widget.GetRenderWindow() if hasattr(self.vtk_widget, "GetRenderWindow") else None
            if rw:
                rw.Render()
                self.vtk_widget.update()
            pts = poly.GetNumberOfPoints()
            cells = poly.GetNumberOfCells()
            return int(pts), int(cells)
        except Exception as e:
            logger.warning(f"Render failed after loading STL: {e}")
            return None


