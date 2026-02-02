"""
PathVizManager: 路径规划 3D 可视化管理器

职责：
- 管理所有路径相关的 VTK Actors（标记点、路径线、预览标记）
- 处理 VTK 渲染更新
- 与主渲染器和独立窗口渲染器同步
"""

import logging
from typing import List, Tuple, Optional, Dict
try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkActor, vtkPolyDataMapper
    from vtkmodules.vtkFiltersSources import vtkSphereSource
except ImportError:
    vtkRenderer = None
    vtkActor = None
    vtkPolyDataMapper = None
    vtkSphereSource = None

from PyQt5.QtWidgets import QApplication

from surgical_robot_app.utils.logger import get_logger
from surgical_robot_app.vtk_utils.coords import get_model_bounds
from surgical_robot_app.vtk_utils.markers import create_sphere_marker
from surgical_robot_app.vtk_utils.path import create_polyline_actor_from_space_points

logger = get_logger("surgical_robot_app.gui.managers.path_viz_manager")

class PathVizManager:
    def __init__(self, vtk_renderer: vtkRenderer, vtk_widget, on_path_updated_callback: Optional[callable] = None):
        self.vtk_renderer = vtk_renderer
        self.vtk_widget = vtk_widget
        self.on_path_updated = on_path_updated_callback
        
        self.path_actors: List[vtkActor] = []  # 路径可视化 actors
        self.point_actors: List[vtkActor] = []  # 点标记 actors
        
        # 预览标记
        self._preview_marker_actor: Optional[vtkActor] = None
        self._reconstruction_preview_marker_actor: Optional[vtkActor] = None
        
        # 独立窗口引用
        self.reconstruction_window = None
        
        # 深度参考线
        self.depth_reference_lines: List[vtkActor] = []

    def update_vtk_display(self):
        """更新 VTK 显示"""
        if self.vtk_widget and hasattr(self.vtk_widget, "GetRenderWindow"):
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                rw.Render()
                self.vtk_widget.update()

    def add_point_marker(self, space_coord: Tuple[float, float, float], point_type: str):
        """添加点标记"""
        if self.vtk_renderer is None:
            return
        
        colors = {
            'start': (0.0, 1.0, 0.0),
            'waypoint': (1.0, 1.0, 0.0),
            'end': (1.0, 0.0, 0.0),
        }
        color = colors.get(point_type, (1.0, 1.0, 1.0))
        
        marker = create_sphere_marker(self.vtk_renderer, space_coord, color)
        if marker:
            self.point_actors.append(marker)
            if self.on_path_updated:
                try:
                    self.on_path_updated()
                except:
                    pass
        return marker

    def visualize_path(self, path_points: List[Tuple[float, float, float]]):
        """可视化路径"""
        if self.vtk_renderer is None:
            return
            
        # 清除旧路径
        self.clear_path_line()
        
        # 创建新路径
        path_actor = create_polyline_actor_from_space_points(
            self.vtk_renderer,
            path_points,
            color=(1.0, 0.0, 0.0),
            line_width=3.0
        )
        
        if path_actor:
            self.path_actors.append(path_actor)
            self.vtk_renderer.AddActor(path_actor)
            
        if self.on_path_updated:
            try:
                self.on_path_updated()
            except:
                pass
            
        return path_actor

    def visualize_path_points(self, path_points: List[Tuple[float, float, float]]):
        """可视化路径上的所有点"""
        for i, pt in enumerate(path_points):
            if i == 0:
                point_type = 'start'
            elif i == len(path_points) - 1:
                point_type = 'end'
            else:
                point_type = 'waypoint'
            self.add_point_marker(pt, point_type)

    def clear_path_line(self):
        """清除路径线"""
        for actor in self.path_actors:
            if self.vtk_renderer:
                try:
                    self.vtk_renderer.RemoveActor(actor)
                except:
                    pass
        self.path_actors.clear()

    def clear_all_path_viz(self):
        """清除所有路径可视化（点和线）"""
        self.clear_path_line()
        for actor in self.point_actors:
            if self.vtk_renderer:
                try:
                    self.vtk_renderer.RemoveActor(actor)
                except:
                    pass
        self.point_actors.clear()
        
        if self.on_path_updated:
            try:
                self.on_path_updated()
            except:
                pass

    def update_preview_marker(self, x: float, y: float, z: float, mode: str):
        """更新预览标记点"""
        if self.vtk_renderer is None:
            return
            
        # 清除主窗口预览点
        if self._preview_marker_actor:
            try:
                self.vtk_renderer.RemoveActor(self._preview_marker_actor)
            except:
                pass
            self._preview_marker_actor = None
            
        preview_colors = {
            'start': (0.0, 1.0, 0.0),
            'waypoint': (1.0, 1.0, 0.0),
            'end': (1.0, 0.0, 0.0),
        }
        color = preview_colors.get(mode, (0.8, 0.8, 0.8))
        
        try:
            bounds = get_model_bounds(self.vtk_renderer)
            if bounds:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                world_x = xmin + (xmax - xmin) * x / 100.0
                world_y = ymin + (ymax - ymin) * y / 100.0
                world_z = zmin + (zmax - zmin) * z / 100.0
                # 缩小球体比例，从 0.035 改为 0.018
                size = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.018
            else:
                world_x, world_y, world_z = x, y, z
                # 缩小默认大小，从 5.0 改为 2.5
                size = 2.5
                
            sphere = vtkSphereSource()
            sphere.SetCenter(world_x, world_y, world_z)
            sphere.SetRadius(size)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()
            
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color[0], color[1], color[2])
            actor.GetProperty().SetOpacity(0.9)
            
            self.vtk_renderer.AddActor(actor)
            self._preview_marker_actor = actor
            
            # 同步独立窗口
            self._sync_preview_to_reconstruction_window(world_x, world_y, world_z, size, color)
            
            self.update_vtk_display()
            QApplication.processEvents()
            
        except Exception as e:
            logger.warning(f"Error creating preview marker: {e}")

    def _sync_preview_to_reconstruction_window(self, wx, wy, wz, size, color):
        if self.reconstruction_window and hasattr(self.reconstruction_window, 'vtk_renderer') and self.reconstruction_window.vtk_renderer:
            try:
                recon_sphere = vtkSphereSource()
                recon_sphere.SetCenter(wx, wy, wz)
                recon_sphere.SetRadius(size)
                recon_sphere.SetThetaResolution(20)
                recon_sphere.SetPhiResolution(20)
                recon_sphere.Update()
                
                recon_mapper = vtkPolyDataMapper()
                recon_mapper.SetInputConnection(recon_sphere.GetOutputPort())
                
                recon_actor = vtkActor()
                recon_actor.SetMapper(recon_mapper)
                recon_actor.GetProperty().SetColor(color[0], color[1], color[2])
                recon_actor.GetProperty().SetOpacity(0.9)
                
                if self._reconstruction_preview_marker_actor:
                    try:
                        self.reconstruction_window.vtk_renderer.RemoveActor(self._reconstruction_preview_marker_actor)
                    except:
                        pass
                    
                self.reconstruction_window.vtk_renderer.AddActor(recon_actor)
                self._reconstruction_preview_marker_actor = recon_actor
                self.reconstruction_window._update_display()
            except Exception as e:
                logger.warning(f"Error syncing preview to reconstruction window: {e}")

    def clear_preview_marker(self):
        """清除预览标记点"""
        if self._preview_marker_actor and self.vtk_renderer:
            try:
                self.vtk_renderer.RemoveActor(self._preview_marker_actor)
                self.update_vtk_display()
            except Exception as e:
                logger.warning(f"Error clearing preview marker: {e}")
            finally:
                self._preview_marker_actor = None
                
        if self._reconstruction_preview_marker_actor and self.reconstruction_window and hasattr(self.reconstruction_window, 'vtk_renderer') and self.reconstruction_window.vtk_renderer:
            try:
                self.reconstruction_window.vtk_renderer.RemoveActor(self._reconstruction_preview_marker_actor)
                self.reconstruction_window._update_display()
            except Exception as e:
                logger.warning(f"Error clearing reconstruction preview marker: {e}")
            finally:
                self._reconstruction_preview_marker_actor = None

    def clear_depth_reference_lines(self):
        """清除深度参考线"""
        if self.vtk_renderer:
            for line_actor in self.depth_reference_lines:
                try:
                    self.vtk_renderer.RemoveActor(line_actor)
                except:
                    pass
            self.depth_reference_lines = []
            self.update_vtk_display()

