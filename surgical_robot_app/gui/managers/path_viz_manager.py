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
    from vtkmodules.vtkFiltersCore import vtkTubeFilter
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
except ImportError:
    vtkRenderer = None
    vtkActor = None
    vtkPolyDataMapper = None
    vtkSphereSource = None
    vtkTubeFilter = None
    vtkPoints = None
    vtkCellArray = None
    vtkPolyData = None

from PyQt5.QtWidgets import QApplication

from surgical_robot_app.utils.logger import get_logger
from surgical_robot_app.vtk_utils.coords import get_model_bounds, space_to_world
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
        
        # RRT 探索树实时可视化
        self._rrt_tree_actor: Optional[vtkActor] = None
        self._rrt_found_path_actor: Optional[vtkActor] = None
        
        # 独立窗口引用
        self.reconstruction_window = None
        
        # Safety zone tube
        self._safety_tube_actor: Optional[vtkActor] = None
        
        # 路径仿真器械 actor
        self._sim_instrument_actor: Optional[vtkActor] = None
        self._sim_trail_actor: Optional[vtkActor] = None  # 已走过的轨迹
        
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

    # -------------------- RRT 探索树实时可视化 --------------------

    def update_rrt_tree_viz(
        self,
        edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        found_path: Optional[List[Tuple[float, float, float]]] = None,
    ):
        """
        更新 RRT 探索树的实时可视化。
        
        Args:
            edges: 树的边列表 [(parent_space, child_space), ...]，坐标为 [0,100] 空间
            found_path: 如果已找到路径，传入路径点列表（用绿色高亮）
        """
        if self.vtk_renderer is None or vtkPoints is None:
            return
        
        # 获取模型边界用于坐标转换
        bounds = get_model_bounds(self.vtk_renderer)
        if bounds and len(bounds) >= 6:
            b = bounds
        else:
            b = (-50.0, 50.0, -50.0, 50.0, -50.0, 50.0)
        
        # --- 1. 渲染探索树（半透明灰色） ---
        if self._rrt_tree_actor:
            try:
                self.vtk_renderer.RemoveActor(self._rrt_tree_actor)
            except:
                pass
            self._rrt_tree_actor = None
        
        if edges:
            points = vtkPoints()
            lines = vtkCellArray()
            
            for parent_pt, child_pt in edges:
                wp = space_to_world(b, parent_pt)
                wc = space_to_world(b, child_pt)
                id1 = points.InsertNextPoint(wp)
                id2 = points.InsertNextPoint(wc)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(id1)
                lines.InsertCellPoint(id2)
            
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)
            
            mapper = vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.6, 0.6, 0.6)   # 灰色
            actor.GetProperty().SetOpacity(0.25)            # 半透明
            actor.GetProperty().SetLineWidth(1.0)
            
            self.vtk_renderer.AddActor(actor)
            self._rrt_tree_actor = actor
        
        # --- 2. 渲染已找到的路径（绿色高亮） ---
        if self._rrt_found_path_actor:
            try:
                self.vtk_renderer.RemoveActor(self._rrt_found_path_actor)
            except:
                pass
            self._rrt_found_path_actor = None
        
        if found_path and len(found_path) >= 2:
            path_points = vtkPoints()
            path_lines = vtkCellArray()
            
            for pt in found_path:
                w = space_to_world(b, pt)
                path_points.InsertNextPoint(w)
            
            for i in range(len(found_path) - 1):
                path_lines.InsertNextCell(2)
                path_lines.InsertCellPoint(i)
                path_lines.InsertCellPoint(i + 1)
            
            path_polydata = vtkPolyData()
            path_polydata.SetPoints(path_points)
            path_polydata.SetLines(path_lines)
            
            path_mapper = vtkPolyDataMapper()
            path_mapper.SetInputData(path_polydata)
            
            path_actor = vtkActor()
            path_actor.SetMapper(path_mapper)
            path_actor.GetProperty().SetColor(0.0, 1.0, 0.3)  # 绿色
            path_actor.GetProperty().SetOpacity(0.9)
            path_actor.GetProperty().SetLineWidth(3.0)
            
            self.vtk_renderer.AddActor(path_actor)
            self._rrt_found_path_actor = path_actor
        
        # 刷新显示
        self.update_vtk_display()
        QApplication.processEvents()

    def clear_rrt_tree_viz(self):
        """清除 RRT 探索树可视化"""
        if self.vtk_renderer:
            if self._rrt_tree_actor:
                try:
                    self.vtk_renderer.RemoveActor(self._rrt_tree_actor)
                except:
                    pass
                self._rrt_tree_actor = None
            if self._rrt_found_path_actor:
                try:
                    self.vtk_renderer.RemoveActor(self._rrt_found_path_actor)
                except:
                    pass
                self._rrt_found_path_actor = None

    # -------------------- Path Simulation Visualization --------------------

    def update_sim_instrument(self, world_pos: Tuple[float, float, float], size: float = None):
        """
        Update the simulation instrument (sphere) position.
        
        Args:
            world_pos: (x, y, z) in world coordinates
            size: sphere radius; auto-calculated from model bounds if None
        """
        if self.vtk_renderer is None or vtkSphereSource is None:
            return
        
        # Remove old actor
        if self._sim_instrument_actor:
            try:
                self.vtk_renderer.RemoveActor(self._sim_instrument_actor)
            except:
                pass
            self._sim_instrument_actor = None
        
        # Calculate size from model bounds if not provided
        if size is None:
            bounds = get_model_bounds(self.vtk_renderer)
            if bounds:
                xmin, xmax, ymin, ymax, zmin, zmax = bounds
                size = max(xmax - xmin, ymax - ymin, zmax - zmin) * 0.02
            else:
                size = 2.0
        
        sphere = vtkSphereSource()
        sphere.SetCenter(world_pos[0], world_pos[1], world_pos[2])
        sphere.SetRadius(size)
        sphere.SetThetaResolution(24)
        sphere.SetPhiResolution(24)
        sphere.Update()
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.8, 1.0)  # cyan
        actor.GetProperty().SetOpacity(0.95)
        actor.GetProperty().SetSpecular(0.6)
        actor.GetProperty().SetSpecularPower(30)
        
        self.vtk_renderer.AddActor(actor)
        self._sim_instrument_actor = actor

    def update_sim_trail(self, world_points: List[Tuple[float, float, float]]):
        """
        Draw the trail (already-traversed portion) of the simulation path.
        
        Args:
            world_points: list of world-coord points for the trail
        """
        if self.vtk_renderer is None or vtkPoints is None:
            return
        
        # Remove old trail
        if self._sim_trail_actor:
            try:
                self.vtk_renderer.RemoveActor(self._sim_trail_actor)
            except:
                pass
            self._sim_trail_actor = None
        
        if len(world_points) < 2:
            return
        
        points = vtkPoints()
        lines = vtkCellArray()
        for pt in world_points:
            points.InsertNextPoint(pt)
        for i in range(len(world_points) - 1):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)
            lines.InsertCellPoint(i + 1)
        
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 1.0, 0.5)  # bright green
        actor.GetProperty().SetOpacity(0.9)
        actor.GetProperty().SetLineWidth(4.0)
        
        self.vtk_renderer.AddActor(actor)
        self._sim_trail_actor = actor

    def clear_sim_viz(self):
        """Clear all simulation visualization (instrument + trail)."""
        if self.vtk_renderer:
            if self._sim_instrument_actor:
                try:
                    self.vtk_renderer.RemoveActor(self._sim_instrument_actor)
                except:
                    pass
                self._sim_instrument_actor = None
            if self._sim_trail_actor:
                try:
                    self.vtk_renderer.RemoveActor(self._sim_trail_actor)
                except:
                    pass
                self._sim_trail_actor = None

    # -------------------- Safety Zone Tube Visualization --------------------

    def show_safety_tube(
        self,
        path_space_points: List[Tuple[float, float, float]],
        safety_radius_world: float = None,
    ):
        """
        Display a semi-transparent tube around the path representing the safety margin.
        
        Args:
            path_space_points: path in [0,100] normalised space coordinates
            safety_radius_world: tube radius in world units. If None, auto-calculated
                                 as 2% of the model's bounding-box diagonal.
        """
        if self.vtk_renderer is None or vtkTubeFilter is None or vtkPoints is None:
            return
        
        # Remove any existing tube first
        self.clear_safety_tube()
        
        if len(path_space_points) < 2:
            return
        
        bounds = get_model_bounds(self.vtk_renderer)
        if bounds and len(bounds) >= 6:
            b = bounds
        else:
            b = (-50.0, 50.0, -50.0, 50.0, -50.0, 50.0)
        
        # Auto-calculate radius if not provided
        if safety_radius_world is None:
            dx = b[1] - b[0]
            dy = b[3] - b[2]
            dz = b[5] - b[4]
            diagonal = (dx**2 + dy**2 + dz**2) ** 0.5
            safety_radius_world = diagonal * 0.02  # 2% of diagonal
        
        # Build polyline in world coordinates
        points = vtkPoints()
        lines = vtkCellArray()
        
        for pt in path_space_points:
            w = space_to_world(b, pt)
            points.InsertNextPoint(w)
        
        n = len(path_space_points)
        lines.InsertNextCell(n)
        for i in range(n):
            lines.InsertCellPoint(i)
        
        polydata = vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        # Apply tube filter
        tube = vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(safety_radius_world)
        tube.SetNumberOfSides(20)
        tube.CappingOn()
        tube.Update()
        
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.2, 0.9, 0.3)   # green
        actor.GetProperty().SetOpacity(0.18)            # semi-transparent
        actor.GetProperty().SetSpecular(0.1)
        
        # Enable depth peeling for correct transparency rendering
        rw = None
        if self.vtk_widget and hasattr(self.vtk_widget, 'GetRenderWindow'):
            rw = self.vtk_widget.GetRenderWindow()
        if rw:
            rw.SetAlphaBitPlanes(1)
            rw.SetMultiSamples(0)
            self.vtk_renderer.SetUseDepthPeeling(1)
            self.vtk_renderer.SetMaximumNumberOfPeels(8)
            self.vtk_renderer.SetOcclusionRatio(0.1)
        
        self.vtk_renderer.AddActor(actor)
        self._safety_tube_actor = actor
        
        self.update_vtk_display()

    def clear_safety_tube(self):
        """Remove the safety zone tube from the renderer."""
        if self.vtk_renderer and self._safety_tube_actor:
            try:
                self.vtk_renderer.RemoveActor(self._safety_tube_actor)
            except:
                pass
            self._safety_tube_actor = None


