"""
PathUIController: 路径规划 UI 控制器

职责：
- 处理路径规划相关的事件（选点、路径生成、可视化等）
- 管理路径规划 UI 状态
- 不直接依赖 Qt Widget，通过回调与 MainWindow 通信
"""

import time
from typing import Optional, Callable, List, Tuple
from PyQt5.QtWidgets import QMessageBox, QListWidget, QLabel, QDialog
from PyQt5.QtCore import QEvent, Qt, QObject, QTimer

try:
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.utils.error_handler import handle_errors
    from surgical_robot_app.utils.threading_utils import run_in_thread
except ImportError:
    from utils.logger import get_logger
    from utils.threading_utils import run_in_thread
    try:
        from utils.error_handler import handle_errors
    except ImportError:
        # 如果错误处理模块不可用，创建一个空的装饰器
        def handle_errors(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

logger = get_logger("surgical_robot_app.gui.controllers.path_ui_controller")

try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkCellPicker, vtkCommand
    from vtkmodules.vtkCommonCore import vtkCommand
except Exception:
    vtkRenderer = None
    vtkCellPicker = None
    vtkCommand = None

try:
    from surgical_robot_app.path_planning.path_controller import PathController
    from surgical_robot_app.gui.view3d_controller import View3DController
    from surgical_robot_app.vtk_utils.coords import get_model_bounds, world_to_space
except ImportError:
    try:
        from path_planning.path_controller import PathController
        from gui.view3d_controller import View3DController
        from vtk_utils.coords import get_model_bounds, world_to_space
    except ImportError:
        PathController = None  # type: ignore
        View3DController = None  # type: ignore
        get_model_bounds = None
        world_to_space = None
        create_sphere_marker = None
        create_polyline_actor_from_space_points = None


class PathUIController(QObject):
    """路径规划 UI 控制器"""
    
    def __init__(
        self,
        path_controller: PathController,
        view3d_controller: View3DController,
        vtk_renderer: vtkRenderer,
        vtk_widget,
        path_list: QListWidget,
        vtk_status: QLabel,
        parent_widget=None,
    ):
        """
        初始化路径规划 UI 控制器
        
        Args:
            path_controller: 路径规划控制器
            view3d_controller: 3D视图控制器
            vtk_renderer: VTK渲染器
            vtk_widget: VTK Widget
            path_list: 路径点列表控件
            vtk_status: VTK状态标签
            parent_widget: 父窗口
        """
        super().__init__()
        self.path_controller = path_controller
        self.view3d_controller = view3d_controller
        self.vtk_renderer = vtk_renderer
        self.vtk_widget = vtk_widget
        self.path_list = path_list
        self.vtk_status = vtk_status
        self.parent_widget = parent_widget
        
        # UI 状态
        self.pick_mode: Optional[str] = None  # 'start', 'waypoint', 'end'
        
        # 初始化可视化管理器 (瘦身：将 VTK 相关逻辑移至 PathVizManager)
        from surgical_robot_app.gui.managers.path_viz_manager import PathVizManager
        self.viz_manager = PathVizManager(
            vtk_renderer=self.vtk_renderer,
            vtk_widget=self.vtk_widget,
            on_path_updated_callback=lambda: self.on_path_updated() if self.on_path_updated else None
        )
        
        # 初始化业务逻辑服务 (瘦身：将算法协调逻辑移至 PathService)
        from surgical_robot_app.services.path_service import PathService
        self.path_service = PathService(self.path_controller)
        
        # 初始化路径评估器
        from surgical_robot_app.path_planning.path_evaluator import PathEvaluator
        self.path_evaluator = PathEvaluator()
        
        # 模型表面选点状态（用于两次点击模式）
        self.model_surface_pick_data: Dict[str, Dict] = {}  # 存储两次点击数据
        
        # RRT 回放相关
        self._rrt_snapshots: List[dict] = []        # 存储规划过程中的快照
        self._replay_timer: Optional[QTimer] = None  # 回放定时器
        self._replay_index: int = 0                  # 当前回放帧索引
        self._replay_speed_ms: int = 80              # 回放速度（毫秒/帧）
        self._is_replaying: bool = False              # 是否正在回放
        
        # Path simulation 相关
        self._sim_timer: Optional[QTimer] = None
        self._sim_world_points: List[Tuple[float, float, float]] = []  # 插值后的世界坐标点
        self._sim_index: int = 0
        self._sim_speed_ms: int = 30           # 帧间隔 (ms)
        self._sim_steps_per_segment: int = 5   # 每两个路径点之间插值的步数
        self._is_simulating: bool = False
        
        # Safety zone
        self._safety_zone_visible: bool = False
        
        # 回调函数
        self.on_path_generated: Optional[Callable] = None
        self.on_path_reset: Optional[Callable] = None
        self.on_path_updated: Optional[Callable] = None  # 路径更新回调（用于同步到独立窗口）
    
    @property
    def reconstruction_window(self):
        """代理到 viz_manager"""
        return self.viz_manager.reconstruction_window
    
    @reconstruction_window.setter
    def reconstruction_window(self, value):
        """代理到 viz_manager"""
        self.viz_manager.reconstruction_window = value
    
    def handle_set_pick_mode(self, mode: str):
        """设置选点模式（智能选择：有体数据用切片对话框，STL模型用坐标平面对话框）"""
        # 检查是否已生成路径
        if self.path_controller.path_points and len(self.path_controller.path_points) > 0:
            QMessageBox.warning(
                self.parent_widget,
                "Path Already Generated",
                "Please click 'Reset Path' first before picking new points."
            )
            return
        
        # 检查是否有数据
        if not hasattr(self.parent_widget, 'data_manager'):
            QMessageBox.warning(
                self.parent_widget,
                "No Data",
                "Please load volume data or 3D model first."
            )
            return
        
        data_manager = self.parent_widget.data_manager
        volume = data_manager.get_volume()
        # 检查是否有有效的体数据（不仅存在，还要有实际数据）
        has_volume = volume is not None and volume.size > 0 and len(volume.shape) == 3
        
        # 检查是否有3D模型（通过检查renderer中是否有模型actors，排除坐标轴）
        has_model = False
        if self.vtk_renderer:
            try:
                actors = self.vtk_renderer.GetActors()
                actors.InitTraversal()
                while True:
                    actor = actors.GetNextItem()
                    if actor is None:
                        break
                    # 检查是否是模型actor（不是坐标轴）
                    mapper = actor.GetMapper()
                    if mapper:
                        input_data = mapper.GetInput()
                        if input_data:
                            try:
                                num_points = input_data.GetNumberOfPoints()
                                if num_points > 100:  # 模型通常有很多点
                                    has_model = True
                                    break
                            except:
                                pass
            except:
                pass
        
        # 智能选择选点方式
        if has_volume:
            # 有体数据：使用切片视图对话框
            self._handle_set_pick_mode_with_volume(mode, data_manager)
        elif has_model:
            # 只有STL模型：使用坐标平面选点对话框
            self._handle_set_pick_mode_with_stl(mode)
        else:
            QMessageBox.warning(
                self.parent_widget,
                "No Data",
                "Please load volume data or 3D model first."
            )
    
    def _handle_set_pick_mode_with_volume(self, mode: str, data_manager):
        """使用体数据的选点模式（弹窗对话框）"""
        # 设置选点模式
        self.pick_mode = mode
        
        # 打开选点对话框
        try:
            from surgical_robot_app.gui.dialogs.path_point_picker_dialog import PathPointPickerDialog
        except ImportError:
            try:
                from gui.dialogs.path_point_picker_dialog import PathPointPickerDialog
            except ImportError:
                logger.error("PathPointPickerDialog not available, falling back to old method")
                self._handle_set_pick_mode_old(mode)
                return
        
        dialog = PathPointPickerDialog(
            data_manager=data_manager,
            point_type=mode,
            parent=self.parent_widget
        )
        
        # 连接信号
        def on_point_selected(x, y, z):
            # 处理选中的点（pick_mode 已经设置）
            self._process_picked_point(x, y, z)
        
        dialog.point_selected.connect(on_point_selected)
        
        # 显示对话框
        result = dialog.exec_()
        
        if result == QDialog.Accepted:
            # 点已通过信号处理
            mode_names = {'start': 'Start', 'waypoint': 'Waypoint', 'end': 'End'}
            if self.vtk_status:
                self.vtk_status.setText(
                    f"{mode_names.get(mode, mode)} point selected via dialog"
                )
        else:
            # 用户取消，清除选点模式
            self.pick_mode = None
            if self.vtk_status:
                self.vtk_status.setText("Point selection cancelled")
    
    def _collect_all_model_polydata(self):
        """
        收集场景中所有模型的 PolyData 并合并为一个。
        支持多标签重建后有多个 actor 的情况。
        
        Returns:
            合并后的 vtkPolyData 或 None
        """
        if not self.vtk_renderer:
            return None
        
        from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
        from vtkmodules.vtkCommonCore import vtkPoints
        
        all_points = []
        model_count = 0
        
        actors = self.vtk_renderer.GetActors()
        actors.InitTraversal()
        while True:
            actor = actors.GetNextItem()
            if actor is None:
                break
            mapper = actor.GetMapper()
            if mapper:
                input_data = mapper.GetInput()
                if input_data:
                    try:
                        n = input_data.GetNumberOfPoints()
                        if n > 100:
                            pts = input_data.GetPoints()
                            if pts:
                                for i in range(n):
                                    all_points.append(pts.GetPoint(i))
                                model_count += 1
                    except Exception:
                        pass
        
        if not all_points:
            return None
        
        # 创建合并的 PolyData
        combined = vtkPolyData()
        vtk_pts = vtkPoints()
        vtk_verts = vtkCellArray()
        
        for pt in all_points:
            pid = vtk_pts.InsertNextPoint(pt)
            vtk_verts.InsertNextCell(1)
            vtk_verts.InsertCellPoint(pid)
        
        combined.SetPoints(vtk_pts)
        combined.SetVerts(vtk_verts)
        
        logger.info(f"合并 {model_count} 个模型: 共 {len(all_points)} 个点")
        return combined
    
    def _handle_set_pick_mode_with_stl(self, mode: str):
        """使用STL模型的选点模式（坐标平面选点对话框）"""
        # 设置选点模式
        self.pick_mode = mode
        
        # 导入坐标平面选点对话框
        try:
            from surgical_robot_app.gui.dialogs.coordinate_plane_picker_dialog import CoordinatePlanePickerDialog
        except ImportError:
            try:
                from gui.dialogs.coordinate_plane_picker_dialog import CoordinatePlanePickerDialog
            except ImportError:
                logger.error("CoordinatePlanePickerDialog not available")
                QMessageBox.warning(
                    self.parent_widget,
                    "Error",
                    "Coordinate plane picker dialog is not available."
                )
                self.pick_mode = None
                return
        
        # 获取模型边界（用于参考）
        model_bounds = None
        if get_model_bounds and self.vtk_renderer:
            model_bounds = get_model_bounds(self.vtk_renderer)
        
        # 获取模型PolyData（用于在平面视图中显示投影）
        # 多标签重建后可能有多个模型，需要合并所有模型的点
        model_polydata = None
        if self.vtk_renderer:
            try:
                model_polydata = self._collect_all_model_polydata()
            except Exception as e:
                logger.warning(f"Error getting model PolyData: {e}")
        
        # 打开坐标平面选点对话框
        dialog = CoordinatePlanePickerDialog(
            point_type=mode,
            model_bounds=model_bounds,
            model_polydata=model_polydata,
            parent=self.parent_widget
        )
        
        # 保存对话框引用和当前模式
        self._stl_pick_dialog = dialog
        self._stl_pick_mode = mode
        
        # 连接信号 - 确认选点
        def on_point_selected(x, y, z):
            # 处理选中的点（空间坐标 0-100）
            self.viz_manager.clear_preview_marker()
            self._process_picked_point(x, y, z)
            mode_names = {'start': 'Start', 'waypoint': 'Waypoint', 'end': 'End'}
            if self.vtk_status:
                self.vtk_status.setText(
                    f"{mode_names.get(mode, mode)} point selected via coordinate planes"
                )
            self._stl_pick_dialog = None
        
        dialog.point_selected.connect(on_point_selected)
        
        # 连接信号 - 取消/关闭
        def on_dialog_rejected():
            self.viz_manager.clear_preview_marker()
            self.pick_mode = None
            if self.vtk_status:
                self.vtk_status.setText("Point selection cancelled")
            self._stl_pick_dialog = None
        
        dialog.rejected.connect(on_dialog_rejected)
        
        # 连接信号 - 实时坐标变化，用于3D预览
        def on_coordinates_changed(x, y, z):
            self.viz_manager.update_preview_marker(x, y, z, mode)
        
        dialog.coordinates_changed.connect(on_coordinates_changed)
        
        # 初始化预览点（显示初始位置）
        self.viz_manager.update_preview_marker(50.0, 50.0, 50.0, mode)
        
        # 使用非模态方式显示对话框，这样主窗口可以实时更新
        dialog.setWindowModality(Qt.NonModal)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
    
    def _handle_set_pick_mode_old(self, mode: str):
        """旧的选点模式（备用方法）"""
        self.pick_mode = mode
        mode_names = {'start': 'Start Point', 'waypoint': 'Waypoint', 'end': 'End Point'}
        self.vtk_status.setText(f"Pick Mode: {mode_names.get(mode, mode)} - Click in 3D window or 2D slice views")
        
        # 安装VTK交互器事件过滤器
        if self.vtk_widget and hasattr(self.vtk_widget, "GetInteractor"):
            interactor = self.vtk_widget.GetInteractor()
            if interactor and vtkCommand:
                # 移除旧的回调
                if hasattr(self, '_vtk_pick_callback'):
                    try:
                        interactor.RemoveObserver(self._vtk_pick_callback)
                    except:
                        pass
                # 添加新的回调
                try:
                    self._vtk_pick_callback = interactor.AddObserver(
                        vtkCommand.LeftButtonPressEvent,
                        self._on_vtk_pick_point
                    )
                except Exception as e:
                    logger.error(f"Error registering callback: {e}")
    
    def handle_vtk_click(self, x: int, y: int):
        """处理VTK窗口的点击事件（目前用于体数据的辅助选点）"""
        if self.pick_mode is None:
            return
        
        # 直接使用单次点击模式
        self._handle_single_click(x, y)
    
    def _handle_single_click(self, x: int, y: int):
        """处理单次点击（直接使用点击位置）"""
        # 使用picker获取3D坐标
        try:
            if vtkCellPicker is None:
                return
            
            picker = vtkCellPicker()
            picker.Pick(x, y, 0, self.vtk_renderer)
            
            if picker.GetCellId() >= 0:
                # 点击在模型上
                world_pos = picker.GetPickPosition()
                x_world, y_world, z_world = world_pos[0], world_pos[1], world_pos[2]
            else:
                # 点击在空白处
                rw = self.vtk_widget.GetRenderWindow()
                if not rw:
                    return
                
                camera = self.vtk_renderer.GetActiveCamera()
                
                # 使用世界坐标转换
                self.vtk_renderer.SetDisplayPoint(x, y, 0.5)
                self.vtk_renderer.DisplayToWorld()
                world_coords = self.vtk_renderer.GetWorldPoint()
                
                if world_coords[3] != 0.0:
                    x_world = world_coords[0] / world_coords[3]
                    y_world = world_coords[1] / world_coords[3]
                    z_world = world_coords[2] / world_coords[3]
                else:
                    # 备用方案：使用相机焦点
                    focal = camera.GetFocalPoint()
                    x_world, y_world, z_world = focal[0], focal[1], focal[2]
        except Exception as e:
            logger.error(f"Error picking point: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 处理选中的点
        self._process_picked_point(x_world, y_world, z_world)
    
    def _process_picked_point(self, x_world: float, y_world: float, z_world: float):
        """处理选中的点，转换坐标并添加标记"""
        if self.pick_mode is None:
            logger.warning("pick_mode 为 None，无法处理选中的点")
            return
        
        logger.info(f"🔵 _process_picked_point 被调用: pick_mode={self.pick_mode}, coords=({x_world}, {y_world}, {z_world})")
        
        # 如果坐标已经是空间坐标（范围 [0, 100]），直接使用
        # 否则尝试从世界坐标转换
        if 0 <= x_world <= 100 and 0 <= y_world <= 100 and 0 <= z_world <= 100:
            # 已经是空间坐标
            space_coord = (x_world, y_world, z_world)
            logger.info(f"✅ 坐标已经是空间坐标: {space_coord}")
        else:
            # 需要从世界坐标转换
            if get_model_bounds is None or world_to_space is None:
                logger.error("❌ get_model_bounds 或 world_to_space 不可用")
                return
            bounds = get_model_bounds(self.vtk_renderer)
            if bounds is None:
                logger.error("❌ 无法获取模型边界")
                return
            space_coord = world_to_space(bounds, (x_world, y_world, z_world))
            logger.info(f"✅ 从世界坐标转换: ({x_world}, {y_world}, {z_world}) -> {space_coord}")
        
        space_x, space_y, space_z = space_coord
        current_pick_mode = self.pick_mode  # 保存当前模式，因为后面会设置为 None
        
        # 根据模式保存点
        if current_pick_mode == 'start':
            self.path_controller.set_start(space_coord)
            self.viz_manager.add_point_marker(space_coord, 'start')
            self._update_path_list_display()
            logger.info(f"✅ Start 点已设置: {space_coord}")
        elif current_pick_mode == 'waypoint':
            self.path_controller.add_waypoint(space_coord)
            self.viz_manager.add_point_marker(space_coord, 'waypoint')
            self._update_path_list_display()
            logger.info(f"✅ Waypoint 已添加: {space_coord}")
        elif current_pick_mode == 'end':
            self.path_controller.set_end(space_coord)
            self.viz_manager.add_point_marker(space_coord, 'end')
            self._update_path_list_display()
            logger.info(f"✅ End 点已设置: {space_coord}")
        
        # 更新显示（必须在添加标记后调用）
        self.viz_manager.update_vtk_display()
        logger.info("✅ VTK 显示已更新")
        
        # 清除 pick_mode（在更新显示后）
        self.pick_mode = None
        if self.vtk_status:
            mode_name = {'start': 'Start', 'waypoint': 'Waypoint', 'end': 'End'}.get(current_pick_mode, 'Point')
            self.vtk_status.setText(f"{mode_name} point selected: ({space_x:.2f}, {space_y:.2f}, {space_z:.2f})")
    
    def _on_vtk_pick_point(self, obj, event):
        """处理VTK 3D窗口的鼠标点击事件（VTK回调版本）"""
        if self.pick_mode is None:
            return
        
        interactor = obj
        if not hasattr(interactor, "GetEventPosition"):
            return
        
        # 获取点击位置
        pos = interactor.GetEventPosition()
        self.handle_vtk_click(pos[0], pos[1])
    
    def handle_generate_path(self, *args):
        """处理异步生成路径事件"""
        if not self.path_controller.can_generate_path():
            QMessageBox.warning(
                self.parent_widget,
                "Cannot Generate Path",
                "Please set start point and end point first."
            )
            return
        
        # 1. 准备障碍物数据 (主线程执行，因为涉及 VTK Actor 遍历)
        data_manager = getattr(self.parent_widget, 'data_manager', None)
        obstacle_set = self.path_service.prepare_obstacles(data_manager, self.vtk_renderer)
        
        if not obstacle_set:
            QMessageBox.warning(
                self.parent_widget,
                "No Obstacle Data",
                "Cannot generate path without obstacle data.\n\n"
                "Please load volume data or STL model first."
            )
            return
        
        # UI 状态
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(True)
            self.parent_widget.recon_progress.setValue(0)
        self.vtk_status.setText("Planning path in background...")
        
        # 清空旧的回放数据
        self._rrt_snapshots.clear()
        self._set_replay_button_enabled(False)
        
        # 2. 启动异步路径规划（含 RRT 树实时可视化）
        run_in_thread(
            self,
            self.path_service.plan_path,
            on_finished=self._on_path_planning_finished,
            on_error=self._on_path_planning_error,
            on_progress=self._on_path_planning_progress,
            on_tree_update=self._on_rrt_tree_update,
            smooth=True
        )

    def _on_path_planning_progress(self, p):
        """进度回调"""
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setValue(p)

    def _on_rrt_tree_update(self, data: dict):
        """
        RRT 探索树实时可视化回调（在主线程执行）。
        
        data 包含:
            - edges: [(parent_pt, child_pt), ...] 规划空间坐标
            - goal: 目标点
            - iteration / total: 当前迭代 / 总迭代
            - found_path (可选): 已找到的路径点列表
            - segment / total_segments (可选): 段索引
        """
        try:
            edges = data.get('edges', [])
            found_path = data.get('found_path', None)
            iteration = data.get('iteration', 0)
            total = data.get('total', 1)
            segment = data.get('segment', 0)
            total_segments = data.get('total_segments', 1)
            
            # 将规划空间坐标转换为 UI 空间坐标 [0,100]
            converted_edges = []
            for parent_pt, child_pt in edges:
                ui_p = self.path_controller._physical_to_space(parent_pt)
                ui_c = self.path_controller._physical_to_space(child_pt)
                converted_edges.append((ui_p, ui_c))
            
            converted_path = None
            if found_path:
                converted_path = [
                    self.path_controller._physical_to_space(pt) for pt in found_path
                ]
            
            # 保存快照用于回放（已转换为 UI 坐标）
            self._rrt_snapshots.append({
                'edges': converted_edges,
                'found_path': converted_path,
                'iteration': iteration,
                'total': total,
                'segment': segment,
                'total_segments': total_segments,
            })
            
            # 更新可视化
            self.viz_manager.update_rrt_tree_viz(converted_edges, converted_path)
            
            # 更新状态文字
            if self.vtk_status:
                seg_info = f" (Segment {segment + 1}/{total_segments})" if total_segments > 1 else ""
                if found_path:
                    self.vtk_status.setText(f"RRT: Path found! {len(edges)} nodes{seg_info}")
                else:
                    self.vtk_status.setText(
                        f"RRT: Exploring... iter {iteration}/{total}, {len(edges)} nodes{seg_info}"
                    )
        except Exception as e:
            logger.warning(f"RRT tree visualization error: {e}")

    def _on_path_planning_finished(self, path_points):
        """规划完成回调"""
        # 隐藏进度条
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        
        # 清除 RRT 探索树可视化（保留最终路径）
        self.viz_manager.clear_rrt_tree_viz()
        
        # 启用回放按钮（如果有快照数据）
        if self._rrt_snapshots:
            self._set_replay_button_enabled(True)
        
        # 启用仿真按钮和安全区域复选框
        self._set_simulate_button_enabled(True)
        self._set_safety_zone_enabled(True)
            
        if path_points:
            self.viz_manager.visualize_path(path_points)
            self._update_path_list_display()
            if self.on_path_generated:
                self.on_path_generated(path_points)
            
            self.vtk_status.setText(f"Path generated: {len(path_points)} points")
            # 评估并显示报告
            self.handle_evaluate_path(show_dialog=True)
        else:
            self._on_path_planning_error("Path generated is empty.")

    def _on_path_planning_error(self, error_msg):
        """规划错误回调"""
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        
        # 清除 RRT 探索树可视化
        self.viz_manager.clear_rrt_tree_viz()
            
        logger.error(f"Path planning error: {error_msg}")
        
        # 检查是否是 RRT 失败（通常抛出 RuntimeError）
        if "RRT" in error_msg or "blocked" in error_msg.lower():
            reply = QMessageBox.question(
                self.parent_widget,
                "Path Generation Failed",
                f"{error_msg}\n\nWould you like to create a simple straight-line path instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                simple_path = self.path_service.generate_simple_path()
                self._on_path_planning_finished(simple_path)
                return

        QMessageBox.warning(self.parent_widget, "Error", f"Path generation error: {error_msg}")
        self.vtk_status.setText("Path planning failed")
    
    def handle_evaluate_path(self, *args, show_dialog: bool = True):
        """评估当前路径并显示报告"""
        path_points = self.path_controller.get_planner_path_points()
        if not path_points or len(path_points) < 2:
            if show_dialog:
                QMessageBox.warning(self.parent_widget, "Evaluation", "No path to evaluate.")
            return None
            
        collision_checker = getattr(self.path_controller, '_collision_checker', None)
        if not collision_checker:
            # 尝试重新准备障碍物
            data_manager = getattr(self.parent_widget, 'data_manager', None)
            self.path_service.prepare_obstacles(data_manager, self.vtk_renderer)
            collision_checker = getattr(self.path_controller, '_collision_checker', None)
            
        if not collision_checker:
            if show_dialog:
                QMessageBox.warning(self.parent_widget, "Evaluation", "Collision checker not available.")
            return None
            
        report = self.path_evaluator.evaluate(path_points, collision_checker)
        
        if show_dialog:
            msg = (
                f"### Path Quality Report ###\n\n"
                f"Overall Score: {report['total_score']:.1f} / 100\n"
                f"----------------------------\n"
                f"1. Length: {report['length']:.2f} units\n"
                f"2. Safety Score: {report['safety']['safety_score']:.1f} / 100\n"
                f"   - Min Distance: {report['safety']['min_distance']:.2f}\n"
                f"   - Avg Distance: {report['safety']['avg_distance']:.2f}\n"
                f"3. Smoothness Score: {report['smoothness']['smoothness_score']:.1f} / 100\n"
                f"   - Avg Curvature: {report['smoothness']['avg_curvature']:.3f} rad\n"
            )
            QMessageBox.information(self.parent_widget, "Path Evaluation", msg)
            
        return report

    def handle_save_path(self, *args):
        """处理保存路径事件"""
        if not self.path_controller.path_points:
            QMessageBox.warning(self.parent_widget, "No Path", "No path to save.")
            return
        
        from PyQt5.QtWidgets import QFileDialog
        from pathlib import Path
        
        default_path = Path(__file__).resolve().parent.parent.parent.parent / "path_data"
        default_path.mkdir(exist_ok=True)
        
        # 使用患者ID生成默认文件名
        default_filename = "path.txt"
        if hasattr(self.parent_widget, 'patient_context') and self.parent_widget.patient_context:
            p_id = self.parent_widget.patient_context.get('patient_id', 'unknown')
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_filename = f"path_{p_id}_{timestamp}.txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self.parent_widget,
            "Save Path",
            str(default_path / default_filename),
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for point in self.path_controller.path_points:
                        f.write(f"{point[0]},{point[1]},{point[2]}\n")
                QMessageBox.information(self.parent_widget, "Success", f"Path saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self.parent_widget, "Error", f"Failed to save path: {str(e)}")
    
    def handle_reset_path(self, *args):
        """处理重置路径事件"""
        # 停止正在进行的回放和仿真
        self._stop_replay()
        if self._is_simulating:
            self._stop_simulation()
        
        # 清除路径控制器中的数据
        self.path_controller.clear_path()
        
        # 清除可视化（含 RRT 探索树 + 仿真 + 安全区域）
        self.viz_manager.clear_rrt_tree_viz()
        self.viz_manager.clear_sim_viz()
        self.viz_manager.clear_safety_tube()
        self.viz_manager.clear_all_path_viz()
        
        # 清空回放数据
        self._rrt_snapshots.clear()
        self._set_replay_button_enabled(False)
        self._set_simulate_button_enabled(False)
        self._set_safety_zone_enabled(False)
        self._safety_zone_visible = False
        
        # 清除路径列表
        self.path_list.clear()
        
        # 重置选点模式
        self.pick_mode = None
        
        # 更新显示
        self.viz_manager.update_vtk_display()
        
        # 调用回调
        if self.on_path_reset:
            self.on_path_reset()
        
        self.vtk_status.setText("Path reset")
    
    def handle_undo(self, *args):
        """处理撤销操作"""
        if self.path_controller.undo():
            self._refresh_after_history_change()
            if self.vtk_status:
                self.vtk_status.setText("Undo successful")
            logger.info("Undo handled in UI")
        else:
            if self.vtk_status:
                self.vtk_status.setText("Nothing to undo")

    def handle_redo(self, *args):
        """处理重做操作"""
        if self.path_controller.redo():
            self._refresh_after_history_change()
            if self.vtk_status:
                self.vtk_status.setText("Redo successful")
            logger.info("Redo handled in UI")
        else:
            if self.vtk_status:
                self.vtk_status.setText("Nothing to redo")

    def _refresh_after_history_change(self):
        """在撤销/重做或路径大幅更新后刷新 UI 和可视化"""
        # 1. 清除旧的可视化
        self.viz_manager.clear_all_path_viz()
        
        # 2. 重新添加点标记
        # 如果已经生成了路径，只显示关键点（起点/终点/中间点）
        if self.path_controller.path_points:
            if self.path_controller.start_point:
                self.viz_manager.add_point_marker(self.path_controller.start_point, 'start')
            for wp in self.path_controller.waypoints:
                self.viz_manager.add_point_marker(wp, 'waypoint')
            if self.path_controller.end_point:
                self.viz_manager.add_point_marker(self.path_controller.end_point, 'end')
            # 仍然绘制完整路径线
            self.viz_manager.visualize_path(self.path_controller.path_points)
        else:
            # 如果没有生成路径，只显示用户手动设置的几个关键控制点
            if self.path_controller.start_point:
                self.viz_manager.add_point_marker(self.path_controller.start_point, 'start')
            for wp in self.path_controller.waypoints:
                self.viz_manager.add_point_marker(wp, 'waypoint')
            if self.path_controller.end_point:
                self.viz_manager.add_point_marker(self.path_controller.end_point, 'end')
            
        # 3. 更新路径列表显示
        self._update_path_list_display()
        
        # 4. 更新 VTK 显示
        self.viz_manager.update_vtk_display()
    
    def _update_path_list_display(self):
        """更新路径列表显示"""
        self.path_list.clear()
        
        if self.path_controller.start_point:
            sp = self.path_controller.start_point
            self.path_list.addItem(f"Start: ({sp[0]:.2f}, {sp[1]:.2f}, {sp[2]:.2f})")
        
        for i, wp in enumerate(self.path_controller.waypoints):
            self.path_list.addItem(f"Waypoint {i+1}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})")
        
        if self.path_controller.end_point:
            ep = self.path_controller.end_point
            self.path_list.addItem(f"End: ({ep[0]:.2f}, {ep[1]:.2f}, {ep[2]:.2f})")
        
        # 如果有生成的路径，添加路径点
        if self.path_controller.path_points:
            # 获取简要评估信息
            report = self.handle_evaluate_path(show_dialog=False)
            if report:
                self.path_list.addItem(f"--- Path Score: {report['total_score']:.1f} ---")
            else:
                self.path_list.addItem("--- Generated Path ---")

            for i, pt in enumerate(self.path_controller.path_points):
                self.path_list.addItem(f"  [{i}]: ({pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f})")
    
    def handle_path_list_double_click(self, item):
        """处理路径列表双击事件 - 编辑路径点"""
        logger.info(f"🔵🔵🔵 路径列表双击事件触发！item={item}")
        
        if item is None:
            logger.warning("❌ item is None")
            return
        
        text = item.text()
        logger.info(f"双击项文本: '{text}'")
        
        stripped = text.strip()

        # 1) 生成的路径点（格式：  [n]: (x, y, z)）
        if stripped.startswith("[") and "]:" in stripped:
            # 提取索引
            try:
                idx = int(stripped.split("]")[0].split("[")[1])
                logger.info(f"解析出索引: {idx}")
                self._open_path_point_edit_dialog(idx)
            except Exception as e:
                logger.error(f"解析路径点索引失败: {e}")
            return

        # 2) Start/End 行双击编辑
        if stripped.startswith("Start:"):
            self._open_path_point_edit_dialog(0)
            return
        if stripped.startswith("End:"):
            if self.path_controller.path_points:
                self._open_path_point_edit_dialog(len(self.path_controller.path_points) - 1)
            return

        logger.info(f"不是路径点项，跳过")
    
    def _open_path_point_edit_dialog(self, point_index: int):
        """打开路径点编辑对话框"""
        logger.info(f"_open_path_point_edit_dialog 被调用, point_index={point_index}")
        
        path_points = self.path_controller.path_points
        logger.info(f"path_points 数量: {len(path_points) if path_points else 0}")
        
        if not path_points or point_index < 0 or point_index >= len(path_points):
            logger.warning(f"无效的索引或没有路径点")
            return
        
        # 导入编辑对话框
        try:
            from surgical_robot_app.gui.dialogs.path_point_edit_dialog import PathPointEditDialog
        except ImportError:
            try:
                from gui.dialogs.path_point_edit_dialog import PathPointEditDialog
            except ImportError:
                logger.error("PathPointEditDialog not available")
                QMessageBox.warning(
                    self.parent_widget,
                    "Error",
                    "Path point edit dialog is not available."
                )
                return
        
        current_point = path_points[point_index]
        
        # 获取模型PolyData和碰撞检测器
        model_polydata = None
        collision_checker = self.path_controller._collision_checker if hasattr(self.path_controller, '_collision_checker') else None
        
        # 获取模型边界（用于坐标转换）
        model_bounds = None
        if self.vtk_renderer:
            try:
                from surgical_robot_app.vtk_utils.coords import get_model_bounds
                model_bounds = get_model_bounds(self.vtk_renderer)
                model_polydata = self._collect_all_model_polydata()
            except Exception as e:
                logger.warning(f"获取模型数据失败: {e}")
        
        # 创建编辑对话框
        dialog = PathPointEditDialog(
            point_index=point_index,
            current_coords=current_point,
            all_path_points=list(path_points),
            model_polydata=model_polydata,
            model_bounds=model_bounds,  # 传递边界信息
            collision_checker=collision_checker,
            parent=self.parent_widget
        )
        
        # 连接信号
        dialog.point_updated.connect(self._on_path_point_updated)
        dialog.point_deleted.connect(self._on_path_point_deleted)
        dialog.preview_requested.connect(lambda x, y, z: self._preview_path_point_edit(point_index, x, y, z))
        
        # 使用非模态显示以支持实时预览
        dialog.setWindowModality(Qt.NonModal)
        self._edit_dialog = dialog
        self._edit_point_index = point_index
        dialog.show()
        dialog.raise_()
    
    def _on_path_point_updated(self, index: int, x: float, y: float, z: float):
        """
        处理路径点更新 — 对受影响的前后两段进行 RRT 局部重规划。
        如果重规划失败则退化为直线连接。
        """
        path_points = list(self.path_controller.path_points)
        if not (0 <= index < len(path_points)):
            return
            
        new_pos = (x, y, z)
        
        # 尝试对受影响的段进行局部重规划
        replan_success = False
        if self.path_controller.rrt_planner is not None:
            new_full_path = []
            new_full_path.extend(path_points[:max(0, index)])
            
            seg1_ok = False
            seg2_ok = False
            
            # [前一点 -> 新点]
            if index > 0:
                seg1 = self.path_service.plan_local_segment(path_points[index-1], new_pos)
                if seg1 and len(seg1) >= 2:
                    new_full_path.extend(seg1[1:])
                    seg1_ok = True
                else:
                    new_full_path.append(new_pos)
            else:
                new_full_path.append(new_pos)
                seg1_ok = True
                
            # [新点 -> 后一点]
            if index < len(path_points) - 1:
                seg2 = self.path_service.plan_local_segment(new_pos, path_points[index+1])
                if seg2 and len(seg2) >= 2:
                    new_full_path.extend(seg2[1:])
                    seg2_ok = True
                else:
                    new_full_path.append(path_points[index+1])
            else:
                seg2_ok = True
            
            # 保留后续路径
            if index < len(path_points) - 2:
                new_full_path.extend(path_points[index+2:])
                
            path_points = new_full_path
            self.path_controller.path_points = path_points
            replan_success = seg1_ok and seg2_ok
        else:
            path_points[index] = new_pos
            self.path_controller.path_points = path_points
        
        # 同步起点/终点标记位置
        updated_points = self.path_controller.path_points
        if updated_points:
            self.path_controller.start_point = updated_points[0]
            self.path_controller.end_point = updated_points[-1]
            
        # 清除旧的回放数据（路径已变更，旧快照不再有效）
        self._rrt_snapshots.clear()
        self._set_replay_button_enabled(False)
        
        # 刷新显示
        self.viz_manager.clear_preview_marker()
        self.viz_manager.clear_rrt_tree_viz()
        self._refresh_after_history_change()
        
        if self.vtk_status:
            if replan_success:
                self.vtk_status.setText(f"Point #{index+1} updated (path replanned)")
            else:
                self.vtk_status.setText(f"Point #{index+1} updated (replan failed, straight line)")
    
    def _on_path_point_deleted(self, index: int):
        """处理路径点删除"""
        path_points = list(self.path_controller.path_points)
        if 0 < index < len(path_points) - 1:  # 不能删除起点和终点
            del path_points[index]
            self.path_controller.path_points = path_points
            
            # 同步起点/终点标记
            if path_points:
                self.path_controller.start_point = path_points[0]
                self.path_controller.end_point = path_points[-1]
            
            # 清除预览点
            self.viz_manager.clear_preview_marker()
            
            # 清除旧的回放数据（路径已变更）
            self._rrt_snapshots.clear()
            self._set_replay_button_enabled(False)
            self.viz_manager.clear_rrt_tree_viz()
            
            # 重新可视化路径（使用统一逻辑，已包含列表更新和 VTK 刷新）
            self._refresh_after_history_change()
            
            if self.vtk_status:
                self.vtk_status.setText(f"Path point #{index + 1} deleted")
            
            logger.info(f"路径点 {index} 已删除")
    
    def _preview_path_point_edit(self, index: int, x: float, y: float, z: float):
        """实时预览路径点编辑"""
        # 更新预览标记
        self.viz_manager.update_preview_marker(x, y, z, 'waypoint')

    # -------------------- RRT 探索过程回放 --------------------

    def _set_replay_button_enabled(self, enabled: bool):
        """设置 Replay 按钮可用状态"""
        btn = getattr(self.parent_widget, 'btn_replay_path', None)
        if btn:
            btn.setEnabled(enabled)

    def _set_replay_button_text(self, text: str):
        """设置 Replay 按钮文字"""
        btn = getattr(self.parent_widget, 'btn_replay_path', None)
        if btn:
            btn.setText(text)

    def handle_replay_rrt(self, *args):
        """处理 RRT 回放按钮点击 — 切换回放/停止"""
        if self._is_replaying:
            self._stop_replay()
        else:
            self._start_replay()

    def _start_replay(self):
        """开始 RRT 探索过程回放"""
        if not self._rrt_snapshots:
            if self.vtk_status:
                self.vtk_status.setText("No RRT data to replay")
            return
        
        # 暂时隐藏最终路径线
        self.viz_manager.clear_path_line()
        
        # 初始化回放状态
        self._replay_index = 0
        self._is_replaying = True
        self._set_replay_button_text("Stop")
        
        # 创建定时器
        if self._replay_timer is None:
            self._replay_timer = QTimer(self)
            self._replay_timer.timeout.connect(self._replay_tick)
        
        self._replay_timer.start(self._replay_speed_ms)
        logger.info(f"RRT 回放开始: {len(self._rrt_snapshots)} 帧, {self._replay_speed_ms}ms/帧")

    def _stop_replay(self):
        """停止 RRT 回放"""
        if self._replay_timer:
            self._replay_timer.stop()
        
        self._is_replaying = False
        self._set_replay_button_text("Replay")
        
        # 清除回放的探索树，恢复最终路径
        self.viz_manager.clear_rrt_tree_viz()
        if self.path_controller.path_points:
            self.viz_manager.visualize_path(self.path_controller.path_points)
        self.viz_manager.update_vtk_display()
        
        if self.vtk_status:
            self.vtk_status.setText("Replay stopped")
        logger.info("RRT 回放停止")

    def _replay_tick(self):
        """回放定时器的每一帧"""
        if self._replay_index >= len(self._rrt_snapshots):
            # 回放结束：在最后一帧停留片刻后恢复
            self._replay_timer.stop()
            # 延迟 1 秒后自动清理
            QTimer.singleShot(1500, self._stop_replay)
            if self.vtk_status:
                self.vtk_status.setText("Replay finished")
            return
        
        snapshot = self._rrt_snapshots[self._replay_index]
        edges = snapshot.get('edges', [])
        found_path = snapshot.get('found_path', None)
        iteration = snapshot.get('iteration', 0)
        total = snapshot.get('total', 1)
        segment = snapshot.get('segment', 0)
        total_segments = snapshot.get('total_segments', 1)
        
        # 渲染当前帧
        self.viz_manager.update_rrt_tree_viz(edges, found_path)
        
        # 更新状态文字
        if self.vtk_status:
            frame_info = f"[{self._replay_index + 1}/{len(self._rrt_snapshots)}]"
            seg_info = f" Seg {segment + 1}/{total_segments}" if total_segments > 1 else ""
            if found_path:
                self.vtk_status.setText(f"Replay {frame_info}: Path found! {len(edges)} nodes{seg_info}")
            else:
                self.vtk_status.setText(f"Replay {frame_info}: iter {iteration}/{total}, {len(edges)} nodes{seg_info}")
        
        self._replay_index += 1

    # -------------------- Path Simulation (Instrument Animation) --------------------

    def _set_simulate_button_enabled(self, enabled: bool):
        btn = getattr(self.parent_widget, 'btn_simulate_path', None)
        if btn:
            btn.setEnabled(enabled)

    def _set_simulate_button_text(self, text: str):
        btn = getattr(self.parent_widget, 'btn_simulate_path', None)
        if btn:
            btn.setText(text)

    def handle_simulate_path(self, *args):
        """Toggle path simulation start / stop."""
        if self._is_simulating:
            self._stop_simulation()
        else:
            self._start_simulation()

    def _build_sim_world_points(self) -> List[Tuple[float, float, float]]:
        """
        Build an interpolated list of world-coordinate points for smooth animation.
        Takes the current path_points (space coords [0,100]), converts each segment
        into `_sim_steps_per_segment` intermediate world-coord points.
        """
        from surgical_robot_app.vtk_utils.coords import get_model_bounds, space_to_world
        
        path_points = self.path_controller.path_points
        if not path_points or len(path_points) < 2:
            return []
        
        bounds = get_model_bounds(self.vtk_renderer)
        if not bounds or len(bounds) < 6:
            bounds = (-50.0, 50.0, -50.0, 50.0, -50.0, 50.0)
        
        world_pts: List[Tuple[float, float, float]] = []
        steps = self._sim_steps_per_segment
        
        for i in range(len(path_points) - 1):
            p0 = path_points[i]
            p1 = path_points[i + 1]
            for s in range(steps):
                t = s / steps
                sp = (
                    p0[0] + (p1[0] - p0[0]) * t,
                    p0[1] + (p1[1] - p0[1]) * t,
                    p0[2] + (p1[2] - p0[2]) * t,
                )
                world_pts.append(space_to_world(bounds, sp))
        
        # Add the final point
        world_pts.append(space_to_world(bounds, path_points[-1]))
        return world_pts

    def _start_simulation(self):
        """Start instrument simulation along the path."""
        self._sim_world_points = self._build_sim_world_points()
        if not self._sim_world_points:
            if self.vtk_status:
                self.vtk_status.setText("No path to simulate")
            return
        
        # Dim the original path line to make trail more visible
        self.viz_manager.clear_path_line()
        
        self._sim_index = 0
        self._is_simulating = True
        self._set_simulate_button_text("Stop")
        self._set_replay_button_enabled(False)  # disable replay during sim
        
        if self._sim_timer is None:
            self._sim_timer = QTimer(self)
            self._sim_timer.timeout.connect(self._sim_tick)
        
        self._sim_timer.start(self._sim_speed_ms)
        logger.info(f"Path simulation started: {len(self._sim_world_points)} frames, {self._sim_speed_ms}ms/frame")

    def _stop_simulation(self):
        """Stop the simulation and restore the path."""
        if self._sim_timer:
            self._sim_timer.stop()
        
        self._is_simulating = False
        self._set_simulate_button_text("Simulate")
        
        # Clean up simulation actors, restore path line
        self.viz_manager.clear_sim_viz()
        if self.path_controller.path_points:
            self.viz_manager.visualize_path(self.path_controller.path_points)
        self.viz_manager.update_vtk_display()
        
        # Re-enable replay if snapshots exist
        if self._rrt_snapshots:
            self._set_replay_button_enabled(True)
        
        if self.vtk_status:
            self.vtk_status.setText("Simulation stopped")
        logger.info("Path simulation stopped")

    def _sim_tick(self):
        """One animation frame of the path simulation."""
        if self._sim_index >= len(self._sim_world_points):
            # Reached the end – pause briefly then auto-stop
            self._sim_timer.stop()
            if self.vtk_status:
                self.vtk_status.setText("Simulation complete")
            QTimer.singleShot(2000, self._stop_simulation)
            return
        
        pos = self._sim_world_points[self._sim_index]
        
        # Update instrument sphere
        self.viz_manager.update_sim_instrument(pos)
        
        # Update trail (all points from start to current)
        if self._sim_index > 0:
            self.viz_manager.update_sim_trail(self._sim_world_points[:self._sim_index + 1])
        
        # Update status
        if self.vtk_status:
            progress = int((self._sim_index + 1) / len(self._sim_world_points) * 100)
            self.vtk_status.setText(
                f"Simulating... {progress}% [{self._sim_index + 1}/{len(self._sim_world_points)}]"
            )
        
        # Render
        self.viz_manager.update_vtk_display()
        
        self._sim_index += 1

    # -------------------- Safety Zone Toggle --------------------

    def _set_safety_zone_enabled(self, enabled: bool):
        chk = getattr(self.parent_widget, 'chk_safety_zone', None)
        if chk:
            chk.setEnabled(enabled)
            if not enabled:
                chk.setChecked(False)

    def handle_toggle_safety_zone(self, checked: bool):
        """Handle the Safety Zone checkbox toggled on/off."""
        if checked:
            path_points = self.path_controller.path_points
            if path_points and len(path_points) >= 2:
                # Use the configured safety_radius converted to world-scale
                safety_radius_world = self._get_safety_radius_world()
                self.viz_manager.show_safety_tube(path_points, safety_radius_world)
                self._safety_zone_visible = True
                if self.vtk_status:
                    self.vtk_status.setText("Safety zone displayed")
            else:
                if self.vtk_status:
                    self.vtk_status.setText("No path for safety zone")
                # Uncheck since nothing to show
                chk = getattr(self.parent_widget, 'chk_safety_zone', None)
                if chk:
                    chk.setChecked(False)
        else:
            self.viz_manager.clear_safety_tube()
            self.viz_manager.update_vtk_display()
            self._safety_zone_visible = False

    def _get_safety_radius_world(self) -> float:
        """
        Convert the path controller's safety_radius (in normalised [0,100] space)
        to world coordinates using the model bounds.
        """
        from surgical_robot_app.vtk_utils.coords import get_model_bounds
        
        # Get the configured safety radius (default 5.0 in normalised space)
        safety_r = getattr(self.path_controller, 'rrt_safety_radius', 5.0)
        
        bounds = get_model_bounds(self.vtk_renderer)
        if bounds and len(bounds) >= 6:
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            avg_extent = (dx + dy + dz) / 3.0
            # safety_r is in [0,100] space, convert proportionally
            return avg_extent * safety_r / 100.0
        else:
            return safety_r  # fallback: use raw value

