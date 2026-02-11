from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QSlider,
    QGroupBox,
    QRadioButton,
    QFileDialog,
    QLabel,
    QMessageBox,
    QAbstractItemView,
    QSizePolicy,
    QGridLayout,
    QScrollArea,
    QProgressBar,
    QApplication,
    QFrame,
    QShortcut,
)
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent, QPainter, QPen, QBrush, QKeySequence
from PyQt5.QtCore import Qt, QEvent, QTimer, pyqtSignal
import logging  # 用于错误处理装饰器的 log_level 参数

# 使用统一的导入工具
try:
    from surgical_robot_app.utils.import_helper import safe_import, safe_import_with_aliases
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.utils.error_handler import handle_errors
    from surgical_robot_app.gui.managers.view3d_manager import View3DManager
    from surgical_robot_app.gui.managers.event_dispatcher import EventDispatcher
    from surgical_robot_app.gui.managers.point_selection_manager import PointSelectionManager
    from surgical_robot_app.config.settings import get_config
except ImportError:
    from utils.import_helper import safe_import, safe_import_with_aliases
    from utils.logger import get_logger
    try:
        from utils.error_handler import handle_errors
    except ImportError:
        # 如果错误处理模块不可用，创建一个空的装饰器
        def handle_errors(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
    try:
        from gui.managers.view3d_manager import View3DManager
        from gui.managers.event_dispatcher import EventDispatcher
        from gui.managers.point_selection_manager import PointSelectionManager
        from config.settings import get_config
    except ImportError:
        View3DManager = None
        EventDispatcher = None
        PointSelectionManager = None
        get_config = None

# 初始化日志系统
logger = get_logger("surgical_robot_app.gui.main_window")

# 导入 SequenceReader
SequenceReader = safe_import(
    ['surgical_robot_app.data_io.sequence_reader', 'data_io.sequence_reader'],
    item_names='SequenceReader',
    default=None
)

# 导入 VTK Widget（特殊处理，有多个尝试路径和日志信息）
try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
    logger.info("VTK PyQt5 binding ready")
except Exception:
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
        logger.info("VTK PyQt5 binding ready (vtkmodules)")
    except Exception:
        QVTKWidget = None
        logger.warning("VTK Qt binding not available")
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
# 导入vtkCellArray - 在VTK中，它应该在vtkCommonDataModel中
def _get_vtk_cell_array_fallback():
    """vtkCellArray 的回退函数"""
    try:
        import vtkmodules.all as vtk
        return vtk.vtkCellArray
    except:
        raise ImportError("Cannot import vtkCellArray, please check VTK installation")

vtkCellArray = safe_import(
    ['vtkmodules.vtkCommonDataModel', 'vtkmodules.vtkCommonCore'],
    item_names='vtkCellArray',
    default=None,
    fallback_func=_get_vtk_cell_array_fallback
)
from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkIOGeometry import vtkSTLWriter, vtkSTLReader
from vtkmodules.vtkFiltersSources import vtkSphereSource
# 导入路径规划相关
AStarPlanner, space_to_grid, grid_to_space, create_obstacle_grid = safe_import(
    ['surgical_robot_app.path_planning.a_star_planner', 'path_planning.a_star_planner'],
    item_names=['AStarPlanner', 'space_to_grid', 'grid_to_space', 'create_obstacle_grid'],
    default=(None, None, None, None)
)
from pathlib import Path
import time
import os
import importlib
import numpy as np
from typing import List, Tuple
# 导入 cv2（可选）
cv2 = safe_import(['cv2'], default=None)

"""
为保持 main_window 更专注于 UI 逻辑，这里通过 try/except 引入若干工具模块：
- segmentation.hsv_threshold: HSV 阈值分割
- vtk_utils.coords / vtk_utils.markers: VTK 坐标与可视化工具
"""

# HSV 阈值分割工具（拆分到 segmentation 模块，便于复用与测试）
def _get_hsv_threshold_fallback():
    """HSV 阈值分割的回退函数"""
    DEFAULT_THRESHOLDS = {
            "h1_low": 0,
            "h1_high": 10,
            "h2_low": 160,
            "h2_high": 180,
            "s_low": 50,
            "s_high": 255,
            "v_low": 50,
            "v_high": 255,
        }
    
    def hsv_compute_threshold_mask(slice_img, thresholds):
        return None
    
    def apply_threshold_all(volume, thresholds):
        return [], None
    
    return DEFAULT_THRESHOLDS, hsv_compute_threshold_mask, apply_threshold_all

DEFAULT_THRESHOLDS, hsv_compute_threshold_mask, apply_threshold_all = safe_import(
    ['surgical_robot_app.segmentation.hsv_threshold', 'segmentation.hsv_threshold'],
    item_names=['DEFAULT_THRESHOLDS', 'compute_threshold_mask', 'apply_threshold_all'],
    default=(None, None, None),
    fallback_func=_get_hsv_threshold_fallback
)
# 重命名 compute_threshold_mask 为 hsv_compute_threshold_mask
hsv_compute_threshold_mask = hsv_compute_threshold_mask or (lambda slice_img, thresholds: None)

# VTK 坐标与标记/路径工具（用于坐标转换和创建球体/折线）
def _get_vtk_utils_fallback():
    """VTK 工具的回退函数"""
    def get_model_bounds(renderer):
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

    def world_to_space(bounds, world_coord, default=50.0):
        x_world, y_world, z_world = world_coord
        if bounds is None:
            return (
                max(0.0, min(100.0, x_world + 50.0)),
                max(0.0, min(100.0, y_world + 50.0)),
                max(0.0, min(100.0, z_world + 50.0)),
            )
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
        return float(space_x), float(space_y), float(space_z)

    def space_to_world(bounds, space_coord):
        x, y, z = space_coord
        if bounds is None:
            return x - 50.0, y - 50.0, z - 50.0
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        world_x = x_min + (x / 100.0) * (x_max - x_min)
        world_y = y_min + (y / 100.0) * (y_max - y_min)
        world_z = z_min + (z / 100.0) * (z_max - z_min)
        return float(world_x), float(world_y), float(world_z)

    def estimate_radius(bounds, ratio=0.02, default=2.0):
        if bounds is None:
            return float(default)
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        size = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if size <= 0:
            return float(default)
        return float(size * ratio)

    def create_sphere_marker(renderer, space_coord, color, radius_ratio=0.02):
        return None

    def create_polyline_actor_from_space_points(renderer, path_points, color=(1.0, 0.0, 0.0), line_width=3.0):
        return None
    
    return {
        'get_model_bounds': get_model_bounds,
        'world_to_space': world_to_space,
        'space_to_world': space_to_world,
        'estimate_radius': estimate_radius,
        'create_sphere_marker': create_sphere_marker,
        'create_polyline_actor_from_space_points': create_polyline_actor_from_space_points,
    }

# 导入 VTK 工具（使用回退函数）
vtk_utils_result = safe_import_with_aliases(
    ['surgical_robot_app.vtk_utils.coords', 'vtk_utils.coords'],
    imports={
        'get_model_bounds': 'get_model_bounds',
        'world_to_space': 'world_to_space',
        'space_to_world': 'space_to_world',
        'estimate_radius': 'estimate_radius',
    },
    defaults={
        'get_model_bounds': None,
        'world_to_space': None,
        'space_to_world': None,
        'estimate_radius': None,
    },
    fallback_funcs={
        'get_model_bounds': lambda: _get_vtk_utils_fallback()['get_model_bounds'],
        'world_to_space': lambda: _get_vtk_utils_fallback()['world_to_space'],
        'space_to_world': lambda: _get_vtk_utils_fallback()['space_to_world'],
        'estimate_radius': lambda: _get_vtk_utils_fallback()['estimate_radius']
    },
)

vtk_markers_result = safe_import(
    ['surgical_robot_app.vtk_utils.markers', 'vtk_utils.markers'],
    item_names='create_sphere_marker',
    default=None,
    fallback_func=lambda: _get_vtk_utils_fallback()['create_sphere_marker']
)

vtk_path_result = safe_import(
    ['surgical_robot_app.vtk_utils.path', 'vtk_utils.path'],
    item_names='create_polyline_actor_from_space_points',
    default=None,
    fallback_func=lambda: _get_vtk_utils_fallback()['create_polyline_actor_from_space_points']
)

# 解包导入结果
get_model_bounds = vtk_utils_result.get('get_model_bounds') or _get_vtk_utils_fallback()['get_model_bounds']
world_to_space = vtk_utils_result.get('world_to_space') or _get_vtk_utils_fallback()['world_to_space']
space_to_world = vtk_utils_result.get('space_to_world') or _get_vtk_utils_fallback()['space_to_world']
estimate_radius = vtk_utils_result.get('estimate_radius') or _get_vtk_utils_fallback()['estimate_radius']
create_sphere_marker = vtk_markers_result or _get_vtk_utils_fallback()['create_sphere_marker']
create_polyline_actor_from_space_points = vtk_path_result or _get_vtk_utils_fallback()['create_polyline_actor_from_space_points']

# 3D 切片与 2D/3D 坐标几何工具
def _get_geometry_fallback():
    """几何工具的回退函数"""
    def extract_slice(volume, plane_type, slice_pos_normalized):
        return volume

    def plane_click_to_space_coord(plane_type, slice_pos_normalized, img_x_norm, img_y_norm):
        x = img_x_norm * 100.0
        y = img_y_norm * 100.0
        z = slice_pos_normalized
        return (x, y), (x, y, z)

    def merge_two_planes_to_3d(plane1_type, coords1, plane2_type, coords2):
        return coords1
    
    return extract_slice, plane_click_to_space_coord, merge_two_planes_to_3d

extract_slice, plane_click_to_space_coord, merge_two_planes_to_3d = safe_import(
    ['surgical_robot_app.geometry.slice_geometry', 'geometry.slice_geometry'],
    item_names=['extract_slice', 'plane_click_to_space_coord', 'merge_two_planes_to_3d'],
    default=(None, None, None),
    fallback_func=_get_geometry_fallback
)

# 路径规划控制器
PathController = safe_import(
    ['surgical_robot_app.path_planning.path_controller', 'path_planning.path_controller'],
    item_names='PathController',
    default=None
)

# SAM2 分割控制器
SAM2Controller = safe_import(
    ['surgical_robot_app.segmentation.sam2_controller', 'segmentation.sam2_controller'],
    item_names='SAM2Controller',
    default=None
)

# 手动画笔分割控制器
ManualSegController = safe_import(
    ['surgical_robot_app.segmentation.manual_controller', 'segmentation.manual_controller'],
    item_names='ManualSegController',
    default=None
)

# 3D 视图控制器
View3DController = safe_import(
    ['surgical_robot_app.gui.view3d_controller', 'gui.view3d_controller'],
    item_names='View3DController',
    default=None
)

# 数据管理器
DataManager = safe_import(
    ['surgical_robot_app.data_io.data_manager', 'data_io.data_manager'],
    item_names='DataManager',
    default=None
)

# UI 构建器
logger.info("开始导入UI构建器...")
build_data_import_ui = safe_import(
    ['surgical_robot_app.gui.ui_builders.data_import_ui', 'gui.ui_builders.data_import_ui'],
    item_names='build_data_import_ui',
    default=None
)
build_slice_editor_ui = safe_import(
    ['surgical_robot_app.gui.ui_builders.slice_editor_ui', 'gui.ui_builders.slice_editor_ui'],
    item_names='build_slice_editor_ui',
    default=None
)
build_sam2_ui = safe_import(
    ['surgical_robot_app.gui.ui_builders.sam2_ui', 'gui.ui_builders.sam2_ui'],
    item_names='build_sam2_ui',
    default=None
)
build_view3d_ui = safe_import(
    ['surgical_robot_app.gui.ui_builders.view3d_ui', 'gui.ui_builders.view3d_ui'],
    item_names='build_view3d_ui',
    default=None
)
build_multi_slice_ui = safe_import(
    ['surgical_robot_app.gui.ui_builders.multi_slice_ui', 'gui.ui_builders.multi_slice_ui'],
    item_names='build_multi_slice_ui',
    default=None
)
logger.info(f"导入完成 - build_data_import_ui={build_data_import_ui}")

# GUI 控制器
DataImportController = safe_import(
    ['surgical_robot_app.gui.controllers.data_import_controller', 'gui.controllers.data_import_controller'],
    item_names='DataImportController',
    default=None
)
SliceEditorController = safe_import(
    ['surgical_robot_app.gui.controllers.slice_editor_controller', 'gui.controllers.slice_editor_controller'],
    item_names='SliceEditorController',
    default=None
)
SAM2UIController = safe_import(
    ['surgical_robot_app.gui.controllers.sam2_ui_controller', 'gui.controllers.sam2_ui_controller'],
    item_names='SAM2UIController',
    default=None
)
PathUIController = safe_import(
    ['surgical_robot_app.gui.controllers.path_ui_controller', 'gui.controllers.path_ui_controller'],
    item_names='PathUIController',
    default=None
)
MultiSliceController = safe_import(
    ['surgical_robot_app.gui.controllers.multi_slice_controller', 'gui.controllers.multi_slice_controller'],
    item_names='MultiSliceController',
    default=None
)


def _load_core_reconstruct_3d():
    """
    动态加载 3D 重建核心函数，避免因包名中含数字导致语法错误
    优先尝试包内路径 'surgical_robot_app.3d_recon.recon_core'
    退而求其次尝试相对路径 '3d_recon.recon_core'
    """
    try:
        module = importlib.import_module("surgical_robot_app.3d_recon.recon_core")
        return getattr(module, "reconstruct_3d", None)
    except Exception:
        try:
            module = importlib.import_module("3d_recon.recon_core")
            return getattr(module, "reconstruct_3d", None)
        except Exception:
            return None


class MainView(QWidget):
    back_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.patient_context = None
        self.setWindowTitle("Surgical Robot App")
        self.reader = SequenceReader()
        
        # 初始化数据管理器（统一管理数据状态）
        self.data_manager = DataManager()
        
        # 加载配置
        if get_config is not None:
            self.config = get_config()
        else:
            # 回退：如果配置模块不可用，使用默认值
            from dataclasses import dataclass
            @dataclass
            class DefaultConfig:
                class PathPlanningConfig:
                    grid_size = (100, 100, 105)
                    obstacle_expansion = 4
                class SegmentationConfig:
                    brush_size = 10
                path_planning = PathPlanningConfig()
                segmentation = SegmentationConfig()
            self.config = DefaultConfig()
        
        # 为了向后兼容，保留这些属性的引用（指向 data_manager）
        # 这样可以逐步迁移，不会一次性破坏所有代码
        self.brush_size = self.config.segmentation.brush_size
        # 手动分割控制器管理 mask_history
        self.manual_controller = ManualSegController(brush_size=self.brush_size)
        # 已移除，现在由 slice_editor_ctrl 管理
        self.threshold_collapsed = False
        
        # SAM2 分割相关（控制器负责模型与提示管理）
        self.sam2_controller = SAM2Controller()
        self.sam2_picking_mode = False  # 是否处于提示点选择模式（仅 UI 层使用）
        self.sam2_prompt_point_type = "positive"  # 'positive' or 'negative'
        
        # 路径规划相关
        # 注意：pick_mode、path_actors 现在由 PathUIController 统一管理
        # point_actors 保留在 MainWindow 中，用于管理3D点标记（蓝色标记，非路径点）
        self.point_actors = []  # 3D点标记 actors（蓝色标记，用于3D点选择，不是路径点）
        grid_size = self.config.path_planning.grid_size
        obstacle_expansion = self.config.path_planning.obstacle_expansion
        
        
        # 使用 RRT 路径规划（基于点云）
        # PathController 管理起点/终点/中间点和 RRT 路径生成
        self.path_controller = PathController(
            planner=None,  # RRT 不需要 A* planner
            grid_size=grid_size, 
            obstacle_expansion=obstacle_expansion,
            use_rrt=True,  # 启用 RRT
            rrt_step_size=2.0,  # RRT 步长
            rrt_safety_radius=2.0  # 安全半径
        )
        
        # 保留 A* planner 以备后用（如果需要切换回 A*）
        self.planner = AStarPlanner(grid_size=grid_size)
        
        # 初始化控制器属性为 None（在 _init_controllers 中会被设置）
        # 这样可以避免在 eventFilter 等方法中访问未初始化的属性
        self.path_ui_ctrl = None
        self.data_import_ctrl = None
        self.slice_editor_ctrl = None
        self.sam2_ui_ctrl = None
        self.multi_slice_ctrl = None
        
        # 初始化管理器属性为 None（在 _init_managers 中会被设置）
        self.view3d_manager = None
        self.event_dispatcher = None
        self.point_selection_manager = None
        
        # 3D切片窗口相关（保留用于向后兼容，实际由 PointSelectionManager 管理）
        self.slice_view_volume = None  # 用于切片显示的volume数据
        self.slice_positions = {'coronal': 0, 'sagittal': 0, 'axial': 0}  # 当前切片位置
        self.slice_pick_mode = False  # 是否启用3D取点模式
        self.computed_3d_points = []  # 计算出的完整3D坐标点
        
        # 向后兼容：这些属性将在 _init_managers 中指向 PointSelectionManager
        self.picked_3d_points = []  # 将指向 point_selection_manager.picked_3d_points
        self.slice_pick_data = {}  # 将指向 point_selection_manager.slice_pick_data
        self.slice_markers = {}  # 将指向 point_selection_manager.slice_markers
        self.point_actors = []  # 将指向 point_selection_manager.point_actors
        
        # 测量UI构建时间
        ui_start = time.time()
        self._build_ui()
        ui_time = time.time() - ui_start
        logger.info(f"UI构建耗时: {ui_time:.2f} 秒")
        
        # 初始化控制器（在 UI 构建完成后）
        self._init_controllers()
        
        # 连接所有信号（在控制器初始化后）
        self._connect_signals()
        
        # 初始化管理器（在 UI 和控制器初始化后，确保所有引用都已设置）
        self._init_managers()
    
    def _init_managers(self):
        """初始化管理器（View3DManager, EventDispatcher, PointSelectionManager）"""
        # 初始化 View3DManager
        if View3DManager is not None:
            self.view3d_manager = View3DManager(
                data_manager=self.data_manager,
                vtk_renderer=self.vtk_renderer,
                vtk_widget=self.vtk_widget,
                view3d_controller=self.view3d,
                btn_recon=self.btn_recon,
                recon_progress=self.recon_progress,
                vtk_status=self.vtk_status,
                parent_widget=self
            )
        else:
            self.view3d_manager = None
        
        # 初始化 EventDispatcher（注意：此时 point_selection_manager 还未初始化，稍后更新）
        if EventDispatcher is not None:
            self.event_dispatcher = EventDispatcher(
                main_window=self,
                slice_editor_ctrl=self.slice_editor_ctrl,
                sam2_ui_ctrl=self.sam2_ui_ctrl,
                multi_slice_ctrl=None,  # 2D Slice Views 面板已删除
                path_ui_ctrl=self.path_ui_ctrl,
                image_label=self.image_label,
                vtk_widget=self.vtk_widget,
                point_selection_manager=None  # 稍后更新
            )
        else:
            self.event_dispatcher = None
        
        # 2D Slice Views 面板已删除，PointSelectionManager 不再需要
        self.point_selection_manager = None
    
    def _init_controllers(self):
        """初始化 GUI 控制器"""
        # 初始化数据导入控制器（UI 元素已在 _build_ui 中确保存在）
        if DataImportController is not None and self.file_list is not None and self.btn_undo is not None:
            self.data_import_ctrl = DataImportController(
                self.data_manager,
                self.reader,
                self.file_list,
                self.btn_undo,
                parent_widget=self,
            )
            # 设置回调
            self.data_import_ctrl.on_volume_loaded = self._on_volume_loaded
            self.data_import_ctrl.on_volume_cleared = self._on_volume_cleared
            self.data_import_ctrl.on_slice_slider_update = self._on_slice_slider_update
        else:
            self.data_import_ctrl = None
        
        # 初始化切片编辑器控制器（UI 元素已在 _build_ui 中确保存在）
        if SliceEditorController is not None and self.image_label is not None and self.slice_slider is not None and self.h1_low is not None:
            threshold_sliders = {
                'h1_low': self.h1_low,
                'h1_high': self.h1_high,
                'h2_low': self.h2_low,
                'h2_high': self.h2_high,
                's_low': self.s_low,
                's_high': self.s_high,
                'v_low': self.v_low,
                'v_high': self.v_high,
            }
            self.slice_editor_ctrl = SliceEditorController(
                self.data_manager,
                self.manual_controller,
                self.sam2_controller,
                self.image_label,
                self.slice_label,
                self.slice_slider,
                threshold_sliders,
                radio_auto=self.radio_auto,
                btn_apply_all=self.btn_apply_all,
                parent_widget=self,
            )
            # 设置 SAM2 点击回调
            self.slice_editor_ctrl.on_sam2_click = self._handle_sam2_click
            # 退出 SAM2 picking 模式的回调将在 sam2_ui_ctrl 初始化后设置
        else:
            self.slice_editor_ctrl = None
        
        # 初始化 SAM2 UI 控制器
        if SAM2UIController is not None and self.sam2_status is not None:
            self.sam2_ui_ctrl = SAM2UIController(
                self.sam2_controller,
                self.data_manager,
                self.sam2_status,
                self.radio_auto,
                self.radio_manual,
                self.btn_add_positive,
                self.btn_add_negative,
                self.btn_undo_positive,
                self.btn_clear_prompts,
                self.btn_sam2_volume_3d,
                parent_widget=self,
                btn_add_box=self.btn_add_box,
                btn_switch_mask=self.btn_switch_mask,
            )
            # 设置回调
            self.sam2_ui_ctrl.on_segmentation_mode_changed = self._on_sam2_segmentation_mode_changed
            self.sam2_ui_ctrl.on_slice_display_update = lambda idx: (
                self.slice_editor_ctrl.update_slice_display(idx) if self.slice_editor_ctrl else self.update_slice_display(idx)
            )
            self.sam2_ui_ctrl.on_mask_updated = lambda: None  # 可以添加更新逻辑
            # 设置获取当前切片索引的函数
            self.sam2_ui_ctrl.set_current_slice_getter(lambda: self.slice_slider.value() if self.slice_slider else None)
            
            # 如果 slice_editor_ctrl 已初始化，设置双向回调以避免模式冲突
            if self.slice_editor_ctrl:
                # 当进入擦除模式时，退出 SAM2 picking 模式
                self.slice_editor_ctrl.on_exit_sam2_picking = (
                    lambda: self.sam2_ui_ctrl.exit_picking_mode() if self.sam2_ui_ctrl else None
                )
                # 当进入 SAM2 picking 模式时，退出擦除模式（同时更新按钮状态）
                def exit_brush_eraser_mode():
                    """退出画笔和擦除模式"""
                    if self.slice_editor_ctrl:
                        self.slice_editor_ctrl.handle_brush_mode(False)
                        self.slice_editor_ctrl.handle_eraser_mode(False)
                    if self.btn_brush_mode:
                        self.btn_brush_mode.setChecked(False)
                    if self.btn_eraser_mode:
                        self.btn_eraser_mode.setChecked(False)
                self.sam2_ui_ctrl.on_exit_eraser_mode = exit_brush_eraser_mode
        else:
            self.sam2_ui_ctrl = None
        
        # 初始化路径规划 UI 控制器（UI 元素已在 _build_ui 中确保存在）
        if PathUIController is not None and self.vtk_renderer is not None:
            self.path_ui_ctrl = PathUIController(
                self.path_controller,
                self.view3d,
                self.vtk_renderer,
                self.vtk_widget,
                self.path_list,
                self.vtk_status,
                parent_widget=self,
            )
            # 设置回调
            self.path_ui_ctrl.on_path_generated = self._on_path_generated
            self.path_ui_ctrl.on_path_reset = self._on_path_reset
            # 设置路径更新回调，用于同步到独立窗口
            def sync_path_updates():
                self._sync_to_reconstruction_window()
            self.path_ui_ctrl.on_path_updated = sync_path_updates
            
            # 连接撤销/重做快捷键
            self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
            self.shortcut_undo.activated.connect(self.path_ui_ctrl.handle_undo)
            
            self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
            self.shortcut_redo.activated.connect(self.path_ui_ctrl.handle_redo)
            
            # 支持 Ctrl+Shift+Z 作为重做的另一种方式
            self.shortcut_redo_alt = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
            self.shortcut_redo_alt.activated.connect(self.path_ui_ctrl.handle_redo)
        else:
            self.path_ui_ctrl = None
        
        # 2D Slice Views 面板已删除，MultiSliceController 不再需要
        self.multi_slice_ctrl = None
    
    def _on_sam2_segmentation_mode_changed(self, checked: bool):
        """SAM2分割模式变化回调"""
        if checked:
            self.sam2_picking_mode = False
        # 可以添加其他逻辑
    
    def _on_path_generated(self, path_points):
        """路径生成回调"""
        # 禁用取点按钮（UI 元素已在 _build_ui 中确保存在）
        if self.btn_pick_start:
            self.btn_pick_start.setEnabled(False)
        if self.btn_pick_waypoint:
            self.btn_pick_waypoint.setEnabled(False)
        if self.btn_pick_end:
            self.btn_pick_end.setEnabled(False)
        if self.btn_pick_3d:
            self.btn_pick_3d.setEnabled(False)

    def _on_path_reset(self):
        """路径重置回调"""
        # 重新启用取点按钮（UI 元素已在 _build_ui 中确保存在）
        if self.btn_pick_start:
            self.btn_pick_start.setEnabled(True)
        if self.btn_pick_waypoint:
            self.btn_pick_waypoint.setEnabled(True)
        if self.btn_pick_end:
            self.btn_pick_end.setEnabled(True)
        if self.btn_pick_3d:
            self.btn_pick_3d.setEnabled(True)
    
    def _on_sim_speed_changed(self, value: int):
        """Speed slider value changed — update simulation & replay speed."""
        # Update label
        if self.sim_speed_value_label:
            self.sim_speed_value_label.setText(str(value))
        # Map slider 1-10 to frame interval:  1→100ms(slow)  10→10ms(fast)
        frame_ms = max(10, 110 - value * 10)
        if self.path_ui_ctrl:
            self.path_ui_ctrl._sim_speed_ms = frame_ms
            self.path_ui_ctrl._replay_speed_ms = frame_ms
            # If currently animating, apply new speed immediately
            if self.path_ui_ctrl._is_simulating and self.path_ui_ctrl._sim_timer:
                self.path_ui_ctrl._sim_timer.setInterval(frame_ms)
            if self.path_ui_ctrl._is_replaying and self.path_ui_ctrl._replay_timer:
                self.path_ui_ctrl._replay_timer.setInterval(frame_ms)

    def _on_3d_point_added(self, coord_3d):
        """3D点添加回调"""
        logger.debug(f"3D point added: {coord_3d}")
    
    def _on_path_point_set_from_list(self, point_type: str, space_coord):
        """从列表设置路径点回调（统一通过 path_ui_ctrl 处理）"""
        if self.path_ui_ctrl:
            # 临时设置 pick_mode，然后统一处理
            old_pick_mode = self.path_ui_ctrl.pick_mode
            self.path_ui_ctrl.pick_mode = point_type
            self.path_ui_ctrl._process_picked_point(space_coord[0], space_coord[1], space_coord[2])
            # 恢复原来的 pick_mode
            self.path_ui_ctrl.pick_mode = old_pick_mode
    
    def _on_volume_loaded(self, volume, metadata):
        """当 volume 加载完成时的回调"""
        # 更新主切片编辑器滑块
        self.slice_slider.setEnabled(True)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(max(0, volume.shape[0] - 1))
        self.slice_slider.setValue(0)
        self.slice_label.setText("Slice: 0")
        
        # 2D Slice Views 面板已删除，不再需要更新多视图切片
        
        # 更新切片显示
        if self.slice_editor_ctrl:
            self.slice_editor_ctrl.update_slice_display(0)
        else:
            self.update_slice_display(0)
    
    def _on_volume_cleared(self):
        """当 volume 被清除时的回调"""
        self.slice_slider.setEnabled(False)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.image_label.setText("No image")
    
    def _on_slice_slider_update(self, min_val, max_val, value):
        """更新切片滑块的回调"""
        self.slice_slider.setMinimum(min_val)
        self.slice_slider.setMaximum(max_val)
        self.slice_slider.setValue(value)

    def _handle_sam2_click(self, x: int, y: int) -> bool:
        """
        处理 SAM2 点击事件（委托给 SAM2 UI 控制器）
        
        Args:
            x: 点击的 x 坐标（图像坐标）
            y: 点击的 y 坐标（图像坐标）
        
        Returns:
            bool: 是否成功处理
        """
        if self.sam2_ui_ctrl:
            return self.sam2_ui_ctrl.handle_sam2_click(x, y)
        return False

    def set_patient_context(self, context):
        self.patient_context = context
        if self.patient_context:
            info_text = f"Current Patient: {context['name']} (ID: {context['patient_id']}) | Case: {context['case_no']} | Type: {context['exam_type']}"
            self.lbl_patient_info.setText(info_text)
            logger.info(f"Patient context set: {info_text}")

    def _build_ui(self):
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(0, 0, 0, 0)
        main_vbox.setSpacing(0)
        
        # --- Top Patient Info Bar ---
        self.patient_bar = QFrame()
        self.patient_bar.setFixedHeight(50)
        self.patient_bar.setStyleSheet("""
            QFrame {
                background-color: #F5F7FA;
                border-bottom: 1px solid #DCDCDC;
            }
            QLabel {
                color: #2C3E50;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        bar_layout = QHBoxLayout(self.patient_bar)
        
        self.lbl_patient_info = QLabel("Current Patient: Not Set")
        bar_layout.addWidget(self.lbl_patient_info)
        
        bar_layout.addStretch()
        
        self.btn_back = QPushButton("Back")
        self.btn_exit = QPushButton("Exit")
        for btn in [self.btn_back, self.btn_exit]:
            btn.setFixedSize(80, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #BDC3C7;
                    border-radius: 4px;
                    color: #2C3E50;
                    font-weight: normal;
                }
                QPushButton:hover { background-color: #ECF0F1; }
            """)
        
        self.btn_back.clicked.connect(self.back_clicked.emit)
        self.btn_exit.clicked.connect(self.exit_clicked.emit)
        
        bar_layout.addWidget(self.btn_back)
        bar_layout.addWidget(self.btn_exit)
        
        main_vbox.addWidget(self.patient_bar)

        # --- Original UI Content ---
        content_widget = QWidget()
        content_widget.setStyleSheet("""
            QWidget {
                background-color: #F8FAFC;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                padding: 8px 15px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #1565C0; }
            
            /* Secondary/Outlined Button Style */
            QPushButton#secondary_btn {
                background-color: transparent;
                color: #1E88E5;
                border: 1px solid #1E88E5;
            }
            QPushButton#secondary_btn:hover {
                background-color: #E3F2FD;
            }

            QGroupBox {
                font-weight: bold;
                border: 1px solid #E0E4E8;
                border-radius: 8px;
                margin-top: 15px;
                background-color: white;
                padding-top: 10px; /* Add internal padding */
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #1E88E5;
                font-size: 14px;
            }
            QListWidget {
                border: 1px solid #E0E4E8;
                border-radius: 4px;
                background-color: #FFFFFF;
            }
            QSlider::handle:horizontal {
                background: #1E88E5;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        self.root_layout = QHBoxLayout(content_widget)
        self.root_layout.setContentsMargins(15, 5, 15, 15)
        self.root_layout.setSpacing(15)
        main_vbox.addWidget(content_widget)

        left_container = QWidget()
        left_layout = QVBoxLayout()
        left_container.setLayout(left_layout)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left_container)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 垂直滚动条按需显示
        left_scroll.setSizeAdjustPolicy(QScrollArea.AdjustToContents)  # 自动调整大小以适应内容
        self.root_layout.addWidget(left_scroll, 1)  # 左边：1份

        # 调试：检查UI构建器是否可用
        logger.debug(f"build_data_import_ui: {build_data_import_ui}")
        logger.debug(f"build_slice_editor_ui: {build_slice_editor_ui}")
        logger.debug(f"build_sam2_ui: {build_sam2_ui}")
        logger.debug(f"build_view3d_ui: {build_view3d_ui}")
        logger.debug(f"build_multi_slice_ui: {build_multi_slice_ui}")

        # 使用 UI 构建器创建数据导入 UI（应该在最上面）
        # 确保所有 UI 元素都被创建，即使构建器不可用也设置为 None
        if build_data_import_ui is not None:
            logger.debug("创建数据导入UI...")
            data_ui = build_data_import_ui()
            self.btn_select = data_ui['btn_select']
            self.btn_import = data_ui['btn_import']
            self.file_list = data_ui['file_list']
            self.btn_delete = data_ui['btn_delete']
            self.btn_undo = data_ui['btn_undo']
            left_layout.addWidget(data_ui['group'])
            logger.debug(f"数据导入UI已添加，left_layout.count()={left_layout.count()}")
        else:
            logger.warning("build_data_import_ui 不可用，设置默认值")
            # 设置为 None，避免后续 hasattr 检查
            self.btn_select = None
            self.btn_import = None
            self.file_list = None
            self.btn_delete = None
            self.btn_undo = None

        # 使用 UI 构建器创建切片编辑器 UI
        # 确保所有 UI 元素都被创建，即使构建器不可用也设置为 None
        if build_slice_editor_ui is not None:
            logger.debug("创建切片编辑器UI...")
            slice_ui = build_slice_editor_ui(self.data_manager.thresholds)
            self.image_label = slice_ui['image_label']
            self.btn_brush_mode = slice_ui['btn_brush_mode']
            self.btn_eraser_mode = slice_ui['btn_eraser_mode']
            self.slice_label = slice_ui['slice_label']
            self.slice_slider = slice_ui['slice_slider']
            self.btn_toggle_threshold = slice_ui['btn_toggle_threshold']
            self.threshold_group = slice_ui['threshold_group']
            self.h1_low = slice_ui['h1_low']
            self.h1_high = slice_ui['h1_high']
            self.h2_low = slice_ui['h2_low']
            self.h2_high = slice_ui['h2_high']
            self.s_low = slice_ui['s_low']
            self.s_high = slice_ui['s_high']
            self.v_low = slice_ui['v_low']
            self.v_high = slice_ui['v_high']
            self.btn_reset_th = slice_ui['btn_reset_th']
            self.btn_apply_all = slice_ui['btn_apply_all']
            self.btn_save_masks = slice_ui['btn_save_masks']
            self.btn_clear_masks = slice_ui['btn_clear_masks']
            left_layout.addWidget(slice_ui['group'])
            logger.debug(f"切片编辑器UI已添加，left_layout.count()={left_layout.count()}")
        else:
            logger.warning("build_slice_editor_ui 不可用，设置默认值")
            # 设置为 None，避免后续 hasattr 检查
            self.image_label = None
            self.btn_brush_mode = None
            self.btn_eraser_mode = None
            self.slice_label = None
            self.slice_slider = None
            self.btn_toggle_threshold = None
            self.threshold_group = None
            self.h1_low = None
            self.h1_high = None
            self.h2_low = None
            self.h2_high = None
            self.s_low = None
            self.s_high = None
            self.v_low = None
            self.v_high = None
            self.btn_reset_th = None
            self.btn_apply_all = None
            self.btn_save_masks = None
            self.btn_clear_masks = None

        # 使用 UI 构建器创建 3D 视图 UI
        # 确保所有 UI 元素都被创建，即使构建器不可用也设置为 None
        if build_view3d_ui is not None:
            logger.debug("创建3D视图UI...")
            view3d_ui = build_view3d_ui(parent_widget=self)
            self.vtk_widget = view3d_ui['vtk_widget']
            self.btn_recon = view3d_ui['btn_recon']
            self.btn_open_model = view3d_ui['btn_open_model']
            self.btn_reset_cam = view3d_ui['btn_reset_cam']
            self.btn_clear_3d = view3d_ui['btn_clear_3d']
            self.btn_open_window = view3d_ui.get('btn_open_window', None)
            self.recon_progress = view3d_ui['recon_progress']
            self.vtk_status = view3d_ui['vtk_status']
            self.btn_pick_start = view3d_ui['btn_pick_start']
            self.btn_pick_waypoint = view3d_ui['btn_pick_waypoint']
            self.btn_pick_end = view3d_ui['btn_pick_end']
            self.path_list = view3d_ui['path_list']
            self.btn_generate_path = view3d_ui['btn_generate_path']
            self.btn_replay_path = view3d_ui['btn_replay_path']
            self.btn_simulate_path = view3d_ui['btn_simulate_path']
            self.btn_save_path = view3d_ui['btn_save_path']
            self.btn_reset_path = view3d_ui['btn_reset_path']
            self.sim_speed_slider = view3d_ui['sim_speed_slider']
            self.sim_speed_value_label = view3d_ui['sim_speed_value_label']
            self.chk_safety_zone = view3d_ui['chk_safety_zone']
            
            # 初始化 VTK renderer
            self.vtk_renderer = vtkRenderer()
            if self.vtk_widget and hasattr(self.vtk_widget, "GetRenderWindow"):
                rw = self.vtk_widget.GetRenderWindow()
                if rw:
                    rw.AddRenderer(self.vtk_renderer)
                    # 使用配置中的背景色
                    bg_color = getattr(self, 'config', None)
                    if bg_color and hasattr(bg_color, 'view3d'):
                        bg = bg_color.view3d.background_color
                    else:
                        bg = (0.1, 0.1, 0.1)
                    self.vtk_renderer.SetBackground(*bg)
            if self.vtk_widget:
                self.vtk_widget.setMinimumHeight(300)
            
            # 初始化 VTK interactor
            try:
                if self.vtk_widget and hasattr(self.vtk_widget, "Initialize"):
                    self.vtk_widget.Initialize()
                if self.vtk_widget and hasattr(self.vtk_widget, "Start"):
                    self.vtk_widget.Start()
                if self.vtk_widget and hasattr(self.vtk_widget, "GetInteractor"):
                    interactor = self.vtk_widget.GetInteractor()
                    if interactor:
                        interactor.Enable()
                        logger.debug("VTK interactor enabled")
            except Exception as e:
                logger.warning(f"Could not start VTK interactor: {e}")
            
            # 安装事件过滤器
            if self.vtk_widget and hasattr(self.vtk_widget, "installEventFilter"):
                self.vtk_widget.installEventFilter(self)
            
            # 3D 视图控制器
            self.view3d = View3DController(self.vtk_renderer, self.vtk_widget) if self.vtk_renderer and self.vtk_widget else None
            
            self.root_layout.addWidget(view3d_ui['middle_panel'], 3)  # 中间：3份
            logger.debug(f"3D视图UI已添加，root_layout.count()={self.root_layout.count()}")
        else:
            logger.warning("build_view3d_ui 不可用，设置默认值")
            # 设置为 None，避免后续 hasattr 检查
            self.vtk_widget = None
            self.vtk_renderer = vtkRenderer()  # renderer 仍然需要创建
            self.btn_recon = None
            self.btn_open_model = None
            self.btn_reset_cam = None
            self.btn_clear_3d = None
            self.btn_open_window = None
            self.recon_progress = None
            self.vtk_status = None
            self.btn_pick_start = None
            self.btn_pick_waypoint = None
            self.btn_pick_end = None
            self.path_list = None
            self.btn_generate_path = None
            self.btn_replay_path = None
            self.btn_simulate_path = None
            self.btn_save_path = None
            self.btn_reset_path = None
            self.sim_speed_slider = None
            self.sim_speed_value_label = None
            self.chk_safety_zone = None
            self.view3d = None
        
        # 2D Slice Views 面板已删除，选点功能已移到弹窗对话框
        
        # 设置为 None，避免后续 hasattr 检查
        self.btn_toggle_slice_view = None
        self.right_scroll = None
        self.right_content = None
        self.coronal_label = None
        self.coronal_slider = None
        self.sagittal_label = None
        self.sagittal_slider = None
        self.axial_label = None
        self.axial_slider = None
        self.btn_pick_3d = None
        self.btn_clear_3d_points = None
        self.coord_list = None
        self.btn_set_start_from_list = None
        self.btn_add_waypoint_from_list = None
        self.btn_set_end_from_list = None
        self.right_panel_widget = None
        
        # 设置布局比例（两列布局：左边和中间）
        self.root_layout.setStretch(0, 2)   # 左边：1份
        self.root_layout.setStretch(2, 4)   # 中间：4份（增加中间面板的宽度）

        # 使用 UI 构建器创建 SAM2 UI
        # 确保所有 UI 元素都被创建，即使构建器不可用也设置为 None
        if build_sam2_ui is not None:
            logger.debug("创建SAM2 UI...")
            sam2_ui = build_sam2_ui()
            self.radio_manual = sam2_ui['radio_manual']
            self.radio_auto = sam2_ui['radio_auto']
            self.sam2_group = sam2_ui['sam2_group']
            self.btn_load_sam2 = sam2_ui['btn_load_sam2']
            self.sam2_status = sam2_ui['sam2_status']
            # prompt_mode_group, radio_point_prompt, radio_box_prompt 已删除
            self.btn_add_positive = sam2_ui['btn_add_positive']
            self.btn_add_negative = sam2_ui['btn_add_negative']
            self.btn_add_box = sam2_ui['btn_add_box']
            self.btn_switch_mask = sam2_ui['btn_switch_mask']
            self.btn_undo_positive = sam2_ui['btn_undo_positive']
            self.btn_clear_prompts = sam2_ui['btn_clear_prompts']
            self.btn_start_seg = sam2_ui['btn_start_seg']
            self.btn_sam2_volume_3d = sam2_ui['btn_sam2_volume_3d']
            # 标签选择器
            self.combo_label = sam2_ui.get('combo_label')
            self.btn_add_label = sam2_ui.get('btn_add_label')
            self.btn_remove_label = sam2_ui.get('btn_remove_label')
            # 保存引用以便后续启用/禁用
            self.btn_add_positive_ref = self.btn_add_positive
            self.btn_add_negative_ref = self.btn_add_negative
            self.btn_add_box_ref = self.btn_add_box # 增加引用
            self.btn_clear_prompts_ref = self.btn_clear_prompts
            left_layout.addWidget(sam2_ui['seg_group'])
            logger.debug(f"SAM2 UI已添加，left_layout.count()={left_layout.count()}")
        else:
            logger.warning("build_sam2_ui 不可用，设置默认值")
            # 设置为 None，避免后续 hasattr 检查
            self.radio_manual = None
            self.radio_auto = None
            self.sam2_group = None
            self.btn_load_sam2 = None
            self.sam2_status = None
            self.prompt_mode_group = None
            self.radio_point_prompt = None
            self.radio_box_prompt = None
            self.btn_add_positive = None
            self.btn_add_negative = None
            self.btn_add_box = None
            self.btn_undo_positive = None
            self.btn_clear_prompts = None
            self.btn_start_seg = None
            self.btn_sam2_volume_3d = None
            self.combo_label = None
            self.btn_add_label = None
            self.btn_remove_label = None
            self.btn_add_positive_ref = None
            self.btn_add_negative_ref = None
            self.btn_add_box_ref = None
            self.btn_clear_prompts_ref = None
        
        # 设置左侧布局的拉伸比例（在所有组件添加完成后）
        # 数据导入UI：按内容大小（stretch=0）
        # 切片编辑器：占据更多空间（stretch=2）
        # SAM2 UI：按内容大小（stretch=0）
        widget_count = left_layout.count()
        for i in range(widget_count):
            # 找到切片编辑器（通常是第二个组件，索引1）
            if i == 1 and widget_count >= 2:
                left_layout.setStretch(i, 2)  # 切片编辑器占据更多空间
            else:
                left_layout.setStretch(i, 0)  # 其他组件按内容大小
        
        logger.info(f"UI构建完成 - left_layout.count()={left_layout.count()}, root_layout.count()={self.root_layout.count()}")
    
    def _connect_signals(self):
        """连接所有UI信号到控制器方法（在控制器初始化后调用）"""
        # 连接数据导入事件（只检查控制器是否存在，UI 元素已在 _build_ui 中确保存在）
        if self.data_import_ctrl:
            if self.btn_select:
                self.btn_select.clicked.connect(self.data_import_ctrl.handle_select_files)
            if self.btn_import:
                self.btn_import.clicked.connect(self.data_import_ctrl.handle_import)
            if self.btn_delete:
                self.btn_delete.clicked.connect(self.data_import_ctrl.handle_exclude_selected)
            if self.btn_undo:
                self.btn_undo.clicked.connect(self.data_import_ctrl.handle_undo_exclusion)
        
        # 连接切片编辑器事件（只检查控制器是否存在）
        if self.slice_editor_ctrl:
            if self.slice_slider:
                self.slice_slider.valueChanged.connect(self.slice_editor_ctrl.handle_slice_change)
            if self.btn_brush_mode:
                self.btn_brush_mode.clicked.connect(self.slice_editor_ctrl.handle_brush_mode)
            if self.btn_eraser_mode:
                self.btn_eraser_mode.clicked.connect(self.slice_editor_ctrl.handle_eraser_mode)
            
            # 连接阈值相关事件
            if self.h1_low:
                for sname, s in [("h1_low", self.h1_low), ("h1_high", self.h1_high), ("h2_low", self.h2_low), 
                                 ("h2_high", self.h2_high), ("s_low", self.s_low), ("s_high", self.s_high),
                                 ("v_low", self.v_low), ("v_high", self.v_high)]:
                    if s:
                        s.valueChanged.connect(lambda v, name=sname: self.slice_editor_ctrl.handle_threshold_change(name, v))
                if self.btn_reset_th:
                    self.btn_reset_th.clicked.connect(self.slice_editor_ctrl.handle_reset_thresholds)
                if self.btn_apply_all:
                    self.btn_apply_all.clicked.connect(self.slice_editor_ctrl.handle_apply_threshold_all)
                if self.btn_save_masks:
                    self.btn_save_masks.clicked.connect(lambda: self.slice_editor_ctrl.handle_save_masks(self))
                if self.btn_clear_masks:
                    self.btn_clear_masks.clicked.connect(self.slice_editor_ctrl.handle_clear_masks)
                if self.btn_toggle_threshold and self.threshold_group:
                    def toggle_threshold():
                        """切换阈值组显示/隐藏并更新按钮文本"""
                        self.slice_editor_ctrl.handle_toggle_threshold(self.threshold_group)
                        # 更新按钮文本
                        self.btn_toggle_threshold.setText("Threshold ▸" if self.slice_editor_ctrl.threshold_collapsed else "Threshold ▾")
                    self.btn_toggle_threshold.clicked.connect(toggle_threshold)
        
        # 连接3D视图事件（UI 元素已在 _build_ui 中确保存在）
        if self.btn_recon:
            self.btn_recon.clicked.connect(self.on_reconstruct_3d)
        if self.btn_clear_3d:
            self.btn_clear_3d.clicked.connect(self.on_clear_3d_view)
        if self.btn_open_model:
            self.btn_open_model.clicked.connect(self.on_open_model)
        if self.btn_reset_cam:
            self.btn_reset_cam.clicked.connect(self.on_reset_camera)
        if self.btn_open_window:
            self.btn_open_window.clicked.connect(self.on_open_reconstruction_window)
        
        # 连接路径规划事件（只检查控制器是否存在）
        logger.info(f"🔵 _connect_signals: path_ui_ctrl={self.path_ui_ctrl}, path_list={self.path_list}")
        if self.path_ui_ctrl:
            logger.info("🔵 path_ui_ctrl 存在，开始连接路径相关信号")
            if self.btn_pick_start:
                self.btn_pick_start.clicked.connect(lambda: self.path_ui_ctrl.handle_set_pick_mode('start'))
            if self.btn_pick_waypoint:
                self.btn_pick_waypoint.clicked.connect(lambda: self.path_ui_ctrl.handle_set_pick_mode('waypoint'))
            if self.btn_pick_end:
                self.btn_pick_end.clicked.connect(lambda: self.path_ui_ctrl.handle_set_pick_mode('end'))
            if self.btn_generate_path:
                self.btn_generate_path.clicked.connect(self.path_ui_ctrl.handle_generate_path)
            if self.btn_replay_path:
                self.btn_replay_path.clicked.connect(self.path_ui_ctrl.handle_replay_rrt)
            if self.btn_simulate_path:
                self.btn_simulate_path.clicked.connect(self.path_ui_ctrl.handle_simulate_path)
            if self.sim_speed_slider:
                self.sim_speed_slider.valueChanged.connect(self._on_sim_speed_changed)
            if self.chk_safety_zone:
                self.chk_safety_zone.toggled.connect(self.path_ui_ctrl.handle_toggle_safety_zone)
            if self.btn_save_path:
                self.btn_save_path.clicked.connect(self.path_ui_ctrl.handle_save_path)
            if self.btn_reset_path:
                self.btn_reset_path.clicked.connect(self.path_ui_ctrl.handle_reset_path)
            # 连接路径列表双击事件 - 编辑路径点
            logger.info(f"🔵 准备连接 path_list 双击信号, path_list={self.path_list}, type={type(self.path_list)}")
            # 使用 hasattr 和 is not None 双重检查
            if hasattr(self, 'path_list') and self.path_list is not None:
                try:
                    self.path_list.itemDoubleClicked.connect(self.path_ui_ctrl.handle_path_list_double_click)
                    logger.info("✅ path_list.itemDoubleClicked 已连接到 handle_path_list_double_click")
                except Exception as e:
                    logger.error(f"❌ 连接 path_list 双击信号失败: {e}")
            else:
                logger.warning(f"❌ path_list 不可用: hasattr={hasattr(self, 'path_list')}, is_not_none={self.path_list is not None if hasattr(self, 'path_list') else False}")
        else:
            logger.warning("❌ path_ui_ctrl 为 None，跳过路径信号连接")
        
        # 连接SAM2相关信号（只检查控制器是否存在）
        if self.sam2_ui_ctrl:
            if self.btn_load_sam2:
                self.btn_load_sam2.clicked.connect(self.sam2_ui_ctrl.handle_load_model)
            if self.radio_auto:
                self.radio_auto.toggled.connect(self.sam2_ui_ctrl.handle_segmentation_mode_changed)
            # radio_point_prompt 和 radio_box_prompt 已删除，无需连接
            if self.btn_add_positive:
                self.btn_add_positive.clicked.connect(lambda: self.sam2_ui_ctrl.handle_set_prompt_mode('positive'))
            if self.btn_add_negative:
                self.btn_add_negative.clicked.connect(lambda: self.sam2_ui_ctrl.handle_set_prompt_mode('negative'))
            if hasattr(self, 'btn_add_box') and self.btn_add_box:
                self.btn_add_box.clicked.connect(lambda: self.sam2_ui_ctrl.handle_set_prompt_mode('box'))
            
            # 设置预览框更新回调
            def update_preview(box):
                if self.slice_editor_ctrl:
                    self.slice_editor_ctrl.update_slice_display(self.slice_slider.value())
            self.sam2_ui_ctrl.on_box_preview_update = update_preview
            if self.btn_undo_positive:
                self.btn_undo_positive.clicked.connect(self.sam2_ui_ctrl.handle_undo_last_positive)
            if self.btn_clear_prompts:
                self.btn_clear_prompts.clicked.connect(self.sam2_ui_ctrl.handle_clear_prompts)
            if self.btn_start_seg:
                self.btn_start_seg.clicked.connect(self.sam2_ui_ctrl.handle_start_segmentation)
            if hasattr(self, 'btn_switch_mask') and self.btn_switch_mask:
                self.btn_switch_mask.clicked.connect(self.sam2_ui_ctrl.handle_switch_mask)
            if self.btn_sam2_volume_3d:
                self.btn_sam2_volume_3d.clicked.connect(self.sam2_ui_ctrl.handle_sam2_volume_3d)
        
        # 标签选择器信号连接
        if self.combo_label:
            self.combo_label.currentIndexChanged.connect(self._on_label_changed)
            # 初始化标签列表
            self._refresh_label_combo()
        if self.btn_add_label:
            self.btn_add_label.clicked.connect(self._on_add_label)
        if self.btn_remove_label:
            self.btn_remove_label.clicked.connect(self._on_remove_label)
        
        # 2D Slice Views 面板已删除，相关信号连接已移除
        
        # 安装事件过滤器（UI 元素已在 _build_ui 中确保存在）
        if self.image_label:
            self.image_label.installEventFilter(self)


    def on_reconstruct_3d(self):
        """3D 重建（委托给 View3DManager）"""
        logger.info("🔵 on_reconstruct_3d() 被调用")
        if self.view3d_manager:
            logger.info("✅ View3DManager 存在，调用 reconstruct_3d()")
            self.view3d_manager.reconstruct_3d()
            # 等待渲染完成后再同步
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            # 同步到3D Reconstruction独立窗口（如果存在）
            self._sync_to_reconstruction_window()
        else:
            # 回退：如果管理器不可用，显示警告
            logger.error("❌ View3DManager 不可用")
            QMessageBox.warning(self, "3D Reconstruction", "View3DManager not available.")

    def on_clear_3d_view(self):
        """清除3D视图（委托给 View3DManager）"""
        if self.view3d_manager:
            self.view3d_manager.clear_3d_view(
                path_ui_ctrl=self.path_ui_ctrl,
                multi_slice_ctrl=self.multi_slice_ctrl
            )
            # 同步到3D Reconstruction独立窗口（如果存在）
            self._sync_to_reconstruction_window()

    def on_open_model(self):
        """打开STL模型（委托给 View3DManager）"""
        if self.view3d_manager:
            self.view3d_manager.open_model()
            # 等待渲染完成后再同步
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            # 同步到3D Reconstruction独立窗口（如果存在）
            self._sync_to_reconstruction_window()
        else:
            QMessageBox.warning(self, "Open Model", "View3DManager not available.")

    def on_reset_camera(self):
        """重置相机（委托给 View3DManager）"""
        if self.view3d_manager:
            self.view3d_manager.reset_camera()
            # 同步到3D Reconstruction独立窗口（如果存在）
            self._sync_to_reconstruction_window()
    
    def _sync_to_reconstruction_window(self):
        """同步主窗口的3D视图内容到3D Reconstruction独立窗口"""
        if hasattr(self, 'reconstruction_window') and self.reconstruction_window is not None:
            try:
                # 更新引用（以防它们发生变化）
                if self.view3d_manager and hasattr(self.view3d_manager, 'coordinate_system'):
                    self.reconstruction_window.source_coordinate_system = self.view3d_manager.coordinate_system
                self.reconstruction_window.source_path_ui_controller = self.path_ui_ctrl
                self.reconstruction_window.source_renderer = self.vtk_renderer
                self.reconstruction_window.sync_from_source()
            except Exception as e:
                logger.warning(f"同步到3D Reconstruction独立窗口时出错: {e}")
    
    def on_open_reconstruction_window(self):
        """打开完整的3D Reconstruction独立窗口"""
        try:
            from surgical_robot_app.gui.dialogs.view3d_reconstruction_window import View3DReconstructionWindow
        except ImportError:
            try:
                from gui.dialogs.view3d_reconstruction_window import View3DReconstructionWindow
            except ImportError:
                logger.error("View3DReconstructionWindow not available")
                QMessageBox.warning(self, "Error", "3D Reconstruction window is not available.")
                return
        
        # 获取必要的引用
        coordinate_system = None
        if self.view3d_manager and hasattr(self.view3d_manager, 'coordinate_system'):
            coordinate_system = self.view3d_manager.coordinate_system
        
        # 如果窗口已存在，则显示并同步；否则创建新窗口
        if not hasattr(self, 'reconstruction_window') or self.reconstruction_window is None:
            self.reconstruction_window = View3DReconstructionWindow(
                parent=self,
                main_view=self,
                source_renderer=self.vtk_renderer,
                coordinate_system=coordinate_system,
                path_ui_controller=self.path_ui_ctrl,
                view3d_manager=self.view3d_manager,
                data_manager=self.data_manager
            )
            # 连接窗口关闭信号，同步状态并清理引用
            def on_window_closing():
                # 同步状态回主窗口
                if self.reconstruction_window:
                    self.reconstruction_window.sync_to_main_view()
                # 清理引用
                setattr(self, 'reconstruction_window', None)
                # 清除路径控制器中的独立窗口引用
                if self.path_ui_ctrl:
                    self.path_ui_ctrl.reconstruction_window = None
            
            self.reconstruction_window.window_closing.connect(on_window_closing)
            
            # 将独立窗口引用传递给路径控制器，用于同步预览标记
            if self.path_ui_ctrl:
                self.path_ui_ctrl.reconstruction_window = self.reconstruction_window
        else:
            # 更新引用（以防它们发生变化）
            self.reconstruction_window.source_renderer = self.vtk_renderer
            if self.view3d_manager and hasattr(self.view3d_manager, 'coordinate_system'):
                self.reconstruction_window.source_coordinate_system = self.view3d_manager.coordinate_system
            self.reconstruction_window.path_ui_controller = self.path_ui_ctrl
            self.reconstruction_window.view3d_manager = self.view3d_manager
            self.reconstruction_window.data_manager = self.data_manager
        
        # 同步内容并显示窗口
        if self.reconstruction_window:
            # 显示窗口（必须在同步前显示，确保VTK widget已初始化）
            self.reconstruction_window.show()
            self.reconstruction_window.raise_()
            self.reconstruction_window.activateWindow()
            
            # 处理事件，确保窗口完全显示
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            # 同步内容（在窗口显示后）
            self.reconstruction_window.sync_from_source()

    # ========== 3D重建和视图控制方法 ==========
    

    def eventFilter(self, obj, event):
        """事件过滤器：将事件委托给 EventDispatcher"""
        # 使用 getattr 安全访问，因为 event_dispatcher 可能在初始化过程中还未设置
        event_dispatcher = getattr(self, 'event_dispatcher', None)
        if event_dispatcher:
            if event_dispatcher.filter_event(obj, event):
                return True
        return False
    
    # ========== 切换切片视图显示 ==========
    # 注意：路径规划相关的方法已迁移到 PathUIController
    # 这里只保留必要的工具方法
    
    def update_vtk_display(self):
        """更新VTK显示（委托给 View3DManager）"""
        if self.view3d_manager:
            self.view3d_manager.update_display()
        else:
            # 回退：如果管理器不可用，直接更新
            try:
                if hasattr(self.vtk_widget, "GetRenderWindow"):
                    rw = self.vtk_widget.GetRenderWindow()
                    if rw:
                        rw.Render()
                        self.vtk_widget.update()
                        QApplication.processEvents()
            except Exception as e:
                logger.error(f"Error updating VTK display: {e}")
    
    # ========== 3D切片窗口相关方法（已迁移到 MultiSliceController） ==========
    # 注意：多视图切片相关的方法已迁移到 MultiSliceController
    # 以下方法作为回退保留（如果控制器不可用）
    
    def handle_slice_click(self, plane_type: str, x: int, y: int, label_widget):
        """
        处理切片窗口的点击事件（委托给 PointSelectionManager）
        
        Args:
            plane_type: 切片类型 ('coronal', 'sagittal', 'axial')
            x: 点击的X坐标（窗口坐标）
            y: 点击的Y坐标（窗口坐标）
            label_widget: 点击的标签控件
        """
        if self.point_selection_manager:
            self.point_selection_manager.handle_slice_click(plane_type, x, y, label_widget)
        else:
            # 回退：如果管理器不可用，显示警告
            QMessageBox.warning(self, "3D Point Pick", "PointSelectionManager not available")
    
    def draw_slice_marker(self, plane_type: str, click_2d: Tuple[float, float], label_widget):
        """
        在切片窗口中绘制标记点（委托给 PointSelectionManager）
        
        Args:
            plane_type: 切片类型
            click_2d: 2D坐标（归一化到0-100）
            label_widget: 标签控件
        """
        if self.point_selection_manager:
            self.point_selection_manager.draw_slice_marker(plane_type, click_2d, label_widget)
    
    def calculate_3d_coordinate(self):
        """
        通过两个切片窗口的交集计算完整的3D坐标（委托给 PointSelectionManager）
        """
        if self.point_selection_manager:
            self.point_selection_manager.calculate_3d_coordinate()
    
    def update_all_slice_markers(self, coord_3d: Tuple[float, float, float]):
        """
        在所有三个切片窗口的对应位置显示标记点（委托给 PointSelectionManager）
        
        Args:
            coord_3d: 3D坐标 (x, y, z)，范围[0, 100]
        """
        if self.point_selection_manager:
            self.point_selection_manager.update_all_slice_markers(coord_3d)
    
    def add_3d_point_marker(self, coord_3d: Tuple[float, float, float]):
        """
        在3D窗口中添加3D点标记（委托给 PointSelectionManager）
        
        Args:
            coord_3d: 3D坐标 (x, y, z)，范围[0, 100]
        """
        if self.point_selection_manager:
            self.point_selection_manager.add_3d_point_marker(coord_3d)
    
    def clear_slice_markers(self):
        """
        清除切片窗口中的标记（委托给 PointSelectionManager）
        """
        if self.point_selection_manager:
            self.point_selection_manager.clear_slice_markers()
        # 2D Slice Views 面板已删除，不再需要处理
    
    # on_enable_3d_pick, on_clear_3d_points, on_set_path_point_from_list 方法已删除
    # （2D Slice Views 面板已删除，选点功能已移到弹窗对话框）
    
    # ==================== 多标签管理 ====================
    
    def _refresh_label_combo(self):
        """刷新标签下拉框"""
        if not self.combo_label or not self.data_manager:
            return
        
        self.combo_label.blockSignals(True)
        self.combo_label.clear()
        
        from surgical_robot_app.gui.ui_builders.sam2_ui import _create_color_icon
        
        for label_id in self.data_manager.get_all_label_ids():
            name = self.data_manager.label_names.get(label_id, f"Label {label_id}")
            color = self.data_manager.get_label_color(label_id)
            icon = _create_color_icon(color)
            self.combo_label.addItem(icon, f"{name}", label_id)
        
        # 选中当前标签
        current = self.data_manager.get_current_label()
        for i in range(self.combo_label.count()):
            if self.combo_label.itemData(i) == current:
                self.combo_label.setCurrentIndex(i)
                break
        
        self.combo_label.blockSignals(False)
    
    def _on_label_changed(self, index):
        """标签下拉框选择变化"""
        if not self.combo_label or not self.data_manager:
            return
        label_id = self.combo_label.itemData(index)
        if label_id is not None:
            self.data_manager.set_current_label(label_id)
            # 刷新切片显示
            if self.slice_editor_ctrl and hasattr(self.slice_editor_ctrl, 'slice_slider'):
                current_idx = self.slice_editor_ctrl.slice_slider.value()
                self.slice_editor_ctrl.update_slice_display(current_idx)
    
    def _on_add_label(self):
        """添加新标签"""
        if not self.data_manager:
            return
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New Label", "Label name:", text=f"Label {self.data_manager._next_label_id}")
        if ok and name.strip():
            label_id = self.data_manager.add_label(name.strip())
            self.data_manager.set_current_label(label_id)
            self._refresh_label_combo()
    
    def _on_remove_label(self):
        """删除当前标签"""
        if not self.data_manager:
            return
        current = self.data_manager.get_current_label()
        label_names = self.data_manager.get_label_names()
        
        if len(label_names) <= 1:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Info", "At least one label is required.")
            return
        
        from PyQt5.QtWidgets import QMessageBox
        name = label_names.get(current, f"Label {current}")
        reply = QMessageBox.question(
            self, "Delete Label",
            f"Delete \"{name}\" and all its segmentation data?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.data_manager.remove_label(current)
            self._refresh_label_combo()
            # 刷新切片显示
            if self.slice_editor_ctrl and hasattr(self.slice_editor_ctrl, 'slice_slider'):
                current_idx = self.slice_editor_ctrl.slice_slider.value()
                self.slice_editor_ctrl.update_slice_display(current_idx)
    
    def on_toggle_slice_view(self):
        """
        切换2D切片视图面板的显示/隐藏
        """
        if self.right_scroll:
            if self.right_scroll.isVisible():
                # 隐藏滚动区域（包含所有内容）
                self.right_scroll.setVisible(False)
                self.btn_toggle_slice_view.setText("▲ Show")
                # 初始化 right_panel_collapsed（如果不存在）
                if not hasattr(self, 'right_panel_collapsed'):
                    self.right_panel_collapsed = False
                self.right_panel_collapsed = True
                # 调整布局比例，给中间更多空间
                if self.root_layout:
                    self.root_layout.setStretch(0, 2)   # 左边：2份
                    self.root_layout.setStretch(1, 3)   # 中间：3份
                    self.root_layout.setStretch(2, 0)   # 右边：0份（最小化）
            else:
                # 显示滚动区域
                self.right_scroll.setVisible(True)
                self.btn_toggle_slice_view.setText("▼ Hide")
                # 初始化 right_panel_collapsed（如果不存在）
                if not hasattr(self, 'right_panel_collapsed'):
                    self.right_panel_collapsed = False
                self.right_panel_collapsed = False
                # 恢复布局比例 1:3:2
                if self.root_layout:
                    self.root_layout.setStretch(0, 1)   # 左边：1份
                    self.root_layout.setStretch(1, 3)   # 中间：3份
                    self.root_layout.setStretch(2, 2)   # 右边：2份
            
            # 强制更新布局以确保隐藏/显示生效
            if self.right_panel_widget:
                self.right_panel_widget.update()
                self.right_panel_widget.adjustSize()
    
    def closeEvent(self, event):
        """
        窗口关闭事件 - 正确清理 VTK 资源以避免 OpenGL 错误
        """
        try:
            # 清理碰撞可视化
            if hasattr(self, 'path_ui_ctrl') and self.path_ui_ctrl:
                if hasattr(self.path_ui_ctrl, 'viz_manager') and self.path_ui_ctrl.viz_manager:
                    self.path_ui_ctrl.viz_manager.clear_collision_visualization()
            
            # 清理 VTK 渲染窗口
            if hasattr(self, 'vtk_widget') and self.vtk_widget:
                try:
                    rw = self.vtk_widget.GetRenderWindow()
                    if rw:
                        # 先移除所有渲染器
                        if hasattr(self, 'vtk_renderer') and self.vtk_renderer:
                            self.vtk_renderer.RemoveAllViewProps()
                        # 完成渲染窗口
                        rw.Finalize()
                        # 释放交互器
                        iren = rw.GetInteractor()
                        if iren:
                            iren.TerminateApp()
                except Exception as e:
                    pass  # 忽略清理过程中的错误
            
            # 关闭独立的 3D 重建窗口
            if hasattr(self, 'reconstruction_window') and self.reconstruction_window:
                try:
                    self.reconstruction_window.close()
                except Exception:
                    pass
                    
        except Exception as e:
            pass  # 确保关闭事件不会因为清理错误而失败
        
        event.accept()