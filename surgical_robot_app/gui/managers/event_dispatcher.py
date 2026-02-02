"""
事件分发器

负责：
- 事件过滤和分发
- 将事件路由到相应的控制器
"""

from PyQt5.QtCore import QEvent, Qt
from typing import Optional

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.gui.managers.event_dispatcher")


class EventDispatcher:
    """事件分发器，负责将事件路由到相应的控制器"""
    
    def __init__(
        self,
        main_window=None,
        image_label=None,
        vtk_widget=None,
        slice_editor_ctrl=None,
        sam2_ui_ctrl=None,
        multi_slice_ctrl=None,
        path_ui_ctrl=None,
        point_selection_manager=None
    ):
        """
        初始化事件分发器
        
        Args:
            main_window: 主窗口（用于访问其他组件）
            image_label: 图像标签（用于切片编辑器）
            vtk_widget: VTK 窗口组件
            slice_editor_ctrl: 切片编辑器控制器
            sam2_ui_ctrl: SAM2 UI 控制器
            multi_slice_ctrl: 多视图切片控制器
            path_ui_ctrl: 路径UI控制器
            point_selection_manager: 3D点选择管理器
        """
        self.main_window = main_window
        self.image_label = image_label
        self.vtk_widget = vtk_widget
        self.slice_editor_ctrl = slice_editor_ctrl
        self.sam2_ui_ctrl = sam2_ui_ctrl
        self.multi_slice_ctrl = multi_slice_ctrl
        self.path_ui_ctrl = path_ui_ctrl
        self.point_selection_manager = point_selection_manager
    
    def filter_event(self, obj, event) -> bool:
        """
        过滤并分发事件
        
        Args:
            obj: 事件对象
            event: 事件
        
        Returns:
            bool: 如果事件已被处理，返回 True；否则返回 False
        """
        # 使用 getattr 安全访问控制器（可能在初始化之前被调用）
        slice_editor_ctrl = getattr(self, 'slice_editor_ctrl', None) or self.slice_editor_ctrl
        sam2_ui_ctrl = getattr(self, 'sam2_ui_ctrl', None) or self.sam2_ui_ctrl
        multi_slice_ctrl = getattr(self, 'multi_slice_ctrl', None) or self.multi_slice_ctrl
        path_ui_ctrl = getattr(self, 'path_ui_ctrl', None) or self.path_ui_ctrl
        
        # 处理图像标签的鼠标事件（委托给切片编辑器控制器）
        if obj is self.image_label:
            if slice_editor_ctrl:
                sam2_picking_mode = (sam2_ui_ctrl and 
                                   hasattr(sam2_ui_ctrl, 'sam2_picking_mode') and 
                                   sam2_ui_ctrl.sam2_picking_mode)
                if slice_editor_ctrl.handle_mouse_event(obj, event, sam2_picking_mode):
                    return True
        
        # 处理切片窗口的鼠标点击事件（委托给多视图切片控制器或 PointSelectionManager）
        if hasattr(obj, 'property') and obj.property('plane_type'):
            plane_type = obj.property('plane_type')
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                # 优先使用 PointSelectionManager（如果可用）
                point_selection_manager = getattr(self, 'point_selection_manager', None) or self.point_selection_manager
                if point_selection_manager and self.main_window and getattr(self.main_window, 'slice_pick_mode', False):
                    point_selection_manager.handle_slice_click(plane_type, event.x(), event.y(), obj)
                    return True
                # 回退：使用 multi_slice_ctrl
                elif multi_slice_ctrl and multi_slice_ctrl.slice_pick_mode:
                    multi_slice_ctrl.handle_slice_click(plane_type, event.x(), event.y(), obj)
                    return True
        
        # 处理VTK widget的鼠标事件（委托给路径UI控制器）
        if obj is self.vtk_widget:
            if path_ui_ctrl and path_ui_ctrl.pick_mode is not None:
                if event.type() == QEvent.MouseButtonPress:
                    if event.button() == Qt.LeftButton:
                        pos = event.pos()
                        path_ui_ctrl.handle_vtk_click(pos.x(), pos.y())
                        return True
        
        return False
    
    def update_controllers(
        self,
        slice_editor_ctrl=None,
        sam2_ui_ctrl=None,
        multi_slice_ctrl=None,
        path_ui_ctrl=None
    ):
        """
        更新控制器引用（在控制器初始化后调用）
        
        Args:
            slice_editor_ctrl: 切片编辑器控制器
            sam2_ui_ctrl: SAM2 UI 控制器
            multi_slice_ctrl: 多视图切片控制器
            path_ui_ctrl: 路径UI控制器
        """
        if slice_editor_ctrl is not None:
            self.slice_editor_ctrl = slice_editor_ctrl
        if sam2_ui_ctrl is not None:
            self.sam2_ui_ctrl = sam2_ui_ctrl
        if multi_slice_ctrl is not None:
            self.multi_slice_ctrl = multi_slice_ctrl
        if path_ui_ctrl is not None:
            self.path_ui_ctrl = path_ui_ctrl

