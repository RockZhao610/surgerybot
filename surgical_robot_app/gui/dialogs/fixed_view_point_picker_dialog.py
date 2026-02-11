"""
固定视角选点对话框

在对齐模型后，使用固定视角（轴向+冠状）进行选点
"""

from typing import Optional, Tuple
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGridLayout, QMessageBox
)
from PyQt5.QtCore import pyqtSignal, Qt
import numpy as np
import logging

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__) if get_logger else None

try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkCellPicker, vtkCamera
    from vtkmodules.vtkRenderingQt import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    vtkRenderer = None
    vtkCellPicker = None
    vtkCamera = None
    QVTKRenderWindowInteractor = None


class FixedViewPointPickerDialog(QDialog):
    """固定视角选点对话框"""
    
    # 信号：当3D点被选中时发出
    point_selected = pyqtSignal(float, float, float)  # x, y, z
    
    def __init__(
        self,
        vtk_renderer: Optional[vtkRenderer],
        model_actor=None,
        point_type: str = "waypoint",
        parent=None
    ):
        """
        初始化固定视角选点对话框
        
        Args:
            vtk_renderer: VTK渲染器（包含模型）
            model_actor: 模型Actor（可选，用于获取模型数据）
            point_type: 点类型 ('start', 'waypoint', 'end')
            parent: 父窗口
        """
        super().__init__(parent)
        self.vtk_renderer = vtk_renderer
        self.model_actor = model_actor
        self.point_type = point_type
        
        # 选点状态
        self.axial_click: Optional[Tuple[float, float]] = None  # (x, y) 在轴向视图
        self.coronal_click: Optional[Tuple[float, float]] = None  # (x, z) 在冠状视图
        
        # VTK组件（用于两个固定视角）
        self.axial_renderer: Optional[vtkRenderer] = None
        self.coronal_renderer: Optional[vtkRenderer] = None
        self.axial_widget = None
        self.coronal_widget = None
        
        # 点击标记
        self.axial_marker = None
        self.coronal_marker = None
        
        self._init_ui()
        self._setup_fixed_views()
    
    def _init_ui(self):
        """初始化UI"""
        self.setWindowTitle(f"Select {self.point_type.capitalize()} Point (Fixed Views)")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 标题
        title = QLabel(f"Select {self.point_type.capitalize()} Point")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # 说明
        instruction = QLabel(
            "Step 1: Click on Axial view (XY plane) to set (x, y)\n"
            "Step 2: Click on Coronal view (XZ plane) to set (x, z)"
        )
        instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(instruction)
        
        # 两个视角的容器
        views_container = QWidget()
        views_layout = QHBoxLayout()
        views_container.setLayout(views_layout)
        
        # 轴向视图
        axial_container = QWidget()
        axial_layout = QVBoxLayout()
        axial_container.setLayout(axial_layout)
        
        axial_label = QLabel("Axial View (XY Plane)")
        axial_label.setAlignment(Qt.AlignCenter)
        axial_layout.addWidget(axial_label)
        
        if VTK_AVAILABLE and QVTKRenderWindowInteractor:
            try:
                self.axial_widget = QVTKRenderWindowInteractor()
                self.axial_renderer = vtkRenderer()
                self.axial_widget.GetRenderWindow().AddRenderer(self.axial_renderer)
                self.axial_widget.setMinimumSize(400, 400)
                axial_layout.addWidget(self.axial_widget)
                
                # 安装事件过滤器
                self.axial_widget.installEventFilter(self)
            except Exception as e:
                logger.error(f"创建轴向视图时出错: {e}") if logger else None
                axial_placeholder = QLabel("Axial View\n(Not Available)")
                axial_placeholder.setAlignment(Qt.AlignCenter)
                axial_placeholder.setMinimumSize(400, 400)
                axial_layout.addWidget(axial_placeholder)
        else:
            axial_placeholder = QLabel("Axial View\n(VTK Not Available)")
            axial_placeholder.setAlignment(Qt.AlignCenter)
            axial_placeholder.setMinimumSize(400, 400)
            axial_layout.addWidget(axial_placeholder)
        
        views_layout.addWidget(axial_container)
        
        # 冠状视图
        coronal_container = QWidget()
        coronal_layout = QVBoxLayout()
        coronal_container.setLayout(coronal_layout)
        
        coronal_label = QLabel("Coronal View (XZ Plane)")
        coronal_label.setAlignment(Qt.AlignCenter)
        coronal_layout.addWidget(coronal_label)
        
        if VTK_AVAILABLE and QVTKRenderWindowInteractor:
            try:
                self.coronal_widget = QVTKRenderWindowInteractor()
                self.coronal_renderer = vtkRenderer()
                self.coronal_widget.GetRenderWindow().AddRenderer(self.coronal_renderer)
                self.coronal_widget.setMinimumSize(400, 400)
                coronal_layout.addWidget(self.coronal_widget)
                
                # 安装事件过滤器
                self.coronal_widget.installEventFilter(self)
            except Exception as e:
                logger.error(f"创建冠状视图时出错: {e}") if logger else None
                coronal_placeholder = QLabel("Coronal View\n(Not Available)")
                coronal_placeholder.setAlignment(Qt.AlignCenter)
                coronal_placeholder.setMinimumSize(400, 400)
                coronal_layout.addWidget(coronal_placeholder)
        else:
            coronal_placeholder = QLabel("Coronal View\n(VTK Not Available)")
            coronal_placeholder.setAlignment(Qt.AlignCenter)
            coronal_placeholder.setMinimumSize(400, 400)
            coronal_layout.addWidget(coronal_placeholder)
        
        views_layout.addWidget(coronal_container)
        
        layout.addWidget(views_container)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        btn_confirm = QPushButton("Confirm")
        btn_confirm.clicked.connect(self._confirm_selection)
        btn_confirm.setEnabled(False)  # 初始禁用，直到两个视角都点击
        self.btn_confirm = btn_confirm
        
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        button_layout.addWidget(btn_confirm)
        button_layout.addWidget(btn_cancel)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
    
    def _setup_fixed_views(self):
        """设置固定视角"""
        if not self.vtk_renderer:
            return
        
        # 获取模型数据
        if self.model_actor:
            mapper = self.model_actor.GetMapper()
            if mapper:
                poly_data = mapper.GetInput()
                if poly_data:
                    # 复制模型到两个视角
                    self._copy_model_to_views(poly_data)
        
        # 设置轴向视图相机（从上往下看，XY平面）
        if self.axial_renderer:
            camera = self.axial_renderer.GetActiveCamera()
            camera.SetPosition(0, 0, 100)  # 在Z轴上方
            camera.SetFocalPoint(0, 0, 0)  # 看向原点
            camera.SetViewUp(0, 1, 0)  # Y轴向上
            self.axial_renderer.ResetCamera()
            self.axial_renderer.ResetCameraClippingRange()
        
        # 设置冠状视图相机（从前往后看，XZ平面）
        if self.coronal_renderer:
            camera = self.coronal_renderer.GetActiveCamera()
            camera.SetPosition(0, 100, 0)  # 在Y轴前方
            camera.SetFocalPoint(0, 0, 0)  # 看向原点
            camera.SetViewUp(0, 0, 1)  # Z轴向上
            self.coronal_renderer.ResetCamera()
            self.coronal_renderer.ResetCameraClippingRange()
    
    def _copy_model_to_views(self, poly_data):
        """复制模型到两个视角"""
        try:
            from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
            
            # 轴向视图
            if self.axial_renderer:
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(poly_data)
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.3, 0.3)
                self.axial_renderer.AddActor(actor)
            
            # 冠状视图
            if self.coronal_renderer:
                mapper = vtkPolyDataMapper()
                mapper.SetInputData(poly_data)
                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.3, 0.3)
                self.coronal_renderer.AddActor(actor)
        except Exception as e:
            logger.error(f"复制模型到视角时出错: {e}") if logger else None
    
    def eventFilter(self, obj, event):
        """事件过滤器：处理点击事件"""
        from PyQt5.QtCore import QEvent, Qt
        
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                if obj == self.axial_widget:
                    self._handle_axial_click(event.x(), event.y())
                    return True
                elif obj == self.coronal_widget:
                    self._handle_coronal_click(event.x(), event.y())
                    return True
        
        return False
    
    def _handle_axial_click(self, x: int, y: int):
        """处理轴向视图点击"""
        if not self.axial_renderer or not vtkCellPicker:
            return
        
        try:
            picker = vtkCellPicker()
            rw = self.axial_widget.GetRenderWindow()
            if not rw:
                return
            
            # 获取窗口大小
            size = rw.GetSize()
            picker.Pick(x, size[1] - y, 0, self.axial_renderer)  # VTK坐标Y轴翻转
            
            if picker.GetCellId() >= 0:
                world_pos = picker.GetPickPosition()
                # 轴向视图：XY平面，Z由相机位置确定（这里Z=0，在XY平面上）
                self.axial_click = (world_pos[0], world_pos[1])
                self._draw_axial_marker(world_pos[0], world_pos[1])
                self._update_status()
            else:
                QMessageBox.warning(self, "Invalid Click", "Please click on the model surface")
        except Exception as e:
            logger.error(f"处理轴向视图点击时出错: {e}") if logger else None
    
    def _handle_coronal_click(self, x: int, y: int):
        """处理冠状视图点击"""
        if not self.coronal_renderer or not vtkCellPicker:
            return
        
        try:
            picker = vtkCellPicker()
            rw = self.coronal_widget.GetRenderWindow()
            if not rw:
                return
            
            # 获取窗口大小
            size = rw.GetSize()
            picker.Pick(x, size[1] - y, 0, self.coronal_renderer)  # VTK坐标Y轴翻转
            
            if picker.GetCellId() >= 0:
                world_pos = picker.GetPickPosition()
                # 冠状视图：XZ平面，Y由相机位置确定（这里Y=0，在XZ平面上）
                self.coronal_click = (world_pos[0], world_pos[2])
                self._draw_coronal_marker(world_pos[0], world_pos[2])
                self._update_status()
            else:
                QMessageBox.warning(self, "Invalid Click", "Please click on the model surface")
        except Exception as e:
            logger.error(f"处理冠状视图点击时出错: {e}") if logger else None
    
    def _draw_axial_marker(self, x: float, y: float):
        """在轴向视图上绘制标记"""
        # TODO: 实现标记绘制
        pass
    
    def _draw_coronal_marker(self, x: float, z: float):
        """在冠状视图上绘制标记"""
        # TODO: 实现标记绘制
        pass
    
    def _update_status(self):
        """更新状态"""
        if self.axial_click and self.coronal_click:
            self.btn_confirm.setEnabled(True)
    
    def _confirm_selection(self):
        """确认选择"""
        if not self.axial_click or not self.coronal_click:
            QMessageBox.warning(self, "Incomplete Selection", "Please click on both views")
            return
        
        # 合并坐标
        # 轴向视图提供 (x, y)
        # 冠状视图提供 (x, z)
        x_axial, y = self.axial_click
        x_coronal, z = self.coronal_click
        
        # 如果两个X坐标不一致，取平均值
        x = (x_axial + x_coronal) / 2.0
        
        # 转换为空间坐标 [0, 100]
        # TODO: 根据模型边界进行归一化
        space_coord = (x, y, z)
        
        # 发射信号
        self.point_selected.emit(space_coord[0], space_coord[1], space_coord[2])
        self.accept()

