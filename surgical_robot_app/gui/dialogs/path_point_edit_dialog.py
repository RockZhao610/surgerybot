"""
Path Point Edit Dialog

Allows editing individual path points using coordinate planes (XY, XZ).
Shows the entire path for context and provides real-time 3D preview.
"""

from typing import Optional, Tuple, List, Callable
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGridLayout, QSlider, QSpinBox, QGroupBox, QCheckBox
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QMouseEvent, QPolygonF
from PyQt5.QtCore import QPointF
import logging

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__) if get_logger else logging.getLogger(__name__)


class PathPlaneWidget(QWidget):
    """Widget displaying a coordinate plane with path and editable point."""
    
    point_changed = pyqtSignal(float, float)  # Emits normalized coordinates (0-100)
    
    def __init__(self, plane_type: str = "XY", parent=None):
        super().__init__(parent)
        self.plane_type = plane_type
        self.setMinimumSize(280, 280)
        self.setMaximumSize(380, 380)
        
        # Current editable point (normalized 0-100)
        self.current_point: Optional[Tuple[float, float]] = None
        self.original_point: Optional[Tuple[float, float]] = None
        
        # Path points for context (list of (h, v) tuples)
        self.path_points: List[Tuple[float, float]] = []
        
        # Model projection (for reference)
        self.model_points: List[Tuple[float, float]] = []
        
        # Index of the point being edited
        self.edit_index: int = -1
        
        # Collision state
        self.is_colliding = False
        
        # Dragging state
        self.is_dragging = False
        
        # Styling
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        
        # Axis labels
        if plane_type == "XY":
            self.h_label = "X"
            self.v_label = "Y"
        elif plane_type == "XZ":
            self.h_label = "X"
            self.v_label = "Z"
        else:
            self.h_label = "Y"
            self.v_label = "Z"
    
    def set_path(self, path_points: List[Tuple[float, float]], edit_index: int):
        """Set the path points and which one is being edited."""
        self.path_points = path_points
        self.edit_index = edit_index
        if 0 <= edit_index < len(path_points):
            self.current_point = path_points[edit_index]
            self.original_point = path_points[edit_index]
        self.update()
    
    def set_model_projection(self, points: List[Tuple[float, float]]):
        """Set model projection for reference."""
        self.model_points = points
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 35
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        
        # Draw background
        painter.fillRect(0, 0, w, h, QColor("#1a1a2e"))
        
        # Draw grid
        painter.setPen(QPen(QColor("#2d2d44"), 1))
        grid_count = 10
        for i in range(grid_count + 1):
            x = margin + plot_w * i / grid_count
            painter.drawLine(int(x), margin, int(x), h - margin)
            y = margin + plot_h * i / grid_count
            painter.drawLine(margin, int(y), w - margin, int(y))
        
        # Draw model projection (if available)
        if self.model_points:
            # Draw as a point cloud to show hollow structures (similar to picker dialog)
            painter.setPen(QPen(QColor(76, 175, 80, 160), 2))
            for ph, pv in self.model_points:
                sx = margin + plot_w * ph / 100
                sy = h - margin - plot_h * pv / 100
                painter.drawPoint(int(sx), int(sy))
        
        # Draw path line
        if len(self.path_points) >= 2:
            painter.setPen(QPen(QColor("#FF9800"), 2))
            for i in range(len(self.path_points) - 1):
                p1 = self.path_points[i]
                p2 = self.path_points[i + 1]
                x1 = margin + plot_w * p1[0] / 100
                y1 = h - margin - plot_h * p1[1] / 100
                x2 = margin + plot_w * p2[0] / 100
                y2 = h - margin - plot_h * p2[1] / 100
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw path points
        for i, (ph, pv) in enumerate(self.path_points):
            sx = margin + plot_w * ph / 100
            sy = h - margin - plot_h * pv / 100
            
            if i == self.edit_index:
                continue  # Draw the editing point last (on top)
            elif i == 0:
                # Start point - green
                painter.setBrush(QBrush(QColor("#4CAF50")))
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
            elif i == len(self.path_points) - 1:
                # End point - red
                painter.setBrush(QBrush(QColor("#F44336")))
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
            else:
                # Waypoint - orange
                painter.setBrush(QBrush(QColor("#FF9800")))
                painter.setPen(QPen(QColor("#FFFFFF"), 1))
            
            painter.drawEllipse(int(sx) - 5, int(sy) - 5, 10, 10)
        
        # Draw the editing point (larger, highlighted)
        if self.current_point is not None:
            px, py = self.current_point
            screen_x = margin + plot_w * px / 100
            screen_y = h - margin - plot_h * py / 100
            
            # Draw crosshair
            painter.setPen(QPen(QColor("#00BCD4"), 1, Qt.DashLine))
            painter.drawLine(int(screen_x), margin, int(screen_x), h - margin)
            painter.drawLine(margin, int(screen_y), w - margin, int(screen_y))
            
            # Draw original position (ghost)
            if self.original_point:
                ox, oy = self.original_point
                osx = margin + plot_w * ox / 100
                osy = h - margin - plot_h * oy / 100
                painter.setBrush(QBrush(QColor(0, 188, 212, 80)))
                painter.setPen(QPen(QColor("#00BCD4"), 1, Qt.DashLine))
                painter.drawEllipse(int(osx) - 6, int(osy) - 6, 12, 12)
            
            # Draw current editing point
            point_color = QColor("#F44336") if self.is_colliding else QColor("#00BCD4")
            painter.setBrush(QBrush(point_color))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawEllipse(int(screen_x) - 8, int(screen_y) - 8, 16, 16)
        
        # Draw axes
        painter.setPen(QPen(QColor("#1E88E5"), 2))
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        painter.drawLine(margin, margin, margin, h - margin)
        
        # Draw axis labels
        painter.setPen(QColor("#FFFFFF"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(w - margin + 5, h - margin + 5, self.h_label)
        painter.drawText(margin - 10, margin - 5, self.v_label)
        
        # Draw axis ticks
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor("#888888"))
        for i in range(0, 101, 20):
            x = margin + plot_w * i / 100
            painter.drawText(int(x) - 10, h - margin + 15, str(i))
            y = h - margin - plot_h * i / 100
            painter.drawText(margin - 28, int(y) + 5, str(i))
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._update_point_from_mouse(event)
            self.is_dragging = True
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self._update_point_from_mouse(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        self.is_dragging = False
    
    def _update_point_from_mouse(self, event: QMouseEvent):
        w, h = self.width(), self.height()
        margin = 35
        plot_w = w - 2 * margin
        plot_h = h - 2 * margin
        
        x = event.x()
        y = event.y()
        
        if margin <= x <= w - margin and margin <= y <= h - margin:
            norm_x = (x - margin) / plot_w * 100
            norm_y = (h - margin - y) / plot_h * 100
            
            norm_x = max(0, min(100, norm_x))
            norm_y = max(0, min(100, norm_y))
            
            self.current_point = (norm_x, norm_y)
            self.point_changed.emit(norm_x, norm_y)
            self.update()
    
    def set_point(self, h_val: float, v_val: float):
        """Set the current point programmatically."""
        self.current_point = (h_val, v_val)
        self.update()


class PathPointEditDialog(QDialog):
    """Dialog for editing a path point using coordinate planes."""
    
    point_updated = pyqtSignal(int, float, float, float)  # index, x, y, z
    point_deleted = pyqtSignal(int)  # index
    preview_requested = pyqtSignal(float, float, float)  # x, y, z for real-time preview
    
    def __init__(
        self,
        point_index: int,
        current_coords: Tuple[float, float, float],
        all_path_points: List[Tuple[float, float, float]],
        model_polydata=None,
        model_bounds=None,
        collision_checker=None,
        parent=None
    ):
        super().__init__(parent)
        self.point_index = point_index
        self.original_coords = current_coords
        self.all_path_points = all_path_points
        self.model_polydata = model_polydata
        self.model_bounds = model_bounds  # 模型边界，用于坐标转换
        self.collision_checker = collision_checker
        
        # Current coordinates
        self.coord_x = current_coords[0]
        self.coord_y = current_coords[1]
        self.coord_z = current_coords[2]
        
        self._init_ui()
        self._connect_signals()
        self._setup_path_display()
    
    def _init_ui(self):
        self.setWindowTitle(f"Edit Path Point #{self.point_index + 1}")
        self.setMinimumSize(800, 580)
        self.setStyleSheet("""
            QDialog { background-color: #F5F7FA; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #E0E4E8;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 5px;
                color: #1E88E5;
            }
            QLabel { color: #2C3E50; }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton#secondary_btn {
                background-color: transparent;
                color: #1E88E5;
                border: 1px solid #1E88E5;
            }
            QPushButton#secondary_btn:hover { background-color: #E3F2FD; }
            QPushButton#danger_btn {
                background-color: #F44336;
                color: white;
                border: none;
            }
            QPushButton#danger_btn:hover { background-color: #D32F2F; }
            QSlider::handle:horizontal {
                background: #1E88E5;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #E0E4E8;
                border-radius: 4px;
            }
            QSpinBox {
                padding: 5px;
                border: 1px solid #E0E4E8;
                border-radius: 4px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Title
        point_type = "Start" if self.point_index == 0 else ("End" if self.point_index == len(self.all_path_points) - 1 else f"Waypoint")
        title = QLabel(f"Edit {point_type} Point (#{self.point_index + 1})")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1E88E5;")
        layout.addWidget(title)
        
        # Instruction
        instruction = QLabel("Drag the cyan point in each plane to adjust position. Changes preview in real-time.")
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("color: #7F8C8D; font-size: 11px;")
        layout.addWidget(instruction)
        
        # Plane views
        planes_layout = QHBoxLayout()
        
        # Axial View (XY)
        axial_group = QGroupBox("Axial View (XY) - Top")
        axial_layout = QVBoxLayout(axial_group)
        self.axial_plane = PathPlaneWidget("XY")
        axial_layout.addWidget(self.axial_plane, alignment=Qt.AlignCenter)
        planes_layout.addWidget(axial_group)
        
        # Coronal View (XZ)
        coronal_group = QGroupBox("Coronal View (XZ) - Front")
        coronal_layout = QVBoxLayout(coronal_group)
        self.coronal_plane = PathPlaneWidget("XZ")
        coronal_layout.addWidget(self.coronal_plane, alignment=Qt.AlignCenter)
        planes_layout.addWidget(coronal_group)
        
        layout.addLayout(planes_layout)
        
        # Coordinate fine adjustment
        coord_group = QGroupBox("Fine Adjustment")
        coord_layout = QGridLayout(coord_group)
        coord_layout.setSpacing(8)
        
        for row, (label, attr) in enumerate([("X:", "x"), ("Y:", "y"), ("Z:", "z")]):
            coord_layout.addWidget(QLabel(label), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(getattr(self, f"coord_{attr}")))
            setattr(self, f"slider_{attr}", slider)
            coord_layout.addWidget(slider, row, 1)
            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setValue(int(getattr(self, f"coord_{attr}")))
            setattr(self, f"spin_{attr}", spin)
            coord_layout.addWidget(spin, row, 2)
        
        layout.addWidget(coord_group)
        
        # Collision Warning Label
        self.collision_warning = QLabel("")
        self.collision_warning.setAlignment(Qt.AlignCenter)
        self.collision_warning.setStyleSheet("color: #F44336; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.collision_warning)
        
        # Info display
        info_layout = QHBoxLayout()
        self.original_label = QLabel(f"Original: ({self.original_coords[0]:.1f}, {self.original_coords[1]:.1f}, {self.original_coords[2]:.1f})")
        self.original_label.setStyleSheet("color: #7F8C8D;")
        info_layout.addWidget(self.original_label)
        info_layout.addStretch()
        self.current_label = QLabel(f"Current: ({self.coord_x:.1f}, {self.coord_y:.1f}, {self.coord_z:.1f})")
        self.current_label.setStyleSheet("font-weight: bold; color: #00BCD4;")
        info_layout.addWidget(self.current_label)
        layout.addLayout(info_layout)
        
        # Collision check
        self.collision_check = QCheckBox("Check collision before applying")
        self.collision_check.setChecked(True)
        layout.addWidget(self.collision_check)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("secondary_btn")
        self.btn_reset.setFixedWidth(80)
        btn_layout.addWidget(self.btn_reset)
        
        btn_layout.addStretch()
        
        # Only show delete for waypoints (not start/end)
        if 0 < self.point_index < len(self.all_path_points) - 1:
            self.btn_delete = QPushButton("Delete Point")
            self.btn_delete.setObjectName("danger_btn")
            self.btn_delete.setFixedWidth(100)
            btn_layout.addWidget(self.btn_delete)
        
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.setFixedWidth(100)
        btn_layout.addWidget(self.btn_apply)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondary_btn")
        self.btn_cancel.setFixedWidth(80)
        btn_layout.addWidget(self.btn_cancel)
        
        layout.addLayout(btn_layout)
    
    def _connect_signals(self):
        # Plane clicks
        self.axial_plane.point_changed.connect(self._on_axial_change)
        self.coronal_plane.point_changed.connect(self._on_coronal_change)
        
        # Sliders
        self.slider_x.valueChanged.connect(self._on_slider_x)
        self.slider_y.valueChanged.connect(self._on_slider_y)
        self.slider_z.valueChanged.connect(self._on_slider_z)
        
        # Spinboxes
        self.spin_x.valueChanged.connect(self._on_spin_x)
        self.spin_y.valueChanged.connect(self._on_spin_y)
        self.spin_z.valueChanged.connect(self._on_spin_z)
        
        # Buttons
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_cancel.clicked.connect(self.reject)
        
        if hasattr(self, 'btn_delete'):
            self.btn_delete.clicked.connect(self._on_delete)
    
    def _setup_path_display(self):
        """Setup path display in plane widgets."""
        # Convert 3D path to 2D projections
        xy_points = [(p[0], p[1]) for p in self.all_path_points]
        xz_points = [(p[0], p[2]) for p in self.all_path_points]
        
        self.axial_plane.set_path(xy_points, self.point_index)
        self.coronal_plane.set_path(xz_points, self.point_index)
        
        # Set model projection if available
        if self.model_polydata is not None:
            self._set_model_projections()
    
    def _set_model_projections(self):
        """Extract and set model projections using the same coordinate system as path points."""
        try:
            import numpy as np
            from surgical_robot_app.vtk_utils.coords import world_to_space
            
            points = self.model_polydata.GetPoints()
            if points is None:
                return
            
            n_points = points.GetNumberOfPoints()
            if n_points == 0:
                return
            
            # 采样点（增加采样点数以显示更细致的结构，如中空）
            step = max(1, n_points // 4000)
            all_pts = []
            for i in range(0, n_points, step):
                all_pts.append(points.GetPoint(i))
            all_pts = np.array(all_pts)
            
            # 使用与路径点相同的坐标系统进行转换
            if self.model_bounds is not None:
                bounds = self.model_bounds
            else:
                bounds = self.model_polydata.GetBounds()
                if bounds and len(bounds) >= 6:
                    bounds = (bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5])
                else:
                    return
            
            # 转换世界坐标到空间坐标（0-100）
            space_coords = []
            for pt in all_pts:
                space_pt = world_to_space(bounds, (float(pt[0]), float(pt[1]), float(pt[2])))
                space_coords.append(space_pt)
            space_coords = np.array(space_coords)
            
            # 提取 XY 和 XZ 投影 (直接使用点投影，不计算凸包)
            norm_x = space_coords[:, 0]
            norm_y = space_coords[:, 1]
            norm_z = space_coords[:, 2]
            
            xy_pts = [(float(norm_x[i]), float(norm_y[i])) for i in range(len(norm_x))]
            xz_pts = [(float(norm_x[i]), float(norm_z[i])) for i in range(len(norm_x))]
            
            self.axial_plane.set_model_projection(xy_pts)
            self.coronal_plane.set_model_projection(xz_pts)
            logger.info(f"模型点云投影已设置: {len(xy_pts)} 点")
        except Exception as e:
            logger.warning(f"Error setting model projections: {e}", exc_info=True)
    
    def _on_axial_change(self, x: float, y: float):
        self.coord_x = x
        self.coord_y = y
        self._update_controls()
        self._update_planes()
        self._emit_preview()
    
    def _on_coronal_change(self, x: float, z: float):
        self.coord_x = x
        self.coord_z = z
        self._update_controls()
        self._update_planes()
        self._emit_preview()
    
    def _on_slider_x(self, value):
        self.coord_x = float(value)
        self.spin_x.blockSignals(True)
        self.spin_x.setValue(value)
        self.spin_x.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _on_slider_y(self, value):
        self.coord_y = float(value)
        self.spin_y.blockSignals(True)
        self.spin_y.setValue(value)
        self.spin_y.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _on_slider_z(self, value):
        self.coord_z = float(value)
        self.spin_z.blockSignals(True)
        self.spin_z.setValue(value)
        self.spin_z.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _on_spin_x(self, value):
        self.coord_x = float(value)
        self.slider_x.blockSignals(True)
        self.slider_x.setValue(value)
        self.slider_x.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _on_spin_y(self, value):
        self.coord_y = float(value)
        self.slider_y.blockSignals(True)
        self.slider_y.setValue(value)
        self.slider_y.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _on_spin_z(self, value):
        self.coord_z = float(value)
        self.slider_z.blockSignals(True)
        self.slider_z.setValue(value)
        self.slider_z.blockSignals(False)
        self._update_display()
        self._update_planes()
        self._emit_preview()
    
    def _update_controls(self):
        """Update sliders and spinboxes."""
        for widget in [self.slider_x, self.slider_y, self.slider_z,
                       self.spin_x, self.spin_y, self.spin_z]:
            widget.blockSignals(True)
        
        self.slider_x.setValue(int(self.coord_x))
        self.slider_y.setValue(int(self.coord_y))
        self.slider_z.setValue(int(self.coord_z))
        self.spin_x.setValue(int(self.coord_x))
        self.spin_y.setValue(int(self.coord_y))
        self.spin_z.setValue(int(self.coord_z))
        
        for widget in [self.slider_x, self.slider_y, self.slider_z,
                       self.spin_x, self.spin_y, self.spin_z]:
            widget.blockSignals(False)
        
        self._update_display()
    
    def _update_planes(self):
        """Update plane widgets."""
        self.axial_plane.set_point(self.coord_x, self.coord_y)
        self.coronal_plane.set_point(self.coord_x, self.coord_z)
    
    def _update_display(self):
        """Update the current coordinate display and check for collisions (point and path segments)."""
        self.current_label.setText(f"Current: ({self.coord_x:.1f}, {self.coord_y:.1f}, {self.coord_z:.1f})")
        
        # Real-time collision check
        point_colliding = False
        path_colliding = False
        current_pos = (self.coord_x, self.coord_y, self.coord_z)
        
        if self.collision_checker:
            try:
                # 1. 检查点本身是否碰撞
                point_colliding = self.collision_checker.is_collision(current_pos)
                
                # 2. 检查与邻居点的连线是否碰撞
                if not point_colliding:
                    # 检查与前一个点的连线
                    if self.point_index > 0:
                        prev_point = self.all_path_points[self.point_index - 1]
                        if not self.collision_checker.is_path_collision_free(prev_point, current_pos):
                            path_colliding = True
                    
                    # 检查与后一个点的连线
                    if not path_colliding and self.point_index < len(self.all_path_points) - 1:
                        next_point = self.all_path_points[self.point_index + 1]
                        if not self.collision_checker.is_path_collision_free(current_pos, next_point):
                            path_colliding = True
            except Exception as e:
                logger.warning(f"Collision check failed: {e}")
        
        # Update UI based on collision state
        is_any_collision = point_colliding or path_colliding
        self.axial_plane.is_colliding = is_any_collision
        self.coronal_plane.is_colliding = is_any_collision
        
        # 更新全路径点列表中的当前点，以便绘图更新
        self.all_path_points[self.point_index] = current_pos
        self.axial_plane.set_path([(p[0], p[1]) for p in self.all_path_points], self.point_index)
        self.coronal_plane.set_path([(p[0], p[2]) for p in self.all_path_points], self.point_index)
        
        if point_colliding:
            self.collision_warning.setText("⚠️ WARNING: Point is inside an obstacle!")
            self.current_label.setStyleSheet("font-weight: bold; color: #F44336;")
        elif path_colliding:
            self.collision_warning.setText("⚠️ WARNING: Path segment crosses an obstacle!")
            self.current_label.setStyleSheet("font-weight: bold; color: #FF9800;") # 橙色警告
        else:
            self.collision_warning.setText("")
            self.current_label.setStyleSheet("font-weight: bold; color: #00BCD4;")
        
        self.axial_plane.update()
        self.coronal_plane.update()
    
    def _emit_preview(self):
        """Emit preview signal for real-time 3D update."""
        self.preview_requested.emit(self.coord_x, self.coord_y, self.coord_z)
    
    def _on_reset(self):
        """Reset to original position."""
        self.coord_x, self.coord_y, self.coord_z = self.original_coords
        self._update_controls()
        self._update_planes()
        self._emit_preview()
    
    def _on_apply(self):
        """Apply the changes."""
        # Check collision if enabled
        if self.collision_check.isChecked() and self.collision_checker:
            new_pos = (self.coord_x, self.coord_y, self.coord_z)
            
            # 1. Check point collision
            if self.collision_checker.is_collision(new_pos):
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    "Collision Detected",
                    "The new position collides with an obstacle.\n"
                    "Please adjust the position or uncheck collision detection."
                )
                return
            
            # 2. Check path segment feasibility (before and after the point)
            # Before the point
            if self.point_index > 0:
                prev_pt = self.all_path_points[self.point_index - 1]
                if not self.collision_checker.is_path_collision_free(prev_pt, new_pos):
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Path Feasibility Error",
                        "The path segment BEFORE this point now intersects an obstacle.\n"
                        "Please adjust the position."
                    )
                    return
            
            # After the point
            if self.point_index < len(self.all_path_points) - 1:
                next_pt = self.all_path_points[self.point_index + 1]
                if not self.collision_checker.is_path_collision_free(new_pos, next_pt):
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Path Feasibility Error",
                        "The path segment AFTER this point now intersects an obstacle.\n"
                        "Please adjust the position."
                    )
                    return
        
        self.point_updated.emit(self.point_index, self.coord_x, self.coord_y, self.coord_z)
        self.accept()
    
    def _on_delete(self):
        """Delete this path point."""
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Delete Point",
            f"Are you sure you want to delete path point #{self.point_index + 1}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.point_deleted.emit(self.point_index)
            self.accept()
    
    def get_coordinates(self) -> Tuple[float, float, float]:
        """Return current coordinates."""
        return (self.coord_x, self.coord_y, self.coord_z)

