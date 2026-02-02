"""
Coordinate Plane Point Picker Dialog

For STL model imports (no volume data), allows point selection 
using coordinate planes (XY, XZ planes) to determine 3D coordinates.
Displays model projection in each plane for reference.
"""

from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QWidget, QGridLayout, QSlider, QSpinBox, QGroupBox, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QMouseEvent, QPolygonF, QPainterPath
from PyQt5.QtCore import QPointF
import logging
import numpy as np

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__) if get_logger else logging.getLogger(__name__)


class CoordinatePlaneWidget(QWidget):
    """A widget that displays a coordinate plane with model projection and allows clicking to select a point."""
    
    point_clicked = pyqtSignal(float, float)  # Emits normalized coordinates (0-100)
    
    def __init__(self, plane_type: str = "XY", parent=None):
        """
        Args:
            plane_type: "XY" (Axial), "XZ" (Coronal), or "YZ" (Sagittal)
        """
        super().__init__(parent)
        self.plane_type = plane_type
        self.setMinimumSize(300, 300)
        self.setMaximumSize(400, 400)
        
        # Selected point (normalized 0-100)
        self.selected_point: Optional[Tuple[float, float]] = None
        
        # Model projection points (normalized 0-100)
        self.model_points: List[Tuple[float, float]] = []
        
        # Model bounds for display (normalized 0-100)
        self.model_bounds_2d: Optional[Tuple[float, float, float, float]] = None  # (min_h, max_h, min_v, max_v)
        
        # Styling
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        
        # Axis labels based on plane type
        if plane_type == "XY":
            self.h_label = "X"
            self.v_label = "Y"
        elif plane_type == "XZ":
            self.h_label = "X"
            self.v_label = "Z"
        else:  # YZ
            self.h_label = "Y"
            self.v_label = "Z"
    
    def set_model_projection(self, points: List[Tuple[float, float]], bounds_2d: Optional[Tuple[float, float, float, float]] = None):
        """Set the model projection points to display.
        
        Args:
            points: List of (h, v) points in normalized coordinates (0-100)
            bounds_2d: Optional (min_h, max_h, min_v, max_v) for bounding box
        """
        self.model_points = points
        self.model_bounds_2d = bounds_2d
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
            # Vertical lines
            x = margin + plot_w * i / grid_count
            painter.drawLine(int(x), margin, int(x), h - margin)
            # Horizontal lines
            y = margin + plot_h * i / grid_count
            painter.drawLine(margin, int(y), w - margin, int(y))
        
        # Draw model projection (before axes so it's behind)
        if self.model_points:
            self._draw_model_projection(painter, margin, plot_w, plot_h, w, h)
        
        # Draw model bounding box
        if self.model_bounds_2d:
            self._draw_model_bounds(painter, margin, plot_w, plot_h, w, h)
        
        # Draw axes
        painter.setPen(QPen(QColor("#1E88E5"), 2))
        # Horizontal axis
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        # Vertical axis
        painter.drawLine(margin, margin, margin, h - margin)
        
        # Draw axis labels
        painter.setPen(QColor("#FFFFFF"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(w - margin + 5, h - margin + 5, self.h_label)
        painter.drawText(margin - 10, margin - 5, self.v_label)
        
        # Draw axis ticks and values
        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor("#888888"))
        for i in range(0, 101, 20):
            # Horizontal axis ticks
            x = margin + plot_w * i / 100
            painter.drawText(int(x) - 10, h - margin + 15, str(i))
            # Vertical axis ticks
            y = h - margin - plot_h * i / 100
            painter.drawText(margin - 28, int(y) + 5, str(i))
        
        # Draw selected point (on top)
        if self.selected_point is not None:
            px, py = self.selected_point
            # Convert to screen coordinates
            screen_x = margin + plot_w * px / 100
            screen_y = h - margin - plot_h * py / 100
            
            # Draw crosshair
            painter.setPen(QPen(QColor("#FF5722"), 1, Qt.DashLine))
            painter.drawLine(int(screen_x), margin, int(screen_x), h - margin)
            painter.drawLine(margin, int(screen_y), w - margin, int(screen_y))
            
            # Draw point
            painter.setBrush(QBrush(QColor("#FF5722")))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.drawEllipse(int(screen_x) - 7, int(screen_y) - 7, 14, 14)
    
    def _draw_model_projection(self, painter: QPainter, margin: int, plot_w: int, plot_h: int, w: int, h: int):
        """Draw the model projection points as a cloud of points."""
        if not self.model_points:
            return
            
        # Draw as a dense point cloud to show hollow structures
        # Use a slightly transparent green for a better look
        painter.setPen(QPen(QColor(76, 175, 80, 180), 2)) # Higher opacity for points
        
        for ph, pv in self.model_points:
            sx = margin + plot_w * ph / 100
            sy = h - margin - plot_h * pv / 100
            painter.drawPoint(int(sx), int(sy))
    
    def _draw_model_bounds(self, painter: QPainter, margin: int, plot_w: int, plot_h: int, w: int, h: int):
        """Draw the model bounding box."""
        min_h, max_h, min_v, max_v = self.model_bounds_2d
        
        # Convert to screen coordinates
        x1 = margin + plot_w * min_h / 100
        x2 = margin + plot_w * max_h / 100
        y1 = h - margin - plot_h * max_v / 100  # Note: y is inverted
        y2 = h - margin - plot_h * min_v / 100
        
        # Draw bounding box
        painter.setPen(QPen(QColor("#FFC107"), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            w, h = self.width(), self.height()
            margin = 35
            plot_w = w - 2 * margin
            plot_h = h - 2 * margin
            
            # Convert screen coordinates to normalized (0-100)
            x = event.x()
            y = event.y()
            
            # Check if click is within the plot area
            if margin <= x <= w - margin and margin <= y <= h - margin:
                norm_x = (x - margin) / plot_w * 100
                norm_y = (h - margin - y) / plot_h * 100
                
                # Clamp to valid range
                norm_x = max(0, min(100, norm_x))
                norm_y = max(0, min(100, norm_y))
                
                self.selected_point = (norm_x, norm_y)
                self.point_clicked.emit(norm_x, norm_y)
                self.update()
    
    def set_point(self, h_val: float, v_val: float):
        """Set the selected point programmatically."""
        self.selected_point = (h_val, v_val)
        self.update()
    
    def clear_point(self):
        """Clear the selected point."""
        self.selected_point = None
        self.update()


class CoordinatePlanePickerDialog(QDialog):
    """Dialog for selecting 3D points using coordinate planes with model projection."""
    
    point_selected = pyqtSignal(float, float, float)  # x, y, z (space coordinates 0-100)
    coordinates_changed = pyqtSignal(float, float, float)  # Real-time coordinate updates for preview
    
    def __init__(self, point_type: str = "waypoint", model_bounds=None, model_polydata=None, parent=None):
        """
        Args:
            point_type: 'start', 'waypoint', or 'end'
            model_bounds: Optional tuple (xmin, xmax, ymin, ymax, zmin, zmax) for reference
            model_polydata: Optional VTK PolyData for model projection
            parent: Parent widget
        """
        super().__init__(parent)
        self.point_type = point_type
        self.model_bounds = model_bounds
        self.model_polydata = model_polydata
        
        # Current coordinates (space coordinates 0-100)
        self.coord_x = 50.0
        self.coord_y = 50.0
        self.coord_z = 50.0
        
        self._init_ui()
        self._connect_signals()
        
        # Set model projections if polydata is available
        if model_polydata is not None:
            self._set_model_projections()
        elif model_bounds is not None:
            self._set_bounds_only()
    
    def _init_ui(self):
        self.setWindowTitle(f"Select {self.point_type.capitalize()} Point")
        self.setMinimumSize(850, 600)
        self.setStyleSheet("""
            QDialog {
                background-color: #F5F7FA;
            }
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
            QLabel {
                color: #2C3E50;
            }
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#secondary_btn {
                background-color: transparent;
                color: #1E88E5;
                border: 1px solid #1E88E5;
            }
            QPushButton#secondary_btn:hover {
                background-color: #E3F2FD;
            }
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
        layout.setSpacing(15)
        
        # Title
        title = QLabel(f"Select {self.point_type.capitalize()} Point")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1E88E5; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instruction = QLabel(
            "Click on each plane to set coordinates. The green area shows the model projection.\n"
            "• Axial View (XY): Sets X and Y coordinates\n"
            "• Coronal View (XZ): Sets X and Z coordinates"
        )
        instruction.setAlignment(Qt.AlignCenter)
        instruction.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        layout.addWidget(instruction)
        
        # Plane views
        planes_layout = QHBoxLayout()
        
        # Axial View (XY Plane)
        axial_group = QGroupBox("Axial View (XY Plane) - Top View")
        axial_layout = QVBoxLayout(axial_group)
        self.axial_plane = CoordinatePlaneWidget("XY")
        axial_layout.addWidget(self.axial_plane, alignment=Qt.AlignCenter)
        planes_layout.addWidget(axial_group)
        
        # Coronal View (XZ Plane)
        coronal_group = QGroupBox("Coronal View (XZ Plane) - Front View")
        coronal_layout = QVBoxLayout(coronal_group)
        self.coronal_plane = CoordinatePlaneWidget("XZ")
        coronal_layout.addWidget(self.coronal_plane, alignment=Qt.AlignCenter)
        planes_layout.addWidget(coronal_group)
        
        layout.addLayout(planes_layout)
        
        # Coordinate display and fine adjustment
        coord_group = QGroupBox("Coordinate Fine Adjustment")
        coord_layout = QGridLayout(coord_group)
        coord_layout.setSpacing(10)
        
        # X coordinate
        coord_layout.addWidget(QLabel("X:"), 0, 0)
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(0, 100)
        self.slider_x.setValue(50)
        coord_layout.addWidget(self.slider_x, 0, 1)
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 100)
        self.spin_x.setValue(50)
        coord_layout.addWidget(self.spin_x, 0, 2)
        
        # Y coordinate
        coord_layout.addWidget(QLabel("Y:"), 1, 0)
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 100)
        self.slider_y.setValue(50)
        coord_layout.addWidget(self.slider_y, 1, 1)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 100)
        self.spin_y.setValue(50)
        coord_layout.addWidget(self.spin_y, 1, 2)
        
        # Z coordinate
        coord_layout.addWidget(QLabel("Z:"), 2, 0)
        self.slider_z = QSlider(Qt.Horizontal)
        self.slider_z.setRange(0, 100)
        self.slider_z.setValue(50)
        coord_layout.addWidget(self.slider_z, 2, 1)
        self.spin_z = QSpinBox()
        self.spin_z.setRange(0, 100)
        self.spin_z.setValue(50)
        coord_layout.addWidget(self.spin_z, 2, 2)
        
        layout.addWidget(coord_group)
        
        # Result display
        self.result_label = QLabel("Selected: (50.0, 50.0, 50.0)")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2C3E50; "
            "background-color: white; padding: 10px; border-radius: 4px;"
        )
        layout.addWidget(self.result_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.setFixedSize(120, 40)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("secondary_btn")
        self.btn_cancel.setFixedSize(120, 40)
        
        btn_layout.addWidget(self.btn_confirm)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
    
    def _set_model_projections(self):
        """Extract and set model projections from PolyData."""
        if self.model_polydata is None:
            return
        
        try:
            # Get points from PolyData
            points = self.model_polydata.GetPoints()
            if points is None:
                logger.warning("PolyData has no points")
                return
            
            n_points = points.GetNumberOfPoints()
            if n_points == 0:
                logger.warning("PolyData has 0 points")
                return
            
            # Extract all points
            all_points = []
            for i in range(n_points):
                pt = points.GetPoint(i)
                all_points.append(pt)
            
            all_points = np.array(all_points)
            
            # Get bounds
            if self.model_bounds is None:
                xmin, xmax = all_points[:, 0].min(), all_points[:, 0].max()
                ymin, ymax = all_points[:, 1].min(), all_points[:, 1].max()
                zmin, zmax = all_points[:, 2].min(), all_points[:, 2].max()
                self.model_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = self.model_bounds
            
            # Add small padding to avoid division by zero
            x_range = max(xmax - xmin, 0.001)
            y_range = max(ymax - ymin, 0.001)
            z_range = max(zmax - zmin, 0.001)
            
            # Normalize points to 0-100 range
            norm_x = (all_points[:, 0] - xmin) / x_range * 100
            norm_y = (all_points[:, 1] - ymin) / y_range * 100
            norm_z = (all_points[:, 2] - zmin) / z_range * 100
            
            # Sample points for projection (more points to show structure)
            max_points = 3000
            if n_points > max_points:
                indices = np.random.choice(n_points, max_points, replace=False)
                norm_x = norm_x[indices]
                norm_y = norm_y[indices]
                norm_z = norm_z[indices]
            
            # Use direct point projection instead of convex hull to show hollow structures
            xy_points = [(float(norm_x[i]), float(norm_y[i])) for i in range(len(norm_x))]
            xz_points = [(float(norm_x[i]), float(norm_z[i])) for i in range(len(norm_x))]
            
            # Set projections
            self.axial_plane.set_model_projection(
                xy_points,
                (0, 100, 0, 100)  # Bounds in normalized space
            )
            self.coronal_plane.set_model_projection(
                xz_points,
                (0, 100, 0, 100)
            )
            
            logger.info(f"Set model point cloud projections: {len(xy_points)} points")
            
        except Exception as e:
            logger.error(f"Error setting model projections: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_convex_hull_2d(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
        """Compute 2D convex hull of points."""
        try:
            from scipy.spatial import ConvexHull
            points = np.column_stack([x, y])
            
            # Need at least 3 points for convex hull
            if len(points) < 3:
                return [(float(x[i]), float(y[i])) for i in range(len(x))]
            
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            
            return [(float(p[0]), float(p[1])) for p in hull_points]
        except ImportError:
            # Fallback if scipy not available - just return bounding box
            return [
                (float(x.min()), float(y.min())),
                (float(x.max()), float(y.min())),
                (float(x.max()), float(y.max())),
                (float(x.min()), float(y.max())),
            ]
        except Exception as e:
            logger.warning(f"Convex hull calculation failed: {e}")
            # Return bounding box as fallback
            return [
                (float(x.min()), float(y.min())),
                (float(x.max()), float(y.min())),
                (float(x.max()), float(y.max())),
                (float(x.min()), float(y.max())),
            ]
    
    def _set_bounds_only(self):
        """Set model bounds rectangle only (when no PolyData available)."""
        if self.model_bounds is None:
            return
        
        xmin, xmax, ymin, ymax, zmin, zmax = self.model_bounds
        
        # Normalize to 0-100
        x_range = max(xmax - xmin, 0.001)
        y_range = max(ymax - ymin, 0.001)
        z_range = max(zmax - zmin, 0.001)
        
        # For display, we'll center the model in the view
        # Just show bounding box
        self.axial_plane.model_bounds_2d = (0, 100, 0, 100)
        self.coronal_plane.model_bounds_2d = (0, 100, 0, 100)
        
        self.axial_plane.update()
        self.coronal_plane.update()
    
    def _connect_signals(self):
        # Plane clicks
        self.axial_plane.point_clicked.connect(self._on_axial_click)
        self.coronal_plane.point_clicked.connect(self._on_coronal_click)
        
        # Sliders
        self.slider_x.valueChanged.connect(self._on_slider_x_changed)
        self.slider_y.valueChanged.connect(self._on_slider_y_changed)
        self.slider_z.valueChanged.connect(self._on_slider_z_changed)
        
        # Spinboxes
        self.spin_x.valueChanged.connect(self._on_spin_x_changed)
        self.spin_y.valueChanged.connect(self._on_spin_y_changed)
        self.spin_z.valueChanged.connect(self._on_spin_z_changed)
        
        # Buttons
        self.btn_confirm.clicked.connect(self._on_confirm)
        self.btn_cancel.clicked.connect(self.reject)
    
    def _on_axial_click(self, x: float, y: float):
        """Handle click on Axial (XY) plane."""
        self.coord_x = x
        self.coord_y = y
        self._update_controls()
        self._update_planes()
    
    def _on_coronal_click(self, x: float, z: float):
        """Handle click on Coronal (XZ) plane."""
        self.coord_x = x  # X is shared between planes
        self.coord_z = z
        self._update_controls()
        self._update_planes()
    
    def _on_slider_x_changed(self, value: int):
        self.coord_x = float(value)
        self.spin_x.blockSignals(True)
        self.spin_x.setValue(value)
        self.spin_x.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _on_slider_y_changed(self, value: int):
        self.coord_y = float(value)
        self.spin_y.blockSignals(True)
        self.spin_y.setValue(value)
        self.spin_y.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _on_slider_z_changed(self, value: int):
        self.coord_z = float(value)
        self.spin_z.blockSignals(True)
        self.spin_z.setValue(value)
        self.spin_z.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _on_spin_x_changed(self, value: int):
        self.coord_x = float(value)
        self.slider_x.blockSignals(True)
        self.slider_x.setValue(value)
        self.slider_x.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _on_spin_y_changed(self, value: int):
        self.coord_y = float(value)
        self.slider_y.blockSignals(True)
        self.slider_y.setValue(value)
        self.slider_y.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _on_spin_z_changed(self, value: int):
        self.coord_z = float(value)
        self.slider_z.blockSignals(True)
        self.slider_z.setValue(value)
        self.slider_z.blockSignals(False)
        self._update_result()
        self._update_planes()
    
    def _update_controls(self):
        """Update sliders and spinboxes from current coordinates."""
        # Block signals to prevent feedback loops
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
        
        self._update_result()
    
    def _update_planes(self):
        """Update plane visualizations."""
        self.axial_plane.set_point(self.coord_x, self.coord_y)
        self.coronal_plane.set_point(self.coord_x, self.coord_z)
    
    def _update_result(self):
        """Update the result label and emit real-time coordinate change."""
        self.result_label.setText(
            f"Selected: ({self.coord_x:.1f}, {self.coord_y:.1f}, {self.coord_z:.1f})"
        )
        # Emit real-time coordinate change for 3D preview
        self.coordinates_changed.emit(self.coord_x, self.coord_y, self.coord_z)
    
    def _on_confirm(self):
        """Confirm and emit the selected point."""
        self.point_selected.emit(self.coord_x, self.coord_y, self.coord_z)
        self.accept()
    
    def get_coordinates(self) -> Tuple[float, float, float]:
        """Return the currently selected coordinates."""
        return (self.coord_x, self.coord_y, self.coord_z)
