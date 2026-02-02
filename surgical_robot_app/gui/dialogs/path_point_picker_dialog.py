"""
路径点选点对话框

提供一个独立的弹窗，集中展示三个切片视图（冠状、矢状、轴向），
用户可以在弹窗内通过两次点击（选择两个不同平面）完成3D点选择。
"""

from typing import Optional, Tuple, Dict, Callable
import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QMessageBox,
    QWidget,
    QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush

try:
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.data_io.data_manager import DataManager
    from surgical_robot_app.geometry.slice_geometry import (
        extract_slice,
        plane_click_to_space_coord,
        merge_two_planes_to_3d,
    )
except ImportError:
    try:
        from utils.logger import get_logger
        from data_io.data_manager import DataManager
        from geometry.slice_geometry import (
            extract_slice,
            plane_click_to_space_coord,
            merge_two_planes_to_3d,
        )
    except ImportError:
        get_logger = None
        DataManager = None
        extract_slice = None
        plane_click_to_space_coord = None
        merge_two_planes_to_3d = None

logger = get_logger(__name__) if get_logger else None

try:
    import cv2
except Exception:
    cv2 = None


class PathPointPickerDialog(QDialog):
    """路径点选点对话框"""
    
    # 信号：当3D点被选中时发出
    point_selected = pyqtSignal(float, float, float)  # x, y, z
    
    def __init__(
        self,
        data_manager: DataManager,
        point_type: str = "waypoint",
        parent=None
    ):
        """
        初始化选点对话框
        
        Args:
            data_manager: 数据管理器
            point_type: 点类型 ('start', 'waypoint', 'end')
            parent: 父窗口
        """
        super().__init__(parent)
        self.data_manager = data_manager
        self.point_type = point_type
        
        # 状态
        self.volume: Optional[np.ndarray] = None
        self.slice_positions: Dict[str, float] = {
            'coronal': 50.0,
            'sagittal': 50.0,
            'axial': 50.0
        }
        self.slice_pick_data: Dict[str, Dict] = {}  # 存储两个平面的点击数据
        self.slice_markers: Dict[str, Tuple[float, float]] = {}  # 存储标记位置
        
        # UI组件
        self.slice_labels: Dict[str, QLabel] = {}
        self.slice_sliders: Dict[str, QSlider] = {}
        
        self._init_ui()
        self._load_volume()
        self._update_all_slices()
    
    def _init_ui(self):
        """初始化UI"""
        self.setWindowTitle(f"Select {self.point_type.capitalize()} Point")
        self.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 标题和说明
        title_label = QLabel(
            f"<h3>Select {self.point_type.capitalize()} Point</h3>"
            "<p>Click on <b>two different slice views</b> to determine a 3D point.</p>"
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 状态提示
        self.status_label = QLabel("Step 1: Click on the first slice view (Coronal, Sagittal, or Axial)")
        self.status_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 三个切片视图（使用网格布局）
        slices_container = QWidget()
        slices_layout = QGridLayout()
        slices_container.setLayout(slices_layout)
        
        # 创建三个切片视图
        for idx, (plane_type, title) in enumerate([
            ("coronal", "Coronal (Y)"),
            ("sagittal", "Sagittal (X)"),
            ("axial", "Axial (Z)")
        ]):
            group = QGroupBox(title)
            group_layout = QVBoxLayout()
            group.setLayout(group_layout)
            
            # 切片显示标签
            label = QLabel("No data")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(250, 200)
            label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
            label.setProperty("plane_type", plane_type)
            label.mousePressEvent = lambda e, pt=plane_type: self._on_slice_click(pt, e)
            self.slice_labels[plane_type] = label
            group_layout.addWidget(label)
            
            # 切片位置滑块
            slider = QSlider(Qt.Horizontal)
            slider.setEnabled(False)
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.valueChanged.connect(lambda v, pt=plane_type: self._on_slider_changed(pt, v))
            self.slice_sliders[plane_type] = slider
            group_layout.addWidget(slider)
            
            # 添加到网格（一行三列）
            slices_layout.addWidget(group, 0, idx)
        
        layout.addWidget(slices_container)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_selection)
        button_layout.addWidget(self.btn_clear)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_cancel)
        
        self.btn_confirm = QPushButton("Confirm")
        self.btn_confirm.clicked.connect(self._confirm_selection)
        self.btn_confirm.setEnabled(False)
        self.btn_confirm.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(self.btn_confirm)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _load_volume(self):
        """加载体数据"""
        self.volume = self.data_manager.get_volume()
        if self.volume is None:
            QMessageBox.warning(self, "No Data", "Please load volume data first.")
            return
        
        # 更新滑块范围
        depth, height, width = self.volume.shape[0], self.volume.shape[1], self.volume.shape[2]
        
        self.slice_sliders['coronal'].setMaximum(max(0, height - 1))
        self.slice_sliders['coronal'].setEnabled(True)
        self.slice_sliders['coronal'].setValue(height // 2)
        
        self.slice_sliders['sagittal'].setMaximum(max(0, width - 1))
        self.slice_sliders['sagittal'].setEnabled(True)
        self.slice_sliders['sagittal'].setValue(width // 2)
        
        self.slice_sliders['axial'].setMaximum(max(0, depth - 1))
        self.slice_sliders['axial'].setEnabled(True)
        self.slice_sliders['axial'].setValue(depth // 2)
        
        # 更新切片位置
        self.slice_positions['coronal'] = (height // 2) / max(1, height - 1) * 100.0
        self.slice_positions['sagittal'] = (width // 2) / max(1, width - 1) * 100.0
        self.slice_positions['axial'] = (depth // 2) / max(1, depth - 1) * 100.0
    
    def _on_slider_changed(self, plane_type: str, value: int):
        """处理滑块变化"""
        if self.volume is None:
            return
        
        depth, height, width = self.volume.shape[0], self.volume.shape[1], self.volume.shape[2]
        
        if plane_type == "coronal":
            self.slice_positions[plane_type] = (value / max(1, height - 1)) * 100.0
        elif plane_type == "sagittal":
            self.slice_positions[plane_type] = (value / max(1, width - 1)) * 100.0
        else:  # axial
            self.slice_positions[plane_type] = (value / max(1, depth - 1)) * 100.0
        
        self._update_slice_display(plane_type)
    
    def _on_slice_click(self, plane_type: str, event):
        """处理切片点击事件"""
        if self.volume is None:
            return
        
        label = self.slice_labels[plane_type]
        pix = label.pixmap()
        if pix is None:
            return
        
        # 获取点击位置
        x = event.x()
        y = event.y()
        
        # 计算图像在label中的偏移
        disp_w = pix.width()
        disp_h = pix.height()
        lab_w = label.width()
        lab_h = label.height()
        off_x = (lab_w - disp_w) / 2.0
        off_y = (lab_h - disp_h) / 2.0
        
        # 检查点击是否在图像范围内
        if x < off_x or y < off_y or x > off_x + disp_w or y > off_y + disp_h:
            return
        
        # 获取切片图像尺寸
        if plane_type == "coronal":
            img_h, img_w = self.volume.shape[1], self.volume.shape[2]
        elif plane_type == "sagittal":
            img_h, img_w = self.volume.shape[0], self.volume.shape[2]
        else:  # axial
            img_h, img_w = self.volume.shape[1], self.volume.shape[0]
        
        # 转换坐标
        sx = img_w / disp_w
        sy = img_h / disp_h
        img_x = (x - off_x) * sx
        img_y = (y - off_y) * sy
        
        # 归一化坐标
        img_x_norm = img_x / img_w
        img_y_norm = img_y / img_h
        
        # 检查是否已经选择了这个平面
        if plane_type in self.slice_pick_data:
            # 如果已经选择，清除并重新选择
            QMessageBox.information(
                self,
                "Plane Already Selected",
                f"You have already selected a point on the {plane_type} plane.\n"
                "Please select a different plane for the second click."
            )
            return
        
        # 保存点击数据
        slice_pos_normalized = self.slice_positions[plane_type]
        self.slice_pick_data[plane_type] = {
            'plane': plane_type,
            'slice_pos': slice_pos_normalized,
            'click': (img_x_norm, img_y_norm),
        }
        
        # 保存标记位置（用于显示）
        self.slice_markers[plane_type] = (img_x, img_y)
        
        # 更新显示
        self._update_slice_display(plane_type)
        
        # 更新状态
        if len(self.slice_pick_data) == 1:
            plane_name = {'coronal': 'Coronal', 'sagittal': 'Sagittal', 'axial': 'Axial'}[plane_type]
            self.status_label.setText(
                f"Step 2: Click on a <b>different</b> slice view (not {plane_name})"
            )
            self.status_label.setStyleSheet("font-weight: bold; color: #ff6600; padding: 5px;")
        elif len(self.slice_pick_data) == 2:
            # 计算3D坐标
            self._calculate_3d_coordinate()
    
    def _calculate_3d_coordinate(self):
        """计算3D坐标（从两个不同平面的点击）"""
        if merge_two_planes_to_3d is None or plane_click_to_space_coord is None:
            QMessageBox.warning(self, "Error", "Geometry tools not available.")
            return
        
        if len(self.slice_pick_data) < 2:
            return
        
        # 获取两个不同平面的数据
        planes = list(self.slice_pick_data.keys())
        if len(planes) < 2:
            return
        
        plane1_type = planes[0]
        plane2_type = planes[1]
        
        data1 = self.slice_pick_data[plane1_type]
        data2 = self.slice_pick_data[plane2_type]
        
        # 转换为空间坐标
        try:
            coords1_2d, coords1_3d = plane_click_to_space_coord(
                plane1_type,
                data1['slice_pos'] / 100.0,
                data1['click'][0],
                data1['click'][1],
            )
            
            coords2_2d, coords2_3d = plane_click_to_space_coord(
                plane2_type,
                data2['slice_pos'] / 100.0,
                data2['click'][0],
                data2['click'][1],
            )
            
            # 合并两个平面的坐标
            coord_3d = merge_two_planes_to_3d(plane1_type, coords1_3d, plane2_type, coords2_3d)
            
            if coord_3d is not None:
                x_final, y_final, z_final = coord_3d
                
                # 更新状态
                self.status_label.setText(
                    f"✅ 3D Point: ({x_final:.2f}, {y_final:.2f}, {z_final:.2f}) - Click 'Confirm' to use this point"
                )
                self.status_label.setStyleSheet("font-weight: bold; color: #00aa00; padding: 5px;")
                
                # 启用确认按钮
                self.btn_confirm.setEnabled(True)
                
                # 在所有三个切片上显示标记
                self._update_all_slice_markers(coord_3d)
                
                # 保存最终坐标
                self.final_3d_coord = coord_3d
            else:
                QMessageBox.warning(self, "Error", "Failed to calculate 3D coordinate.")
        except Exception as e:
            logger.error(f"Error calculating 3D coordinate: {e}") if logger else None
            QMessageBox.warning(self, "Error", f"Failed to calculate 3D coordinate: {str(e)}")
    
    def _update_all_slice_markers(self, coord_3d: Tuple[float, float, float]):
        """在所有三个切片上显示标记"""
        x, y, z = coord_3d
        
        # 为每个平面计算2D坐标
        if 'coronal' in self.slice_labels:
            coronal_2d = (x / 100.0 * self.volume.shape[2], z / 100.0 * self.volume.shape[0])
            self.slice_markers['coronal'] = coronal_2d
        
        if 'sagittal' in self.slice_labels:
            sagittal_2d = (y / 100.0 * self.volume.shape[1], z / 100.0 * self.volume.shape[0])
            self.slice_markers['sagittal'] = sagittal_2d
        
        if 'axial' in self.slice_labels:
            axial_2d = (x / 100.0 * self.volume.shape[2], y / 100.0 * self.volume.shape[1])
            self.slice_markers['axial'] = axial_2d
        
        # 更新所有切片显示
        for plane_type in ['coronal', 'sagittal', 'axial']:
            self._update_slice_display(plane_type)
    
    def _update_slice_display(self, plane_type: str):
        """更新指定切片的显示"""
        if self.volume is None or extract_slice is None:
            return
        
        label = self.slice_labels[plane_type]
        slice_pos_normalized = self.slice_positions[plane_type]
        
        try:
            # 提取切片
            slice_img = extract_slice(self.volume, plane_type, slice_pos_normalized)
            
            # 转换切片图像为显示格式
            if slice_img.ndim == 2:
                # 灰度图像
                if slice_img.dtype != np.uint8:
                    if cv2 is not None:
                        slice_img = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    else:
                        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-8) * 255).astype(np.uint8)
                
                # 转换为RGB
                rgb_img = np.stack([slice_img] * 3, axis=-1)
            elif slice_img.ndim == 3:
                rgb_img = slice_img.copy()
                if cv2 is not None:
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            else:
                label.setText("Unsupported format")
                return
            
            # 绘制标记
            if plane_type in self.slice_markers:
                marker_pos = self.slice_markers[plane_type]
                x, y = marker_pos
                if 0 <= x < rgb_img.shape[1] and 0 <= y < rgb_img.shape[0]:
                    radius = 5
                    y_min = max(0, int(y) - radius)
                    y_max = min(rgb_img.shape[0], int(y) + radius + 1)
                    x_min = max(0, int(x) - radius)
                    x_max = min(rgb_img.shape[1], int(x) + radius + 1)
                    for py in range(y_min, y_max):
                        for px in range(x_min, x_max):
                            if ((px - x)**2 + (py - y)**2) <= radius**2:
                                rgb_img[py, px] = [255, 0, 0]  # 红色标记
            
            # 高亮已选择的平面
            if plane_type in self.slice_pick_data:
                # 添加黄色边框
                rgb_img[0:3, :] = [255, 255, 0]  # 上边框
                rgb_img[-3:, :] = [255, 255, 0]  # 下边框
                rgb_img[:, 0:3] = [255, 255, 0]  # 左边框
                rgb_img[:, -3:] = [255, 255, 0]  # 右边框
            
            # 转换为 QPixmap
            h, w, _ = rgb_img.shape
            qimg = QImage(rgb_img.data, w, h, w * 3, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            
            # 适应标签大小
            if label.width() > 0 and label.height() > 0:
                pix = pix.scaled(
                    label.width(),
                    label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            
            label.setPixmap(pix)
            
            # 更新边框样式
            if plane_type in self.slice_pick_data:
                label.setStyleSheet("border: 3px solid #ffaa00; background-color: #fff8e1;")
            else:
                label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        
        except Exception as e:
            logger.error(f"Error updating slice display: {e}") if logger else None
            import traceback
            traceback.print_exc()
    
    def _update_all_slices(self):
        """更新所有切片显示"""
        for plane_type in ['coronal', 'sagittal', 'axial']:
            self._update_slice_display(plane_type)
    
    def _clear_selection(self):
        """清除选择"""
        self.slice_pick_data.clear()
        self.slice_markers.clear()
        self.final_3d_coord = None
        self.btn_confirm.setEnabled(False)
        self.status_label.setText("Step 1: Click on the first slice view (Coronal, Sagittal, or Axial)")
        self.status_label.setStyleSheet("font-weight: bold; color: #0066cc; padding: 5px;")
        self._update_all_slices()
    
    def _confirm_selection(self):
        """确认选择"""
        if not hasattr(self, 'final_3d_coord') or self.final_3d_coord is None:
            QMessageBox.warning(self, "No Point Selected", "Please select a point first.")
            return
        
        # 发出信号
        x, y, z = self.final_3d_coord
        self.point_selected.emit(x, y, z)
        
        # 关闭对话框
        self.accept()
    
    def get_selected_point(self) -> Optional[Tuple[float, float, float]]:
        """获取选中的点"""
        if hasattr(self, 'final_3d_coord'):
            return self.final_3d_coord
        return None

