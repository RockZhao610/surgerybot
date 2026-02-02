"""
MultiSliceController: 多视图切片控制器

职责：
- 处理多视图切片相关的事件（切片位置变化、点击、3D坐标计算等）
- 管理切片显示和标记
- 不直接依赖 Qt Widget，通过回调与 MainWindow 通信
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSlider, QListWidget, QMessageBox

try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.gui.controllers.multi_slice_controller")

try:
    import cv2
except Exception:
    cv2 = None

try:
    from surgical_robot_app.data_io.data_manager import DataManager
    from surgical_robot_app.geometry.slice_geometry import (
        extract_slice,
        plane_click_to_space_coord,
        merge_two_planes_to_3d,
    )
except ImportError:
    try:
        from data_io.data_manager import DataManager
        from geometry.slice_geometry import (
            extract_slice,
            plane_click_to_space_coord,
            merge_two_planes_to_3d,
        )
    except ImportError:
        DataManager = None  # type: ignore
        extract_slice = None
        plane_click_to_space_coord = None
        merge_two_planes_to_3d = None


class MultiSliceController:
    """多视图切片控制器"""
    
    def __init__(
        self,
        data_manager: DataManager,
        slice_labels: Dict[str, QLabel],
        slice_sliders: Dict[str, QSlider],
        coord_list: QListWidget,
        parent_widget=None,
    ):
        """
        初始化多视图切片控制器
        
        Args:
            data_manager: 数据管理器
            slice_labels: 切片标签字典 {'coronal': label, 'sagittal': label, 'axial': label}
            slice_sliders: 切片滑块字典 {'coronal': slider, 'sagittal': slider, 'axial': slider}
            coord_list: 3D点列表控件
            parent_widget: 父窗口
        """
        self.data_manager = data_manager
        self.slice_labels = slice_labels
        self.slice_sliders = slice_sliders
        self.coord_list = coord_list
        self.parent_widget = parent_widget
        
        # 状态
        self.slice_view_volume: Optional[np.ndarray] = None
        self.slice_positions: Dict[str, float] = {'coronal': 50.0, 'sagittal': 50.0, 'axial': 50.0}
        self.slice_pick_data: Dict[str, Dict] = {}
        self.slice_markers: Dict[str, List] = {}
        self.slice_pick_mode: bool = False
        self.picked_3d_points: List[Tuple[float, float, float]] = []
        self.computed_3d_points: List[Tuple[float, float, float]] = []
        
        # 回调函数
        self.on_3d_point_added: Optional[Callable] = None
        self.on_path_point_set: Optional[Callable] = None
    
    def set_slice_view_volume(self, volume: Optional[np.ndarray]):
        """设置用于切片显示的 volume"""
        self.slice_view_volume = volume
        if volume is not None:
            # 更新滑块范围
            depth, height, width = volume.shape[0], volume.shape[1], volume.shape[2]
            if 'coronal' in self.slice_sliders:
                self.slice_sliders['coronal'].setMaximum(max(0, height - 1))
                self.slice_sliders['coronal'].setEnabled(True)
            if 'sagittal' in self.slice_sliders:
                self.slice_sliders['sagittal'].setMaximum(max(0, width - 1))
                self.slice_sliders['sagittal'].setEnabled(True)
            if 'axial' in self.slice_sliders:
                self.slice_sliders['axial'].setMaximum(max(0, depth - 1))
                self.slice_sliders['axial'].setEnabled(True)
        else:
            # 禁用所有滑块
            for slider in self.slice_sliders.values():
                slider.setEnabled(False)
                slider.setMaximum(0)
    
    def handle_slice_position_changed(self, plane_type: str, value: int):
        """处理切片位置变化"""
        volume = self.slice_view_volume
        if volume is None:
            volume = self.data_manager.get_volume()
        
        if volume is not None:
            depth, height, width = volume.shape[0], volume.shape[1], volume.shape[2]
            
            if plane_type == "coronal":
                normalized_pos = (value / max(1, height - 1)) * 100.0
                self.slice_positions[plane_type] = normalized_pos
            elif plane_type == "sagittal":
                normalized_pos = (value / max(1, width - 1)) * 100.0
                self.slice_positions[plane_type] = normalized_pos
            else:  # axial
                normalized_pos = (value / max(1, depth - 1)) * 100.0
                self.slice_positions[plane_type] = normalized_pos
        else:
            self.slice_positions[plane_type] = float(value)
        
        # 更新切片显示
        self.update_slice_display(plane_type)
    
    def update_slice_display(self, plane_type: str):
        """更新指定切片的显示"""
        volume = self.slice_view_volume
        if volume is None:
            volume = self.data_manager.get_volume()
        
        if volume is None:
            if plane_type in self.slice_labels:
                self.slice_labels[plane_type].setText("No data")
            return
        
        if extract_slice is None:
            return
        
        # 获取切片位置
        slice_pos_normalized = self.slice_positions[plane_type]
        
        # 提取切片
        try:
            slice_img = extract_slice(volume, plane_type, slice_pos_normalized)
            
            if plane_type not in self.slice_labels:
                return
            
            label = self.slice_labels[plane_type]
            
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
                for marker_pos, marker_color in self.slice_markers[plane_type]:
                    x, y = marker_pos
                    if 0 <= x < rgb_img.shape[1] and 0 <= y < rgb_img.shape[0]:
                        radius = 3
                        y_min = max(0, int(y) - radius)
                        y_max = min(rgb_img.shape[0], int(y) + radius + 1)
                        x_min = max(0, int(x) - radius)
                        x_max = min(rgb_img.shape[1], int(x) + radius + 1)
                        for py in range(y_min, y_max):
                            for px in range(x_min, x_max):
                                if ((px - x)**2 + (py - y)**2) <= radius**2:
                                    rgb_img[py, px] = marker_color
            
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
        
        except Exception as e:
            logger.error(f"Error updating slice display: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_slice_click(self, plane_type: str, x: int, y: int, label_widget: QLabel):
        """处理切片点击事件"""
        if not self.slice_pick_mode:
            return
        
        # 获取标签的显示尺寸
        pix = label_widget.pixmap()
        if pix is None:
            return
        
        disp_w = pix.width()
        disp_h = pix.height()
        lab_w = label_widget.width()
        lab_h = label_widget.height()
        
        # 计算偏移
        off_x = (lab_w - disp_w) / 2.0
        off_y = (lab_h - disp_h) / 2.0
        
        # 检查点击是否在图像范围内
        if x < off_x or y < off_y or x > off_x + disp_w or y > off_y + disp_h:
            return
        
        # 获取切片图像尺寸
        volume = self.slice_view_volume
        if volume is None:
            volume = self.data_manager.get_volume()
        
        if volume is None:
            return
        
        if plane_type == "coronal":
            img_h, img_w = volume.shape[1], volume.shape[2]
        elif plane_type == "sagittal":
            img_h, img_w = volume.shape[0], volume.shape[2]
        else:  # axial
            img_h, img_w = volume.shape[1], volume.shape[0]
        
        # 转换坐标
        sx = img_w / disp_w
        sy = img_h / disp_h
        img_x = (x - off_x) * sx
        img_y = (y - off_y) * sy
        
        # 归一化坐标
        img_x_norm = img_x / img_w
        img_y_norm = img_y / img_h
        
        # 保存点击数据
        slice_pos_normalized = self.slice_positions[plane_type]
        self.slice_pick_data[plane_type] = {
            'plane': plane_type,
            'slice_pos': slice_pos_normalized,
            'click': (img_x_norm, img_y_norm),
        }
        
        # 绘制标记
        self._draw_slice_marker(plane_type, (img_x, img_y), label_widget)
        
        # 如果有两个不同平面的点击，计算3D坐标
        if len(self.slice_pick_data) >= 2:
            self._calculate_3d_coordinate()
    
    def _draw_slice_marker(self, plane_type: str, click_2d: Tuple[float, float], label_widget: QLabel):
        """在切片上绘制标记"""
        # 保存标记
        if plane_type not in self.slice_markers:
            self.slice_markers[plane_type] = []
        
        marker_color = [255, 0, 0]  # 红色
        self.slice_markers[plane_type].append((click_2d, marker_color))
        
        # 更新显示
        self.update_slice_display(plane_type)
    
    def _calculate_3d_coordinate(self):
        """计算3D坐标（从两个不同平面的点击）"""
        if merge_two_planes_to_3d is None or plane_click_to_space_coord is None:
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
        
        # 保存3D点
        self.computed_3d_points.append(coord_3d)
        self.picked_3d_points.append(coord_3d)
        
        # 更新列表显示
        self._update_coord_list()
        
        # 在所有切片上更新标记
        self._update_all_slice_markers(coord_3d)
        
        # 调用回调
        if self.on_3d_point_added:
            self.on_3d_point_added(coord_3d)
        
        # 清除点击数据，准备下一次选择
        self.slice_pick_data.clear()
    
    def _update_all_slice_markers(self, coord_3d: Tuple[float, float, float]):
        """在所有切片上更新标记"""
        if plane_click_to_space_coord is None:
            return
        
        # 为每个平面计算2D坐标
        for plane_type in ['coronal', 'sagittal', 'axial']:
            slice_pos_normalized = self.slice_positions[plane_type] / 100.0
            # 这里需要反向计算，从3D坐标到2D坐标
            # 简化处理：直接更新显示
            self.update_slice_display(plane_type)
    
    def _update_coord_list(self):
        """更新3D点列表显示"""
        self.coord_list.clear()
        for i, point in enumerate(self.picked_3d_points):
            x, y, z = point
            self.coord_list.addItem(f"Point {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})")
    
    def handle_enable_3d_pick(self):
        """启用3D点选择模式"""
        self.slice_pick_mode = True
        self.slice_pick_data.clear()
        # 清除旧标记
        for plane_type in self.slice_markers:
            self.slice_markers[plane_type].clear()
        # 更新所有切片显示
        for plane_type in ['coronal', 'sagittal', 'axial']:
            self.update_slice_display(plane_type)
    
    def handle_clear_3d_points(self):
        """清除所有3D点"""
        self.picked_3d_points.clear()
        self.computed_3d_points.clear()
        self.slice_pick_data.clear()
        # 清除所有标记
        for plane_type in self.slice_markers:
            self.slice_markers[plane_type].clear()
        # 更新列表显示
        self._update_coord_list()
        # 更新所有切片显示
        for plane_type in ['coronal', 'sagittal', 'axial']:
            self.update_slice_display(plane_type)
    
    def handle_set_path_point_from_list(self, point_type: str):
        """从列表中选择3D点设置为路径点"""
        if not self.picked_3d_points:
            QMessageBox.warning(self.parent_widget, "No Points", "No 3D points available.")
            return
        
        selected_index = self.coord_list.currentRow()
        if selected_index < 0 or selected_index >= len(self.picked_3d_points):
            QMessageBox.warning(self.parent_widget, "Invalid Selection", "Please select a point from the list.")
            return
        
        space_coord = self.picked_3d_points[selected_index]
        
        # 调用回调设置路径点
        if self.on_path_point_set:
            self.on_path_point_set(point_type, space_coord)
    
    def handle_toggle_slice_view(self):
        """切换切片视图显示/隐藏"""
        # 这个功能在 MainWindow 中处理
        pass

