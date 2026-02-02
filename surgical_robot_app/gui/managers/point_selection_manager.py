"""
3D点选择管理器

负责管理3D点选择相关的功能：
- 处理切片点击事件
- 计算3D坐标
- 绘制切片标记
- 管理3D点列表
"""

from typing import Optional, Tuple, Dict, List

try:
    from PyQt5.QtWidgets import QMessageBox, QLabel, QListWidget
    from PyQt5.QtGui import QPainter, QPen, QBrush, QPixmap
    from PyQt5.QtCore import Qt
    QMessageBox_AVAILABLE = True
except ImportError:
    QMessageBox_AVAILABLE = False
    QMessageBox = None
    QLabel = None
    QListWidget = None
    QPainter = None
    QPen = None
    QBrush = None
    QPixmap = None
    Qt = None

try:
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.geometry.slice_geometry import (
        plane_click_to_space_coord,
        merge_two_planes_to_3d
    )
except ImportError:
    from utils.logger import get_logger
    try:
        from geometry.slice_geometry import (
            plane_click_to_space_coord,
            merge_two_planes_to_3d
        )
    except ImportError:
        plane_click_to_space_coord = None
        merge_two_planes_to_3d = None

logger = get_logger("surgical_robot_app.gui.managers.point_selection_manager")


class PointSelectionManager:
    """3D点选择管理器"""
    
    def __init__(
        self,
        data_manager,
        coord_list: Optional[QListWidget] = None,
        coronal_label: Optional[QLabel] = None,
        sagittal_label: Optional[QLabel] = None,
        axial_label: Optional[QLabel] = None,
        vtk_renderer=None,
        vtk_widget=None,
        path_ui_ctrl=None,
        vtk_status: Optional[QLabel] = None,
        parent_widget=None
    ):
        """
        初始化3D点选择管理器
        
        Args:
            data_manager: 数据管理器
            coord_list: 坐标列表控件
            coronal_label: 冠状面标签
            sagittal_label: 矢状面标签
            axial_label: 横断面标签
            vtk_renderer: VTK渲染器
            vtk_widget: VTK控件
            path_ui_ctrl: 路径UI控制器
            vtk_status: VTK状态标签
            parent_widget: 父窗口
        """
        self.data_manager = data_manager
        self.coord_list = coord_list
        self.coronal_label = coronal_label
        self.sagittal_label = sagittal_label
        self.axial_label = axial_label
        self.vtk_renderer = vtk_renderer
        self.vtk_widget = vtk_widget
        self.path_ui_ctrl = path_ui_ctrl
        self.vtk_status = vtk_status
        self.parent_widget = parent_widget
        
        # 3D点选择相关状态
        self.slice_pick_data: Dict = {}  # 存储两个切片窗口的点击数据
        self.slice_positions = {'coronal': 0, 'sagittal': 0, 'axial': 0}  # 当前切片位置
        self.picked_3d_points: List[Tuple[float, float, float]] = []  # 已选择的3D点列表
        self.slice_markers: Dict[str, List] = {}  # 存储切片窗口中的标记
        self.point_actors: List = []  # 3D点标记 actors（蓝色标记）
        
        # 回调函数
        self.on_3d_point_added = None  # 当3D点被添加时的回调
        self.on_path_point_set = None  # 当路径点被设置时的回调
        self.update_vtk_display_callback = None  # 更新VTK显示的回调
    
    def set_slice_positions(self, positions: Dict[str, int]):
        """设置切片位置"""
        self.slice_positions = positions
    
    def handle_slice_click(self, plane_type: str, x: int, y: int, label_widget: QLabel):
        """
        处理切片窗口的点击事件，获取2D坐标
        
        Args:
            plane_type: 切片类型 ('coronal', 'sagittal', 'axial')
            x: 点击的X坐标（窗口坐标）
            y: 点击的Y坐标（窗口坐标）
            label_widget: 点击的标签控件
        """
        # 检查是否有volume数据
        volume = self.data_manager.get_volume()
        if volume is None:
            if self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Point Pick", "Please load volume data first")
            return
        
        # 获取显示的pixmap
        pix = label_widget.pixmap()
        if pix is None:
            return
        
        # 计算点击位置在图像中的实际坐标
        disp_w = pix.width()
        disp_h = pix.height()
        lab_w = label_widget.width()
        lab_h = label_widget.height()
        
        # 计算图像在label中的偏移
        off_x = (lab_w - disp_w) / 2.0
        off_y = (lab_h - disp_h) / 2.0
        
        # 检查点击是否在图像范围内
        if x < off_x or y < off_y or x > off_x + disp_w or y > off_y + disp_h:
            return
        
        # 获取volume尺寸
        depth, height, width = volume.shape[0], volume.shape[1], volume.shape[2]
        
        # 计算图像坐标（归一化到 0-1）
        img_x_norm = (x - off_x) / disp_w
        img_y_norm = (y - off_y) / disp_h
        
        # 使用几何工具将 2D 点击转换为 3D 空间坐标
        if plane_click_to_space_coord is None:
            logger.error("plane_click_to_space_coord not available")
            return
        
        slice_pos = self.slice_positions.get(plane_type, 0)
        try:
            click_2d, (x_coord, y_coord, z_coord) = plane_click_to_space_coord(
                plane_type,
                slice_pos,
                img_x_norm,
                img_y_norm,
            )
        except Exception as e:
            logger.error(f"Error converting plane click to space coord: {e}")
            return
        
        # 存储点击数据
        if 'plane1' not in self.slice_pick_data:
            # 第一个切片窗口的点击
            self.slice_pick_data['plane1'] = {
                'plane': plane_type,
                'slice_pos': slice_pos,
                'click': click_2d,
                'coords': (x_coord, y_coord, z_coord)
            }
            # 在切片窗口显示标记
            self.draw_slice_marker(plane_type, click_2d, label_widget)
            # 更新信息显示
            if self.coord_list:
                self.coord_list.addItem(
                    f"Plane 1 ({plane_type}): X={x_coord:.1f}, Y={y_coord:.1f}, Z={z_coord:.1f} (partial)"
                )
        elif 'plane2' not in self.slice_pick_data:
            # 第二个切片窗口的点击
            if self.slice_pick_data['plane1']['plane'] == plane_type:
                if self.parent_widget:
                    QMessageBox.warning(
                        self.parent_widget,
                        "3D Point Pick",
                        "Please click on a different plane"
                    )
                return
            
            self.slice_pick_data['plane2'] = {
                'plane': plane_type,
                'slice_pos': slice_pos,
                'click': click_2d,
                'coords': (x_coord, y_coord, z_coord)
            }
            # 在切片窗口显示标记
            self.draw_slice_marker(plane_type, click_2d, label_widget)
            # 计算完整的3D坐标
            self.calculate_3d_coordinate()
        else:
            # 已有两个点击，清除后重新开始
            self.slice_pick_data = {}
            self.clear_slice_markers()
            if self.coord_list:
                self.coord_list.clear()
            # 重新处理当前点击
            self.handle_slice_click(plane_type, x, y, label_widget)
    
    def draw_slice_marker(self, plane_type: str, click_2d: Tuple[float, float], label_widget: QLabel):
        """
        在切片窗口中绘制标记点
        
        Args:
            plane_type: 切片类型
            click_2d: 2D坐标（归一化到0-100）
            label_widget: 标签控件
        """
        pix = label_widget.pixmap()
        if pix is None:
            return
        
        # 创建标记图像
        marked_pix = pix.copy()
        painter = QPainter(marked_pix)
        painter.setPen(QPen(Qt.red, 3))
        painter.setBrush(QBrush(Qt.red))
        
        # 将归一化坐标转换为图像坐标
        w, h = pix.width(), pix.height()
        x_img = (click_2d[0] / 100.0) * w
        y_img = (click_2d[1] / 100.0) * h
        
        # 绘制圆形标记
        radius = 5
        painter.drawEllipse(int(x_img - radius), int(y_img - radius), radius * 2, radius * 2)
        painter.end()
        
        # 更新显示
        label_widget.setPixmap(marked_pix)
        
        # 存储标记信息
        if plane_type not in self.slice_markers:
            self.slice_markers[plane_type] = []
        self.slice_markers[plane_type].append((click_2d, marked_pix))
    
    def calculate_3d_coordinate(self):
        """
        通过两个切片窗口的交集计算完整的3D坐标
        逻辑：两个切片窗口各提供一个坐标维度，通过交集确定完整的XYZ坐标
        """
        if 'plane1' not in self.slice_pick_data or 'plane2' not in self.slice_pick_data:
            return
        
        plane1_data = self.slice_pick_data['plane1']
        plane2_data = self.slice_pick_data['plane2']
        plane1_type = plane1_data['plane']
        plane2_type = plane2_data['plane']
        
        # 获取两个平面的坐标
        coords1 = plane1_data['coords']  # (x, y, z)
        coords2 = plane2_data['coords']  # (x, y, z)
        
        # 使用几何工具合并两个平面的坐标为完整 3D 坐标
        if merge_two_planes_to_3d is None:
            logger.error("merge_two_planes_to_3d not available")
            if self.parent_widget:
                QMessageBox.warning(
                    self.parent_widget,
                    "3D Point Pick",
                    "Failed to calculate 3D coordinate: geometry tools not available"
                )
            return
        
        merged = merge_two_planes_to_3d(plane1_type, coords1, plane2_type, coords2)
        
        if merged is not None:
            x_final, y_final, z_final = merged
            # 完整的3D坐标
            final_3d_coord = (x_final, y_final, z_final)
            
            # 添加到已选择的3D点列表
            self.picked_3d_points.append(final_3d_coord)
            
            # 更新坐标列表显示
            if self.coord_list:
                self.coord_list.addItem(
                    f"3D Point {len(self.picked_3d_points)}: X={x_final:.2f}, Y={y_final:.2f}, Z={z_final:.2f}"
                )
            
            # 在所有三个切片窗口的对应位置显示标记
            self.update_all_slice_markers(final_3d_coord)
            
            # 检查是否有设置路径点模式，如果有则自动添加到路径规划
            if self.path_ui_ctrl and self.path_ui_ctrl.pick_mode is not None:
                # 委托给 PathUIController 处理
                self.path_ui_ctrl._process_picked_point(
                    final_3d_coord[0], final_3d_coord[1], final_3d_coord[2]
                )
                
                # 更新状态
                if self.vtk_status:
                    mode_name = {'start': 'Start', 'waypoint': 'Waypoint', 'end': 'End'}.get(
                        self.path_ui_ctrl.pick_mode, 'Point'
                    )
                    self.vtk_status.setText(
                        f"{mode_name} point set from 2D slice: ({x_final:.2f}, {y_final:.2f}, {z_final:.2f})"
                    )
            else:
                # 如果没有设置路径点模式，只显示蓝色标记
                self.add_3d_point_marker(final_3d_coord)
                
                # 更新状态
                if self.vtk_status:
                    self.vtk_status.setText(
                        f"3D Point {len(self.picked_3d_points)} selected: ({x_final:.2f}, {y_final:.2f}, {z_final:.2f})"
                    )
                
                # 调用回调
                if self.on_3d_point_added:
                    self.on_3d_point_added(final_3d_coord)
            
            # 清除当前点击数据，准备下一次选择
            self.slice_pick_data = {}
            self.clear_slice_markers()
        else:
            if self.parent_widget:
                QMessageBox.warning(
                    self.parent_widget,
                    "3D Point Pick",
                    "Failed to calculate 3D coordinate"
                )
    
    def update_all_slice_markers(self, coord_3d: Tuple[float, float, float]):
        """
        在所有三个切片窗口的对应位置显示标记点
        
        Args:
            coord_3d: 3D坐标 (x, y, z)，范围[0, 100]
        """
        x, y, z = coord_3d
        
        # 更新冠状面（显示X-Z，Y由切片位置确定）
        if self.coronal_label and self.coronal_label.pixmap() is not None:
            coronal_2d = (x, z)
            self.draw_slice_marker('coronal', coronal_2d, self.coronal_label)
        
        # 更新矢状面（显示Y-Z，X由切片位置确定）
        if self.sagittal_label and self.sagittal_label.pixmap() is not None:
            sagittal_2d = (y, z)
            self.draw_slice_marker('sagittal', sagittal_2d, self.sagittal_label)
        
        # 更新横断面（显示X-Y，Z由切片位置确定）
        if self.axial_label and self.axial_label.pixmap() is not None:
            axial_2d = (x, y)
            self.draw_slice_marker('axial', axial_2d, self.axial_label)
    
    def add_3d_point_marker(self, coord_3d: Tuple[float, float, float]):
        """
        在3D窗口中添加3D点标记
        
        Args:
            coord_3d: 3D坐标 (x, y, z)，范围[0, 100]
        """
        if self.vtk_renderer is None:
            return
        
        try:
            from surgical_robot_app.vtk_utils.markers import create_sphere_marker
        except ImportError:
            try:
                from vtk_utils.markers import create_sphere_marker
            except ImportError:
                logger.error("create_sphere_marker not available")
                return
        
        # 使用通用 VTK 工具创建蓝色 3D 点标记
        actor = create_sphere_marker(
            self.vtk_renderer,
            coord_3d,
            color=(0.0, 0.0, 1.0),
            radius_ratio=0.02
        )
        if actor is not None:
            self.point_actors.append(actor)
            if self.update_vtk_display_callback:
                self.update_vtk_display_callback()
    
    def clear_slice_markers(self):
        """清除切片窗口中的标记，重新显示原始切片图像"""
        # 清除标记存储
        self.slice_markers = {}
        # 注意：实际的切片图像更新应该由 MultiSliceController 处理
    
    def clear_3d_points(self):
        """清除所有3D点和标记"""
        # 清除3D点列表
        self.picked_3d_points = []
        
        # 清除切片点击数据
        self.slice_pick_data = {}
        
        # 清除切片窗口标记
        self.clear_slice_markers()
        
        # 清除3D窗口中的点标记
        if self.vtk_renderer:
            for actor in self.point_actors:
                self.vtk_renderer.RemoveActor(actor)
        self.point_actors = []
        
        # 清空坐标列表
        if self.coord_list:
            self.coord_list.clear()
    
    def enable_3d_pick(self):
        """启用3D点选择模式"""
        # 这个方法主要用于更新UI状态，实际逻辑由调用者处理
        if self.vtk_status:
            self.vtk_status.setText(
                "3D Point Pick Mode: Click on any two slice windows to select a 3D point"
            )
    
    def set_path_point_from_list(self, point_type: str, selected_index: int):
        """
        从坐标列表中选择的点设置为路径点
        
        Args:
            point_type: 点类型 ('start', 'waypoint', 'end')
            selected_index: 选中的索引
        """
        if selected_index < 0 or selected_index >= len(self.picked_3d_points):
            if self.parent_widget:
                QMessageBox.warning(
                    self.parent_widget,
                    "Error",
                    "Invalid point index"
                )
            return
        
        selected_point = self.picked_3d_points[selected_index]
        
        if self.path_ui_ctrl:
            # 临时设置 pick_mode
            old_pick_mode = self.path_ui_ctrl.pick_mode
            self.path_ui_ctrl.pick_mode = point_type
            
            # 处理点
            self.path_ui_ctrl._process_picked_point(
                selected_point[0], selected_point[1], selected_point[2]
            )
            
            # 恢复原来的 pick_mode
            if old_pick_mode is None:
                self.path_ui_ctrl.pick_mode = None
            else:
                self.path_ui_ctrl.pick_mode = old_pick_mode
            
            # 更新状态
            if self.vtk_status:
                point_name = {'start': 'Start', 'waypoint': 'Waypoint', 'end': 'End'}.get(
                    point_type, 'Point'
                )
                self.vtk_status.setText(
                    f"{point_name} point set from 3D point {selected_index + 1}"
                )
            
            # 调用回调
            if self.on_path_point_set:
                self.on_path_point_set(point_type, selected_point)

