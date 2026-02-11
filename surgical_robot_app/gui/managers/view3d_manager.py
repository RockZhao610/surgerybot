"""
3D 视图管理器

负责：
- 3D 重建
- 3D 视图控制（清除、打开模型、重置相机）
- VTK 显示更新
"""

from typing import Optional, Tuple
from pathlib import Path
import time
import logging

try:
    from PyQt5.QtWidgets import QMessageBox, QApplication
    from PyQt5.QtCore import QObject
except ImportError:
    QMessageBox = None
    QApplication = None
    QObject = object

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
        def handle_errors(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

logger = get_logger("surgical_robot_app.gui.managers.view3d_manager")

# 导入 VTK 相关
try:
    from vtkmodules.vtkRenderingCore import vtkRenderer, vtkPolyDataMapper, vtkActor
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    from vtkmodules.vtkFiltersModeling import vtkOutlineFilter, vtkFillHolesFilter
    from vtkmodules.vtkIOGeometry import vtkSTLWriter
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    VTK_AVAILABLE = True
    logger.info("✅ VTK 模块导入成功")
except ImportError as e:
    VTK_AVAILABLE = False
    vtkRenderer = None
    vtkPolyDataMapper = None
    vtkActor = None
    vtkPolyDataNormals = None
    vtkFillHolesFilter = None
    vtkOutlineFilter = None
    vtkSTLWriter = None
    vtkPolyData = None
    logger.error(f"❌ VTK 模块导入失败: {e}")
except Exception as e:
    VTK_AVAILABLE = False
    vtkRenderer = None
    vtkPolyDataMapper = None
    vtkActor = None
    vtkPolyDataNormals = None
    vtkFillHolesFilter = None
    vtkOutlineFilter = None
    vtkSTLWriter = None
    vtkPolyData = None
    logger.error(f"❌ VTK 模块导入时发生未知错误: {e}")


def _load_core_reconstruct_3d():
    """动态加载 3D 重建核心函数"""
    import importlib
    try:
        module = importlib.import_module("surgical_robot_app.3d_recon.recon_core")
        return getattr(module, "reconstruct_3d", None)
    except Exception:
        try:
            module = importlib.import_module("3d_recon.recon_core")
            return getattr(module, "reconstruct_3d", None)
        except Exception:
            return None


class View3DManager(QObject):
    """3D 视图管理器"""
    
    def __init__(
        self,
        data_manager,
        vtk_renderer: Optional[vtkRenderer],
        vtk_widget,
        view3d_controller,
        btn_recon=None,
        recon_progress=None,
        vtk_status=None,
        parent_widget=None
    ):
        super().__init__()
        # 坐标系可视化器（初始化时不显示，只在3D重建后显示）
        self.coordinate_system: Optional['CoordinateSystemVisualizer'] = None
        try:
            from surgical_robot_app.vtk_utils.coordinate_system import CoordinateSystemVisualizer
            if vtk_renderer:
                self.coordinate_system = CoordinateSystemVisualizer(vtk_renderer)
                # 不在这里显示，等待3D重建完成后再显示
        except ImportError:
            try:
                from vtk_utils.coordinate_system import CoordinateSystemVisualizer
                if vtk_renderer:
                    self.coordinate_system = CoordinateSystemVisualizer(vtk_renderer)
                    # 不在这里显示，等待3D重建完成后再显示
            except ImportError:
                logger.warning("CoordinateSystemVisualizer not available")
                self.coordinate_system = None
        """
        初始化 3D 视图管理器
        
        Args:
            data_manager: 数据管理器
            vtk_renderer: VTK 渲染器
            vtk_widget: VTK 窗口组件
            view3d_controller: 3D 视图控制器
            btn_recon: 重建按钮
            recon_progress: 重建进度条
            vtk_status: VTK 状态标签
            parent_widget: 父窗口（用于显示消息框）
        """
        self.data_manager = data_manager
        self.vtk_renderer = vtk_renderer
        self.vtk_widget = vtk_widget
        self.view3d = view3d_controller
        self.btn_recon = btn_recon
        self.recon_progress = recon_progress
        self.vtk_status = vtk_status
        self.parent_widget = parent_widget
        
        # 从配置中获取填充参数
        try:
            from surgical_robot_app.config.settings import get_config
            config = get_config()
            self.morph_kernel_size = config.view3d.morph_kernel_size
            self.morph_iterations = config.view3d.morph_iterations
            self.vtk_hole_size = config.view3d.vtk_hole_size
            self.enable_contour_filling = config.view3d.enable_contour_filling
        except Exception:
            # 默认值（更保守的设置）
            self.morph_kernel_size = 3
            self.morph_iterations = 1
            self.vtk_hole_size = 1000.0
            self.enable_contour_filling = True
    
    @handle_errors(parent_widget=None, show_message=True, context="3D Reconstruction")
    def reconstruct_3d(self, *args):
        """执行异步 3D 重建（支持多标签）"""
        logger.info("🔵 reconstruct_3d() 被调用 (异步方式)")
        
        if not VTK_AVAILABLE:
            error_msg = "VTK is not available"
            logger.error(f"❌ {error_msg}")
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return
        
        if self.vtk_renderer is None:
            error_msg = "VTK renderer not initialized"
            logger.error(f"❌ {error_msg}")
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return
        
        volume = self.data_manager.get_volume()
        if volume is None:
            error_msg = "No volume data loaded. Please import data first."
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return
        
        seg_mask_volume = self.data_manager.get_seg_mask_volume()
        if seg_mask_volume is None:
            # 尝试从 masks 列表准备
            masks = self.data_manager.get_masks()
            if masks and len(masks) == volume.shape[0]:
                import numpy as np
                first_mask = next((m for m in masks if m is not None), None)
                if first_mask is not None:
                    seg_mask_volume = np.stack([m if m is not None else np.zeros_like(first_mask) for m in masks], axis=0)
                    self.data_manager.set_seg_mask_volume(seg_mask_volume)
            
        if seg_mask_volume is None:
            error_msg = "No segmentation mask found. Please perform segmentation first."
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return

        # UI 准备
        if self.btn_recon:
            self.btn_recon.setEnabled(False)
        if self.recon_progress:
            self.recon_progress.setVisible(True)
            self.recon_progress.setValue(0)
        if self.vtk_status:
            self.vtk_status.setText("Reconstructing in background...")

        # 准备任务参数
        spacing = None
        metadata = self.data_manager.get_metadata()
        if isinstance(metadata, dict) and "Spacing" in metadata:
            s = metadata["Spacing"]
            if isinstance(s, (list, tuple)) and len(s) == 3:
                spacing = (float(s[0]), float(s[1]), float(s[2]))

        # 获取需要重建的标签列表及其颜色
        import numpy as np
        unique_labels = np.unique(seg_mask_volume)
        unique_labels = [int(v) for v in unique_labels if v > 0]
        
        if not unique_labels:
            error_msg = "Segmentation mask is empty. Please perform segmentation first."
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            if self.btn_recon: self.btn_recon.setEnabled(True)
            if self.recon_progress: self.recon_progress.setVisible(False)
            return
        
        # 收集标签颜色信息
        label_colors = {}
        for label_id in unique_labels:
            label_colors[label_id] = self.data_manager.get_label_color_float(label_id)
        
        logger.info(f"多标签重建: 标签={unique_labels}, 颜色={label_colors}")

        # 启动线程
        run_in_thread(
            self,
            self._reconstruct_3d_task,
            on_finished=self._on_reconstruction_finished,
            on_error=self._on_reconstruction_error,
            on_progress=self._on_reconstruction_progress,
            vol=seg_mask_volume,
            spacing=spacing,
            label_ids=unique_labels,
            label_colors=label_colors
        )

    def _reconstruct_3d_task(self, vol, spacing=None, label_ids=None, label_colors=None, progress_callback=None):
        """
        在后台线程执行的重建任务（支持多标签）
        
        对每个标签分别提取二值体、预处理、Marching Cubes
        """
        import numpy as np
        import cv2
        
        # 兼容旧调用（单标签）
        if label_ids is None:
            label_ids = [1]
            vol = np.where(vol > 128, 255, 0).astype(np.uint8)
        
        core_reconstruct_3d = _load_core_reconstruct_3d()
        if core_reconstruct_3d is None:
            raise RuntimeError("Core reconstruction module not available")
        
        results = []  # [(label_id, poly, image, color), ...]
        total_labels = len(label_ids)
        
        for idx, label_id in enumerate(label_ids):
            # 提取此标签的二值体
            binary_vol = (vol == label_id).astype(np.uint8) * 255
            
            # 检查是否有数据
            if np.sum(binary_vol > 0) == 0:
                logger.warning(f"标签 {label_id} 无数据，跳过")
                continue
            
            # 预处理（形态学 + 填充）
            filled_vol = np.zeros_like(binary_vol)
            for z in range(binary_vol.shape[0]):
                mask_slice = binary_vol[z].copy()
                if np.sum(mask_slice > 0) == 0:
                    filled_vol[z] = mask_slice
                    continue
                
                kernel_size = getattr(self, 'morph_kernel_size', 3)
                iterations = getattr(self, 'morph_iterations', 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask_slice = cv2.morphologyEx(mask_slice, cv2.MORPH_CLOSE, kernel, iterations=iterations)
                
                enable_contour = getattr(self, 'enable_contour_filling', True)
                if enable_contour:
                    contours, _ = cv2.findContours(mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filled_mask = np.zeros_like(mask_slice)
                    if contours:
                        cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)
                        mask_slice = filled_mask
                filled_vol[z] = mask_slice
            
            # 进度回调
            def _label_progress(p, _idx=idx, _total=total_labels):
                if progress_callback:
                    overall = int((_idx * 100 + p) / _total)
                    progress_callback(overall)
            
            # Marching Cubes
            poly, image = core_reconstruct_3d(filled_vol, spacing=spacing, threshold=128, progress_cb=_label_progress)
            
            color = label_colors.get(label_id, (1.0, 0.3, 0.3)) if label_colors else (1.0, 0.3, 0.3)
            results.append((label_id, poly, image, color))
            
            logger.info(f"标签 {label_id} 重建完成: "
                        f"points={poly.GetNumberOfPoints() if poly else 0}")
        
        if not results:
            raise RuntimeError("All label reconstructions failed")
        
        return results

    def _on_reconstruction_progress(self, p):
        """更新进度条"""
        if self.recon_progress:
            self.recon_progress.setValue(p)

    def _on_reconstruction_finished(self, result):
        """重建完成后的 UI 更新（支持多标签）"""
        # result = [(label_id, poly, image, color), ...]
        results = result
        
        if not results:
            self._on_reconstruction_error("Empty results")
            return
        
        self.vtk_renderer.RemoveAllViewProps()
        
        total_points = 0
        first_image = None
        saved_files = []
        
        for label_id, poly, image, color in results:
            if poly is None or poly.GetNumberOfPoints() == 0:
                logger.warning(f"标签 {label_id} 表面为空，跳过")
                continue
            
            if first_image is None:
                first_image = image
            
            # 后处理 (VTK 过滤器)
            try:
                if vtkFillHolesFilter is not None:
                    hole_size = getattr(self, 'vtk_hole_size', 1000.0)
                    fill_holes = vtkFillHolesFilter()
                    fill_holes.SetInputData(poly)
                    fill_holes.SetHoleSize(hole_size)
                    fill_holes.Update()
                    poly = fill_holes.GetOutput()
            except Exception as e:
                logger.warning(f"VTK FillHoles failed for label {label_id}: {e}")

            # 创建 Actor
            mapper = vtkPolyDataMapper()
            normals = vtkPolyDataNormals()
            normals.SetInputData(poly)
            normals.SetFeatureAngle(60.0)
            normals.Update()
            mapper.SetInputData(normals.GetOutput())
            mapper.ScalarVisibilityOff()
            
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetSpecular(0.5)
            actor.GetProperty().SetSpecularPower(20.0)
            
            self.vtk_renderer.AddActor(actor)
            total_points += poly.GetNumberOfPoints()
            
            # 保存 STL 文件
            try:
                result_dir = Path(__file__).resolve().parent.parent.parent / "3d_recon" / "result"
                result_dir.mkdir(parents=True, exist_ok=True)
                label_name = self.data_manager.label_names.get(label_id, f"label_{label_id}")
                # 清理文件名中的非法字符
                safe_name = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in label_name)
                out_path = result_dir / f"model_{safe_name}_{int(time.time())}.stl"
                from vtkmodules.vtkIOGeometry import vtkSTLWriter
                writer = vtkSTLWriter()
                writer.SetInputData(poly)
                writer.SetFileName(str(out_path))
                writer.Write()
                saved_files.append(str(out_path))
                logger.info(f"Saved label {label_id}: {out_path}")
            except Exception as e:
                logger.warning(f"Failed to save STL for label {label_id}: {e}")
        
        # 添加体积轮廓线
        if first_image is not None:
            try:
                outline = vtkOutlineFilter()
                outline.SetInputData(first_image)
                outline.Update()
                outline_mapper = vtkPolyDataMapper()
                outline_mapper.SetInputData(outline.GetOutput())
                outline_actor = vtkActor()
                outline_actor.SetMapper(outline_mapper)
                outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
                self.vtk_renderer.AddActor(outline_actor)
            except Exception as e:
                logger.warning(f"Failed to create outline: {e}")
        
        self.vtk_renderer.ResetCamera()
        
        if self.coordinate_system:
            bounds = self.vtk_renderer.ComputeVisiblePropBounds()
            self.coordinate_system.auto_center_and_size_from_bounds(tuple(bounds))
            
        self.update_display()
        
        label_count = len([r for r in results if r[1] is not None and r[1].GetNumberOfPoints() > 0])
        if self.vtk_status:
            self.vtk_status.setText(f"Render complete: {label_count} labels, {total_points} points")

        if self.btn_recon: self.btn_recon.setEnabled(True)
        if self.recon_progress: self.recon_progress.setVisible(False)
        
        # 触发主界面的同步
        if hasattr(self.parent_widget, '_sync_to_reconstruction_window'):
            self.parent_widget._sync_to_reconstruction_window()

    def _on_reconstruction_error(self, error_msg):
        """处理重建错误"""
        logger.error(f"Reconstruction Error: {error_msg}")
        if self.vtk_status: self.vtk_status.setText("Error occurred")
        if self.btn_recon: self.btn_recon.setEnabled(True)
        if self.recon_progress: self.recon_progress.setVisible(False)
        if QMessageBox and self.parent_widget:
            QMessageBox.critical(self.parent_widget, "3D Reconstruction", f"Error: {error_msg}")

    @handle_errors(parent_widget=None, show_message=False, context="Clear 3D View", log_level=logging.WARNING)
    def clear_3d_view(self, *args, path_ui_ctrl=None, multi_slice_ctrl=None):
        """
        清除3D视图中的所有对象
        
        Args:
            path_ui_ctrl: 路径UI控制器（可选）
            multi_slice_ctrl: 多视图切片控制器（可选）
        """
        if not VTK_AVAILABLE or self.vtk_renderer is None:
            return
        
        # 先重置路径规划和 3D 点相关数据
        if path_ui_ctrl:
            path_ui_ctrl.handle_reset_path()
        if multi_slice_ctrl:
            multi_slice_ctrl.handle_clear_3d_points()

        # 隐藏坐标轴（如果存在）
        if self.coordinate_system:
            self.coordinate_system.hide_coordinate_system()

        # 再统一清空渲染窗口中所有对象
        self.vtk_renderer.RemoveAllViewProps()
        # 重新渲染空场景
        if self.vtk_widget:
            try:
                rw = self.vtk_widget.GetRenderWindow()
                if rw:
                    rw.Render()
                    self.vtk_widget.update()
            except Exception:
                pass
        
        # 重置 3D 重建 UI 状态
        if self.recon_progress:
            self.recon_progress.setVisible(False)
            self.recon_progress.setValue(0)
        if self.btn_recon:
            self.btn_recon.setEnabled(True)
        if self.vtk_status:
            self.vtk_status.setText("VTK: 3D view cleared")
    
    @handle_errors(parent_widget=None, show_message=True, context="Open STL Model")
    def open_model(self, *args):
        """打开并加载 STL 模型文件"""
        if not VTK_AVAILABLE or self.vtk_widget is None:
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "Open Model", "VTK Qt widget not available.")
            return
        
        from PyQt5.QtWidgets import QFileDialog
        base_dir = Path(__file__).resolve().parent.parent.parent / "3d_recon" / "result"
        base_dir.mkdir(parents=True, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self.parent_widget,
            "Open STL Model",
            str(base_dir),
            "STL Files (*.stl)"
        )
        if not file_path:
            return
        
        if self.vtk_status:
            self.vtk_status.setText("Loading model...")

        result = self.view3d.load_stl_model(file_path)
        if result is None:
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "Open Model", "Failed to load STL model.")
            return
        
        pts, cells = result
        
        # 更新坐标系位置和大小（根据模型边界）
        if self.coordinate_system and self.vtk_renderer:
            try:
                bounds = self.vtk_renderer.ComputeVisiblePropBounds()
                if bounds and len(bounds) >= 6:
                    # 从配置中获取坐标系大小比例
                    try:
                        from surgical_robot_app.config.settings import get_config
                        config = get_config()
                        size_ratio = config.view3d.coordinate_system_size_ratio
                    except Exception:
                        size_ratio = 0.2  # 默认值
                    self.coordinate_system.auto_center_and_size_from_bounds(tuple(bounds), size_ratio=size_ratio)
            except Exception as e:
                logger.warning(f"自动调整坐标系位置和大小时出错: {e}")
        
        if self.vtk_status:
            self.vtk_status.setText(f"Model loaded — points: {pts}, cells: {cells}")
        logger.info(f"模型已加载: points={pts}, cells={cells}")
    
    @handle_errors(parent_widget=None, show_message=False, context="Reset Camera", log_level=logging.WARNING)
    def reset_camera(self, *args):
        """重置相机"""
        if not VTK_AVAILABLE or self.vtk_widget is None:
            return
        
        camera_info = self.view3d.reset_camera()
        if camera_info:
            pos = camera_info.position
            focal = camera_info.focal
            view_up = camera_info.view_up
            logger.debug(f"Camera Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            logger.debug(f"Camera Focal Point: ({focal[0]:.2f}, {focal[1]:.2f}, {focal[2]:.2f})")
            logger.debug(f"Camera View Up: ({view_up[0]:.2f}, {view_up[1]:.2f}, {view_up[2]:.2f})")
            if self.vtk_status:
                self.vtk_status.setText(f"Camera reset - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        else:
            if self.vtk_status:
                self.vtk_status.setText("Camera reset failed")
    
    def update_display(self):
        """更新VTK显示"""
        if not VTK_AVAILABLE or self.vtk_widget is None:
            return
        
        try:
            if hasattr(self.vtk_widget, "GetRenderWindow"):
                rw = self.vtk_widget.GetRenderWindow()
                if rw:
                    rw.Render()
                    self.vtk_widget.update()
                    if QApplication:
                        QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating VTK display: {e}")

