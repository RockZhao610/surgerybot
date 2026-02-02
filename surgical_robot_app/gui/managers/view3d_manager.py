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
        """执行异步 3D 重建"""
        logger.info("🔵 reconstruct_3d() 被调用 (异步方式)")
        
        if not VTK_AVAILABLE:
            error_msg = "VTK 不可用"
            logger.error(f"❌ {error_msg}")
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return
        
        if self.vtk_renderer is None:
            error_msg = "VTK 渲染器未初始化"
            logger.error(f"❌ {error_msg}")
            if QMessageBox and self.parent_widget:
                QMessageBox.warning(self.parent_widget, "3D Reconstruction", error_msg)
            return
        
        volume = self.data_manager.get_volume()
        if volume is None:
            error_msg = "未加载体积数据，请先导入数据"
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
            error_msg = "未找到分割掩码，请先进行分割操作"
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

        # 启动线程
        run_in_thread(
            self,
            self._reconstruct_3d_task,
            on_finished=self._on_reconstruction_finished,
            on_error=self._on_reconstruction_error,
            on_progress=self._on_reconstruction_progress,
            vol=seg_mask_volume,
            spacing=spacing
        )

    def _reconstruct_3d_task(self, vol, spacing=None, progress_callback=None):
        """在后台线程执行的重建任务"""
        import numpy as np
        import cv2
        
        # 1. 预处理 (二值化 + 填充)
        vol = vol.astype(np.uint8) if vol.dtype != np.uint8 else vol
        vol = np.where(vol > 128, 255, 0).astype(np.uint8)
        
        filled_vol = np.zeros_like(vol)
        for z in range(vol.shape[0]):
            mask = vol[z].copy()
            if np.sum(mask > 0) == 0:
                filled_vol[z] = mask
                continue
            
            kernel_size = getattr(self, 'morph_kernel_size', 3)
            iterations = getattr(self, 'morph_iterations', 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            enable_contour = getattr(self, 'enable_contour_filling', True)
            if enable_contour:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                filled_mask = np.zeros_like(mask)
                if contours:
                    cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)
                    mask = filled_mask
            filled_vol[z] = mask
            
        # 2. 调用核心重建逻辑
        core_reconstruct_3d = _load_core_reconstruct_3d()
        if core_reconstruct_3d is None:
            raise RuntimeError("Core reconstruction module not available")

        poly, image = core_reconstruct_3d(filled_vol, spacing=spacing, threshold=128, progress_cb=progress_callback)
        return poly, image

    def _on_reconstruction_progress(self, p):
        """更新进度条"""
        if self.recon_progress:
            self.recon_progress.setValue(p)

    def _on_reconstruction_finished(self, result):
        """重建完成后的 UI 更新 (在主线程执行)"""
        poly, image = result
        
        if poly is None or poly.GetNumberOfPoints() == 0:
            self._on_reconstruction_error("Empty surface extracted")
            return

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
            logger.warning(f"VTK FillHoles failed: {e}")

        # 创建 Actor 并更新视图
        mapper = vtkPolyDataMapper()
        normals = vtkPolyDataNormals()
        normals.SetInputData(poly)
        normals.SetFeatureAngle(60.0)
        normals.Update()
        mapper.SetInputData(normals.GetOutput())
        mapper.ScalarVisibilityOff()
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.3, 0.3)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(20.0)
        
        outline = vtkOutlineFilter()
        outline.SetInputData(image)
        outline.Update()
        outline_mapper = vtkPolyDataMapper()
        outline_mapper.SetInputData(outline.GetOutput())
        outline_actor = vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        
        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.AddActor(outline_actor)
        self.vtk_renderer.ResetCamera()
        
        if self.coordinate_system:
            bounds = self.vtk_renderer.ComputeVisiblePropBounds()
            self.coordinate_system.auto_center_and_size_from_bounds(tuple(bounds))
            
        self.update_display()
        
        if self.vtk_status:
            self.vtk_status.setText(f"Render complete: {poly.GetNumberOfPoints()} points")
        
        # 保存 STL 文件
        try:
            result_dir = Path(__file__).resolve().parent.parent.parent / "3d_recon" / "result"
            result_dir.mkdir(parents=True, exist_ok=True)
            out_path = result_dir / f"model_{int(time.time())}.stl"
            from vtkmodules.vtkIOGeometry import vtkSTLWriter
            writer = vtkSTLWriter()
            writer.SetInputData(poly)
            writer.SetFileName(str(out_path))
            writer.Write()
            logger.info(f"Saved: {out_path}")
        except Exception as e:
            logger.warning(f"Failed to save STL: {e}")

        if self.btn_recon: self.btn_recon.setEnabled(True)
        if self.recon_progress: self.recon_progress.setVisible(False)
        
        # 触发主界面的同步 (由于 MainView 持有 View3DManager 引用并连接了 reconstruction_window)
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

