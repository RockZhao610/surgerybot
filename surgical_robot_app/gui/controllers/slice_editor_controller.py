"""
SliceEditorController: 切片编辑器控制器

职责：
- 处理切片显示、编辑相关的事件
- 管理 HSV 阈值控制
- 处理鼠标事件（画笔/擦除）
- 不直接依赖 Qt Widget，通过回调与 MainWindow 通信
"""

import numpy as np
import logging
from typing import Optional, Callable, Tuple
from PyQt5.QtCore import QEvent, Qt, QObject
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent
from PyQt5.QtWidgets import QLabel, QSlider, QFileDialog, QMessageBox

try:
    import cv2
except Exception:
    cv2 = None

try:
    from surgical_robot_app.data_io.data_manager import DataManager
    from surgical_robot_app.segmentation.manual_controller import ManualSegController
    from surgical_robot_app.segmentation.sam2_controller import SAM2Controller
    from surgical_robot_app.segmentation.hsv_threshold import (
        compute_threshold_mask as hsv_compute_threshold_mask,
        apply_threshold_all,
        DEFAULT_THRESHOLDS,
    )
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.utils.error_handler import handle_errors
    from surgical_robot_app.utils.threading_utils import run_in_thread
except ImportError:
    try:
        from data_io.data_manager import DataManager
        from segmentation.manual_controller import ManualSegController
        from segmentation.sam2_controller import SAM2Controller
        from segmentation.hsv_threshold import (
            compute_threshold_mask as hsv_compute_threshold_mask,
            apply_threshold_all,
            DEFAULT_THRESHOLDS,
        )
        from utils.logger import get_logger
        from utils.threading_utils import run_in_thread
        from utils.error_handler import handle_errors
    except ImportError:
        DataManager = None  # type: ignore
        ManualSegController = None  # type: ignore
        SAM2Controller = None  # type: ignore
        hsv_compute_threshold_mask = None
        apply_threshold_all = None
        DEFAULT_THRESHOLDS = {}
        def handle_errors(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def run_in_thread(*args, **kwargs):
            return None, None
        def get_logger(name):
            return logging.getLogger(name)

logger = get_logger("surgical_robot_app.gui.controllers.slice_editor_controller")


class SliceEditorController(QObject):
    """切片编辑器控制器"""
    
    def __init__(
        self,
        data_manager: DataManager,
        manual_controller: ManualSegController,
        sam2_controller: Optional[SAM2Controller],
        image_label: QLabel,
        slice_label: QLabel,
        slice_slider: QSlider,
        threshold_sliders: dict,
        radio_auto=None,  # SAM2 自动模式单选按钮
        btn_apply_all=None, # 应用到所有切片按钮
        parent_widget=None,
    ):
        """
        初始化切片编辑器控制器
        
        Args:
            data_manager: 数据管理器
            manual_controller: 手动分割控制器
            sam2_controller: SAM2 控制器（可选）
            image_label: 图像显示标签
            slice_label: 切片标签
            slice_slider: 切片滑块
            threshold_sliders: 阈值滑块字典
            radio_auto: SAM2 自动模式单选按钮（可选）
            btn_apply_all: 应用到所有切片按钮（可选）
            parent_widget: 父窗口
        """
        super().__init__()
        self.data_manager = data_manager
        self.manual_controller = manual_controller
        self.sam2_controller = sam2_controller
        self.image_label = image_label
        self.slice_label = slice_label
        self.slice_slider = slice_slider
        self.threshold_sliders = threshold_sliders
        self.radio_auto = radio_auto
        self.btn_apply_all = btn_apply_all
        self.parent_widget = parent_widget
        
        # 状态
        self.brush_mode = False  # 画笔模式
        self.eraser_mode = False  # 擦除模式
        self.brush_size = manual_controller.brush_size if hasattr(manual_controller, 'brush_size') else 10
        self.threshold_collapsed = False
        
        # 回调函数
        self.on_sam2_click: Optional[Callable] = None
        self.on_exit_sam2_picking: Optional[Callable] = None  # 退出 SAM2 picking 模式的回调
    
    def handle_slice_change(self, value: int):
        """处理切片变化"""
        self.slice_label.setText(f"Slice: {value}")
        self.update_slice_display(value)
    
    def update_slice_display(self, index: int):
        """更新切片显示"""
        volume = self.data_manager.get_volume()
        if volume is None:
            return
        if index < 0 or index >= volume.shape[0]:
            return
        
        slice_img = volume[index]
        mask = self.data_manager.get_mask(index)
        
        # 在自动分割（SAM2）模式下，不叠加阈值分割结果
        use_threshold = True
        if self.radio_auto is not None and self.radio_auto.isChecked():
            use_threshold = False
        
        # 计算阈值掩码
        thmask = None
        if use_threshold and hsv_compute_threshold_mask is not None:
            thmask = hsv_compute_threshold_mask(slice_img, self.data_manager.thresholds)
        
        # 合并掩码
        combined = thmask if mask is None else (
            np.maximum(thmask, mask) if thmask is not None else mask
        )
        
        # 转换为显示图像
        if isinstance(slice_img, np.ndarray):
            if slice_img.ndim == 2:
                base = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2RGB) if cv2 is not None else np.stack([slice_img]*3, axis=-1)
                
                # 移除绿色阈值覆盖（已禁用，恢复原始显示）
                # if thmask is not None:
                #     green_overlay = base.copy()
                #     green_alpha = 0.25
                #     green_mask = thmask == 0
                #     green_overlay[green_mask] = [0, 255, 0]
                #     base = (green_overlay * green_alpha + base * (1 - green_alpha)).astype(np.uint8)
                
                # 添加红色掩码覆盖
                if combined is not None:
                    overlay_red = base.copy()
                    overlay_red[combined > 0] = [255, 0, 0]
                    alpha_red = 0.4
                    base = (overlay_red * alpha_red + base * (1 - alpha_red)).astype(np.uint8)
                
                # 绘制 SAM2 提示
                if self.sam2_controller and index == self.slice_slider.value():
                    # 获取当前框 (优先级：预览框 > 已确认框)
                    sam2_ui_ctrl = getattr(self.parent_widget, 'sam2_ui_ctrl', None)
                    active_box = None
                    is_preview = False
                    
                    if sam2_ui_ctrl and sam2_ui_ctrl.current_box:
                        active_box = sam2_ui_ctrl.current_box
                        is_preview = True
                    elif self.sam2_controller.prompt_box:
                        active_box = self.sam2_controller.prompt_box
                    
                    # 1. 绘制选区视觉增强 (Spotlight 效果)
                    if active_box:
                        x1, y1, x2, y2 = active_box
                        # 创建一个半透明遮罩层
                        overlay = base.copy()
                        # 变暗非选区
                        mask_ext = np.ones(base.shape[:2], dtype=bool)
                        mask_ext[y1:y2, x1:x2] = False
                        overlay[mask_ext] = (overlay[mask_ext] * 0.5).astype(np.uint8)
                        # 稍微提亮选区或加个淡蓝色底
                        if is_preview:
                            overlay[y1:y2, x1:x2] = (overlay[y1:y2, x1:x2] * 0.9 + np.array([0, 50, 50]) * 0.1).astype(np.uint8)
                        base = overlay

                        # 绘制框边框
                        color = [255, 255, 0] if is_preview else [0, 255, 255]
                        thickness = 1 if is_preview else 2
                        if cv2 is not None:
                            cv2.rectangle(base, (x1, y1), (x2, y2), color, thickness)
                    
                    # 2. 绘制点提示
                    for px, py, label in self.sam2_controller.prompt_points:
                        color = [0, 255, 0] if label == 1 else [255, 0, 0]
                        radius = 5
                        if cv2 is not None:
                            cv2.circle(base, (px, py), radius, color, -1)
                        else:
                            y_min = max(0, py - radius)
                            y_max = min(base.shape[0], py + radius + 1)
                            x_min = max(0, px - radius)
                            x_max = min(base.shape[1], px + radius + 1)
                            for y in range(y_min, y_max):
                                for x in range(x_min, x_max):
                                    if ((x - px)**2 + (y - py)**2) <= radius**2:
                                        base[y, x] = color
                
                h, w = base.shape[:2]
                qimg = QImage(base.data, w, h, w*3, QImage.Format_RGB888)
                
            elif slice_img.ndim == 3 and slice_img.shape[2] == 3:
                img = slice_img.copy()
                if cv2 is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 绘制 SAM2 提示
                if self.sam2_controller and index == self.slice_slider.value():
                    sam2_ui_ctrl = getattr(self.parent_widget, 'sam2_ui_ctrl', None)
                    active_box = None
                    is_preview = False
                    
                    if sam2_ui_ctrl and sam2_ui_ctrl.current_box:
                        active_box = sam2_ui_ctrl.current_box
                        is_preview = True
                    elif self.sam2_controller.prompt_box:
                        active_box = self.sam2_controller.prompt_box

                    # 1. 绘制选区视觉增强 (Spotlight 效果)
                    if active_box:
                        x1, y1, x2, y2 = active_box
                        overlay = img.copy()
                        # 变暗非选区
                        mask_ext = np.ones(img.shape[:2], dtype=bool)
                        mask_ext[y1:y2, x1:x2] = False
                        overlay[mask_ext] = (overlay[mask_ext] * 0.5).astype(np.uint8)
                        img = overlay

                        # 绘制框边框
                        color = [255, 255, 0] if is_preview else [0, 255, 255]
                        thickness = 1 if is_preview else 2
                        if cv2 is not None:
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    
                    # 2. 绘制点提示
                    for px, py, label in self.sam2_controller.prompt_points:
                        color = [0, 255, 0] if label == 1 else [255, 0, 0]
                        radius = 5
                        if cv2 is not None:
                            cv2.circle(img, (px, py), radius, color, -1)
                        else:
                            y_min = max(0, py - radius)
                            y_max = min(img.shape[0], py + radius + 1)
                            x_min = max(0, px - radius)
                            x_max = min(img.shape[1], px + radius + 1)
                            for y in range(y_min, y_max):
                                for x in range(x_min, x_max):
                                    if ((x - px)**2 + (y - py)**2) <= radius**2:
                                        img[y, x] = color
                
                # 移除绿色阈值覆盖（已禁用，恢复原始显示）
                # if thmask is not None:
                #     green_overlay = img.copy()
                #     green_alpha = 0.25
                #     green_mask = thmask == 0
                #     green_overlay[green_mask] = [0, 255, 0]
                #     img = (green_overlay * green_alpha + img * (1 - green_alpha)).astype(np.uint8)
                
                if combined is not None:
                    overlay_red = img.copy()
                    overlay_red[combined > 0] = [255, 0, 0]
                    alpha_red = 0.4
                    img = (overlay_red * alpha_red + img * (1 - alpha_red)).astype(np.uint8)
                
                h, w, _ = img.shape
                qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                self.image_label.setText("Unsupported image shape")
                return
            
            pix = QPixmap.fromImage(qimg)
            # 适应 label 大小显示
            if self.image_label.width() > 0 and self.image_label.height() > 0:
                pix = pix.scaled(
                    self.image_label.width(), 
                    self.image_label.height(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
            self.image_label.setPixmap(pix)
        else:
            self.image_label.setText("No image data")
    
    def handle_threshold_change(self, name: str, value: int):
        """处理阈值变化"""
        self.data_manager.set_threshold(name, value)
        self.update_slice_display(self.slice_slider.value())
    
    def handle_reset_thresholds(self, *args):
        """重置阈值"""
        self.data_manager.reset_thresholds()
        # 更新滑块值
        for name, slider in self.threshold_sliders.items():
            slider.setValue(self.data_manager.get_threshold(name))
        self.update_slice_display(self.slice_slider.value())
    
    @handle_errors(parent_widget=None, show_message=True, context="Batch Thresholding")
    def handle_apply_threshold_all(self, *args):
        """异步应用阈值到所有切片"""
        volume = self.data_manager.get_volume()
        if volume is None:
            QMessageBox.warning(self.parent_widget, "Warning", "No volume loaded.")
            return
        if apply_threshold_all is None:
            return
        
        # UI 准备
        if self.btn_apply_all:
            self.btn_apply_all.setEnabled(False)
        
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(True)
            self.parent_widget.recon_progress.setValue(0)
            
        # 启动异步线程
        run_in_thread(
            self,
            apply_threshold_all,
            on_finished=self._on_threshold_all_finished,
            on_error=self._on_threshold_all_error,
            on_progress=self._on_threshold_all_progress,
            volume=volume,
            thresholds=self.data_manager.thresholds
        )

    def _on_threshold_all_progress(self, p):
        """更新批量阈值分割进度"""
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setValue(p)

    def _on_threshold_all_finished(self, result):
        """批量阈值分割完成回调"""
        masks, seg_volume = result
        
        # 恢复 UI 状态
        if self.btn_apply_all:
            self.btn_apply_all.setEnabled(True)
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
            
        self.data_manager.masks = masks
        self.data_manager.seg_mask_volume = seg_volume
        self.update_slice_display(self.slice_slider.value())
        
        QMessageBox.information(
            self.parent_widget, 
            "Success", 
            f"Threshold applied to all {len(masks)} slices."
        )

    def _on_threshold_all_error(self, error_msg):
        """批量阈值分割错误回调"""
        logger.error(f"Batch threshold error: {error_msg}")
        if self.btn_apply_all:
            self.btn_apply_all.setEnabled(True)
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        QMessageBox.critical(self.parent_widget, "Error", f"Batch thresholding failed:\n{error_msg}")
    
    @handle_errors(parent_widget=None, show_message=True, context="Save Masks")
    def handle_save_masks(self, *args, parent_widget=None):
        """异步保存掩码"""
        if not self.data_manager.masks:
            QMessageBox.warning(self.parent_widget, "Warning", "No masks to save.")
            return
            
        # 建议默认目录名包含患者ID
        default_dir_name = "masks_export"
        if parent_widget and hasattr(parent_widget, 'patient_context') and parent_widget.patient_context:
            p_id = parent_widget.patient_context.get('patient_id', 'unknown')
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            default_dir_name = f"masks_{p_id}_{timestamp}"
            
        directory = QFileDialog.getExistingDirectory(parent_widget, "Select Output Folder", "")
        if not directory:
            return
            
        # 在选定目录下创建子目录
        from pathlib import Path
        export_path = Path(directory) / default_dir_name
        export_path.mkdir(parents=True, exist_ok=True)
        
        # UI 准备
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(True)
            self.parent_widget.recon_progress.setValue(0)
            
        try:
            from surgical_robot_app.data_io.file_handler import FileHandler
        except ImportError:
            from data_io.file_handler import FileHandler
        fh = FileHandler()
        
        # 启动异步线程
        run_in_thread(
            self,
            fh.save_masks,
            on_finished=lambda _: self._on_save_masks_finished(export_path),
            on_error=self._on_save_masks_error,
            on_progress=self._on_save_masks_progress,
            directory=str(export_path),
            masks=self.data_manager.masks
        )

    def _on_save_masks_progress(self, p):
        """更新保存进度"""
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setValue(p)

    def _on_save_masks_finished(self, export_path):
        """保存完成回调"""
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        QMessageBox.information(self.parent_widget, "Success", f"Masks saved to:\n{export_path}")

    def _on_save_masks_error(self, error_msg):
        """保存错误回调"""
        logger.error(f"Save masks error: {error_msg}")
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        QMessageBox.critical(self.parent_widget, "Error", f"Failed to save masks:\n{error_msg}")
    
    def handle_clear_masks(self, *args):
        """清除所有掩码"""
        self.data_manager.clear_all_masks()
        volume = self.data_manager.get_volume()
        if volume is not None:
            self.update_slice_display(self.slice_slider.value())
    
    def handle_toggle_threshold(self, threshold_group):
        """切换阈值组显示/隐藏"""
        self.threshold_collapsed = not self.threshold_collapsed
        threshold_group.setVisible(not self.threshold_collapsed)
        # 强制更新布局以确保隐藏/显示生效
        if hasattr(threshold_group, 'parent') and threshold_group.parent():
            threshold_group.parent().update()
        # 更新按钮文本（需要外部设置）
    
    def handle_brush_mode(self, checked: bool, *args):
        """切换画笔模式"""
        self.brush_mode = checked
        # 如果画笔模式被激活，关闭擦除模式（互斥）
        if checked:
            self.eraser_mode = False
            # 如果进入画笔模式，退出 SAM2 picking 模式，避免冲突
            if self.on_exit_sam2_picking:
                self.on_exit_sam2_picking()
    
    def handle_eraser_mode(self, checked: bool, *args):
        """切换擦除模式"""
        self.eraser_mode = checked
        # 如果擦除模式被激活，关闭画笔模式（互斥）
        if checked:
            self.brush_mode = False
            # 如果进入擦除模式，退出 SAM2 picking 模式，避免冲突
            if self.on_exit_sam2_picking:
                self.on_exit_sam2_picking()
    
    def handle_mouse_event(self, obj, event: QEvent, sam2_picking_mode: bool = False) -> bool:
        """
        处理鼠标事件（画笔/擦除/SAM2点击/SAM2画框）
        """
        if obj is not self.image_label:
            return False
        
        et = event.type()
        if et not in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
            return False
        
        # 对于移动事件，必须按下左键
        if et == QEvent.MouseMove and not (event.buttons() & Qt.LeftButton):
            return False
            
        volume = self.data_manager.get_volume()
        if volume is None:
            return True
        
        idx = self.slice_slider.value()
        slice_img = volume[idx]
        h, w = slice_img.shape[:2]
        
        pix = self.image_label.pixmap()
        if pix is None:
            return True
        
        disp_w = pix.width()
        disp_h = pix.height()
        lab_w = self.image_label.width()
        lab_h = self.image_label.height()
        x = event.x()
        y = event.y()
        
        off_x = (lab_w - disp_w) / 2.0
        off_y = (lab_h - disp_h) / 2.0
        
        # 坐标映射：将点击位置映射到原始图像坐标
        sx = w / disp_w
        sy = h / disp_h
        ix = int((x - off_x) * sx)
        iy = int((y - off_y) * sy)
        
        # 边界检查
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        
        # --- SAM2 交互逻辑 (B 方案：支持框选和精修点) ---
        if sam2_picking_mode:
            # 获取 SAM2UIController 的引用 (通过 MainView)
            sam2_ui_ctrl = getattr(self.parent_widget, 'sam2_ui_ctrl', None)
            if not sam2_ui_ctrl:
                return False

            if sam2_ui_ctrl.sam2_prompt_mode == "box":
                if et == QEvent.MouseButtonPress:
                    return sam2_ui_ctrl.handle_sam2_mouse_press(ix, iy)
                elif et == QEvent.MouseMove:
                    return sam2_ui_ctrl.handle_sam2_mouse_move(ix, iy)
                elif et == QEvent.MouseButtonRelease:
                    processed = sam2_ui_ctrl.handle_sam2_mouse_release(ix, iy)
                    # 释放后强制更新一次显示，以清除预览框并显示分割结果
                    self.update_slice_display(idx)
                    return processed
            else:
                # 点模式
                if et == QEvent.MouseButtonPress:
                    return sam2_ui_ctrl.handle_sam2_click(ix, iy)
            return True

        # --- 手动分割模式 (画笔/擦除) ---
        if not sam2_picking_mode:
            if et == QEvent.MouseButtonRelease:
                return True # 手动模式不处理释放
                
            shape = self.data_manager.get_volume_shape()
            if shape:
                depth, h, w = shape[:3]
                self.data_manager.ensure_masks((depth, h, w))
                if self.data_manager.masks[idx] is None:
                    self.data_manager.masks[idx] = np.zeros((h, w), dtype=np.uint8)
                
                if self.eraser_mode:
                    self.data_manager.masks = self.manual_controller.apply_eraser(
                        self.data_manager.masks, idx, (ix, iy), self.brush_size,
                    )
                else:
                    self.data_manager.masks = self.manual_controller.apply_brush(
                        self.data_manager.masks, idx, (ix, iy), self.brush_size,
                    )
                
                self.data_manager._update_seg_mask_volume()
                self.update_slice_display(idx)
                return True
        
        return False

