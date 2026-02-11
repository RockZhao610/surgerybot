"""
SAM2UIController: SAM2 UI 控制器

职责：
- 处理 SAM2 UI 相关的事件（模型加载、提示控制、分割等）
- 管理 SAM2 UI 状态（按钮启用/禁用、状态显示等）
- 不直接依赖 Qt Widget，通过回调与 MainWindow 通信
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLabel, QRadioButton, QPushButton
from PyQt5.QtCore import QObject

try:
    from surgical_robot_app.data_io.data_manager import DataManager
    from surgical_robot_app.segmentation.sam2_controller import SAM2Controller
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.utils.error_handler import handle_errors
    from surgical_robot_app.utils.threading_utils import run_in_thread
except ImportError:
    try:
        from data_io.data_manager import DataManager
        from segmentation.sam2_controller import SAM2Controller
        from utils.logger import get_logger
        from utils.threading_utils import run_in_thread
        from utils.error_handler import handle_errors
    except ImportError:
        DataManager = None  # type: ignore
        SAM2Controller = None  # type: ignore
        def handle_errors(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        def get_logger(name):
            return logging.getLogger(name)
        def run_in_thread(*args, **kwargs):
            return None, None

logger = get_logger("surgical_robot_app.gui.controllers.sam2_ui_controller")


class SAM2UIController(QObject):
    """SAM2 UI 控制器"""
    
    def __init__(
        self,
        sam2_controller: SAM2Controller,
        data_manager: DataManager,
        sam2_status: QLabel,
        radio_auto: QRadioButton,
        radio_manual: QRadioButton,
        btn_add_positive: QPushButton,
        btn_add_negative: QPushButton,
        btn_undo_positive: QPushButton,
        btn_clear_prompts: QPushButton,
        btn_sam2_volume_3d: QPushButton,
        parent_widget=None,
        btn_add_box: QPushButton = None,
        btn_switch_mask: QPushButton = None, # 新增
        prompt_mode_group=None, # 保留以兼容旧代码，但不再使用
    ):
        """
        初始化 SAM2 UI 控制器
        """
        super().__init__()
        self.sam2_controller = sam2_controller
        self.data_manager = data_manager
        self.sam2_status = sam2_status
        self.radio_auto = radio_auto
        self.radio_manual = radio_manual
        self.btn_add_positive = btn_add_positive
        self.btn_add_negative = btn_add_negative
        self.btn_undo_positive = btn_undo_positive
        self.btn_clear_prompts = btn_clear_prompts
        self.btn_sam2_volume_3d = btn_sam2_volume_3d
        self.btn_add_box = btn_add_box
        self.btn_switch_mask = btn_switch_mask
        self.parent_widget = parent_widget
        
        # UI 状态
        self.sam2_picking_mode = False
        self.sam2_prompt_mode = "point"  # 'point' or 'box'
        self.sam2_prompt_point_type = "positive"
        
        # 框选交互状态
        self.box_start_point: Optional[Tuple[int, int]] = None
        self.current_box: Optional[Tuple[int, int, int, int]] = None
        
        # 回调函数
        self.on_segmentation_mode_changed: Optional[Callable] = None
        self.on_slice_display_update: Optional[Callable] = None
        self.on_mask_updated: Optional[Callable] = None
        self.on_box_preview_update: Optional[Callable] = None  # 用于绘制蓝色预览框
    
    def handle_load_model(self, *args):
        """处理加载 SAM2 模型事件"""
        # 建议的模型目录
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        models_dir = project_root / "models" / "sam2"
        default_dir = str(models_dir) if models_dir.exists() else str(project_root / "models")
        
        # 选择模型文件
        model_path, _ = QFileDialog.getOpenFileName(
            self.parent_widget,
            "Select SAM2 Model File",
            default_dir,
            "PyTorch Model (*.pt);;All Files (*)"
        )
        
        if not model_path:
            return
        
        try:
            # 检查文件是否存在
            if not os.path.exists(model_path):
                QMessageBox.warning(self.parent_widget, "Error", f"Model file not found:\n{model_path}")
                self.sam2_status.setText("SAM2: File not found")
                return
            
            # 检查文件大小
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size < 10:
                QMessageBox.warning(
                    self.parent_widget,
                    "Error",
                    f"Model file seems too small ({file_size:.1f} MB).\n"
                    "Please check if the download completed correctly."
                )
                self.sam2_status.setText("SAM2: File too small")
                return
            
            # 加载模型
            self.sam2_status.setText("SAM2: Loading...")
            if self.parent_widget:
                self.parent_widget.repaint()
            
            success = self.sam2_controller.load_model(model_path)
            
            if success:
                self.sam2_status.setText("SAM2: Model loaded")
                self.sam2_status.setStyleSheet("color: #27AE60; font-weight: bold;")
                
                # 启用 SAM2 相关控件
                if self.btn_add_positive: self.btn_add_positive.setEnabled(True)
                if self.btn_add_negative: self.btn_add_negative.setEnabled(True)
                if self.btn_add_box: self.btn_add_box.setEnabled(True)
                if self.btn_switch_mask: self.btn_switch_mask.setEnabled(True) # 启用切换
                if self.btn_undo_positive: self.btn_undo_positive.setEnabled(True)
                if self.btn_clear_prompts: self.btn_clear_prompts.setEnabled(True)
                if self.btn_sam2_volume_3d: self.btn_sam2_volume_3d.setEnabled(True)
            else:
                self.sam2_status.setText("SAM2: Load failed")
                QMessageBox.warning(self.parent_widget, "Error", "Failed to load SAM2 model.")
        
        except Exception as e:
            self.sam2_status.setText("SAM2: Error")
            QMessageBox.critical(self.parent_widget, "Error", f"Error loading SAM2 model:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def handle_segmentation_mode_changed(self, checked: bool):
        """处理分割模式变化事件"""
        if checked:  # 自动模式（SAM2）
            self.sam2_status.setText("SAM2: Auto mode active")
            # 如果模型未加载，提示用户
            if not self.sam2_controller.is_model_loaded():
                self.sam2_status.setText("SAM2: Please load model first")
        else:  # 手动模式
            self.sam2_picking_mode = False
            if hasattr(self, 'sam2_status'):
                self.sam2_status.setText("SAM2: Manual mode active")
        
        # 调用回调
        if self.on_segmentation_mode_changed:
            self.on_segmentation_mode_changed(checked)
    
    def handle_prompt_mode_changed(self):
        """处理提示模式变化事件"""
        # 这个功能在 MainWindow 中处理，这里只是占位
        pass
    
    def handle_set_prompt_mode(self, mode: str):
        """设置提示模式并同步按钮状态"""
        if mode == "positive":
            self.sam2_prompt_mode = "point"
            self.sam2_prompt_point_type = "positive"
            if self.btn_add_positive: self.btn_add_positive.setChecked(True)
            if self.btn_add_negative: self.btn_add_negative.setChecked(False)
            if self.btn_add_box: self.btn_add_box.setChecked(False)
        elif mode == "negative":
            self.sam2_prompt_mode = "point"
            self.sam2_prompt_point_type = "negative"
            if self.btn_add_positive: self.btn_add_positive.setChecked(False)
            if self.btn_add_negative: self.btn_add_negative.setChecked(True)
            if self.btn_add_box: self.btn_add_box.setChecked(False)
        elif mode == "box":
            self.sam2_prompt_mode = "box"
            if self.btn_add_positive: self.btn_add_positive.setChecked(False)
            if self.btn_add_negative: self.btn_add_negative.setChecked(False)
            if self.btn_add_box: self.btn_add_box.setChecked(True)

        self.sam2_picking_mode = True
        status_text = {
            "positive": "Click to add positive point",
            "negative": "Click to add negative point",
            "box": "Drag to draw selection box"
        }.get(mode if mode == "box" else self.sam2_prompt_point_type)
        self.sam2_status.setText(f"SAM2: {status_text}")
        
        # 通知退出擦除模式
        if hasattr(self, 'on_exit_eraser_mode') and self.on_exit_eraser_mode:
            self.on_exit_eraser_mode()
    
    def handle_sam2_mouse_press(self, x: int, y: int):
        """处理鼠标按下事件 (针对框选模式)"""
        if not self.sam2_picking_mode or self.sam2_prompt_mode != "box":
            return False
        
        # 按下时初始化起点，并清空旧的 Box 提示以开始新一轮动态显示
        self.box_start_point = (x, y)
        self.current_box = None
        self.sam2_controller.clear_prompts() # 开始新框选前清空
        return True

    def handle_sam2_mouse_move(self, x: int, y: int):
        """处理鼠标移动事件 (显示实时预览框 + 实时动态分割)"""
        if not self.sam2_picking_mode or self.sam2_prompt_mode != "box" or not self.box_start_point:
            return False
        
        x1, y1 = self.box_start_point
        x2, y2 = x, y
        
        # 1. 更新当前 Box 坐标 [x_min, y_min, x_max, y_max]
        self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        
        # 2. 将 Box 设置到模型控制器中
        self.sam2_controller.set_box_prompt(*self.current_box)
        
        # 3. 实时执行分割（动态预览）
        # 注意：这里我们快速执行分割，但不保存到 DataManager 的永久掩码中，
        # 或者直接保存以实现“即见即所得”
        self._execute_segmentation()
        
        # 4. 通知 UI 绘制黄色预览框
        if self.on_box_preview_update:
            self.on_box_preview_update(self.current_box)
        return True

    def handle_sam2_mouse_release(self, x: int, y: int):
        """处理鼠标释放事件 (完成框选并锁定结果)"""
        if not self.sam2_picking_mode or self.sam2_prompt_mode != "box" or not self.box_start_point:
            return False
        
        # 释放时只需做最后的清理和状态重置，因为分割已经在 Move 过程中实时完成了
        self.box_start_point = None
        if self.on_box_preview_update: 
            self.on_box_preview_update(None)
        
        self.sam2_status.setText("SAM2: Box confirmed. Use points to refine if needed.")
        return True

    def handle_sam2_click(self, x: int, y: int) -> bool:
        """处理点击事件 (针对精修点模式)"""
        if not self.sam2_controller.is_model_loaded():
            return False
        
        # 如果是点模式，直接添加点并分割
        if self.sam2_picking_mode and self.sam2_prompt_mode == "point":
            label = 1 if self.sam2_prompt_point_type == "positive" else 0
            self.sam2_controller.add_point_prompt(x, y, label)
            return self._execute_segmentation()
        
        return False

    def _execute_segmentation(self) -> bool:
        """核心分割执行逻辑 (B 方案：框点混合预测)"""
        volume = self.data_manager.get_volume()
        if volume is None: return False
        
        current_idx = self._get_current_slice_index()
        if current_idx is None: return False
        
        slice_img = volume[current_idx]
        
        try:
            # 准备 RGB 图像
            if slice_img.ndim == 2:
                rgb_img = np.stack([slice_img] * 3, axis=-1)
            else:
                rgb_img = slice_img
            
            # 执行组合预测 (混合框与点)
            mask = self.sam2_controller.segment_slice(rgb_img)
            
            if mask is not None:
                shape = self.data_manager.get_volume_shape()
                if shape:
                    self.data_manager.ensure_masks((shape[0], mask.shape[0], mask.shape[1]))
                    self.data_manager.set_mask(current_idx, mask)
                
                if self.on_slice_display_update:
                    self.on_slice_display_update(current_idx)
                if self.on_mask_updated:
                    self.on_mask_updated()
                return True
        except Exception as e:
            logger.error(f"Mixed segmentation failed: {e}")
            return False
        return False
    
    def handle_switch_mask(self, *args):
        """处理切换掩码候选层级"""
        if not self.sam2_controller.is_model_loaded():
            return
        
        mask = self.sam2_controller.switch_mask_candidate()
        if mask is not None:
            current_idx = self._get_current_slice_index()
            if current_idx is not None:
                self.data_manager.set_mask(current_idx, mask)
                # 更新视觉
                if self.on_slice_display_update:
                    self.on_slice_display_update(current_idx)
                
                # 显示当前是哪个层级
                level_names = ["Whole/Object", "Part", "Sub-part"]
                idx = self.sam2_controller.current_candidate_idx
                name = level_names[idx] if idx < len(level_names) else f"Candidate {idx}"
                self.sam2_status.setText(f"SAM2: Switched to {name}")
    
    def handle_undo_last_positive(self, *args):
        """撤销最后一次添加的正点"""
        if not self.sam2_controller.is_model_loaded():
            QMessageBox.warning(self.parent_widget, "Warning", "SAM2 model not loaded.")
            return
        
        # 撤销最后一个正点
        removed = self.sam2_controller.undo_last_positive_point()
        
        if removed:
            # 清除对应的 mask
            current_idx = self._get_current_slice_index()
            if current_idx is not None:
                self.data_manager.set_mask(current_idx, None)
                
                # 更新显示
                if self.on_slice_display_update:
                    self.on_slice_display_update(current_idx)
            
            self.sam2_status.setText("SAM2: Last positive point removed")
        else:
            QMessageBox.information(self.parent_widget, "Info", "No positive points to undo.")
    
    def handle_clear_prompts(self, *args):
        """清除所有提示"""
        self.sam2_controller.clear_prompts()
        self.sam2_picking_mode = False
        self.sam2_status.setText("SAM2: Prompts cleared")
        
        # 更新显示
        current_idx = self._get_current_slice_index()
        if current_idx is not None and self.on_slice_display_update:
            self.on_slice_display_update(current_idx)
    
    def exit_picking_mode(self):
        """退出 SAM2 picking 模式"""
        self.sam2_picking_mode = False
        if self.btn_add_positive: self.btn_add_positive.setChecked(False)
        if self.btn_add_negative: self.btn_add_negative.setChecked(False)
        if self.btn_add_box: self.btn_add_box.setChecked(False)
        if self.sam2_status:
            self.sam2_status.setText("SAM2: Picking mode exited")
    
    @handle_errors(parent_widget=None, show_message=True, context="SAM2 Volume Segmentation")
    def handle_sam2_volume_3d(self, *args):
        """处理异步 SAM2 批量分割事件"""
        if not self.sam2_controller.is_model_loaded():
            QMessageBox.warning(self.parent_widget, "Warning", "SAM2 model not loaded.")
            return
        
        volume = self.data_manager.get_volume()
        if volume is None:
            QMessageBox.warning(self.parent_widget, "Warning", "No volume loaded.")
            return
        
        # 检查是否有提示点
        if not self.sam2_controller.prompt_points:
            QMessageBox.warning(self.parent_widget, "Warning", "Please add at least one prompt point first.")
            return
        
        # 获取当前切片索引
        current_idx = self._get_current_slice_index()
        if current_idx is None:
            QMessageBox.warning(self.parent_widget, "Error", "Cannot determine current slice index.")
            return
            
        # UI 准备
        self.sam2_status.setText("SAM2: Planning volume task...")
        if self.btn_sam2_volume_3d:
            self.btn_sam2_volume_3d.setEnabled(False)
        
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(True)
            self.parent_widget.recon_progress.setValue(0)
            
        # 启动异步线程
        run_in_thread(
            self,
            self.sam2_controller.segment_volume,
            on_finished=self._on_volume_seg_finished,
            on_error=self._on_volume_seg_error,
            on_progress=self._on_volume_seg_progress,
            volume=volume,
            slice_idx=current_idx
        )

    def _on_volume_seg_progress(self, p):
        """更新批量分割进度"""
        self.sam2_status.setText(f"SAM2: Processing volume... {p}%")
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setValue(p)

    def _on_volume_seg_finished(self, masks):
        """批量分割完成回调"""
        # 恢复 UI 状态
        if self.btn_sam2_volume_3d:
            self.btn_sam2_volume_3d.setEnabled(True)
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
            
        if masks is not None and masks.size > 0:
            # 逐切片写入当前标签（保留其他标签的分割数据）
            for i in range(masks.shape[0]):
                slice_mask = masks[i]
                if slice_mask is not None and np.sum(slice_mask > 0) > 0:
                    self.data_manager.set_mask(i, slice_mask)
            
            # 更新显示
            current_idx = self._get_current_slice_index()
            if current_idx is not None and self.on_slice_display_update:
                self.on_slice_display_update(current_idx)
            
            # 更新掩码回调
            if self.on_mask_updated:
                self.on_mask_updated()
            
            self.sam2_status.setText(f"SAM2: Volume processed ({masks.shape[0]} slices)")
            QMessageBox.information(
                self.parent_widget,
                "Success",
                f"SAM2 volume segmentation completed.\n{masks.shape[0]} slices processed."
            )
        else:
            self._on_volume_seg_error("Segmentation returned no result.")

    def _on_volume_seg_error(self, error_msg):
        """批量分割错误回调"""
        logger.error(f"SAM2 volume error: {error_msg}")
        self.sam2_status.setText("SAM2: Error occurred")
        if self.btn_sam2_volume_3d:
            self.btn_sam2_volume_3d.setEnabled(True)
        if hasattr(self.parent_widget, 'recon_progress'):
            self.parent_widget.recon_progress.setVisible(False)
        QMessageBox.critical(self.parent_widget, "Error", f"SAM2 volume segmentation error:\n{error_msg}")
    
    def handle_start_segmentation(self, *args):
        """处理开始分割事件（单张切片）"""
        if not self.sam2_controller.is_model_loaded():
            QMessageBox.warning(self.parent_widget, "Warning", "SAM2 model not loaded.")
            return
        
        volume = self.data_manager.get_volume()
        if volume is None:
            QMessageBox.warning(self.parent_widget, "Warning", "No volume loaded.")
            return
        
        current_idx = self._get_current_slice_index()
        if current_idx is None:
            return
        
        slice_img = volume[current_idx]
        
        try:
            # 准备图像
            if slice_img.ndim == 2:
                rgb_img = slice_img.copy()
                if rgb_img.max() > 255:
                    rgb_img = (rgb_img / rgb_img.max() * 255).astype('uint8')
                rgb_img = rgb_img.astype('uint8')
                rgb_img = rgb_img[:, :, None] if rgb_img.ndim == 2 else rgb_img
                if rgb_img.shape[2] == 1:
                    rgb_img = rgb_img.repeat(3, axis=2)
            elif slice_img.ndim == 3:
                rgb_img = slice_img.copy()
                if rgb_img.shape[2] == 1:
                    rgb_img = rgb_img.repeat(3, axis=2)
                rgb_img = rgb_img.astype('uint8')
            else:
                QMessageBox.warning(self.parent_widget, "Error", "Unsupported image format.")
                return
            
            # 执行分割
            mask = self.sam2_controller.segment_slice(rgb_img)
            
            if mask is not None:
                # 保存分割结果
                shape = self.data_manager.get_volume_shape()
                if shape:
                    depth = shape[0]
                    self.data_manager.ensure_masks((depth, mask.shape[0], mask.shape[1]))
                    self.data_manager.set_mask(current_idx, mask)
                
                # 更新显示
                if self.on_slice_display_update:
                    self.on_slice_display_update(current_idx)
                
                # 更新掩码回调
                if self.on_mask_updated:
                    self.on_mask_updated()
                
                self.sam2_status.setText("SAM2: Segmentation complete")
            else:
                QMessageBox.warning(self.parent_widget, "Error", "SAM2 segmentation returned no result.")
        
        except Exception as e:
            QMessageBox.warning(self.parent_widget, "Error", f"SAM2 segmentation failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _get_current_slice_index(self) -> Optional[int]:
        """获取当前切片索引（需要从 MainWindow 获取）"""
        # 这个方法需要通过回调从 MainWindow 获取当前切片索引
        if hasattr(self, '_current_slice_getter'):
            return self._current_slice_getter()
        return None
    
    def set_current_slice_getter(self, getter: Callable[[], Optional[int]]):
        """设置获取当前切片索引的函数"""
        self._current_slice_getter = getter

