"""
切片编辑器 UI 构建器

职责：创建切片编辑器相关的 UI 组件（主切片视图、HSV阈值控制等）
"""

from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QSlider,
    QLabel,
    QWidget,
    QSizePolicy,
)
from PyQt5.QtCore import Qt


def build_slice_editor_ui(thresholds: dict) -> dict:
    """
    构建切片编辑器 UI
    
    Args:
        thresholds: 阈值字典
    
    Returns:
        dict: 包含所有 UI 控件的字典
    """
    # 创建主容器
    image_group = QGroupBox("Slice View")
    image_layout = QVBoxLayout()
    image_group.setLayout(image_layout)
    
    # 图像显示标签
    image_label = QLabel("No image")
    image_label.setAlignment(Qt.AlignCenter)
    image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    image_label.setMinimumHeight(400)
    image_layout.addWidget(image_label)
    
    # 画笔和擦除模式控制（使用GridLayout实现对齐）
    brush_eraser_control = QWidget()
    brush_eraser_layout = QGridLayout()
    brush_eraser_layout.setContentsMargins(0, 0, 0, 0)
    brush_eraser_control.setLayout(brush_eraser_layout)
    
    # 画笔模式按钮
    btn_brush_mode = QPushButton("Brush")
    btn_brush_mode.setCheckable(True)
    btn_brush_mode.setFixedHeight(30)
    
    # 擦除模式按钮
    btn_eraser_mode = QPushButton("Eraser")
    btn_eraser_mode.setCheckable(True)
    btn_eraser_mode.setFixedHeight(30)
    
    # 掩码管理按钮
    btn_save_masks = QPushButton("Save Masks")
    btn_save_masks.setFixedHeight(30)
    
    btn_clear_masks = QPushButton("Clear Masks")
    btn_clear_masks.setObjectName("secondary_btn")
    btn_clear_masks.setFixedHeight(30)
    
    # 将按钮添加到布局
    brush_eraser_layout.addWidget(btn_brush_mode, 0, 0)
    brush_eraser_layout.addWidget(btn_eraser_mode, 0, 1)
    brush_eraser_layout.addWidget(btn_save_masks, 1, 0)
    brush_eraser_layout.addWidget(btn_clear_masks, 1, 1)
    
    image_layout.addWidget(brush_eraser_control)
    
    # 切片标签和滑块
    slice_info_layout = QHBoxLayout()
    slice_label = QLabel("Slice: 0")
    slice_label.setStyleSheet("color: #7F8C8D; font-weight: bold;")
    slice_slider = QSlider(Qt.Horizontal)
    slice_slider.setEnabled(False)
    slice_slider.setMinimum(0)
    slice_slider.setMaximum(0)
    
    slice_info_layout.addWidget(slice_label)
    slice_info_layout.addWidget(slice_slider)
    image_layout.addLayout(slice_info_layout)
    
    # 阈值折叠按钮
    btn_toggle_threshold = QPushButton("Threshold Controls ▾")
    btn_toggle_threshold.setObjectName("secondary_btn")
    btn_toggle_threshold.setFixedHeight(25)
    image_layout.addWidget(btn_toggle_threshold)
    
    # 阈值组
    threshold_group = QGroupBox("Threshold")
    th_layout = QGridLayout()
    threshold_group.setLayout(th_layout)
    
    # 创建阈值滑块
    h1_low = QSlider(Qt.Horizontal)
    h1_high = QSlider(Qt.Horizontal)
    h2_low = QSlider(Qt.Horizontal)
    h2_high = QSlider(Qt.Horizontal)
    s_low = QSlider(Qt.Horizontal)
    s_high = QSlider(Qt.Horizontal)
    v_low = QSlider(Qt.Horizontal)
    v_high = QSlider(Qt.Horizontal)
    
    # 设置滑块范围
    for s in [h1_low, h1_high, h2_low, h2_high]:
        s.setRange(0, 180)
    for s in [s_low, s_high, v_low, v_high]:
        s.setRange(0, 255)
    
    # 设置滑块初始值
    h1_low.setValue(thresholds.get("h1_low", 0))
    h1_high.setValue(thresholds.get("h1_high", 10))
    h2_low.setValue(thresholds.get("h2_low", 160))
    h2_high.setValue(thresholds.get("h2_high", 180))
    s_low.setValue(thresholds.get("s_low", 50))
    s_high.setValue(thresholds.get("s_high", 255))
    v_low.setValue(thresholds.get("v_low", 50))
    v_high.setValue(thresholds.get("v_high", 255))
    
    # 添加标签和滑块到布局
    th_layout.addWidget(QLabel("Hue1 Low"), 0, 0)
    th_layout.addWidget(h1_low, 0, 1)
    th_layout.addWidget(QLabel("Hue1 High"), 1, 0)
    th_layout.addWidget(h1_high, 1, 1)
    th_layout.addWidget(QLabel("Hue2 Low"), 2, 0)
    th_layout.addWidget(h2_low, 2, 1)
    th_layout.addWidget(QLabel("Hue2 High"), 3, 0)
    th_layout.addWidget(h2_high, 3, 1)
    th_layout.addWidget(QLabel("Sat Low"), 4, 0)
    th_layout.addWidget(s_low, 4, 1)
    th_layout.addWidget(QLabel("Sat High"), 5, 0)
    th_layout.addWidget(s_high, 5, 1)
    th_layout.addWidget(QLabel("Val Low"), 6, 0)
    th_layout.addWidget(v_low, 6, 1)
    th_layout.addWidget(QLabel("Val High"), 7, 0)
    th_layout.addWidget(v_high, 7, 1)
    
    # 设置 GridLayout 的列拉伸比例，确保滑块列可以扩展适应容器宽度
    th_layout.setColumnStretch(0, 0)  # 标签列不拉伸
    th_layout.setColumnStretch(1, 1)  # 滑块列可以拉伸
    
    # 阈值控制按钮
    btn_reset_th = QPushButton("Reset")
    btn_reset_th.setObjectName("secondary_btn")
    btn_apply_all = QPushButton("Apply to All")
    btn_apply_all.setObjectName("secondary_btn")
    
    # 添加到GridLayout
    th_layout.addWidget(btn_reset_th, 8, 0)
    th_layout.addWidget(btn_apply_all, 8, 1)
    
    image_layout.addWidget(threshold_group)
    
    return {
        'group': image_group,
        'image_label': image_label,
        'btn_brush_mode': btn_brush_mode,
        'btn_eraser_mode': btn_eraser_mode,
        'slice_label': slice_label,
        'slice_slider': slice_slider,
        'btn_toggle_threshold': btn_toggle_threshold,
        'threshold_group': threshold_group,
        'h1_low': h1_low,
        'h1_high': h1_high,
        'h2_low': h2_low,
        'h2_high': h2_high,
        's_low': s_low,
        's_high': s_high,
        'v_low': v_low,
        'v_high': v_high,
        'btn_reset_th': btn_reset_th,
        'btn_apply_all': btn_apply_all,
        'btn_save_masks': btn_save_masks,
        'btn_clear_masks': btn_clear_masks,
    }

