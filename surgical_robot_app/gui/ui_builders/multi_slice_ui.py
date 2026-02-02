"""
多视图切片 UI 构建器

职责：创建多视图切片相关的 UI 组件（三个正交切片视图、3D点选择等）
"""

from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QSlider,
    QWidget,
    QScrollArea,
)
from PyQt5.QtCore import Qt


def build_multi_slice_ui() -> dict:
    """
    构建多视图切片 UI
    
    Returns:
        dict: 包含所有 UI 控件的字典
    """
    # 右边面板：2D切片视图
    right_panel = QGroupBox("2D Slice Views")
    right_layout = QVBoxLayout()
    right_panel.setLayout(right_layout)
    
    # 添加折叠/展开按钮
    right_header = QWidget()
    right_header_layout = QHBoxLayout()
    right_header.setLayout(right_header_layout)
    btn_toggle_slice_view = QPushButton("▼ Hide")
    btn_toggle_slice_view.setMaximumWidth(80)
    right_header_layout.addWidget(btn_toggle_slice_view)
    right_header_layout.addStretch()
    right_layout.addWidget(right_header)
    
    # 创建滚动区域
    right_scroll = QScrollArea()
    right_scroll.setWidgetResizable(True)
    right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    
    # 创建内容容器
    right_content = QWidget()
    right_content_layout = QVBoxLayout()
    right_content.setLayout(right_content_layout)
    
    # 创建三个切片窗口的容器
    slices_container = QWidget()
    slices_container_layout = QVBoxLayout()
    slices_container.setLayout(slices_container_layout)
    
    # 创建三个切片视图（冠状、矢状、横断）
    slice_views = {}
    for plane_type, title in [("coronal", "Coronal (Y)"), ("sagittal", "Sagittal (X)"), ("axial", "Axial (Z)")]:
        group = QGroupBox(title)
        layout = QVBoxLayout()
        group.setLayout(layout)
        
        # 切片显示标签
        label = QLabel("No data")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(150)
        label.setProperty("plane_type", plane_type)
        layout.addWidget(label)
        
        # 切片位置滑块
        slider = QSlider(Qt.Horizontal)
        slider.setEnabled(False)
        slider.setMinimum(0)
        slider.setMaximum(0)
        slider.setMinimumHeight(30)  # 设置最小高度，确保滑块可见
        slider.setMaximumHeight(30)  # 设置最大高度，保持一致的显示
        layout.addWidget(slider)
        
        slices_container_layout.addWidget(group)
        slice_views[plane_type] = {
            'group': group,
            'label': label,
            'slider': slider,
        }
    
    right_content_layout.addWidget(slices_container)
    
    # 将内容容器添加到滚动区域
    right_scroll.setWidget(right_content)
    
    # 将滚动区域添加到布局
    right_layout.addWidget(right_scroll)
    
    return {
        'right_panel': right_panel,
        'btn_toggle_slice_view': btn_toggle_slice_view,
        'right_scroll': right_scroll,
        'right_content': right_content,
        'coronal_group': slice_views['coronal']['group'],
        'coronal_label': slice_views['coronal']['label'],
        'coronal_slider': slice_views['coronal']['slider'],
        'sagittal_group': slice_views['sagittal']['group'],
        'sagittal_label': slice_views['sagittal']['label'],
        'sagittal_slider': slice_views['sagittal']['slider'],
        'axial_group': slice_views['axial']['group'],
        'axial_label': slice_views['axial']['label'],
        'axial_slider': slice_views['axial']['slider'],
    }

