"""
SAM2 UI 构建器

职责：创建 SAM2 相关的 UI 组件（含多标签选择器）
"""

from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QRadioButton,
    QLabel,
    QWidget,
    QComboBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QColor


def _create_color_icon(color_rgb, size=16):
    """创建纯色方块图标"""
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(*color_rgb))
    return QIcon(pixmap)


def build_sam2_ui() -> dict:
    """
    构建 SAM2 UI
    
    Returns:
        dict: 包含所有 UI 控件的字典
    """
    # 分割模式选择
    seg_group = QGroupBox("Segmentation")
    seg_layout = QVBoxLayout()
    seg_group.setLayout(seg_layout)
    
    radio_manual = QRadioButton("Manual Segmentation")
    radio_auto = QRadioButton("Automatic Segmentation (SAM2)")
    radio_manual.setChecked(True)
    seg_layout.addWidget(radio_manual)
    seg_layout.addWidget(radio_auto)
    
    # ===== 标签选择器 =====
    label_group = QGroupBox("Label")
    label_layout = QVBoxLayout()
    label_layout.setSpacing(6)
    label_group.setLayout(label_layout)
    
    # 标签下拉框 + 按钮行
    label_row = QWidget()
    label_row_layout = QHBoxLayout()
    label_row_layout.setContentsMargins(0, 0, 0, 0)
    label_row_layout.setSpacing(6)
    label_row.setLayout(label_row_layout)
    
    combo_label = QComboBox()
    combo_label.setMinimumHeight(30)
    combo_label.setToolTip("Select current segmentation label")
    # 默认标签会在 MainView 中通过 DataManager 初始化
    
    btn_add_label = QPushButton("+ Add")
    btn_add_label.setFixedHeight(30)
    btn_add_label.setToolTip("Add a new segmentation label")
    btn_add_label.setStyleSheet("""
        QPushButton { background-color: #27AE60; color: white; font-weight: bold; font-size: 12px; border-radius: 4px; padding: 0 8px; }
        QPushButton:hover { background-color: #2ECC71; }
    """)
    
    btn_remove_label = QPushButton("- Del")
    btn_remove_label.setFixedHeight(30)
    btn_remove_label.setToolTip("Delete the current label and its segmentation data")
    btn_remove_label.setStyleSheet("""
        QPushButton { background-color: #E74C3C; color: white; font-weight: bold; font-size: 12px; border-radius: 4px; padding: 0 8px; }
        QPushButton:hover { background-color: #EC7063; }
    """)
    
    label_row_layout.addWidget(combo_label, 1)
    label_row_layout.addWidget(btn_add_label)
    label_row_layout.addWidget(btn_remove_label)
    
    label_layout.addWidget(label_row)
    seg_layout.addWidget(label_group)
    
    # SAM2 设置组
    sam2_group = QGroupBox("SAM2 Settings")
    sam2_layout = QVBoxLayout()
    sam2_group.setLayout(sam2_layout)
    
    # Load SAM2 按钮
    load_sam2_container = QWidget()
    load_sam2_layout = QHBoxLayout()
    load_sam2_layout.setContentsMargins(0, 0, 0, 0)
    load_sam2_container.setLayout(load_sam2_layout)
    btn_load_sam2 = QPushButton("Load SAM2")
    btn_load_sam2.setFixedHeight(35)
    load_sam2_layout.addWidget(btn_load_sam2)
    sam2_layout.addWidget(load_sam2_container)
    
    sam2_status = QLabel("SAM2: Not loaded")
    sam2_status.setStyleSheet("color: #7F8C8D; font-style: italic;")
    sam2_layout.addWidget(sam2_status)
    
    # 提示控制按钮组
    prompt_buttons_container = QWidget()
    prompt_buttons_layout = QGridLayout()
    prompt_buttons_layout.setContentsMargins(0, 0, 0, 0)
    prompt_buttons_layout.setSpacing(8)
    prompt_buttons_container.setLayout(prompt_buttons_layout)
    
    # 创建互斥按钮组（类似工具栏）
    btn_add_positive = QPushButton("＋ Pos Point")
    btn_add_positive.setCheckable(True)
    btn_add_positive.setStyleSheet("""
        QPushButton { background-color: #F8FAF8; color: #27AE60; border: 1px solid #27AE60; font-weight: bold; }
        QPushButton:checked { background-color: #27AE60; color: white; }
        QPushButton:hover:!checked { background-color: #EBF5EB; }
    """)
    
    btn_add_negative = QPushButton("－ Neg Point")
    btn_add_negative.setCheckable(True)
    btn_add_negative.setStyleSheet("""
        QPushButton { background-color: #FAF8F8; color: #E74C3C; border: 1px solid #E74C3C; font-weight: bold; }
        QPushButton:checked { background-color: #E74C3C; color: white; }
        QPushButton:hover:!checked { background-color: #FDEDED; }
    """)
    
    btn_add_box = QPushButton("⬜ Box Select")
    btn_add_box.setCheckable(True)
    btn_add_box.setStyleSheet("""
        QPushButton { background-color: #FAF9F0; color: #F1C40F; border: 1px solid #F1C40F; font-weight: bold; }
        QPushButton:checked { background-color: #F1C40F; color: white; }
        QPushButton:hover:!checked { background-color: #FEF9E7; }
    """)
    
    btn_switch_mask = QPushButton("🔄 Switch Level")
    btn_switch_mask.setStyleSheet("""
        QPushButton { background-color: #F4F6F7; color: #7F8C8D; border: 1px solid #7F8C8D; font-weight: bold; }
        QPushButton:hover { background-color: #E5E8E8; }
    """)
    
    btn_clear_prompts = QPushButton("Clear All")
    btn_clear_prompts.setObjectName("secondary_btn")
    
    btn_undo_positive = QPushButton("Undo Last")
    btn_undo_positive.setObjectName("secondary_btn")

    # 布局排列
    prompt_buttons_layout.addWidget(btn_add_positive, 0, 0)
    prompt_buttons_layout.addWidget(btn_add_negative, 0, 1)
    prompt_buttons_layout.addWidget(btn_add_box, 1, 0)
    prompt_buttons_layout.addWidget(btn_switch_mask, 1, 1) # 替换原来的 Clear 位置
    prompt_buttons_layout.addWidget(btn_undo_positive, 2, 0)
    prompt_buttons_layout.addWidget(btn_clear_prompts, 2, 1) # Clear 移到这里
    
    sam2_layout.addWidget(prompt_buttons_container)
    
    # 默认禁用
    btn_add_positive.setEnabled(False)
    btn_add_negative.setEnabled(False)
    btn_add_box.setEnabled(False)
    btn_switch_mask.setEnabled(False) # 默认禁用
    btn_undo_positive.setEnabled(False)
    btn_clear_prompts.setEnabled(False)
    
    seg_layout.addWidget(sam2_group)
    
    # 分割按钮
    seg_buttons_container = QWidget()
    seg_buttons_layout = QHBoxLayout()
    seg_buttons_layout.setContentsMargins(0, 0, 0, 0)
    seg_buttons_container.setLayout(seg_buttons_layout)
    
    btn_start_seg = QPushButton("Start Seg")
    btn_start_seg.setFixedHeight(35)
    btn_sam2_volume_3d = QPushButton("SAM2 Vol")
    btn_sam2_volume_3d.setFixedHeight(35)
    btn_sam2_volume_3d.setEnabled(False)
    
    seg_buttons_layout.addWidget(btn_start_seg)
    seg_buttons_layout.addWidget(btn_sam2_volume_3d)
    seg_layout.addWidget(seg_buttons_container)
    
    return {
        'seg_group': seg_group,
        'radio_manual': radio_manual,
        'radio_auto': radio_auto,
        'sam2_group': sam2_group,
        'btn_load_sam2': btn_load_sam2,
        'sam2_status': sam2_status,
        'btn_add_positive': btn_add_positive,
        'btn_add_negative': btn_add_negative,
        'btn_add_box': btn_add_box,
        'btn_switch_mask': btn_switch_mask,
        'btn_undo_positive': btn_undo_positive,
        'btn_clear_prompts': btn_clear_prompts,
        'btn_start_seg': btn_start_seg,
        'btn_sam2_volume_3d': btn_sam2_volume_3d,
        # 标签选择器
        'label_group': label_group,
        'combo_label': combo_label,
        'btn_add_label': btn_add_label,
        'btn_remove_label': btn_remove_label,
    }

