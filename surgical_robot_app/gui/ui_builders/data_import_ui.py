"""
数据导入 UI 构建器

职责：创建数据导入相关的 UI 组件
"""

from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QListWidget,
    QAbstractItemView,
    QWidget,
)
from PyQt5.QtCore import Qt


def build_data_import_ui() -> dict:
    """
    构建数据导入 UI
    
    Returns:
        dict: 包含所有 UI 控件的字典
    """
    # 创建主容器
    import_group = QGroupBox("Data Import")
    import_layout = QVBoxLayout()
    import_group.setLayout(import_layout)
    
    # 按钮容器（使用GridLayout实现对齐）
    import_buttons = QWidget()
    import_buttons_layout = QGridLayout()
    import_buttons.setLayout(import_buttons_layout)
    
    btn_select = QPushButton("Select Folder")
    btn_select.setFixedWidth(110)
    btn_import = QPushButton("Import")
    btn_import.setFixedWidth(110)
    
    import_buttons_layout.addWidget(btn_select, 0, 0, alignment=Qt.AlignHCenter)
    import_buttons_layout.addWidget(btn_import, 0, 1, alignment=Qt.AlignHCenter)
    import_buttons_layout.setColumnStretch(0, 1)
    import_buttons_layout.setColumnStretch(1, 1)
    import_layout.addWidget(import_buttons)
    
    # 文件列表
    file_list = QListWidget()
    file_list.setFixedHeight(120) # Constrain height to avoid pushing other elements down
    file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    import_layout.addWidget(file_list)
    
    # 操作按钮容器
    action_buttons_container = QWidget()
    action_buttons_layout = QHBoxLayout()
    action_buttons_layout.setContentsMargins(0, 0, 0, 0)
    action_buttons_container.setLayout(action_buttons_layout)
    
    btn_delete = QPushButton("Exclude")
    btn_delete.setObjectName("secondary_btn")
    
    btn_undo = QPushButton("Undo")
    btn_undo.setObjectName("secondary_btn")
    btn_undo.setEnabled(False)
    
    action_buttons_layout.addWidget(btn_delete)
    action_buttons_layout.addWidget(btn_undo)
    import_layout.addWidget(action_buttons_container)
    
    return {
        'group': import_group,
        'btn_select': btn_select,
        'btn_import': btn_import,
        'file_list': file_list,
        'btn_delete': btn_delete,
        'btn_undo': btn_undo,
    }

