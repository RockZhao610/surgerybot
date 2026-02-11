"""
3D视图 UI 构建器

职责：创建 3D 视图和路径规划相关的 UI 组件
"""

from PyQt5.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QWidget,
    QProgressBar,
    QSlider,
    QCheckBox,
)
from PyQt5.QtCore import Qt

try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
except Exception:
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
    except Exception:
        QVTKWidget = None


def build_view3d_ui(parent_widget=None) -> dict:
    """
    构建 3D 视图 UI
    
    Args:
        parent_widget: 父窗口（用于创建 VTK Widget）
    
    Returns:
        dict: 包含所有 UI 控件的字典
    """
    # 3D重建面板
    middle_panel = QGroupBox("3D Reconstruction")
    middle_layout = QVBoxLayout()
    middle_panel.setLayout(middle_layout)
    
    # VTK Widget
    if QVTKWidget is None:
        vtk_widget = QLabel("VTK Qt bindings not available")
        vtk_widget.setAlignment(Qt.AlignCenter)
    else:
        vtk_widget = QVTKWidget(parent_widget)
    middle_layout.addWidget(vtk_widget)
    
    # 3D重建控制按钮
    btn_recon = QPushButton("Reconstruct 3D")
    btn_recon.setFixedHeight(40) # Make primary action stand out
    middle_layout.addWidget(btn_recon)
    
    # Horizontal layout for secondary 3D tools
    tools_3d_layout = QHBoxLayout()
    btn_open_model = QPushButton("Open Model")
    btn_open_model.setObjectName("secondary_btn")
    
    btn_reset_cam = QPushButton("Reset Camera")
    btn_reset_cam.setObjectName("secondary_btn")
    
    btn_clear_3d = QPushButton("Clear View")
    btn_clear_3d.setObjectName("secondary_btn")
    
    btn_open_window = QPushButton("Open in Window")
    btn_open_window.setObjectName("secondary_btn")
    btn_open_window.setToolTip("Open complete 3D Reconstruction interface in a separate window")
    
    tools_3d_layout.addWidget(btn_open_model)
    tools_3d_layout.addWidget(btn_reset_cam)
    tools_3d_layout.addWidget(btn_clear_3d)
    tools_3d_layout.addWidget(btn_open_window)
    middle_layout.addLayout(tools_3d_layout)
    
    # 进度条
    recon_progress = QProgressBar()
    recon_progress.setFixedHeight(12)
    recon_progress.setTextVisible(False)
    recon_progress.setRange(0, 100)
    recon_progress.setValue(0)
    recon_progress.setVisible(False)
    middle_layout.addWidget(recon_progress)
    
    # 状态标签
    vtk_status = QLabel("VTK: Initialized" if QVTKWidget is not None else "VTK: Not available")
    vtk_status.setStyleSheet("color: #7F8C8D; font-style: italic; font-size: 11px;")
    middle_layout.addWidget(vtk_status)
    
    # 路径规划面板
    path_group = QGroupBox("Path Planning")
    path_layout = QVBoxLayout()
    path_group.setLayout(path_layout)
    
    # 选点模式按钮
    pick_buttons = QWidget()
    pick_buttons_layout = QHBoxLayout()
    pick_buttons_layout.setContentsMargins(0, 0, 0, 0)
    pick_buttons.setLayout(pick_buttons_layout)
    btn_pick_start = QPushButton("Start Point")
    btn_pick_waypoint = QPushButton("Waypoint")
    btn_pick_end = QPushButton("End Point")
    pick_buttons_layout.addWidget(btn_pick_start)
    pick_buttons_layout.addWidget(btn_pick_waypoint)
    pick_buttons_layout.addWidget(btn_pick_end)
    path_layout.addWidget(pick_buttons)
    
    # 路径点列表
    path_list = QListWidget()
    path_list.setMaximumHeight(80)
    path_layout.addWidget(QLabel("Path Point Sequence:"))
    path_layout.addWidget(path_list)
    
    # 路径控制按钮
    path_control = QWidget()
    path_control_layout = QHBoxLayout()
    path_control_layout.setContentsMargins(0, 0, 0, 0)
    path_control.setLayout(path_control_layout)
    btn_generate_path = QPushButton("Generate")
    btn_generate_path.setFixedHeight(35)
    
    btn_replay_path = QPushButton("Replay")
    btn_replay_path.setObjectName("secondary_btn")
    btn_replay_path.setFixedHeight(35)
    btn_replay_path.setToolTip("Replay the RRT exploration animation")
    btn_replay_path.setEnabled(False)  # 初始禁用，规划完成后启用
    
    btn_simulate_path = QPushButton("Simulate")
    btn_simulate_path.setObjectName("secondary_btn")
    btn_simulate_path.setFixedHeight(35)
    btn_simulate_path.setToolTip("Animate instrument movement along the planned path")
    btn_simulate_path.setEnabled(False)  # 初始禁用，规划完成后启用
    
    btn_save_path = QPushButton("Save")
    btn_save_path.setObjectName("secondary_btn")
    btn_save_path.setFixedHeight(35)
    
    btn_reset_path = QPushButton("Reset")
    btn_reset_path.setObjectName("secondary_btn")
    btn_reset_path.setFixedHeight(35)
    
    path_control_layout.addWidget(btn_generate_path, 2)
    path_control_layout.addWidget(btn_replay_path, 1)
    path_control_layout.addWidget(btn_simulate_path, 1)
    path_control_layout.addWidget(btn_save_path, 1)
    path_control_layout.addWidget(btn_reset_path, 1)
    path_layout.addWidget(path_control)
    
    # Simulation speed slider row
    sim_speed_row = QWidget()
    sim_speed_layout = QHBoxLayout()
    sim_speed_layout.setContentsMargins(0, 0, 0, 0)
    sim_speed_layout.setSpacing(6)
    sim_speed_row.setLayout(sim_speed_layout)
    
    sim_speed_label = QLabel("Speed:")
    sim_speed_label.setFixedWidth(42)
    sim_speed_label.setStyleSheet("color: #7F8C8D; font-size: 11px;")
    
    sim_speed_slider = QSlider(Qt.Horizontal)
    sim_speed_slider.setRange(1, 10)   # 1=slowest, 10=fastest
    sim_speed_slider.setValue(5)        # default medium
    sim_speed_slider.setTickPosition(QSlider.TicksBelow)
    sim_speed_slider.setTickInterval(1)
    sim_speed_slider.setToolTip("Adjust simulation & replay speed (1=slow, 10=fast)")
    
    sim_speed_value_label = QLabel("5")
    sim_speed_value_label.setFixedWidth(18)
    sim_speed_value_label.setAlignment(Qt.AlignCenter)
    sim_speed_value_label.setStyleSheet("color: #7F8C8D; font-size: 11px;")
    
    sim_speed_layout.addWidget(sim_speed_label)
    sim_speed_layout.addWidget(sim_speed_slider, 1)
    sim_speed_layout.addWidget(sim_speed_value_label)
    path_layout.addWidget(sim_speed_row)
    
    # Safety zone checkbox
    chk_safety_zone = QCheckBox("Show Safety Zone")
    chk_safety_zone.setToolTip("Display a semi-transparent tube around the path representing the safety margin")
    chk_safety_zone.setEnabled(False)  # enabled after path is generated
    chk_safety_zone.setStyleSheet("color: #7F8C8D; font-size: 11px;")
    path_layout.addWidget(chk_safety_zone)
    
    middle_layout.addWidget(path_group)
    
    return {
        'middle_panel': middle_panel,
        'vtk_widget': vtk_widget,
        'btn_recon': btn_recon,
        'btn_open_model': btn_open_model,
        'btn_reset_cam': btn_reset_cam,
        'btn_clear_3d': btn_clear_3d,
        'btn_open_window': btn_open_window,
        'recon_progress': recon_progress,
        'vtk_status': vtk_status,
        'path_group': path_group,
        'btn_pick_start': btn_pick_start,
        'btn_pick_waypoint': btn_pick_waypoint,
        'btn_pick_end': btn_pick_end,
        'path_list': path_list,
        'btn_generate_path': btn_generate_path,
        'btn_replay_path': btn_replay_path,
        'btn_simulate_path': btn_simulate_path,
        'btn_save_path': btn_save_path,
        'btn_reset_path': btn_reset_path,
        'sim_speed_slider': sim_speed_slider,
        'sim_speed_value_label': sim_speed_value_label,
        'chk_safety_zone': chk_safety_zone,
    }

