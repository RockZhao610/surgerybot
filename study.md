# Surgical Robot App — 4-Day Study Guide

> 目标：4 天系统掌握项目全部代码，按「底层 → 上层」「数据 → 算法 → 界面」的顺序推进。
> 每天约 2-3 小时阅读量。建议边看代码边在本文件打 ✅。

---

## Day 1: 项目骨架 + 数据层 + 工具层

> 目标：理解项目如何启动、数据如何流转、通用工具的作用。

### 1.1 入口与窗口切换
- [ ] `main.py` (56 行) — 程序入口，创建 QApplication 和 MainWindow
- [ ] `gui/main_window.py` (139 行) — QStackedWidget 管理三个页面
- [ ] `gui/views/welcome_view.py` (81 行) — 欢迎页
- [ ] `gui/views/patient_info_view.py` (179 行) — 患者信息页
- [ ] **理解启动流程**：`main()` → `MainWindow()` → `WelcomeView` → `PatientInfoView` → `MainView`

### 1.2 配置管理
- [ ] `config/settings.py` (279 行) — 所有配置的 dataclass 定义
  - 重点看：`PathPlanningConfig`、`SegmentationConfig`、`View3DConfig`
  - 理解 `get_config()` 单例模式和 `load_config()` 从 JSON 加载
- [ ] `config/__init__.py` (26 行) — 导出接口

### 1.3 数据管理
- [ ] `data_io/data_manager.py` (408 行) — **核心数据中心**，必须彻底理解
  - volume / masks / metadata 的存取
  - 多标签管理：`add_label()`, `remove_label()`, `set_current_label()`
  - `get_seg_mask_volume()` — 多标签 mask 如何组装为 3D 体
  - `DEFAULT_LABEL_COLORS` — 预定义颜色表
- [ ] `data_io/sequence_reader.py` (158 行) — DICOM/PNG 序列读取
- [ ] `data_io/file_handler.py` (36 行) — mask/路径点保存

### 1.4 通用工具
- [ ] `utils/logger.py` (163 行) — 日志系统
- [ ] `utils/error_handler.py` (244 行) — `@handle_errors` 装饰器，自动 try-catch + QMessageBox
- [ ] `utils/threading_utils.py` (89 行) — `Worker` + `run_in_thread()`，后台线程执行耗时任务
  - 重点看 `tree_update` 信号是如何自动注入 `tree_callback` 的
- [ ] `utils/import_helper.py` (241 行) — 安全导入，浏览即可

### 1.5 Day 1 检验
读完后你应该能回答：
1. 程序启动后经过哪几个页面？切换逻辑在哪里？
2. DataManager 中 `masks` 和 `seg_mask_volume` 的关系是什么？
3. `run_in_thread()` 如何把后台任务的结果回调到主线程？

---

## Day 2: GUI 框架 + 2D 切片编辑 + 分割

> 目标：理解 UI 的构建方式（Builder → View → Controller 三层分离），以及分割的完整流程。

### 2.1 UI 构建器（UI Builders）
- [ ] `gui/ui_builders/data_import_ui.py` (79 行) — 数据导入面板
- [ ] `gui/ui_builders/slice_editor_ui.py` (181 行) — 切片编辑面板
- [ ] `gui/ui_builders/sam2_ui.py` (208 行) — SAM2 分割面板 + 标签选择器
- [ ] `gui/ui_builders/view3d_ui.py` (213 行) — 3D 视图 + 路径规划面板
- [ ] **理解模式**：每个 builder 返回一个 `dict`，MainView 通过 key 拿到控件引用

### 2.2 主视图（MainView）— 最大的文件
- [ ] `gui/views/main_view.py` (1691 行) — **不需要逐行看**，按块理解：
  - `__init__()` → 初始化数据、控制器、管理器
  - `_build_ui()` → 调用各 UI Builder 组装界面
  - `_init_controllers()` → 创建各控制器
  - `_connect_signals()` → 信号-槽连接（重要！理解按钮如何触发逻辑）
  - `_init_managers()` → 初始化 View3DManager、EventDispatcher
  - 标签管理方法：`_refresh_label_combo()`, `_on_add_label()`, `_on_remove_label()`

### 2.3 事件分发
- [ ] `gui/managers/event_dispatcher.py` (131 行) — 鼠标/键盘事件路由到正确的控制器

### 2.4 数据导入控制器
- [ ] `gui/controllers/data_import_controller.py` (216 行) — Select Folder / Import / Undo

### 2.5 切片编辑控制器
- [ ] `gui/controllers/slice_editor_controller.py` (605 行)
  - 切片显示：`update_slice_display()`
  - 多标签 2D 覆盖：`_apply_multilabel_overlay()`
  - 画笔/擦除模式
  - HSV 阈值分割（单片 + 批量异步）
  - 保存 mask

### 2.6 手动分割
- [ ] `segmentation/manual_controller.py` (150 行) — 画笔历史 + undo
- [ ] `segmentation/hsv_threshold.py` (130 行) — HSV 阈值算法

### 2.7 SAM2 分割
- [ ] `segmentation/sam2_segmenter.py` (277 行) — SAM2 底层推理（点/框提示 → mask）
- [ ] `segmentation/sam2_controller.py` (258 行) — SAM2 状态管理（提示点、模式、全卷分割）
- [ ] `gui/controllers/sam2_ui_controller.py` (544 行)
  - 加载模型
  - 正/负点、框选模式
  - 单片分割 vs 全卷分割 (`handle_sam2_volume`)
  - 理解 mask 如何写入 DataManager 的当前标签

### 2.8 Day 2 检验
读完后你应该能回答：
1. 用户画完一笔后，mask 数据经过了哪些函数？最终存在哪里？
2. SAM2 全卷分割的流程是什么？它如何利用 `run_in_thread`？
3. 多标签分割时，切换标签后再分割，为什么不会覆盖其他标签？

---

## Day 3: 3D 重建 + VTK 工具 + 对话框

> 目标：理解 mask → 3D 模型的完整链路，以及 VTK 坐标系。

### 3.1 VTK 工具层
- [ ] `vtk_utils/coords.py` (116 行) — **核心坐标转换**
  - `get_model_bounds()` — 获取模型边界
  - `world_to_space()` — 世界坐标 → [0,100] 归一化空间
  - `space_to_world()` — [0,100] 归一化空间 → 世界坐标
- [ ] `vtk_utils/markers.py` (70 行) — 创建球体标记
- [ ] `vtk_utils/path.py` (87 行) — 创建折线 Actor
- [ ] `vtk_utils/coordinate_system.py` (269 行) — XYZ 坐标轴 Actor

### 3.2 3D 重建核心
- [ ] `3d_recon/recon_core.py` (64 行) — Marching Cubes 算法封装
  - 输入：mask 体 + spacing
  - 输出：vtkPolyData（三角网格）

### 3.3 3D 视图管理器
- [ ] `gui/managers/view3d_manager.py` (568 行) — **重点**
  - `reconstruct_3d()` — 多标签异步重建入口
  - `_reconstruct_3d_task()` — 逐标签提取二值体 → Marching Cubes → 填孔 → 法线
  - `_on_reconstruction_finished()` — 创建多个 Actor（每个标签一个颜色）
  - `open_stl_model()` — 加载外部 STL
  - STL 文件保存逻辑

### 3.4 3D 视图控制器
- [ ] `gui/view3d_controller.py` (158 行) — 相机控制、STL 加载、清除

### 3.5 对话框
- [ ] `gui/dialogs/path_point_picker_dialog.py` (501 行) — 三视图选点（体数据）
- [ ] `gui/dialogs/coordinate_plane_picker_dialog.py` (660 行) — XY/XZ 平面选点（STL）
- [ ] `gui/dialogs/path_point_edit_dialog.py` (741 行) — 路径点编辑（拖拽 + 碰撞检测）
- [ ] `gui/dialogs/view3d_expanded_window.py` (642 行) — 独立 3D 窗口
- [ ] `gui/dialogs/view3d_reconstruction_window.py` (343 行) — 3D 重建独立窗口

### 3.6 辅助
- [ ] `gui/managers/point_selection_manager.py` (474 行) — 3D 点选择
- [ ] `gui/utils/model_surface_picker.py` (339 行) — 两次点击表面选点
- [ ] `geometry/slice_geometry.py` (148 行) — 切片几何计算

### 3.7 Day 3 检验
读完后你应该能回答：
1. 从用户点击 "Reconstruct 3D" 到模型显示出来，经过哪些函数？
2. `space_to_world()` 和 `world_to_space()` 的公式是什么？为什么需要两套坐标？
3. 多标签重建时，每个标签的颜色是从哪里获取的？

---

## Day 4: 路径规划 + 可视化 + 仿真

> 目标：理解 RRT/A* 算法、碰撞检测、以及路径的可视化和仿真。

### 4.1 碰撞检测基础
- [ ] `path_planning/point_cloud_utils.py` (389 行) — 从 mask/mesh 生成点云 + KD-tree 碰撞检测
- [ ] `path_planning/sdf_utils.py` (567 行) — SDF（符号距离场）碰撞检测
  - 理解 SDF 的含义：负值 = 内部，正值 = 外部
  - 三线性插值查询

### 4.2 路径规划算法
- [ ] `path_planning/rrt_planner.py` (252 行) — **RRT 算法核心**
  - `plan()` — 单段规划 + `tree_callback` 实时回调
  - `plan_multi_segment()` — 多段规划（经过中间点）
  - `smooth_path()` — 路径平滑
  - 理解 goal_bias、step_size、safety_radius 的作用
- [ ] `path_planning/a_star_planner.py` (453 行) — A* 算法（备用）
- [ ] `path_planning/path_evaluator.py` (151 行) — 路径质量评估（长度、安全距离、曲率）

### 4.3 路径控制器
- [ ] `path_planning/path_controller.py` (1130 行) — **最大的后端文件**
  - 起点/终点/中间点管理 + undo/redo
  - `set_obstacle_from_volume()` — 从 mask 体创建碰撞检测器
  - `set_obstacle_from_mesh()` — 从 STL 网格创建碰撞检测器
  - `generate_path()` — 调用 RRT 多段规划 + 后处理（重采样/平滑/碰撞复验）
  - `_space_to_physical()` / `_physical_to_space()` — 坐标转换
  - 路径后处理管线：`_post_process_path()` → `_resample_path()` → `_smooth_path()` → `_simplify_ramer_douglas_peucker()`

### 4.4 路径服务层
- [ ] `services/path_service.py` (149 行) — 桥接 DataManager + PathController
  - `prepare_obstacles()` — 多标签 mask → 二值障碍 → 碰撞检测器
  - `plan_path()` — 调用 PathController 规划
  - `plan_local_segment()` — 局部段重规划（用于编辑路径点后）

### 4.5 路径可视化管理器
- [ ] `gui/managers/path_viz_manager.py` (613 行)
  - 点标记：`add_point_marker()`
  - 路径线：`visualize_path()`
  - RRT 探索树：`update_rrt_tree_viz()` / `clear_rrt_tree_viz()`
  - 仿真动画：`update_sim_instrument()` / `update_sim_trail()` / `clear_sim_viz()`
  - 安全管道：`show_safety_tube()` / `clear_safety_tube()`

### 4.6 路径 UI 控制器
- [ ] `gui/controllers/path_ui_controller.py` (1384 行) — **最大的 GUI 文件**，分块理解：
  - **选点模式** (L138-L390)：智能选择对话框（体数据 vs STL）
  - **路径生成** (L536-L640)：异步规划 + RRT 实时可视化回调
  - **RRT 回放** (L1090-L1200)：QTimer 逐帧动画
  - **路径仿真** (L1210-L1330)：器械球体沿路径移动 + 轨迹绘制
  - **安全区域** (L1335-L1380)：vtkTubeFilter 半透明管道
  - **路径点编辑** (L977-L1070)：局部 RRT 重规划
  - **路径评估** (L694-L730)：评估并显示报告
  - **保存/重置** (L732-L810)

### 4.7 Day 4 检验
读完后你应该能回答：
1. RRT 算法的主循环做了什么？`goal_bias` 是怎么用的？
2. 用户点击 "Generate" 后，数据经过了 `PathService` → `PathController` → `RRTPlanner` 的哪些方法？
3. 实时 RRT 可视化的数据流是什么？（后台线程 → 信号 → 主线程渲染）
4. 编辑路径点后，局部重规划是怎么实现的？如果失败怎么办？
5. Safety Zone 管道的半径是怎么计算的？

---

## 附录：文件一览表（按行数排序）

| 行数 | 文件 | 模块 |
|------|------|------|
| 1691 | gui/views/main_view.py | GUI 主视图 |
| 1384 | gui/controllers/path_ui_controller.py | 路径 UI 控制器 |
| 1130 | gui/controllers/path_controller.py | 路径规划控制器 |
| 741 | gui/dialogs/path_point_edit_dialog.py | 路径点编辑对话框 |
| 660 | gui/dialogs/coordinate_plane_picker_dialog.py | 坐标平面选点 |
| 642 | gui/dialogs/view3d_expanded_window.py | 独立 3D 窗口 |
| 613 | gui/managers/path_viz_manager.py | 路径可视化 |
| 605 | gui/controllers/slice_editor_controller.py | 切片编辑控制器 |
| 568 | gui/managers/view3d_manager.py | 3D 视图管理器 |
| 567 | path_planning/sdf_utils.py | SDF 碰撞检测 |
| 544 | gui/controllers/sam2_ui_controller.py | SAM2 UI 控制器 |
| 501 | gui/dialogs/path_point_picker_dialog.py | 三视图选点 |
| 474 | gui/managers/point_selection_manager.py | 3D 点选择 |
| 453 | path_planning/a_star_planner.py | A* 算法 |
| 410 | gui/controllers/multi_slice_controller.py | 多视图切片 |
| 408 | data_io/data_manager.py | 数据管理中心 |
| 389 | path_planning/point_cloud_utils.py | 点云碰撞检测 |
| 343 | gui/dialogs/view3d_reconstruction_window.py | 重建独立窗口 |
| 339 | gui/utils/model_surface_picker.py | 表面选点 |
| 279 | config/settings.py | 配置管理 |
| 277 | segmentation/sam2_segmenter.py | SAM2 推理 |
| 269 | vtk_utils/coordinate_system.py | 坐标系 Actor |
| 258 | segmentation/sam2_controller.py | SAM2 控制器 |
| 252 | path_planning/rrt_planner.py | RRT 算法 |
| 244 | utils/error_handler.py | 错误处理 |
| 241 | utils/import_helper.py | 安全导入 |
| 216 | gui/controllers/data_import_controller.py | 数据导入控制器 |
| 213 | gui/ui_builders/view3d_ui.py | 3D 视图 UI |
| 208 | gui/ui_builders/sam2_ui.py | SAM2 UI |
| 181 | gui/ui_builders/slice_editor_ui.py | 切片编辑 UI |
| 179 | gui/views/patient_info_view.py | 患者信息页 |
| 163 | utils/logger.py | 日志系统 |
| 158 | data_io/sequence_reader.py | 序列读取 |
| 158 | gui/view3d_controller.py | 3D 视图控制 |
| 151 | path_planning/path_evaluator.py | 路径评估 |
| 150 | segmentation/manual_controller.py | 手动分割 |
| 149 | services/path_service.py | 路径服务 |
| 148 | geometry/slice_geometry.py | 切片几何 |
| 139 | gui/main_window.py | 主窗口 |
| 131 | gui/managers/event_dispatcher.py | 事件分发 |
| 130 | segmentation/hsv_threshold.py | HSV 阈值 |
| 116 | vtk_utils/coords.py | 坐标转换 |
| 110 | gui/ui_builders/multi_slice_ui.py | 多视图 UI |
| 89 | utils/threading_utils.py | 线程工具 |
| 87 | vtk_utils/path.py | 折线 Actor |
| 81 | gui/views/welcome_view.py | 欢迎页 |
| 79 | gui/ui_builders/data_import_ui.py | 导入 UI |
| 70 | vtk_utils/markers.py | 球体标记 |
| 64 | 3d_recon/recon_core.py | Marching Cubes |
| 56 | main.py | 入口 |
| 36 | data_io/file_handler.py | 文件保存 |

---

## 架构总结图

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│                  MainWindow                         │
│         ┌──────────┼──────────┐                     │
│    WelcomeView  PatientInfo  MainView               │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────────┐
        │         MainView              │
        │  ┌──────────────────────┐     │
        │  │    UI Builders       │     │
        │  │ (data_import_ui,     │     │
        │  │  slice_editor_ui,    │     │
        │  │  sam2_ui, view3d_ui) │     │
        │  └──────────────────────┘     │
        │             │ dict of widgets │
        │  ┌──────────▼──────────┐     │
        │  │   Controllers       │     │
        │  │ (data_import_ctrl,  │     │
        │  │  slice_editor_ctrl, │     │
        │  │  sam2_ui_ctrl,      │     │
        │  │  path_ui_ctrl)      │     │
        │  └──────────┬──────────┘     │
        │             │                 │
        │  ┌──────────▼──────────┐     │
        │  │    Managers          │     │
        │  │ (view3d_manager,    │     │
        │  │  path_viz_manager,  │     │
        │  │  event_dispatcher)  │     │
        │  └──────────┬──────────┘     │
        └─────────────┼────────────────┘
                      │
    ┌─────────────────┼─────────────────────┐
    │                 │                     │
┌───▼───┐     ┌──────▼──────┐     ┌────────▼────────┐
│DataMgr│     │ Segmentation│     │  Path Planning  │
│       │     │ (SAM2,HSV,  │     │ (RRT, A*,       │
│volume │     │  manual)    │     │  SDF, PointCloud│
│masks  │     └─────────────┘     │  PathController)│
│labels │                         └─────────────────┘
└───────┘                                 │
    │                              ┌──────▼──────┐
    │                              │ PathService  │
    │                              │ (bridge)     │
    │                              └─────────────┘
    │
┌───▼────────┐    ┌──────────────┐
│ VTK Utils  │    │  3D Recon    │
│ (coords,   │    │ (Marching    │
│  markers,  │    │  Cubes)      │
│  path)     │    └──────────────┘
└────────────┘
```

---

> **Tip**: 建议用 IDE 的 "Go to Definition" 功能跟踪函数调用链，比纯看代码效率高很多。
