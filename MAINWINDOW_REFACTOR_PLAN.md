# MainWindow 进一步拆分计划

## 当前 MainWindow 的职责分析

### 应该保留在 MainWindow 的职责（窗口管理）
1. **UI 构建和布局管理**
   - `_build_ui()` - 构建UI组件
   - `_connect_signals()` - 连接信号和槽
   - `on_toggle_slice_view()` - 切换视图显示/隐藏

2. **控制器初始化和协调**
   - `_init_controllers()` - 初始化所有控制器
   - 控制器之间的协调和通信

3. **数据加载回调（简化版）**
   - `_on_volume_loaded()` - 委托给控制器
   - `_on_volume_cleared()` - 委托给控制器

### 应该拆分出去的职责

#### 1. 3D点选择控制器（3DPointController）
**职责**：管理3D点选择相关的所有逻辑
- `handle_slice_click()` - 处理切片点击
- `calculate_3d_coordinate()` - 计算3D坐标
- `on_enable_3d_pick()` - 启用3D取点模式
- `on_clear_3d_points()` - 清除3D点
- `on_set_path_point_from_list()` - 从列表设置路径点
- `add_3d_point_marker()` - 添加3D点标记
- `_on_3d_point_added()` - 3D点添加回调
- `_on_path_point_set_from_list()` - 路径点设置回调

**状态管理**：
- `slice_pick_data` - 切片点击数据
- `picked_3d_points` - 已选择的3D点
- `slice_pick_mode` - 3D取点模式
- `computed_3d_points` - 计算出的3D点
- `point_actors` - 3D点标记actors

#### 2. 切片标记控制器（SliceMarkerController）
**职责**：管理切片窗口中的标记绘制和更新
- `draw_slice_marker()` - 绘制切片标记
- `update_all_slice_markers()` - 更新所有切片标记
- `clear_slice_markers()` - 清除切片标记

**状态管理**：
- `slice_markers` - 切片标记存储

#### 3. 3D重建控制器（ReconstructionController）
**职责**：管理3D重建相关的所有操作
- `on_reconstruct_3d()` - 3D重建
- `on_clear_3d_view()` - 清除3D视图
- `on_open_model()` - 打开模型
- `on_reset_camera()` - 重置相机
- `update_vtk_display()` - 更新VTK显示

#### 4. 事件过滤器（EventFilterController）
**职责**：统一处理所有事件过滤逻辑
- `eventFilter()` - 事件过滤（委托给相应的控制器）

## 拆分步骤

### 第一步：创建 3D点选择控制器
- 文件：`surgical_robot_app/gui/controllers/point3d_controller.py`
- 将3D点选择相关的方法和状态迁移过去

### 第二步：创建切片标记控制器
- 文件：`surgical_robot_app/gui/controllers/slice_marker_controller.py`
- 将切片标记相关的方法迁移过去

### 第三步：创建3D重建控制器
- 文件：`surgical_robot_app/gui/controllers/reconstruction_controller.py`
- 将3D重建相关的方法迁移过去

### 第四步：简化事件过滤器
- 在 MainWindow 中保留 `eventFilter`，但只做路由，具体逻辑委托给控制器

### 第五步：清理 MainWindow
- 移除已迁移的方法
- 保留窗口管理和协调逻辑
- 简化回调方法

## 预期效果

### MainWindow 将只包含：
1. UI 构建和布局管理
2. 控制器初始化和协调
3. 简化的回调方法（委托给控制器）
4. 事件路由（委托给控制器）

### 代码行数减少：
- 当前 MainWindow: ~1735 行
- 预期 MainWindow: ~800-900 行
- 减少约 50% 的代码

### 可维护性提升：
- 每个控制器职责单一
- 更容易测试和维护
- 更容易扩展新功能

