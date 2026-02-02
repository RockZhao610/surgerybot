# 路径规划选点功能改进方案

## 当前实现分析

### 现有选点方式
1. **3D视图点击**：在VTK 3D窗口中直接点击
   - 优点：直观、快速
   - 缺点：点击空白处坐标转换不准确；难以精确定位

2. **三视图交叉选点**：在两个不同切片视图（冠状/矢状/轴向）中各点击一次
   - 优点：精度较高
   - 缺点：需要两次点击，操作繁琐；需要理解几何关系

## 改进方案

### 方案1：单切片视图选点（推荐 ⭐⭐⭐⭐⭐）
**描述**：在单个切片视图中点击，自动使用当前切片位置作为第三个维度

**实现逻辑**：
- 用户在任意一个切片视图（冠状/矢状/轴向）中点击
- 系统自动使用该切片的当前位置作为第三个坐标维度
- 例如：在冠状面点击 (x, z)，使用当前冠状面位置作为 y 坐标

**优点**：
- 只需一次点击
- 操作简单直观
- 精度足够（切片位置已知）

**代码示例**：
```python
def handle_single_slice_click(self, plane_type: str, x: int, y: int):
    """单切片选点：使用当前切片位置作为第三个维度"""
    volume = self.data_manager.get_volume()
    if volume is None:
        return
    
    # 获取当前切片位置（归一化到 0-1）
    slice_pos_normalized = self.slice_positions[plane_type] / 100.0
    
    # 计算2D坐标（归一化到 0-1）
    img_x_norm, img_y_norm = self._convert_click_to_normalized(plane_type, x, y)
    
    # 转换为3D坐标
    if plane_type == 'coronal':
        # 冠状面：显示 X-Z，Y 由切片位置确定
        x_3d = img_x_norm * 100.0
        y_3d = slice_pos_normalized * 100.0
        z_3d = img_y_norm * 100.0
    elif plane_type == 'sagittal':
        # 矢状面：显示 Y-Z，X 由切片位置确定
        x_3d = slice_pos_normalized * 100.0
        y_3d = img_x_norm * 100.0
        z_3d = img_y_norm * 100.0
    else:  # axial
        # 横断面：显示 X-Y，Z 由切片位置确定
        x_3d = img_x_norm * 100.0
        y_3d = img_y_norm * 100.0
        z_3d = slice_pos_normalized * 100.0
    
    return (x_3d, y_3d, z_3d)
```

---

### 方案2：数值输入对话框（推荐 ⭐⭐⭐⭐）
**描述**：提供对话框直接输入坐标值

**UI设计**：
```
┌─────────────────────────────┐
│  Set Path Point             │
├─────────────────────────────┤
│  Point Type: [Start ▼]     │
│                             │
│  X: [50.0]  (0-100)        │
│  Y: [50.0]  (0-100)         │
│  Z: [50.0]  (0-100)         │
│                             │
│  [Preview in 3D]            │
│                             │
│  [Cancel]  [OK]             │
└─────────────────────────────┘
```

**优点**：
- 精度最高
- 适合精确控制
- 可以输入已知坐标

**实现位置**：
- 在 `PathUIController` 中添加 `handle_manual_input()` 方法
- 使用 `QDialog` 创建输入对话框

---

### 方案3：智能特征点选择（推荐 ⭐⭐⭐）
**描述**：基于图像特征自动选择关键点

**功能**：
1. **中心点选择**：自动选择分割区域的质心
2. **边缘点选择**：选择分割区域的边界点
3. **最远点选择**：选择距离起点最远的点（用于终点）
4. **安全点选择**：选择距离障碍物最远的点

**实现逻辑**：
```python
def select_centroid_point(self, seg_mask_volume: np.ndarray):
    """选择分割区域的质心"""
    # 计算质心
    coords = np.where(seg_mask_volume > 0)
    if len(coords[0]) == 0:
        return None
    
    centroid = (
        np.mean(coords[2]) / seg_mask_volume.shape[2] * 100.0,  # x
        np.mean(coords[1]) / seg_mask_volume.shape[1] * 100.0,  # y
        np.mean(coords[0]) / seg_mask_volume.shape[0] * 100.0   # z
    )
    return centroid

def select_safe_point(self, seg_mask_volume: np.ndarray, reference_point):
    """选择距离障碍物最远的点"""
    # 使用距离变换找到最安全的点
    from scipy.ndimage import distance_transform_edt
    dist_map = distance_transform_edt(seg_mask_volume == 0)
    max_dist_idx = np.unravel_index(np.argmax(dist_map), dist_map.shape)
    # 转换为空间坐标...
    return safe_point
```

---

### 方案4：点编辑功能（推荐 ⭐⭐⭐⭐）
**描述**：提供点的编辑、删除、移动功能

**功能列表**：
1. **删除点**：右键菜单或按钮删除选中的点
2. **移动点**：拖拽3D标记点或切片视图中的点
3. **编辑坐标**：双击点打开编辑对话框
4. **撤销/重做**：支持操作历史

**UI设计**：
```
路径点列表：
┌─────────────────────────────┐
│ [1] Start: (10.0, 20.0, 30.0) [Edit] [Delete] │
│ [2] Waypoint: (40.0, 50.0, 60.0) [Edit] [Delete] │
│ [3] End: (70.0, 80.0, 90.0) [Edit] [Delete] │
└─────────────────────────────┘
```

---

### 方案5：视觉反馈增强（推荐 ⭐⭐⭐）
**描述**：改进选点模式的视觉提示

**改进点**：
1. **高亮当前切片**：选点模式下高亮当前可用的切片视图
2. **鼠标样式**：选点模式下鼠标变为十字准星
3. **预览标记**：鼠标悬停时显示预览点位置
4. **状态提示**：清晰显示当前选点模式和剩余步骤

**实现**：
```python
def set_pick_mode(self, mode: str):
    """设置选点模式，并更新视觉反馈"""
    self.pick_mode = mode
    
    # 更新鼠标样式
    if self.vtk_widget:
        self.vtk_widget.setCursor(Qt.CrossCursor)
    
    # 高亮切片视图
    for label in self.slice_labels.values():
        label.setStyleSheet("border: 2px solid yellow;")
    
    # 更新状态文本
    self.vtk_status.setText(
        f"Pick Mode: {mode.upper()} - Click in any slice view or 3D window"
    )
```

---

### 方案6：组合选点模式（推荐 ⭐⭐⭐⭐⭐）
**描述**：结合多种选点方式，用户可选择最适合的方法

**UI设计**：
```
路径点选择
├─ 快速选点
│  ├─ [单切片点击] (默认)
│  └─ [3D视图点击]
├─ 精确选点
│  ├─ [双切片交叉]
│  └─ [数值输入]
└─ 智能选点
   ├─ [质心]
   ├─ [边缘点]
   └─ [安全点]
```

---

## 实施优先级

### 高优先级（立即实施）
1. ✅ **单切片视图选点** - 显著简化操作流程
2. ✅ **数值输入对话框** - 提供精确控制
3. ✅ **点编辑功能** - 提高用户体验

### 中优先级（后续实施）
4. ⚠️ **视觉反馈增强** - 改善交互体验
5. ⚠️ **智能特征点选择** - 提供自动化选项

### 低优先级（可选）
6. ⚪ **组合选点模式** - 整合所有功能

---

## 实施建议

### 第一步：实现单切片选点
修改 `MultiSliceController.handle_slice_click()` 方法，添加单切片选点逻辑。

### 第二步：添加数值输入
在 `PathUIController` 中添加 `QDialog` 用于坐标输入。

### 第三步：实现点编辑
添加右键菜单和拖拽功能。

---

## 代码修改位置

1. **`multi_slice_controller.py`**：
   - 修改 `handle_slice_click()` 支持单切片选点
   - 添加选点模式切换

2. **`path_ui_controller.py`**：
   - 添加 `handle_manual_input()` 方法
   - 添加点编辑相关方法

3. **`point_selection_manager.py`**：
   - 添加智能选点方法
   - 添加点编辑逻辑

