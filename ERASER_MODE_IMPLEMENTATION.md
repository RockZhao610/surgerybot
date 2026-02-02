# 擦除模式实现说明

## 📋 概述

擦除模式允许用户在图像上清除误分割的掩码区域。实现分为三个层次：UI层、控制器层和核心算法层。

---

## 🏗️ 实现架构

```
┌─────────────────────────────────────────┐
│          UI 层 (UI Builder)            │
│  - 擦除模式切换按钮                      │
│  - 按钮状态管理                          │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      控制器层 (SliceEditorController)   │
│  - eraser_mode 状态管理                  │
│  - 鼠标事件处理                          │
│  - 模式切换逻辑                          │
└─────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    核心算法层 (ManualSegController)     │
│  - apply_eraser() 方法                   │
│  - 圆形区域计算                          │
│  - 掩码清除逻辑                          │
└─────────────────────────────────────────┘
```

---

## 📝 代码实现详解

### 1. UI 层：擦除模式按钮

**文件**：`surgical_robot_app/gui/ui_builders/slice_editor_ui.py`

```python
# 擦除模式控制按钮
btn_toggle_eraser = QPushButton("Brush Mode")
btn_toggle_eraser.setCheckable(True)  # 可切换状态
btn_toggle_eraser.setFixedWidth(130)
```

**说明**：
- 按钮是可切换的（`setCheckable(True)`）
- 按钮文本显示为 "Brush Mode"（未选中时）或 "Eraser Mode"（选中时）
- 按钮状态通过 `checked` 信号传递给控制器

---

### 2. 控制器层：模式切换和事件处理

**文件**：`surgical_robot_app/gui/controllers/slice_editor_controller.py`

#### 2.1 模式切换

```python
def handle_toggle_eraser(self, checked: bool):
    """切换擦除模式"""
    self.eraser_mode = checked
```

**说明**：
- `checked=True`：进入擦除模式
- `checked=False`：退出擦除模式（画笔模式）

#### 2.2 鼠标事件处理

```python
def handle_mouse_event(self, obj, event: QEvent, sam2_picking_mode: bool = False) -> bool:
    """处理鼠标事件（画笔/擦除/SAM2点击）"""
    
    # ... 坐标转换逻辑 ...
    
    # 手动分割模式
    if not sam2_picking_mode:
        shape = self.data_manager.get_volume_shape()
        if shape:
            depth, h, w = shape[:3]
            self.data_manager.ensure_masks((depth, h, w))
            
            # 根据 eraser_mode 选择不同的操作
            if self.eraser_mode:
                # 擦除模式：清除掩码
                self.data_manager.masks = self.manual_controller.apply_eraser(
                    self.data_manager.masks,
                    idx,              # 当前切片索引
                    (ix, iy),         # 鼠标点击坐标
                    self.brush_size,  # 擦除笔半径
                )
            else:
                # 画笔模式：绘制掩码
                self.data_manager.masks = self.manual_controller.apply_brush(
                    self.data_manager.masks,
                    idx,
                    (ix, iy),
                    self.brush_size,
                )
            
            # 更新掩码体积和显示
            self.data_manager._update_seg_mask_volume()
            self.update_slice_display(idx)
            return True
```

**关键逻辑**：
1. **坐标转换**：将鼠标在窗口中的坐标转换为图像像素坐标
2. **模式判断**：根据 `self.eraser_mode` 选择调用 `apply_eraser` 或 `apply_brush`
3. **实时更新**：擦除后立即更新掩码体积和显示

---

### 3. 核心算法层：擦除实现

**文件**：`surgical_robot_app/segmentation/manual_controller.py`

#### 3.1 apply_eraser 方法

```python
def apply_eraser(
    self,
    masks: List[np.ndarray],
    slice_idx: int,
    center: Tuple[int, int],
    radius: int,
) -> List[np.ndarray]:
    """
    在指定切片上应用擦除笔（清除圆形区域内的掩码）。
    
    Args:
        masks: 掩码列表
        slice_idx: 当前切片索引
        center: (ix, iy) 像素坐标
        radius: 擦除笔半径
    
    Returns:
        更新后的掩码列表
    """
    if slice_idx < 0 or slice_idx >= len(masks):
        return masks
    
    h, w = masks[slice_idx].shape[:2]
    ix, iy = int(center[0]), int(center[1])
    r = int(radius)
    
    # 记录历史（用于撤销）
    if slice_idx not in self.mask_history:
        self.mask_history[slice_idx] = []
    self.mask_history[slice_idx].append(masks[slice_idx].copy())
    
    # 计算圆形区域的边界框（优化性能）
    y_min = max(0, iy - r)
    y_max = min(h, iy + r + 1)
    x_min = max(0, ix - r)
    x_max = min(w, ix + r + 1)
    
    # 创建圆形掩码
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
    
    # 将圆形区域内的掩码设为 0（清除）
    masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 0
    
    return masks
```

**核心算法**：

1. **边界检查**：确保切片索引有效
2. **历史记录**：保存擦除前的掩码状态（用于撤销）
3. **边界框计算**：只处理圆形区域所在的矩形区域（性能优化）
4. **圆形掩码生成**：使用 `np.ogrid` 创建坐标网格，计算圆形区域
5. **掩码清除**：将圆形区域内的像素值设为 `0`

#### 3.2 与画笔模式的对比

**画笔模式** (`apply_brush`)：
```python
# 将圆形区域内的掩码设为 255（绘制）
masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 255
```

**擦除模式** (`apply_eraser`)：
```python
# 将圆形区域内的掩码设为 0（清除）
masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 0
```

**区别**：
- 画笔模式：`mask = 255`（白色，表示前景）
- 擦除模式：`mask = 0`（黑色，表示背景）

---

## 🔄 完整工作流程

### 1. 用户操作流程

```
用户点击 "Brush Mode" 按钮
    ↓
按钮状态切换（checked=True）
    ↓
触发 handle_toggle_eraser(True)
    ↓
设置 eraser_mode = True
    ↓
用户在图像上拖动鼠标
    ↓
触发 handle_mouse_event()
    ↓
检测到 eraser_mode = True
    ↓
调用 apply_eraser()
    ↓
清除圆形区域内的掩码
    ↓
更新显示
```

### 2. 数据流

```
鼠标事件 (x, y)
    ↓
坐标转换 (ix, iy)
    ↓
apply_eraser(masks, slice_idx, (ix, iy), radius)
    ↓
计算圆形区域
    ↓
masks[slice_idx][圆形区域] = 0
    ↓
更新 data_manager.masks
    ↓
更新 seg_mask_volume
    ↓
更新 UI 显示
```

---

## 🎯 关键实现细节

### 1. 圆形区域计算

使用 `np.ogrid` 创建坐标网格，然后计算距离：

```python
yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
```

**说明**：
- `np.ogrid` 创建开放网格，比 `np.mgrid` 更节省内存
- 使用欧几里得距离公式：`(x - cx)² + (y - cy)² ≤ r²`
- `mask_circle` 是一个布尔数组，标记圆形区域内的像素

### 2. 性能优化

**边界框裁剪**：
```python
y_min = max(0, iy - r)
y_max = min(h, iy + r + 1)
x_min = max(0, ix - r)
x_max = min(w, ix + r + 1)
```

**说明**：
- 只处理圆形区域所在的矩形区域
- 避免处理整个图像，提高性能
- 对于大图像和小的擦除笔，性能提升显著

### 3. 历史记录

```python
if slice_idx not in self.mask_history:
    self.mask_history[slice_idx] = []
self.mask_history[slice_idx].append(masks[slice_idx].copy())
```

**说明**：
- 每个切片维护一个历史栈
- 每次操作前保存掩码副本
- 可用于实现撤销功能

---

## 📊 代码位置总结

| 层次 | 文件 | 关键方法/类 | 行数 |
|------|------|------------|------|
| **UI 层** | `gui/ui_builders/slice_editor_ui.py` | `btn_toggle_eraser` | 47-49 |
| **控制器层** | `gui/controllers/slice_editor_controller.py` | `handle_toggle_eraser()`<br>`handle_mouse_event()` | 273-275<br>344-350 |
| **核心算法层** | `segmentation/manual_controller.py` | `apply_eraser()` | 84-125 |

---

## 🔍 相关代码引用

### 1. 擦除模式切换

```273:275:surgical_robot_app/gui/controllers/slice_editor_controller.py
def handle_toggle_eraser(self, checked: bool):
    """切换擦除模式"""
    self.eraser_mode = checked
```

### 2. 鼠标事件处理（擦除分支）

```344:350:surgical_robot_app/gui/controllers/slice_editor_controller.py
if self.eraser_mode:
    self.data_manager.masks = self.manual_controller.apply_eraser(
        self.data_manager.masks,
        idx,
        (ix, iy),
        self.brush_size,
    )
```

### 3. 核心擦除算法

```84:125:surgical_robot_app/segmentation/manual_controller.py
def apply_eraser(
    self,
    masks: List[np.ndarray],
    slice_idx: int,
    center: Tuple[int, int],
    radius: int,
) -> List[np.ndarray]:
    """
    在指定切片上应用擦除笔（清除圆形区域内的掩码）。
    """
    # ... 边界检查 ...
    
    # 记录历史
    if slice_idx not in self.mask_history:
        self.mask_history[slice_idx] = []
    self.mask_history[slice_idx].append(masks[slice_idx].copy())
    
    # 计算边界框
    y_min = max(0, iy - r)
    y_max = min(h, iy + r + 1)
    x_min = max(0, ix - r)
    x_max = min(w, ix + r + 1)
    
    # 创建圆形掩码
    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
    
    # 清除掩码
    masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 0
    
    return masks
```

---

## 💡 实现特点

### 优点

1. **模块化设计**：UI、控制器、算法分离
2. **性能优化**：边界框裁剪，只处理必要区域
3. **历史记录**：支持撤销功能
4. **实时反馈**：擦除后立即更新显示
5. **代码复用**：与画笔模式共享坐标转换逻辑

### 可改进点

1. **连续擦除**：当前只支持点击擦除，可以支持拖动连续擦除
2. **不同形状**：当前只支持圆形，可以支持矩形、椭圆等
3. **软擦除**：当前是硬擦除（直接设为0），可以支持软擦除（渐变）

---

## 🔗 相关功能

- **画笔模式**：`apply_brush()` - 绘制掩码
- **撤销功能**：使用 `mask_history` 实现
- **HSV 阈值分割**：另一种分割方式
- **SAM2 分割**：自动分割，与手动分割互补

---

## 📝 总结

擦除模式的实现分为三个层次：

1. **UI 层**：提供切换按钮
2. **控制器层**：管理状态，处理鼠标事件，根据模式调用不同方法
3. **核心算法层**：实现圆形区域计算和掩码清除逻辑

核心算法使用 `np.ogrid` 和欧几里得距离公式计算圆形区域，然后将区域内的掩码像素设为 `0`，实现擦除效果。

