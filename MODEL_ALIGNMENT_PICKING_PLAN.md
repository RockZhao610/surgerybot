# 模型对齐 + 固定视角选点方案

## 📋 方案概述

### 核心思路

1. **建立标准坐标系**：在3D视图中心显示标准坐标系（轴向/冠状/矢状）
2. **模型对齐**：提供工具将STL模型旋转对齐到标准坐标系
3. **固定视角选点**：对齐后，使用固定视角（相机）在两个正交平面上选择点

---

## 🎯 功能设计

### 1. 标准坐标系可视化

#### 1.1 坐标系显示

在3D视图中心显示标准坐标系：

```
坐标轴：
- X轴：红色箭头（矢状方向，左右）
- Y轴：绿色箭头（冠状方向，前后）
- Z轴：蓝色箭头（轴向方向，上下）

坐标平面：
- XY平面（轴向视图）：显示网格线（可选）
- XZ平面（冠状视图）：显示网格线（可选）
- YZ平面（矢状视图）：显示网格线（可选）
```

#### 1.2 实现方式

- 使用VTK的 `vtkAxesActor` 显示坐标轴
- 使用 `vtkPlaneSource` 和 `vtkPolyDataMapper` 显示坐标平面（可选）
- 坐标系固定在视图中心（世界坐标原点）
- 坐标系大小根据模型大小自适应

---

### 2. 模型对齐工具

#### 2.1 对齐方式

**方式A：手动对齐（推荐）**
- 提供旋转控制（X/Y/Z轴旋转）
- 提供平移控制（X/Y/Z轴平移）
- 实时预览对齐效果
- 一键重置

**方式B：自动对齐（可选）**
- 基于模型的主轴（Principal Axis）自动对齐
- 或基于模型的边界框（Bounding Box）对齐

#### 2.2 UI设计

```
[模型对齐] 面板
├── [对齐模式]
│   ├── ○ 手动对齐
│   └── ○ 自动对齐
├── [旋转控制]
│   ├── X轴旋转: [滑块] [-90° ~ +90°]
│   ├── Y轴旋转: [滑块] [-90° ~ +90°]
│   └── Z轴旋转: [滑块] [-90° ~ +90°]
├── [平移控制]
│   ├── X轴平移: [滑块] [-100 ~ +100]
│   ├── Y轴平移: [滑块] [-100 ~ +100]
│   └── Z轴平移: [滑块] [-100 ~ +100]
├── [操作按钮]
│   ├── [重置对齐]
│   ├── [应用对齐]
│   └── [取消对齐]
└── [对齐状态]
    └── 显示：已对齐 / 未对齐
```

#### 2.3 对齐逻辑

```python
# 伪代码
1. 用户导入STL模型
2. 模型显示在3D视图中（原始位置）
3. 标准坐标系显示在视图中心
4. 用户调整旋转/平移滑块
5. 实时更新模型位置（相对于坐标系）
6. 用户点击"应用对齐"
7. 保存对齐变换矩阵
8. 后续选点使用对齐后的模型位置
```

---

### 3. 固定视角选点

#### 3.1 选点流程

对齐完成后：

1. **启用固定视角模式**
   - 点击路径规划按钮（Start/Waypoint/End）
   - 系统切换到固定视角模式

2. **显示两个固定视角**
   - **视角1：轴向视图（XY平面）** - 从上往下看
   - **视角2：冠状视图（XZ平面）** - 从前往后看
   - 或提供选择：用户可以选择哪两个视角

3. **在两个视角上各点击一次**
   - 在轴向视图上点击 → 确定X、Y坐标
   - 在冠状视图上点击 → 确定X、Z坐标
   - 系统自动计算3D坐标

#### 3.2 视角显示方式

**选项A：分屏显示（推荐）**
```
┌─────────────────┬─────────────────┐
│   轴向视图      │   冠状视图      │
│   (XY平面)      │   (XZ平面)      │
│                 │                 │
│   [点击这里]    │   [点击这里]    │
│                 │                 │
└─────────────────┴─────────────────┘
```

**选项B：切换显示**
- 显示一个视角，用户点击后切换到另一个视角
- 更简单，但需要切换

**选项C：弹窗对话框（类似现有方案）**
- 打开对话框，显示两个固定视角
- 用户在各视角上点击一次

#### 3.3 坐标计算

```python
# 对齐后的坐标计算
1. 用户在轴向视图（XY平面）点击 → 得到 (x1, y1)
2. 用户在冠状视图（XZ平面）点击 → 得到 (x2, z1)
3. 合并坐标：
   - X = (x1 + x2) / 2  # 或使用其中一个
   - Y = y1
   - Z = z1
4. 转换为空间坐标 [0, 100]
```

---

## 🛠️ 实现架构

### 模块设计

```
surgical_robot_app/
├── gui/
│   ├── utils/
│   │   ├── coordinate_system.py      # 坐标系可视化
│   │   └── model_alignment.py        # 模型对齐工具
│   ├── controllers/
│   │   └── alignment_controller.py   # 对齐控制器
│   └── ui_builders/
│       └── alignment_ui.py            # 对齐UI
```

### 核心类设计

#### 1. CoordinateSystemVisualizer

```python
class CoordinateSystemVisualizer:
    """标准坐标系可视化"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.axes_actor = None
        self.grid_actors = []
    
    def show_coordinate_system(self, center=(0, 0, 0), size=10.0):
        """在指定位置显示坐标系"""
        # 创建坐标轴
        # 创建坐标平面（可选）
        # 添加到renderer
    
    def hide_coordinate_system(self):
        """隐藏坐标系"""
    
    def set_size(self, size):
        """设置坐标系大小"""
```

#### 2. ModelAlignmentTool

```python
class ModelAlignmentTool:
    """模型对齐工具"""
    
    def __init__(self, renderer, model_actor):
        self.renderer = renderer
        self.model_actor = model_actor
        self.transform = vtkTransform()
        self.original_transform = None  # 保存原始变换
    
    def rotate_x(self, angle):
        """绕X轴旋转"""
    
    def rotate_y(self, angle):
        """绕Y轴旋转"""
    
    def rotate_z(self, angle):
        """绕Z轴旋转"""
    
    def translate(self, x, y, z):
        """平移"""
    
    def reset(self):
        """重置对齐"""
    
    def apply(self):
        """应用对齐（保存变换）"""
    
    def get_transform_matrix(self):
        """获取变换矩阵"""
```

#### 3. FixedViewPointPicker

```python
class FixedViewPointPicker:
    """固定视角选点器"""
    
    def __init__(self, renderer, camera, alignment_transform=None):
        self.renderer = renderer
        self.camera = camera
        self.alignment_transform = alignment_transform
        self.view1_type = 'axial'  # 轴向视图
        self.view2_type = 'coronal'  # 冠状视图
    
    def setup_fixed_views(self):
        """设置固定视角"""
        # 设置相机到轴向视图
        # 设置相机到冠状视图
    
    def pick_point_in_view(self, view_type, x, y):
        """在指定视角上选点"""
        # 设置相机到指定视角
        # 使用picker获取点击位置
        # 返回2D投影坐标
    
    def compute_3d_from_two_views(self, click1, click2):
        """从两个视角的点击计算3D坐标"""
        # click1: (x, y) 在轴向视图
        # click2: (x, z) 在冠状视图
        # 合并得到 (x, y, z)
```

---

## 📐 UI设计

### 对齐面板布局

```
┌─────────────────────────────────────┐
│  模型对齐                            │
├─────────────────────────────────────┤
│  对齐模式:                           │
│  ○ 手动对齐  ● 自动对齐              │
│                                      │
│  旋转控制:                           │
│  X轴: [━━━━●━━━━] -45°              │
│  Y轴: [━━━━●━━━━]  0°               │
│  Z轴: [━━━━●━━━━] +30°              │
│                                      │
│  平移控制:                           │
│  X轴: [━━━━●━━━━]  0                │
│  Y轴: [━━━━●━━━━]  0                │
│  Z轴: [━━━━●━━━━]  0                │
│                                      │
│  [重置] [应用对齐] [取消]            │
│                                      │
│  状态: ✓ 已对齐                      │
└─────────────────────────────────────┘
```

### 固定视角选点对话框

```
┌─────────────────────────────────────┐
│  路径点选择（固定视角）               │
├──────────────┬──────────────────────┤
│  轴向视图    │  冠状视图            │
│  (XY平面)    │  (XZ平面)            │
│              │                      │
│  [模型显示]  │  [模型显示]          │
│              │                      │
│  点击位置:   │  点击位置:           │
│  (x, y)      │  (x, z)              │
│              │                      │
├──────────────┴──────────────────────┤
│  [确认] [取消]                       │
└─────────────────────────────────────┘
```

---

## 🔄 工作流程

### 完整流程

```
1. 导入STL模型
   ↓
2. 显示标准坐标系（在视图中心）
   ↓
3. 用户对齐模型（旋转/平移）
   ↓
4. 点击"应用对齐"
   ↓
5. 点击路径规划按钮（Start/Waypoint/End）
   ↓
6. 打开固定视角选点对话框
   ↓
7. 在轴向视图上点击 → 确定 (x, y)
   ↓
8. 在冠状视图上点击 → 确定 (x, z)
   ↓
9. 系统计算3D坐标 → 添加路径点
```

---

## 💻 实现步骤

### Phase 1: 坐标系可视化（1天）

1. 创建 `CoordinateSystemVisualizer` 类
2. 实现坐标轴显示（X/Y/Z轴，不同颜色）
3. 实现坐标平面显示（可选，网格线）
4. 集成到 `View3DManager`

### Phase 2: 模型对齐工具（2-3天）

1. 创建 `ModelAlignmentTool` 类
2. 实现旋转控制（X/Y/Z轴）
3. 实现平移控制（X/Y/Z轴）
4. 实现实时预览
5. 创建对齐UI面板
6. 集成到主窗口

### Phase 3: 固定视角选点（2天）

1. 创建 `FixedViewPointPicker` 类
2. 实现固定视角设置（轴向/冠状）
3. 实现分屏显示或对话框
4. 实现坐标计算逻辑
5. 集成到 `PathUIController`

### Phase 4: 集成和测试（1天）

1. 整合所有模块
2. 测试完整流程
3. 优化用户体验
4. 添加错误处理

**总计**：6-7天

---

## 🎨 技术细节

### 坐标系显示

```python
# 使用VTK Axes Actor
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

axes = vtkAxesActor()
axes.SetXAxisLabelText("X (Sagittal)")
axes.SetYAxisLabelText("Y (Coronal)")
axes.SetZAxisLabelText("Z (Axial)")
axes.SetTotalLength(10, 10, 10)  # 坐标轴长度
renderer.AddActor(axes)
```

### 模型对齐

```python
# 使用VTK Transform
from vtkmodules.vtkCommonTransforms import vtkTransform

transform = vtkTransform()
transform.RotateX(angle_x)
transform.RotateY(angle_y)
transform.RotateZ(angle_z)
transform.Translate(x, y, z)

# 应用到模型
model_actor.SetUserTransform(transform)
```

### 固定视角设置

```python
# 设置相机到轴向视图
camera.SetPosition(0, 0, 100)  # 在Z轴上方
camera.SetFocalPoint(0, 0, 0)  # 看向原点
camera.SetViewUp(0, 1, 0)      # Y轴向上

# 设置相机到冠状视图
camera.SetPosition(0, 100, 0)  # 在Y轴前方
camera.SetFocalPoint(0, 0, 0)  # 看向原点
camera.SetViewUp(0, 0, 1)      # Z轴向上
```

---

## ✅ 优势

1. **更直观**：标准坐标系作为参考，用户容易理解
2. **更精确**：固定视角，减少视角检测误差
3. **更简单**：不需要用户手动旋转视角
4. **更可控**：预先对齐，选点流程标准化
5. **更专业**：符合医学图像处理的标准做法

---

## ⚠️ 注意事项

1. **对齐复杂度**：需要提供简单易用的对齐工具
2. **对齐精度**：对齐不准确会影响选点精度
3. **用户学习成本**：需要学习对齐操作
4. **性能**：实时预览可能影响性能（大模型）

---

## 🎯 推荐实现顺序

1. **先实现坐标系可视化**（最简单，立即可用）
2. **再实现固定视角选点**（核心功能）
3. **最后实现模型对齐**（增强功能，可选）

如果对齐功能太复杂，可以先实现：
- 坐标系可视化
- 固定视角选点（假设模型已经对齐）

---

**最后更新**：2026-01-04

