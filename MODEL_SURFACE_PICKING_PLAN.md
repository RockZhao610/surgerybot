# 3D模型表面两次点击选点方案

## 需求分析

当用户导入local model（STL文件）后，需要一种新的选点方式：
- 不使用切片视图（因为没有体数据）
- 直接在3D视图中点击模型表面
- 通过两次正交视角的点击确定3D坐标
- 显示深度参考线辅助用户

## 实现方案

### 1. 核心逻辑

#### 第一次点击（主视角）
- 用户在某个视角（如轴向）点击模型表面
- 系统获取：
  - 点击位置的世界坐标 `(x1, y1, z1)`
  - 当前相机视角类型（轴向/冠状/矢状）
  - 模型表面法向量（用于深度参考线）
  - 投影坐标（根据视角类型确定）

#### 第二次点击（正交视角）
- 用户切换到正交视角（如冠状/矢状）
- 再次点击模型表面
- 系统获取：
  - 点击位置的世界坐标 `(x2, y2, z2)`
  - 当前相机视角类型
  - 投影坐标

#### 3D坐标计算
- 通过两次正交视角的投影坐标，几何计算确定唯一的3D坐标
- 使用模型边界进行坐标归一化

### 2. 深度参考线

深度参考线用于辅助用户理解点击位置的深度：
- **显示方式**：垂直于当前视角的虚线
- **起点**：点击位置在模型表面
- **终点**：沿法向量方向延伸一定距离（或到模型中心）
- **颜色**：半透明，不遮挡模型

### 3. 视角检测

需要检测当前相机视角类型：
- **轴向（Axial）**：相机看向Z轴方向
- **冠状（Coronal）**：相机看向Y轴方向
- **矢状（Sagittal）**：相机看向X轴方向

检测方法：通过相机的 `GetViewUp()` 和 `GetFocalPoint()` 判断

### 4. 坐标转换

- 世界坐标 → 空间坐标（[0, 100]）：使用 `world_to_space()`
- 空间坐标 → 世界坐标：使用 `space_to_world()`

## 实现细节

### 数据结构

```python
class ModelSurfacePickData:
    """模型表面选点数据"""
    first_click: Optional[Dict] = {
        'world_pos': (x, y, z),
        'view_type': 'axial' | 'coronal' | 'sagittal',
        'projection_2d': (u, v),  # 在当前视角的2D投影
        'normal': (nx, ny, nz),   # 表面法向量
    }
    second_click: Optional[Dict] = {
        'world_pos': (x, y, z),
        'view_type': 'axial' | 'coronal' | 'sagittal',
        'projection_2d': (u, v),
        'normal': (nx, ny, nz),
    }
```

### 关键函数

1. **检测相机视角**
```python
def detect_camera_view_type(camera) -> str:
    """检测当前相机视角类型"""
    view_up = camera.GetViewUp()
    focal = camera.GetFocalPoint()
    position = camera.GetPosition()
    
    # 根据view_up和position判断视角
    # ...
```

2. **获取模型表面法向量**
```python
def get_surface_normal(picker, cell_id) -> Tuple[float, float, float]:
    """获取点击位置处的表面法向量"""
    # 使用VTK获取cell的法向量
    # ...
```

3. **计算投影坐标**
```python
def project_to_view_plane(world_pos, view_type, camera) -> Tuple[float, float]:
    """将世界坐标投影到当前视角的2D平面"""
    # 根据视角类型计算投影
    # ...
```

4. **从两次投影计算3D坐标**
```python
def compute_3d_from_two_projections(
    click1: Dict, 
    click2: Dict
) -> Tuple[float, float, float]:
    """从两次正交视角的投影计算3D坐标"""
    # 几何计算
    # ...
```

5. **绘制深度参考线**
```python
def draw_depth_reference_line(
    renderer,
    start_pos: Tuple[float, float, float],
    normal: Tuple[float, float, float],
    length: float
):
    """绘制深度参考线"""
    # 创建VTK Line actor
    # ...
```

## 实现步骤

1. 创建 `ModelSurfacePicker` 类
2. 实现视角检测功能
3. 实现表面法向量获取
4. 实现投影坐标计算
5. 实现3D坐标计算
6. 实现深度参考线绘制
7. 集成到 `PathUIController`

## 与现有系统的集成

- 修改 `handle_set_pick_mode`：检测是否有体数据
  - 有体数据 → 使用弹窗对话框（切片视图）
  - 只有模型 → 使用3D视图直接点击
- 修改 `handle_vtk_click`：支持两次点击模式
- 添加深度参考线可视化

