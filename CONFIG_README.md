# 配置文件说明 (config.json)

本文档详细说明 `config.json` 中各配置项的含义和推荐值。

---

## 路径规划配置 (path_planning)

### 算法选择

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_rrt` | bool | `true` | `true`: 使用 RRT 算法（推荐）<br>`false`: 使用 A* 算法 |
| `use_sdf` | bool | `true` | `true`: 使用 SDF 碰撞检测（推荐，解决内部空洞问题）<br>`false`: 使用点云碰撞检测 |

### A* 算法参数

> 仅当 `use_rrt = false` 时生效

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `grid_size` | [int, int, int] | `[100, 100, 105]` | 3D 栅格大小 [X, Y, Z] |
| `obstacle_expansion` | int | `4` | 障碍物膨胀半径（栅格单位） |

### RRT 算法参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `rrt_step_size` | float | `2.0` | 0.5 - 5.0 | 每次扩展的步长<br>↓小 = 更精细但更慢<br>↑大 = 更快但可能不够精细 |
| `rrt_goal_bias` | float | `0.1` | 0.0 - 1.0 | 目标偏向概率<br>每次采样有此概率直接向目标扩展<br>↑大 = 更快找到目标但可能陷入局部最优 |
| `rrt_max_iterations` | int | `5000` | 1000 - 50000 | 最大迭代次数<br>↑大 = 更可能找到路径但更慢 |
| `rrt_goal_threshold` | float | `2.0` | 1.0 - 10.0 | 到达目标的距离阈值 |

### 安全距离与器械尺寸

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `safety_radius` | float | `5.0` | 路径点到障碍物的最小安全距离<br>单位：与坐标系一致（物理坐标为 mm，归一化坐标为 0-100 的单位）<br>↑大 = 更安全但可能找不到路径<br>↓小 = 更容易找到路径但风险更高 |
| `instrument_radius` | float | `2.0` | **器械半径**（用于体积碰撞检测）<br>传统检测只考虑路径线，器械检测考虑实际物理尺寸<br>针/细探针: 0.5~1.0<br>内窥镜: 2.0~5.0<br>手术钳: 3.0~6.0 |

**器械碰撞检测原理**：
```
传统检测（路径无体积）：        器械碰撞检测（胶囊体）：
       ○ 路径点                    ╭───╮
       │                           │器械│
       ○                           │体积│
                                   ╰───╯
可能穿过狭窄空间         →     正确检测器械是否能通过
```

### 路径简化

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `simplify_enabled` | bool | `true` | 是否启用路径简化 |
| `simplify_max_points` | int | `50` | 最大路径点数<br>`0` = 不限制点数<br>典型值：20（快速预览）、50（正常）、100（高精度） |
| `simplify_tolerance` | float | `1.0` | RDP 简化容差（点到直线的最大偏离）<br>↑大 = 点更少但可能偏离原路径<br>↓小 = 保留更多细节 |

### SDF 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sdf_use_interpolation` | bool | `true` | `true`: 使用三线性插值（更精确）<br>`false`: 使用最近体素（更快） |

---

## 推荐配置方案

### 方案 1：高精度手术路径

```json
{
  "path_planning": {
    "safety_radius": 8.0,
    "simplify_max_points": 100,
    "simplify_tolerance": 0.5,
    "rrt_step_size": 1.0,
    "rrt_max_iterations": 10000
  }
}
```

### 方案 2：快速预览

```json
{
  "path_planning": {
    "safety_radius": 5.0,
    "simplify_max_points": 20,
    "simplify_tolerance": 2.0,
    "rrt_max_iterations": 3000
  }
}
```

### 方案 3：狭窄空间

```json
{
  "path_planning": {
    "safety_radius": 3.0,
    "rrt_step_size": 1.0,
    "rrt_goal_bias": 0.2,
    "rrt_max_iterations": 15000
  }
}
```

### 方案 4：大血管/神经附近（高安全要求）

```json
{
  "path_planning": {
    "safety_radius": 10.0,
    "simplify_tolerance": 0.3,
    "rrt_step_size": 1.5
  }
}
```

---

## 分割配置 (segmentation)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `brush_size` | int | `10` | 手动分割画笔大小（像素） |
| `radius_ratio` | float | `0.02` | 3D 点标记半径比例 |
| `default_coord` | float | `50.0` | 默认坐标值 |
| `default_radius` | float | `2.0` | 默认半径 |

---

## 3D 视图配置 (view3d)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `marker_radius_ratio` | float | `0.02` | 标记点半径比例 |
| `background_color` | [float, float, float] | `[0.1, 0.1, 0.1]` | 背景色 RGB (0-1) |
| `coordinate_system_size_ratio` | float | `0.2` | 坐标系大小比例 |
| `axes_actor_scale_factor` | float | `0.6` | XYZ 轴缩放因子 |
| `morph_kernel_size` | int | `3` | 形态学操作核大小 |
| `morph_iterations` | int | `1` | 形态学迭代次数 |
| `vtk_hole_size` | float | `1000.0` | VTK 填充孔洞大小 |
| `enable_contour_filling` | bool | `true` | 是否启用轮廓填充 |

---

## HSV 阈值配置 (hsv_threshold)

用于红色区域检测（如血管）。

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `h1_low` | int | `0` | 0-180 | 色相范围 1 下限 |
| `h1_high` | int | `10` | 0-180 | 色相范围 1 上限 |
| `h2_low` | int | `160` | 0-180 | 色相范围 2 下限 |
| `h2_high` | int | `180` | 0-180 | 色相范围 2 上限 |
| `s_low` | int | `50` | 0-255 | 饱和度下限 |
| `s_high` | int | `255` | 0-255 | 饱和度上限 |
| `v_low` | int | `50` | 0-255 | 明度下限 |
| `v_high` | int | `255` | 0-255 | 明度上限 |

---

## UI 配置 (ui)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vtk_widget_min_height` | int | `300` | VTK 控件最小高度 |
| `image_label_min_height` | int | `400` | 图像标签最小高度 |
| `slice_label_min_height` | int | `150` | 切片标签最小高度 |
| `toggle_button_max_width` | int | `80` | 切换按钮最大宽度 |

---

## 注意事项

1. **修改后需重启程序** 才能生效
2. **JSON 格式要求**：确保格式正确，可以用在线 JSON 验证工具检查
3. **备份配置**：修改前建议备份原配置文件
4. 以 `_` 开头的字段是注释说明，不影响程序运行
