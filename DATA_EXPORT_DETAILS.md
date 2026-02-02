# 数据导出和报告生成功能详细说明

## 📋 概述

这个模块将实现完整的数据导出和报告生成功能，让用户可以：
- 导出路径数据到多种格式（JSON/CSV/XML）
- 生成专业的手术规划报告（PDF）
- 导出3D模型到多种格式（STL/OBJ/PLY）
- 导出分割结果到多种格式（PNG/NIfTI/DICOM）

---

## 🎯 功能详细说明

### 1. 路径数据导出（多种格式）

#### 当前状态
- ✅ 已实现：简单的文本格式（CSV坐标点）
- ❌ 未实现：结构化格式（JSON/XML）、元数据、路径评估信息

#### 可实现的导出格式

**1.1 JSON 格式**（推荐，结构化数据）
```json
{
  "version": "1.0",
  "timestamp": "2026-01-04T10:30:00",
  "path_info": {
    "algorithm": "RRT",
    "start_point": [10.5, 20.3, 30.1],
    "end_point": [90.2, 80.5, 70.8],
    "waypoints": [[50.0, 50.0, 50.0]],
    "total_points": 150,
    "path_length": 125.6,
    "safety_margin": 2.0
  },
  "path_points": [
    {"x": 10.5, "y": 20.3, "z": 30.1, "segment": 0},
    {"x": 11.2, "y": 21.0, "z": 30.5, "segment": 0},
    ...
  ],
  "metadata": {
    "patient_id": "P001",
    "procedure_type": "Gallbladder Puncture",
    "planning_date": "2026-01-04"
  }
}
```

**1.2 CSV 格式**（兼容Excel）
```csv
x,y,z,segment,index
10.5,20.3,30.1,0,0
11.2,21.0,30.5,0,1
...
```

**1.3 XML 格式**（标准化格式）
```xml
<path_plan>
  <metadata>
    <algorithm>RRT</algorithm>
    <timestamp>2026-01-04T10:30:00</timestamp>
  </metadata>
  <waypoints>
    <point type="start" x="10.5" y="20.3" z="30.1"/>
    <point type="waypoint" x="50.0" y="50.0" z="50.0"/>
    <point type="end" x="90.2" y="80.5" z="70.8"/>
  </waypoints>
  <path>
    <point x="10.5" y="20.3" z="30.1"/>
    ...
  </path>
</path_plan>
```

**1.4 ROS Path 格式**（机器人系统兼容）
```yaml
header:
  seq: 0
  stamp:
    secs: 1234567890
    nsecs: 0
  frame_id: "base_link"
poses:
  - pose:
      position: {x: 10.5, y: 20.3, z: 30.1}
      orientation: {x: 0, y: 0, z: 0, w: 1}
  ...
```

**实现内容**：
- ✅ 路径点坐标导出
- ✅ 路径元数据（算法、时间戳、参数）
- ✅ 路径评估指标（长度、平滑度、安全性）
- ✅ 分段信息（起点、中间点、终点）
- ✅ 障碍物信息（可选）

---

### 2. 手术规划报告生成（PDF）

#### 功能描述
生成专业的手术规划报告，包含所有关键信息，便于医生审查和存档。

#### 报告内容

**2.1 报告封面**
- 患者信息（ID、姓名、日期）
- 手术类型
- 规划医生
- 报告生成时间

**2.2 执行摘要**
- 规划概述
- 关键参数
- 风险评估

**2.3 图像信息**
- 原始图像信息（尺寸、层数、间距）
- 分割结果统计
- 3D模型信息

**2.4 路径规划详情**
- 路径算法和参数
- 起点、终点、中间点坐标
- 路径可视化（2D投影图、3D示意图）
- 路径评估指标：
  - 总长度
  - 路径点数
  - 平均曲率
  - 最小安全距离
  - 碰撞风险评分

**2.5 3D可视化**
- 3D模型截图（多视角）
- 路径在3D模型上的可视化
- 关键点标记

**2.6 安全分析**
- 路径安全检查结果
- 边界检查结果
- 风险评估

**2.7 技术参数**
- 使用的算法
- 配置参数
- 系统版本信息

**实现技术**：
- 使用 `reportlab` 或 `matplotlib` + `PIL` 生成PDF
- 使用 `matplotlib` 生成图表和可视化
- 使用 `VTK` 渲染3D截图

---

### 3. 3D模型导出（多种格式）

#### 当前状态
- ✅ 已实现：STL格式导出
- ❌ 未实现：OBJ、PLY、VTK格式

#### 可实现的格式

**3.1 STL 格式**（已实现，可增强）
- 二进制STL（当前）
- ASCII STL（可读格式）
- 带颜色的STL（扩展格式）

**3.2 OBJ 格式**（3D建模软件兼容）
```
# OBJ file
v 10.5 20.3 30.1
v 11.2 21.0 30.5
...
f 1 2 3
f 2 3 4
...
```

**3.3 PLY 格式**（点云和网格）
- 支持顶点颜色
- 支持法向量
- 支持纹理坐标

**3.4 VTK 格式**（VTK原生格式）
- 保留所有VTK属性
- 便于在VTK中重新加载

**3.5 3MF 格式**（3D打印）
- 支持多材料
- 支持颜色和纹理

**实现内容**：
- ✅ 多种格式导出
- ✅ 格式转换工具
- ✅ 质量选项（高/中/低精度）
- ✅ 压缩选项

---

### 4. 分割结果导出（多种格式）

#### 当前状态
- ✅ 已实现：PNG序列导出
- ❌ 未实现：NIfTI、DICOM、Numpy格式

#### 可实现的格式

**4.1 PNG 序列**（已实现，可增强）
- 当前：单色PNG
- 增强：带透明度的PNG、彩色PNG

**4.2 NIfTI 格式**（医学图像标准）
- 3D体数据格式
- 保留空间信息（spacing、origin、direction）
- 支持多种数据类型

**4.3 DICOM RTSS 格式**（DICOM RT Structure Set）
- 医学标准格式
- 支持ROI（Region of Interest）定义
- 兼容DICOM查看器

**4.4 Numpy 格式**（Python兼容）
- `.npy` 格式（NumPy原生）
- `.npz` 格式（压缩，支持多个数组）
- 包含元数据

**4.5 ITK/VTK 格式**
- ITK MetaImage格式
- VTK ImageData格式

**实现内容**：
- ✅ 多种格式导出
- ✅ 元数据保留（spacing、origin等）
- ✅ 批量导出
- ✅ 格式转换工具

---

## 🛠️ 实现架构

### 模块结构

```
surgical_robot_app/
├── data_export/              # 新增模块
│   ├── __init__.py
│   ├── path_exporter.py      # 路径导出
│   ├── report_generator.py   # 报告生成
│   ├── model_exporter.py     # 3D模型导出
│   ├── mask_exporter.py      # 分割结果导出
│   └── utils.py              # 工具函数
```

### 核心类设计

**1. PathExporter**
```python
class PathExporter:
    def export_json(self, path_data, file_path, metadata=None)
    def export_csv(self, path_data, file_path)
    def export_xml(self, path_data, file_path, metadata=None)
    def export_ros(self, path_data, file_path)
```

**2. ReportGenerator**
```python
class ReportGenerator:
    def generate_pdf(self, planning_data, output_path)
    def add_cover_page(self, patient_info)
    def add_path_section(self, path_data, visualizations)
    def add_3d_visualization(self, model_data, path_data)
    def add_safety_analysis(self, safety_results)
```

**3. ModelExporter**
```python
class ModelExporter:
    def export_stl(self, poly_data, file_path, binary=True)
    def export_obj(self, poly_data, file_path)
    def export_ply(self, poly_data, file_path, with_colors=True)
    def export_vtk(self, poly_data, file_path)
```

**4. MaskExporter**
```python
class MaskExporter:
    def export_nifti(self, volume, file_path, spacing, origin)
    def export_dicom_rtss(self, masks, file_path, dicom_series)
    def export_numpy(self, volume, file_path, metadata=None)
```

---

## 📊 使用场景示例

### 场景1：完整手术规划导出

**用户操作**：
1. 完成图像分割
2. 完成3D重建
3. 完成路径规划
4. 点击 "Export Planning Report"

**系统生成**：
- ✅ PDF报告（包含所有信息）
- ✅ 路径数据（JSON格式）
- ✅ 3D模型（STL + OBJ格式）
- ✅ 分割结果（NIfTI格式）

**输出目录结构**：
```
export_2026-01-04_10-30-00/
├── report.pdf
├── path_data.json
├── path_data.csv
├── model.stl
├── model.obj
├── masks.nii.gz
└── metadata.json
```

### 场景2：仅导出路径数据

**用户操作**：
1. 完成路径规划
2. 点击 "Export Path" → 选择格式

**系统生成**：
- 选择JSON：生成结构化路径数据
- 选择CSV：生成Excel兼容格式
- 选择XML：生成标准化格式
- 选择ROS：生成ROS兼容格式

### 场景3：批量导出分割结果

**用户操作**：
1. 完成分割
2. 点击 "Export Masks" → 选择格式

**系统生成**：
- PNG序列（用于查看）
- NIfTI格式（用于其他医学软件）
- DICOM RTSS（用于DICOM查看器）

---

## 🎨 UI 增强

### 新增UI元素

**1. 导出菜单/按钮组**
```
[Export Planning] [Export Path] [Export Model] [Export Masks]
```

**2. 导出选项对话框**
- 格式选择（下拉菜单）
- 质量选项（滑块）
- 包含内容（复选框）
- 输出目录选择

**3. 导出进度显示**
- 进度条
- 当前处理项
- 预计剩余时间

---

## 📦 依赖库

### 必需库
- `numpy` - 数据处理
- `vtk` - 3D模型处理
- `pydicom` - DICOM格式支持
- `nibabel` - NIfTI格式支持

### 可选库（推荐）
- `reportlab` - PDF生成
- `matplotlib` - 图表生成
- `Pillow` - 图像处理
- `lxml` - XML处理

---

## ⏱️ 预计工作量

### 阶段1：路径数据导出（1天）
- JSON/CSV/XML格式
- 基础元数据

### 阶段2：3D模型导出（0.5天）
- OBJ/PLY格式
- 格式转换

### 阶段3：分割结果导出（1天）
- NIfTI格式
- DICOM RTSS格式（如果时间允许）

### 阶段4：PDF报告生成（1-2天）
- 报告模板设计
- 内容生成
- 可视化集成

**总计**：3.5-4.5天

---

## 💡 实现建议

### 优先级排序

1. **路径数据导出（JSON/CSV）** - 最常用，最简单
2. **3D模型导出（OBJ）** - 常用格式，易于实现
3. **PDF报告生成** - 提升专业度，但较复杂
4. **分割结果导出（NIfTI）** - 医学标准，但需要额外库

### 实现策略

1. **先实现基础功能**：JSON/CSV路径导出、OBJ模型导出
2. **再实现高级功能**：PDF报告、NIfTI导出
3. **逐步增强**：添加更多格式、优化质量

---

## 🎯 预期效果

完成这个模块后，用户可以：

1. ✅ **完整导出手术规划**：一键导出所有相关数据
2. ✅ **多格式兼容**：支持不同软件和系统
3. ✅ **专业报告**：生成符合医疗标准的PDF报告
4. ✅ **数据存档**：便于后续分析和审查
5. ✅ **系统集成**：导出的数据可直接用于机器人系统

---

**最后更新**：2026-01-04

