# 手术机器人应用 (Surgical Robot App)

一个用于医学图像分割、3D 重建与路径规划的交互式桌面应用程序。本文档**仅覆盖当前已实现功能**：分割、重建、路径规划与 UI 使用，目标是让从未使用过的人也能按步骤完成完整流程。

---

## ✅ 运行环境
- **操作系统**：macOS / Windows / Linux  
- **Python**：3.8+（推荐 3.10+）  
- **可选 GPU**：CUDA 或 Apple MPS  

## 🚀 安装与启动（从零开始）
```bash
git clone <your-repo-url>
cd surgerybot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r surgical_robot_app/requirements.txt
python -m surgical_robot_app.main
```

## 📦 SAM2（可选）
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
模型文件放在 `models/sam2/`（该目录已加入 `.gitignore`，不会提交到仓库）。

---

## 🧭 一次完整使用流程（新手版）
1. **进入系统** → 欢迎页点击 **ENTER SYSTEM**  
2. **填写患者信息** → 点击 **CONFIRM && PROCEED**  
3. **导入数据** → **Select Folder** → **Import**  
4. **分割** → 选择 SAM2 或手动分割  
5. **3D 重建** → **Reconstruct 3D**  
6. **路径规划** → 选点 → 生成路径 → 评估 → 保存  

---

## 🧩 UI 组件与“逐按钮”用法（非常细致）

### 1) Welcome View（欢迎页）
- **ENTER SYSTEM**：进入患者信息页  

### 2) Patient Info View（患者信息页）
- **Name / Patient ID / Case No / Exam Type**：患者信息  
- **CONFIRM && PROCEED**：确认并进入主界面  
- **Back**：返回欢迎页  
- **Exit**：退出程序  

### 3) 主界面结构
主界面由三部分组成：  
- **左侧控制面板**：数据导入、切片编辑、SAM2 分割  
- **中间 3D 视图面板**：重建/模型显示/路径规划  
- **顶部状态栏**：显示当前患者信息  

---

## 📂 数据导入（Data Import Panel）
### 目标
把 DICOM/PNG 序列加载为体数据，或直接加载 STL 模型。

### 按钮说明
- **Select Folder**  
  - 选择包含 DICOM 或 PNG 的文件夹  
  - 仅选择目录，不选单个文件  
- **Import**  
  - 读取文件夹中的序列  
  - 成功后显示 “Series loaded | format: ... | slices: ...”  
- **Open Local Model**  
  - 直接加载 STL 文件  
  - 适用于只做 3D 查看或 STL 路径规划  

### 文件列表
- 显示序列切片文件名  
- 第一行显示加载状态与切片数量  

---

## 🖌️ 切片编辑器（手动分割）
### 目标
逐层编辑掩码，或通过 HSV 阈值快速分割。

### 控件说明
- **Slice Slider**：切换当前切片  
- **Brush**：画笔前景（生成掩码）  
- **Eraser**：擦除掩码  
- **Threshold**：HSV 阈值区（可折叠）  
- **Apply**：应用阈值到当前切片  
- **Apply to All**：批量应用阈值（异步执行）  
- **Save Masks**：保存所有掩码到文件夹  
- **Clear Masks**：清空掩码  

### 推荐操作
- 先用 HSV 阈值粗分割  
- 再用画笔/擦除微调边界  

---

## 🤖 SAM2 分割面板
### 目标
使用 SAM2 自动分割当前切片或整个体数据。

### 按钮说明
- **Load SAM2 Model**：加载 `.pt` 权重  
- **Add Positive / Negative**：在图像上点击添加正/负样本点  
- **Add Box**：拖拽矩形框，支持实时预览  
- **Switch Mask**：切换 Whole / Part / Sub‑part 候选  
- **Start Segmentation**：对当前切片分割  
- **SAM2 Volume**：全卷分割（双向追踪）  

### 注意
- 未加载模型无法使用 SAM2  
- 全卷分割需要至少一个提示点  

---

## 🧱 3D 重建与视图
### 目标
由掩码生成 3D 模型并展示。

### 按钮说明
- **Reconstruct 3D**  
  - 基于掩码生成三角网格  
  - 自动填孔并显示坐标轴  
- **Open Model**  
  - 加载 STL 文件  
- **Reset Camera**  
  - 重置视角与缩放  
- **Open in Window**  
  - 独立 3D 窗口显示  

---

## 🧭 路径规划面板
### 目标
生成从起点到终点（经中间点）的路径，并评估质量。

### 按钮说明
- **Pick Start / Add Waypoint / Pick End**  
  - 进入选点流程  
  - 有体数据：弹窗三视图  
  - 仅 STL：坐标平面选点  
- **Generate Path**  
  - 使用 RRT 或 A* 生成路径  
- **Save Path**  
  - 保存路径点列表  
- **Reset Path**  
  - 清空路径与选点  

### 路径列表
- 显示 Start / Waypoint / End  
- 显示路径评分  
- 显示路径点 `[n]: (x,y,z)`  
- 双击任意点可编辑  

---

## ✏️ 路径点编辑（双击列表）
### 功能
- 在 XY / XZ 平面精确拖拽  
- 实时碰撞检测  
- 支持删除中间点  

### 操作步骤
1. 双击列表点（Start/End/路径点）  
2. 在 XY / XZ 平面拖动  
3. 点击 **Apply** 保存  

---

## 🎯 选点机制（非常详细）

### A. 有体数据时（DICOM/PNG）
- 点击 **Pick Start/Waypoint/End** 后弹出 **PathPointPickerDialog**  
- 弹窗显示三视图（轴向/矢状/冠状）  
- **两次点击完成一个 3D 点**：  
  1) 第一次点击确定一个平面坐标  
  2) 第二次在另一个平面补齐第三个坐标  
- 完成后自动换算为 **空间坐标** 并写入路径点列表  

### B. 仅 STL 模型时
- 点击 **Pick Start/Waypoint/End** 后弹出 **CoordinatePlanePickerDialog**  
- **XY 平面**确定 X/Y  
- **XZ 平面**确定 X/Z  
- 提供滑块与数值输入框做细调  
- 3D 视图实时预览点位置  

### C. 坐标系统说明
- **体数据**：路径规划使用物理坐标（mm）  
- **STL 模型**：使用归一化坐标（0-100）  

---

## 🧪 数据格式
- **DICOM**：目录内 `.dcm`  
- **PNG**：连续切片序列  
- **STL**：三角网格模型  

---

## 🧩 关键实现说明（已实现部分）
- **路径规划坐标**：体数据在有 `Spacing` 时使用物理坐标（mm）；STL 仍使用归一化坐标（0-100）  
- **路径后处理**：生成路径后重采样 + 平滑 + 碰撞复验  
- **路径点编辑**：支持双击编辑 Start/End 与路径点  

---

## ⚙️ 配置（可选）
在项目根目录添加 `config.json`：
```json
{
  "path_planning": {
    "grid_size": [100, 100, 105],
    "obstacle_expansion": 4,
    "use_rrt": true,
    "rrt_step_size": 2.0,
    "rrt_goal_bias": 0.1,
    "rrt_max_iterations": 5000,
    "rrt_goal_threshold": 2.0,
    "rrt_safety_radius": 5.0
  }
}
```

---

## 📁 项目结构
```
surgerybot/
├── surgical_robot_app/          # 主应用代码
│   ├── main.py                  # 应用入口
│   ├── config/                  # 配置管理
│   ├── data_io/                 # 数据输入输出
│   ├── gui/                     # 图形界面
│   ├── segmentation/            # 分割模块
│   ├── path_planning/           # 路径规划
│   ├── 3d_recon/                # 3D 重建
│   ├── vtk_utils/               # VTK 工具
│   └── utils/                   # 通用工具
├── models/                      # 模型文件（忽略）
├── logs/                        # 日志
└── README.md
```

---

## 🧹 仓库忽略内容
以下内容已加入 `.gitignore`：  
- `models/`  
- `surgical_robot_app/test_data/`  
- `path_data/`  
- `surgical_robot_app/3d_recon/result/`  

---

## 🧪 常见问题
**Q: VTK 窗口不显示？**  
A: 确认已安装 `vtk` 并且 Qt 绑定可用。  

**Q: SAM2 模型加载失败？**  
A: 确认已安装 `sam2` 且模型文件在 `models/sam2/`。  

**Q: 路径仍穿过物体？**  
A: 增大安全半径或提升点云密度。  

---

## 📌 项目状态（已实现部分）
- 图像分割  
- 3D 重建  
- 路径规划  
- 路径评估  
- UI 工作流  

## 🧾 许可证
待补充

