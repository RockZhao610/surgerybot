# 手术机器人应用 (Surgical Robot App)

一个用于医学图像分割、3D 重建与路径规划的交互式桌面应用程序。支持 SAM2 自动分割、手动分割、3D 可视化、RRT/A* 路径规划与路径评估。

## ✨ 核心功能
- **医学图像分割**：SAM2 交互式分割（点/框提示、双向追踪）、手动分割、HSV 阈值分割
- **3D 可视化与重建**：基于分割掩码的 Marching Cubes 重建、VTK 实时渲染、STL 导入/导出
- **路径规划**：RRT（默认）与 A*，支持起点/终点/中间点、路径评估、撤销/重做与路径点编辑
- **数据管理**：DICOM/PNG 序列导入、掩码保存、患者信息工作流

## 🧭 快速开始
```bash
pip install -r surgical_robot_app/requirements.txt
python -m surgical_robot_app.main
```

## 📦 可选：SAM2
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
模型文件放在 `models/sam2/`（目录被 `.gitignore` 忽略）。

## 🧩 实现说明（近期改动）
- **路径规划坐标**：体数据路径规划在有 `Spacing` 时使用物理坐标（mm），STL 模型仍使用归一化坐标（0-100）
- **路径后处理**：生成路径后进行重采样与平滑，并做碰撞复验
- **路径点编辑**：列表支持双击编辑 Start/End 与路径点

## 📚 文档索引
- `SYSTEM_ARCHITECTURE.md`：系统架构与模块划分
- `WITHOUT_ROBOT_TASKS.md`：无硬件可完成的任务清单
- `DATA_EXPORT_DETAILS.md`：数据导出与格式细节
- `MODEL_POINT_PICKING_GUIDE.md`：模型点选流程说明
- `PATH_POINT_SELECTION_IMPROVEMENTS.md`：路径选点改进记录
- `ERASER_MODE_IMPLEMENTATION.md`：擦除模式实现细节
- `MAINWINDOW_REFACTOR_PLAN.md`：主窗口重构计划
- `MODEL_ALIGNMENT_PICKING_PLAN.md`：模型对齐选点计划
- `MODEL_SURFACE_PICKING_PLAN.md`：模型表面选点计划

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

## 📌 项目状态
- 已实现：图像分割、3D 重建、路径规划、路径评估、UI 工作流
- 未实现：机器人控制、视觉跟踪、远程控制、安全模块、RL 路径优化（有 demo）

## 🧾 许可证
待补充
