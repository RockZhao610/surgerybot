# 手术机器人应用 (Surgical Robot App)

一个用于医学图像分割、3D 重建与路径规划的交互式桌面应用程序。支持 SAM2 自动分割、手动分割、3D 可视化、RRT/A* 路径规划、路径评估与路径点精细编辑。

## ✨ 功能概览
- **医学图像分割**：SAM2 交互式分割（正/负点、矩形框、双向追踪）、手动分割、HSV 阈值分割  
- **3D 可视化与重建**：Marching Cubes 重建、VTK 实时渲染、STL 导入/导出  
- **路径规划**：RRT（默认）与 A*、路径评估、撤销/重做、路径点编辑  
- **数据管理**：DICOM/PNG 导入、掩码保存、患者信息工作流  

## ✅ 运行环境
- **操作系统**：macOS / Windows / Linux  
- **Python**：3.8+（推荐 3.10+）  
- **可选 GPU**：CUDA 或 Apple MPS  

## 🚀 安装与启动
```bash
pip install -r surgical_robot_app/requirements.txt
python -m surgical_robot_app.main
```

## 📦 SAM2（可选）
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```
模型文件放在 `models/sam2/`（该目录已加入 `.gitignore`，不会提交到仓库）。

## 🧭 使用流程（详细）
1. **启动应用**  
   - 欢迎界面 → 进入系统  
2. **患者信息**  
   - 填写姓名、患者 ID、病例号、检查类型  
3. **导入数据**  
   - DICOM / PNG 目录  
   - 或直接打开 STL  
4. **图像分割**  
   - **SAM2**：加载模型 → 选择正/负点或框选 → 单层/全卷分割  
   - **手动**：画笔/擦除/HSV 阈值  
5. **3D 重建**  
   - 点击重建 → 生成 3D 模型 → 可导出 STL  
6. **路径规划**  
   - 设置起点 / 中间点 / 终点  
   - 生成路径 → 评估 → 保存  
7. **路径点编辑**  
   - 双击列表点（Start/End/路径点） → 进入三视图编辑  

## 🧪 数据格式
- **DICOM**：目录内 `.dcm`  
- **PNG**：连续切片序列  
- **STL**：三角网格模型  

## 🧩 关键实现说明
- **路径规划坐标**：体数据在有 `Spacing` 时使用物理坐标（mm）；STL 仍使用归一化坐标（0-100）  
- **路径后处理**：生成路径后重采样 + 平滑 + 碰撞复验  
- **路径点编辑**：支持双击编辑 Start/End 与路径点  

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

## 🧹 仓库忽略内容
以下内容已加入 `.gitignore`：
- `models/`  
- `surgical_robot_app/test_data/`  
- `path_data/`  
- `surgical_robot_app/3d_recon/result/`  

## 🧪 常见问题
**Q: VTK 窗口不显示？**  
A: 确认已安装 `vtk` 并且 Qt 绑定可用。  

**Q: SAM2 模型加载失败？**  
A: 确认已安装 `sam2` 且模型文件在 `models/sam2/`。  

**Q: 路径仍穿过物体？**  
A: 增大安全半径或提升点云密度。  

## 📌 项目状态
- 已实现：分割、3D 重建、路径规划、路径评估、UI 工作流  
- 未实现：机器人控制、视觉跟踪、远程控制、安全模块、RL 路径优化  

## 🧾 许可证
待补充
