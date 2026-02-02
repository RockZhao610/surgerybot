# RL 路径优化 Demo

这是一个独立的demo，展示如何使用 **Gymnasium** 和 **Stable-Baselines3** 来实现路径优化。

## 📋 内容

1. **技术栈介绍** (`RL_TECH_STACK_INTRO.md`)
   - Gymnasium 介绍
   - Stable-Baselines3 介绍
   - 两者关系和使用建议

2. **Demo 代码** (`rl_path_optimization_demo.py`)
   - 简化的2D路径优化环境
   - RL训练示例
   - 对比演示（随机、贪心、RL）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 Gymnasium
pip install gymnasium

# 安装 Stable-Baselines3（可选，用于训练）
# 注意：在 zsh 中需要使用引号
pip install "stable-baselines3[extra]"

# 安装可视化依赖
pip install matplotlib
```

### 2. 运行 Demo

```bash
cd demo
python rl_path_optimization_demo.py
```

### 3. 查看结果

Demo 会生成以下文件：
- `demo/rl_path_random.png` - 随机路径结果
- `demo/rl_path_greedy.png` - 贪心路径结果
- `demo/rl_path_result_episode_*.png` - RL优化路径结果
- `demo/rl_path_model.zip` - 训练好的模型（如果训练）

## 📊 Demo 说明

### 环境设置

- **2D栅格环境** (50x50)
- **起点**: (5, 5)
- **终点**: (45, 45)
- **障碍物**: 中间有一个矩形障碍物

### 对比方法

1. **随机路径**: 随机选择动作
2. **贪心路径**: 直接向目标移动（可能碰撞）
3. **RL优化路径**: 使用强化学习训练的策略

### RL 训练

- **算法**: SAC (Soft Actor-Critic)
- **训练步数**: 5000步（demo中，实际可以更多）
- **状态空间**: 8维（位置、目标、距离等）
- **动作空间**: 3维（方向x, y + 步长）

## 🎯 预期效果

运行demo后，你应该看到：

1. **随机路径**: 路径混乱，可能无法到达目标
2. **贪心路径**: 可能直接撞到障碍物
3. **RL优化路径**: 
   - 避开障碍物
   - 路径相对平滑
   - 成功到达目标

## 📝 注意事项

1. **这是简化版**: 实际项目是3D环境，更复杂
2. **独立运行**: 不会修改你的项目代码
3. **学习目的**: 主要用于理解RL路径优化的概念

## 🔧 自定义

你可以修改以下参数来实验：

```python
# 在 rl_path_optimization_demo.py 中修改

# 环境参数
env = SimplePathEnv(
    grid_size=(50, 50),      # 栅格大小
    start=(5.0, 5.0),        # 起点
    goal=(45.0, 45.0),       # 终点
)

# 训练参数
model = train_rl_agent(
    env, 
    algorithm='SAC',         # 或 'PPO'
    total_timesteps=10000,   # 训练步数
)
```

## 📚 相关文档

- [技术栈介绍](../RL_TECH_STACK_INTRO.md)
- [完整实现方案](../RL_PATH_OPTIMIZATION_PLAN.md)

## ❓ 常见问题

**Q: 为什么是2D而不是3D？**  
A: 为了简化演示和可视化。实际项目中会使用3D环境。

**Q: 训练需要多长时间？**  
A: 在demo中，5000步大约需要1-2分钟。实际项目中可能需要更长时间。

**Q: 可以用于实际项目吗？**  
A: 这个demo是概念验证。实际项目需要更复杂的环境和更长的训练时间。

