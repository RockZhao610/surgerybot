"""
强化学习路径优化 Demo

这是一个独立的demo，展示如何使用 Gymnasium 和 Stable-Baselines3 
来实现路径优化。

注意：这个demo不会修改你的项目代码，可以独立运行。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import sys
from pathlib import Path

# 添加项目路径（用于导入工具函数）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== 第一部分：使用 Gymnasium 定义环境 ====================

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    print("⚠️  Gymnasium 未安装，请运行: pip install gymnasium")
    GYMNASIUM_AVAILABLE = False

try:
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.callbacks import EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  Stable-Baselines3 未安装，请运行: pip install stable-baselines3[extra]")
    SB3_AVAILABLE = False


class SimplePathEnv(gym.Env if GYMNASIUM_AVAILABLE else object):
    """
    简化的路径优化环境（2D版本，便于可视化）
    
    这是一个简化的demo，展示RL路径优化的基本概念。
    实际项目中应该是3D环境。
    
    继承自 gymnasium.Env 以兼容 Stable-Baselines3
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (50, 50),
        start: Tuple[float, float] = (5.0, 5.0),
        goal: Tuple[float, float] = (45.0, 45.0),
        obstacle_map: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
    ):
        """
        初始化环境
        
        Args:
            grid_size: 栅格大小 (width, height)
            start: 起点坐标
            goal: 终点坐标
            obstacle_map: 障碍物地图（可选）
            render_mode: 渲染模式（可选）
        """
        if GYMNASIUM_AVAILABLE:
            super().__init__()
        
        self.grid_size = grid_size
        self.start = np.array(start, dtype=np.float32)
        self.goal = np.array(goal, dtype=np.float32)
        self.render_mode = render_mode
        
        # 创建障碍物地图（如果没有提供）
        if obstacle_map is None:
            self.obstacle_map = self._create_default_obstacles()
        else:
            self.obstacle_map = obstacle_map
        
        # 当前状态
        self.current_pos = self.start.copy()
        self.path = [self.start.copy()]
        self.step_count = 0
        self.max_steps = 500
        
        # 定义动作空间（连续：方向 + 步长）
        # 动作：[dx, dy, step_size]
        # dx, dy: 方向（归一化到[-1, 1]）
        # step_size: 步长 [0.1, 2.0]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.1], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # 定义状态空间
        # 状态：[current_x, current_y, goal_x, goal_y, 
        #        distance_to_goal, min_obstacle_distance, 
        #        path_length, step_count_normalized]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )
    
    def _create_default_obstacles(self) -> np.ndarray:
        """创建默认障碍物（一个矩形障碍物）"""
        obstacle_map = np.zeros(self.grid_size, dtype=bool)
        
        # 在中间创建一个矩形障碍物
        x1, y1 = 15, 15
        x2, y2 = 35, 35
        obstacle_map[y1:y2, x1:x2] = True
        
        return obstacle_map
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.current_pos = self.start.copy()
        self.path = [self.start.copy()]
        self.step_count = 0
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        Args:
            action: [dx, dy, step_size] 或 numpy array
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 确保 action 是 numpy array
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 解析动作
        direction = action[:2]
        step_size = action[2] if len(action) > 2 else 1.0
        
        # 归一化方向
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        
        # 计算下一个位置
        next_pos = self.current_pos + direction * step_size
        
        # 检查边界
        next_pos[0] = np.clip(next_pos[0], 0, self.grid_size[0] - 1)
        next_pos[1] = np.clip(next_pos[1], 0, self.grid_size[1] - 1)
        
        # 检查碰撞
        collision = self._check_collision(next_pos)
        
        # 更新状态
        self.current_pos = next_pos
        self.path.append(next_pos.copy())
        self.step_count += 1
        
        # 计算奖励
        reward = self._compute_reward(collision)
        
        # 检查终止条件
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        reached_goal = distance_to_goal < 2.0
        terminated = reached_goal or collision
        truncated = self.step_count >= self.max_steps
        
        info = {
            'reached_goal': reached_goal,
            'collision': collision,
            'distance_to_goal': distance_to_goal,
        }
        
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info
    
    def _check_collision(self, pos: np.ndarray) -> bool:
        """检查是否碰撞障碍物"""
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
            return self.obstacle_map[y, x]
        return True  # 超出边界视为碰撞
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察（状态）"""
        # 到目标的距离
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        
        # 到最近障碍物的距离
        min_obstacle_dist = self._get_min_obstacle_distance(self.current_pos)
        
        # 路径长度
        path_length = self._compute_path_length()
        
        # 归一化步数
        step_count_norm = self.step_count / self.max_steps
        
        # 组合状态向量
        obs = np.array([
            self.current_pos[0] / self.grid_size[0],  # 归一化x
            self.current_pos[1] / self.grid_size[1],  # 归一化y
            self.goal[0] / self.grid_size[0],         # 归一化目标x
            self.goal[1] / self.grid_size[1],         # 归一化目标y
            distance_to_goal / 100.0,                 # 归一化距离
            min_obstacle_dist / 10.0,                 # 归一化障碍物距离
            path_length / 200.0,                       # 归一化路径长度
            step_count_norm,                          # 归一化步数
        ], dtype=np.float32)
        
        return obs
    
    def _get_min_obstacle_distance(self, pos: np.ndarray) -> float:
        """计算到最近障碍物的距离"""
        x, y = int(pos[0]), int(pos[1])
        min_dist = float('inf')
        
        # 检查周围区域
        search_radius = 10
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x, check_y = x + dx, y + dy
                if (0 <= check_x < self.grid_size[0] and 
                    0 <= check_y < self.grid_size[1] and 
                    self.obstacle_map[check_y, check_x]):
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist < float('inf') else 20.0
    
    def _compute_path_length(self) -> float:
        """计算当前路径长度"""
        if len(self.path) < 2:
            return 0.0
        total_length = 0.0
        for i in range(1, len(self.path)):
            total_length += np.linalg.norm(self.path[i] - self.path[i-1])
        return total_length
    
    def _compute_reward(self, collision: bool) -> float:
        """计算奖励"""
        if collision:
            return -100.0  # 碰撞大惩罚
        
        # 到目标的距离
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal)
        
        # 到达目标
        if distance_to_goal < 2.0:
            return 100.0
        
        # 路径长度惩罚
        step_length = np.linalg.norm(self.path[-1] - self.path[-2]) if len(self.path) > 1 else 0
        length_penalty = -0.1 * step_length
        
        # 安全性奖励（距离障碍物越远越好）
        min_obstacle_dist = self._get_min_obstacle_distance(self.current_pos)
        safety_reward = 1.0 * min_obstacle_dist
        
        # 进度奖励（向目标移动）
        if len(self.path) > 1:
            prev_distance = np.linalg.norm(self.path[-2] - self.goal)
            progress = prev_distance - distance_to_goal
            progress_reward = 0.5 * progress
        else:
            progress_reward = 0.0
        
        # 平滑度奖励（鼓励小角度变化）
        smoothness_reward = 0.0
        if len(self.path) > 2:
            v1 = self.path[-1] - self.path[-2]
            v2 = self.path[-2] - self.path[-3]
            if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_change = np.arccos(np.clip(cos_angle, -1, 1))
                smoothness_reward = -0.5 * angle_change
        
        total_reward = length_penalty + safety_reward + progress_reward + smoothness_reward
        return total_reward
    
    def render(self, save_path: Optional[str] = None):
        """
        可视化环境
        
        Args:
            save_path: 保存路径（可选），如果为None则显示图像
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制障碍物
        obstacle_vis = np.zeros((*self.grid_size, 3))
        obstacle_vis[self.obstacle_map] = [0.3, 0.3, 0.3]  # 灰色障碍物
        
        ax.imshow(obstacle_vis, origin='lower', extent=[0, self.grid_size[0], 0, self.grid_size[1]])
        
        # 绘制起点
        ax.plot(self.start[0], self.start[1], 'go', markersize=15, label='Start', zorder=5)
        
        # 绘制终点
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=15, label='Goal', zorder=5)
        
        # 绘制路径
        if len(self.path) > 1:
            path_array = np.array(self.path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Path', zorder=3)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b.', markersize=5, zorder=4)
        
        # 绘制当前位置
        ax.plot(self.current_pos[0], self.current_pos[1], 'yo', markersize=10, label='Current', zorder=6)
        
        ax.set_xlim(0, self.grid_size[0])
        ax.set_ylim(0, self.grid_size[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Path Optimization Environment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            # 确保目录存在
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_path_obj), dpi=150, bbox_inches='tight')
            print(f"✅ 图像已保存到: {save_path_obj.absolute()}")
        else:
            plt.show()
        
        plt.close()


# ==================== 第二部分：使用 Stable-Baselines3 训练 ====================

def train_rl_agent(env: SimplePathEnv, algorithm: str = 'SAC', total_timesteps: int = 10000):
    """
    训练RL智能体
    
    Args:
        env: 环境
        algorithm: 算法名称 ('SAC' 或 'PPO')
        total_timesteps: 训练步数
    """
    if not SB3_AVAILABLE:
        print("❌ Stable-Baselines3 未安装，无法训练")
        return None
    
    print(f"\n🚀 开始训练 {algorithm} 智能体...")
    print(f"   训练步数: {total_timesteps}")
    
    # 选择算法
    if algorithm == 'SAC':
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
        )
    elif algorithm == 'PPO':
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
        )
    else:
        raise ValueError(f"未知算法: {algorithm}")
    
    # 训练
    model.learn(total_timesteps=total_timesteps)
    
    print("✅ 训练完成！")
    return model


def test_agent(model, env: SimplePathEnv, demo_dir: Path, num_episodes: int = 3):
    """测试训练好的智能体"""
    print(f"\n🧪 测试智能体 ({num_episodes} 个episode)...")
    
    success_count = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        
        if info.get('reached_goal', False):
            success_count += 1
            print(f"  Episode {episode + 1}: ✅ 成功到达目标！奖励: {episode_reward:.2f}")
        else:
            print(f"  Episode {episode + 1}: ❌ 未到达目标。奖励: {episode_reward:.2f}")
        
        # 可视化最后一个episode
        if episode == num_episodes - 1:
            save_path = demo_dir / f"rl_path_result_episode_{episode + 1}.png"
            env.render(save_path=str(save_path))
    
    print(f"\n📊 测试结果:")
    print(f"   成功率: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"   平均奖励: {total_reward/num_episodes:.2f}")


# ==================== 第三部分：对比演示 ====================

def random_path_demo(env: SimplePathEnv, demo_dir: Path):
    """随机路径演示（对比用）"""
    print("\n🎲 随机路径演示（对比用）...")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # 随机动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    save_path = demo_dir / "rl_path_random.png"
    env.render(save_path=str(save_path))
    print(f"   最终距离目标: {info.get('distance_to_goal', 0):.2f}")


def greedy_path_demo(env: SimplePathEnv, demo_dir: Path):
    """贪心路径演示（对比用）"""
    print("\n🎯 贪心路径演示（对比用）...")
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # 贪心策略：直接向目标移动
        direction = env.goal - env.current_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm
        
        action = np.array([direction[0], direction[1], 1.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    save_path = demo_dir / "rl_path_greedy.png"
    env.render(save_path=str(save_path))
    print(f"   最终距离目标: {info.get('distance_to_goal', 0):.2f}")


# ==================== 主函数 ====================

def main():
    """主函数：运行完整的demo"""
    print("=" * 60)
    print("🚀 强化学习路径优化 Demo")
    print("=" * 60)
    
    # 检查依赖
    if not GYMNASIUM_AVAILABLE:
        print("\n❌ 请先安装 Gymnasium: pip install gymnasium")
        return
    
    # 创建demo目录
    demo_dir = Path(__file__).parent
    demo_dir.mkdir(exist_ok=True)
    
    # 1. 创建环境
    print("\n📦 创建环境...")
    env = SimplePathEnv(
        grid_size=(50, 50),
        start=(5.0, 5.0),
        goal=(45.0, 45.0),
    )
    print("✅ 环境创建成功")
    
    # 2. 演示随机路径（对比）
    random_path_demo(env, demo_dir)
    
    # 3. 演示贪心路径（对比）
    greedy_path_demo(env, demo_dir)
    
    # 4. 训练RL智能体（如果SB3可用）
    if SB3_AVAILABLE:
        model = train_rl_agent(env, algorithm='SAC', total_timesteps=5000)
        
        if model:
            # 5. 测试智能体
            test_agent(model, env, demo_dir, num_episodes=3)
            
            # 6. 保存模型
            model_path = demo_dir / "rl_path_model"
            model.save(str(model_path))
            print(f"\n💾 模型已保存到: {model_path}")
    else:
        print("\n⚠️  Stable-Baselines3 未安装，跳过训练步骤")
        print("   可以运行: pip install \"stable-baselines3[extra]\"")
    
    print("\n" + "=" * 60)
    print("✅ Demo 完成！")
    print("=" * 60)
    print("\n📝 说明:")
    print("   - 这个demo是独立的，不会修改你的项目代码")
    print("   - 生成的图像保存在 demo/ 目录")
    print("   - 可以对比随机路径、贪心路径和RL优化路径的效果")


if __name__ == "__main__":
    main()

