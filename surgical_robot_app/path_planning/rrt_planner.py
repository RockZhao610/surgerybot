import numpy as np
import random
from typing import List, Tuple, Optional, Callable
import logging

from surgical_robot_app.path_planning.point_cloud_utils import PointCloudCollisionChecker

logger = logging.getLogger(__name__)

class TreeNode:
    def __init__(self, point: Tuple[float, float, float], parent=None):
        self.point = point
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def get_path_to_root(self) -> List[Tuple[float, float, float]]:
        path = []
        current = self
        while current is not None:
            path.append(current.point)
            current = current.parent
        return path[::-1]  # 反转路径，从起点到终点

class RRTPlanner:
    def __init__(
        self,
        collision_checker: PointCloudCollisionChecker,
        step_size: float = 2.0,
        goal_bias: float = 0.1,
        max_iterations: int = 5000,
        goal_threshold: float = 2.0,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None,
    ):
        self.collision_checker = collision_checker
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.goal_threshold = goal_threshold
        
        # 边界通常是 [0, 100]，也可使用物理坐标边界
        if bounds is not None:
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
            self.min_bounds = np.array([xmin, ymin, zmin], dtype=float)
            self.max_bounds = np.array([xmax, ymax, zmax], dtype=float)
        else:
            self.min_bounds = np.array([0.0, 0.0, 0.0], dtype=float)
            self.max_bounds = np.array([100.0, 100.0, 100.0], dtype=float)

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
        progress_callback: Optional[Callable[[int], None]] = None,
        tree_callback: Optional[Callable[[List], None]] = None
    ) -> Optional[List[Tuple[float, float, float]]]:
        """在起点和终点之间规划路径"""
        start_distance = self.collision_checker.get_distance_to_obstacle(start)
        goal_distance = self.collision_checker.get_distance_to_obstacle(goal)
        
        adjusted_start = start
        adjusted_goal = goal
        
        # 修复之前的缩进错误逻辑
        if start_distance < self.collision_checker.safety_radius:
            if self.collision_checker.is_inside_obstacle(start):
                logger.warning(f"起点在障碍物内部，尝试调整: {start}")
                adjusted_start = self.collision_checker.adjust_point_away_from_obstacle(start)
                if adjusted_start is None:
                    logger.error(f"无法将起点调整到安全位置")
                    return None
                logger.info(f"起点已调整: {start} -> {adjusted_start}")
        
        if goal_distance < self.collision_checker.safety_radius:
            if self.collision_checker.is_inside_obstacle(goal):
                logger.warning(f"终点在障碍物内部，尝试调整: {goal}")
                adjusted_goal = self.collision_checker.adjust_point_away_from_obstacle(goal)
                if adjusted_goal is None:
                    logger.error(f"无法将终点调整到安全位置")
                    return None
                logger.info(f"终点已调整: {goal} -> {adjusted_goal}")

        start = adjusted_start
        goal = adjusted_goal
        
        root = TreeNode(start)
        tree = [root]
        goal_node = None
        
        # 实时可视化：累计树的边
        all_edges: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
        tree_viz_interval = 100  # 每 100 次迭代发送一次可视化数据
        
        for iteration in range(self.max_iterations):
            if progress_callback and iteration % 100 == 0:
                progress_callback(int((iteration / self.max_iterations) * 100))
                
            if random.random() < self.goal_bias:
                random_point = goal
            else:
                random_point = self._random_point()
            
            nearest_node = self._nearest_node(tree, random_point)
            new_point = self._steer(nearest_node.point, random_point)
            
            if self.collision_checker.is_collision(new_point):
                continue
            
            if not self.collision_checker.is_path_collision_free(nearest_node.point, new_point):
                continue
            
            new_node = TreeNode(new_point, nearest_node)
            nearest_node.add_child(new_node)
            tree.append(new_node)
            
            # 记录新边
            all_edges.append((nearest_node.point, new_point))
            
            # 定期发送树可视化数据
            if tree_callback and iteration % tree_viz_interval == 0 and all_edges:
                tree_callback({
                    'edges': list(all_edges),
                    'goal': goal,
                    'iteration': iteration,
                    'total': self.max_iterations,
                })
            
            if self._distance(new_point, goal) <= self.goal_threshold:
                if self.collision_checker.is_path_collision_free(new_point, goal):
                    goal_node = TreeNode(goal, new_node)
                    new_node.add_child(goal_node)
                    all_edges.append((new_point, goal))
                    logger.info(f"RRT找到路径，迭代次数: {iteration + 1}")
                    
                    # 发送最终树数据（含找到的路径）
                    if tree_callback:
                        final_path = goal_node.get_path_to_root()
                        tree_callback({
                            'edges': list(all_edges),
                            'goal': goal,
                            'iteration': iteration + 1,
                            'total': self.max_iterations,
                            'found_path': final_path,
                        })
                    break
        
        if goal_node is not None:
            return goal_node.get_path_to_root()
        
        return None

    def plan_multi_segment(
        self,
        waypoints: List[Tuple[float, float, float]],
        smooth: bool = True,
        progress_callback: Optional[Callable[[int], None]] = None,
        tree_callback: Optional[Callable[[dict], None]] = None
    ) -> Optional[List[Tuple[float, float, float]]]:
        """多段路径规划"""
        if len(waypoints) < 2:
            return None
            
        full_path = []
        num_segments = len(waypoints) - 1
        for i in range(num_segments):
            # 为每一段分配进度区间
            seg_start_p = int((i / num_segments) * 100)
            seg_end_p = int(((i + 1) / num_segments) * 100)
            
            def _seg_progress(p, _sp=seg_start_p, _ep=seg_end_p):
                if progress_callback:
                    progress_callback(_sp + int(p * (_ep - _sp) / 100))

            # 为 tree_callback 包装一个段标识
            def _seg_tree_callback(data, _seg=i, _total=num_segments):
                if tree_callback:
                    data['segment'] = _seg
                    data['total_segments'] = _total
                    tree_callback(data)

            segment = self.plan(
                waypoints[i], waypoints[i+1],
                progress_callback=_seg_progress,
                tree_callback=_seg_tree_callback if tree_callback else None
            )
            if segment is None:
                logger.error(f"无法规划第 {i} 段路径")
                return None
            
            # 对每一段单独进行平滑，确保段与段之间的关键点（Waypoint）被保留
            if smooth and len(segment) > 2:
                segment = self._smooth_path(segment)
                
            if i > 0:
                full_path.extend(segment[1:])
            else:
                full_path.extend(segment)
        
        return full_path

    def _smooth_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """路径平滑：在保持路径点顺序的前提下，尝试跳过不必要的中间节点"""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            best_next = current_idx + 1
            # 从最后面开始往前找，看最远能连接到哪一个点
            for next_idx in range(len(path) - 1, current_idx + 1, -1):
                # 如果两点之间无碰撞，则可以直接连接，跳过中间点
                if self.collision_checker.is_path_collision_free(path[current_idx], path[next_idx], num_check_points=20):
                    best_next = next_idx
                    break
            smoothed.append(path[best_next])
            current_idx = best_next
            
        return smoothed

    def _random_point(self) -> Tuple[float, float, float]:
        point = np.random.uniform(self.min_bounds, self.max_bounds)
        return tuple(point)

    def _nearest_node(self, tree: List[TreeNode], target_point: Tuple[float, float, float]) -> TreeNode:
        distances = [self._distance(node.point, target_point) for node in tree]
        return tree[np.argmin(distances)]

    def _steer(self, from_point: Tuple[float, float, float], to_point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        from_pt = np.array(from_point)
        to_pt = np.array(to_point)
        diff = to_pt - from_pt
        dist = np.linalg.norm(diff)
        
        if dist <= self.step_size:
            return to_point
            
        new_pt = from_pt + (diff / dist) * self.step_size
        return tuple(new_pt)

    def _distance(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _validate_path(self, path: List[Tuple[float, float, float]]) -> bool:
        """最终验证路径是否穿过物体"""
        for i in range(len(path) - 1):
            if not self.collision_checker.is_path_collision_free(path[i], path[i+1]):
                return False
        return True
