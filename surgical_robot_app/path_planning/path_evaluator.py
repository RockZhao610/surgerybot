import numpy as np
from typing import List, Tuple, Dict
import logging

from surgical_robot_app.path_planning.point_cloud_utils import PointCloudCollisionChecker

logger = logging.getLogger(__name__)

class PathEvaluator:
    """
    路径评估系统：负责计算路径的各项质量指标。
    """

    @staticmethod
    def calculate_length(path: List[Tuple[float, float, float]]) -> float:
        """
        计算路径的总长度。
        """
        if not path or len(path) < 2:
            return 0.0
        
        pts = np.array(path)
        diffs = np.diff(pts, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))

    @staticmethod
    def calculate_safety(path: List[Tuple[float, float, float]], 
                         collision_checker: PointCloudCollisionChecker) -> Dict[str, float]:
        """
        评估路径安全性。
        
        Returns:
            Dict containing:
                - min_distance: 路径到障碍物的最小距离
                - avg_distance: 路径到障碍物的平均距离
                - safety_score: 安全评分 (0-100)
        """
        if not path or collision_checker is None:
            return {"min_distance": 0.0, "avg_distance": 0.0, "safety_score": 0.0}

        distances = []
        # 在路径段上均匀采样点进行更精细的距离检测
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            dist_seg = np.linalg.norm(p2 - p1)
            
            # 采样点密度：每单位长度采样一个点
            num_samples = max(2, int(dist_seg) + 1)
            for t in np.linspace(0, 1, num_samples):
                sample_pt = p1 * (1 - t) + p2 * t
                d = collision_checker.get_distance_to_obstacle(tuple(sample_pt))
                distances.append(d)
        
        if not distances:
            return {"min_distance": 0.0, "avg_distance": 0.0, "safety_score": 0.0}
            
        min_dist = np.min(distances)
        avg_dist = np.mean(distances)
        
        # 安全评分逻辑：
        # 如果最小距离小于安全半径，分数为 0
        # 如果最小距离大于安全半径的3倍，分为 100
        safety_radius = collision_checker.safety_radius
        if min_dist <= safety_radius:
            safety_score = 0.0
        else:
            # 线性映射 (safety_radius, 3*safety_radius) -> (0, 100)
            safety_score = min(100.0, (min_dist - safety_radius) / (2 * safety_radius) * 100.0)
            
        return {
            "min_distance": float(min_dist),
            "avg_distance": float(avg_dist),
            "safety_score": float(safety_score)
        }

    @staticmethod
    def calculate_smoothness(path: List[Tuple[float, float, float]]) -> Dict[str, float]:
        """
        评估路径平滑度。
        通过计算相邻线段之间的夹角来评估。
        
        Returns:
            Dict containing:
                - total_curvature: 总转角 (弧度)
                - avg_curvature: 平均转角
                - smoothness_score: 平滑度评分 (0-100)
        """
        if not path or len(path) < 3:
            return {"total_curvature": 0.0, "avg_curvature": 0.0, "smoothness_score": 100.0}

        pts = np.array(path)
        vectors = np.diff(pts, axis=0)
        # 归一化向量
        norms = np.linalg.norm(vectors, axis=1)
        # 过滤掉长度为0的向量
        valid_mask = norms > 1e-6
        if np.sum(valid_mask) < 2:
            return {"total_curvature": 0.0, "avg_curvature": 0.0, "smoothness_score": 100.0}
            
        valid_vectors = vectors[valid_mask]
        valid_norms = norms[valid_mask]
        unit_vectors = valid_vectors / valid_norms[:, np.newaxis]
        
        angles = []
        for i in range(len(unit_vectors) - 1):
            v1 = unit_vectors[i]
            v2 = unit_vectors[i+1]
            # 点积求夹角
            cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_theta) # 弧度
            angles.append(angle)
            
        total_curvature = np.sum(angles)
        avg_curvature = np.mean(angles)
        
        # 平滑度评分逻辑：
        # 如果平均转角大于 90度 (pi/2)，平滑度极低
        # 假设平均转角在 0 到 30度 (pi/6) 之间是比较平滑的
        # 映射 pi/2 -> 0, 0 -> 100
        smoothness_score = max(0.0, 100.0 * (1.0 - (avg_curvature / (np.pi / 2))))
        
        return {
            "total_curvature": float(total_curvature),
            "avg_curvature": float(avg_curvature),
            "smoothness_score": float(smoothness_score)
        }

    def evaluate(self, path: List[Tuple[float, float, float]], 
                 collision_checker: PointCloudCollisionChecker) -> Dict[str, any]:
        """
        全面评估路径并返回报告。
        """
        length = self.calculate_length(path)
        safety_metrics = self.calculate_safety(path, collision_checker)
        smoothness_metrics = self.calculate_smoothness(path)
        
        # 计算综合得分 (权值可调)
        # 这里简单取平均值，或者给安全性更高权重
        total_score = (safety_metrics["safety_score"] * 0.5 + 
                       smoothness_metrics["smoothness_score"] * 0.3 + 
                       (100.0 if length > 0 else 0) * 0.2)
        
        return {
            "length": length,
            "safety": safety_metrics,
            "smoothness": smoothness_metrics,
            "total_score": total_score
        }

