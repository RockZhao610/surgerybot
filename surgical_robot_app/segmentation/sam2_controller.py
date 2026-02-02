"""
SAM2Controller: SAM2 分割控制器

职责：
- 管理 SAM2Segmenter 的加载与状态
- 维护提示信息（点/框）
- 对单张切片或整个 volume 执行分割

注意：
- 不依赖 Qt，只做模型与图像/提示之间的协调，便于在 GUI 之外复用或单元测试。
"""

import os
import tempfile
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import cv2

try:
    from surgical_robot_app.segmentation.sam2_segmenter import SAM2Segmenter, create_sam2_segmenter
except ImportError:
    try:
        from segmentation.sam2_segmenter import SAM2Segmenter, create_sam2_segmenter  # type: ignore
    except ImportError:
        SAM2Segmenter = None  # type: ignore
        create_sam2_segmenter = None  # type: ignore

logger = logging.getLogger(__name__)


PromptPoint = Tuple[int, int, int]  # (x, y, label) label: 1=前景, 0=背景


class SAM2Controller:
    def __init__(self, model_type: str = "hiera_large"):
        """
        初始化 SAM2 控制器

        Args:
            model_type: SAM2 模型类型（例如 'hiera_tiny'/'hiera_small' 等）
        """
        self.segmenter: Optional[SAM2Segmenter] = None
        self.model_path: Optional[str] = None
        self.model_type: str = model_type

        # 提示相关
        self.prompt_mode: str = "point"  # 'point' or 'box'
        self.prompt_points: List[PromptPoint] = []
        self.prompt_box: Optional[Tuple[int, int, int, int]] = None

        # 多掩码候选系统
        self.mask_candidates: List[np.ndarray] = []
        self.current_candidate_idx: int = 0

    # -------------------- 模型加载与状态 --------------------

    def load_model(self, model_path: str) -> bool:
        """
        加载 SAM2 模型

        Returns:
            True 表示加载成功，False 表示失败
        """
        if create_sam2_segmenter is None:
            return False

        segmenter = create_sam2_segmenter(model_path=model_path, model_type=self.model_type)
        if segmenter is not None and segmenter.is_model_loaded():
            self.segmenter = segmenter
            self.model_path = model_path
            return True

        self.segmenter = None
        self.model_path = None
        return False

    def is_model_loaded(self) -> bool:
        return self.segmenter is not None and self.segmenter.is_model_loaded()

    # -------------------- 提示管理 --------------------

    def set_prompt_mode(self, mode: str) -> None:
        """设置提示模式：'point' 或 'box'，并清空现有提示"""
        if mode not in ("point", "box"):
            mode = "point"
        self.prompt_mode = mode
        self.clear_prompts()

    def clear_prompts(self) -> None:
        """清除所有提示点和框，并重置候选系统"""
        self.prompt_points = []
        self.prompt_box = None
        self.mask_candidates = []
        self.current_candidate_idx = 0

    def add_point_prompt(self, x: int, y: int, label: int) -> int:
        """
        添加一点提示

        Args:
            x, y: 像素坐标
            label: 1=前景（正样本），0=背景（负样本）

        Returns:
            当前提示点总数
        """
        self.prompt_points.append((int(x), int(y), int(label)))
        return len(self.prompt_points)

    def set_box_prompt(self, x_min: int, y_min: int, x_max: int, y_max: int) -> None:
        """设置框选提示"""
        self.prompt_box = (int(x_min), int(y_min), int(x_max), int(y_max))

    def get_point_prompts(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """返回点提示的坐标和标签列表"""
        coords = [(x, y) for x, y, _ in self.prompt_points]
        labels = [label for _, _, label in self.prompt_points]
        return coords, labels

    def undo_last_positive_point(self) -> Optional[PromptPoint]:
        """
        撤销最后一次添加的正点（label=1）
        
        Returns:
            被撤销的点 (x, y, label)，如果没有正点则返回 None
        """
        # 从后往前查找最后一个正点
        for i in range(len(self.prompt_points) - 1, -1, -1):
            x, y, label = self.prompt_points[i]
            if label == 1:  # 找到正点
                removed_point = self.prompt_points.pop(i)
                return removed_point
        return None

    # -------------------- 分割接口 --------------------

    def segment_slice(self, rgb_slice: np.ndarray) -> np.ndarray:
        """
        使用交互式组合提示（框 + 点）执行分割，支持多掩码模式。
        """
        if not self.is_model_loaded():
            raise RuntimeError("SAM2 model not loaded")
        if self.segmenter is None:
            raise RuntimeError("SAM2 segmenter is None")

        # 获取当前所有的提示信息
        coords, labels = self.get_point_prompts()
        box = self.prompt_box

        # 如果既没有框也没有点，返回空掩码
        if box is None and not coords:
            self.mask_candidates = []
            return np.zeros(rgb_slice.shape[:2], dtype=np.uint8)

        # 调用底层接口执行多掩码混合预测
        candidates = self.segmenter.segment_with_mixed_prompts(
            rgb_slice, 
            point_coords=coords if coords else None, 
            point_labels=labels if labels else None, 
            box=box,
            multimask=True # 获取 3 个层级的掩码
        )

        self.mask_candidates = candidates
        # 确保索引合法
        if self.current_candidate_idx >= len(self.mask_candidates):
            self.current_candidate_idx = 0

        return self.mask_candidates[self.current_candidate_idx] if self.mask_candidates else np.zeros(rgb_slice.shape[:2], dtype=np.uint8)

    def switch_mask_candidate(self) -> Optional[np.ndarray]:
        """在生成的 3 个掩码层级之间循环切换"""
        if not self.mask_candidates:
            return None
        self.current_candidate_idx = (self.current_candidate_idx + 1) % len(self.mask_candidates)
        return self.mask_candidates[self.current_candidate_idx]

    def segment_volume(self, volume: np.ndarray, slice_idx: int, progress_callback: Optional[Callable[[int], None]] = None) -> np.ndarray:
        """
        使用 SAM2 Video Predictor 追踪模式进行 3D 批量分割。
        (通过临时目录中转，解决官方库不支持内存数组的问题)
        """
        if not self.is_model_loaded():
            raise RuntimeError("SAM2 model not loaded")
        
        # 1. 准备提示点
        coords, labels = self.get_point_prompts()
        if not coords:
            raise RuntimeError("No prompt points found on current slice")

        # 2. 创建临时文件夹并保存切片为 JPEG
        tmp_dir = tempfile.mkdtemp()
        try:
            Z = volume.shape[0]
            logger.info(f"Preparing temporary sequence: {Z} frames -> {tmp_dir}")
            
            for z in range(Z):
                frame = volume[z]
                if frame.ndim == 2:
                    frame_rgb = np.stack([frame] * 3, axis=-1)
                else:
                    frame_rgb = frame
                
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8) if frame_rgb.max() <= 1.0 else frame_rgb.astype(np.uint8)
                
                img_path = os.path.join(tmp_dir, f"{z:05d}.jpg")
                cv2.imwrite(img_path, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            # 3. 初始化视频状态
            inference_state = self.segmenter.init_video_state(tmp_dir)
            
            # 4. 在种子层添加提示
            self.segmenter.add_new_points_to_video(
                inference_state=inference_state,
                frame_idx=int(slice_idx),
                obj_id=1,
                points=np.array(coords, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32)
            )

            # 5. 执行双向追踪传播
            masks_3d = np.zeros((Z, volume.shape[1], volume.shape[2]), dtype=np.uint8)
            processed_frames = 0
            
            # 5.1 向后传播 (Forward in time: slice_idx -> Z-1)
            logger.info(f"Starting forward propagation from frame {slice_idx}...")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.segmenter.propagate_video(
                inference_state, start_frame_idx=slice_idx, reverse=False
            ):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
                masks_3d[out_frame_idx] = (mask * 255).astype(np.uint8)
                processed_frames += 1
                if progress_callback:
                    progress_callback(int((processed_frames / Z) * 100))

            # 5.2 向前传播 (Backward in time: slice_idx -> 0)
            logger.info(f"Starting backward propagation from frame {slice_idx}...")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.segmenter.propagate_video(
                inference_state, start_frame_idx=slice_idx, reverse=True
            ):
                mask = (out_mask_logits[0] > 0.0).cpu().numpy().astype(np.uint8).squeeze()
                masks_3d[out_frame_idx] = (mask * 255).astype(np.uint8)
                processed_frames += 1
                if progress_callback:
                    progress_callback(int((processed_frames / Z) * 100))

            return masks_3d

        finally:
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
                logger.info(f"Temporary data cleaned: {tmp_dir}")


