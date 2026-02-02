"""
ManualSegController: 手动画笔分割控制器

职责：
- 管理手动分割的掩码列表和历史记录
- 提供画笔操作（在指定切片上画圆形前景）

注意：
- 不依赖 Qt，只处理 numpy 数组，便于测试。
"""

from typing import Dict, List, Tuple

import numpy as np


class ManualSegController:
    def __init__(self, brush_size: int = 10):
        self.brush_size = int(brush_size)
        # 每个切片一个历史栈：idx -> [mask_copy1, mask_copy2, ...]
        self.mask_history: Dict[int, List[np.ndarray]] = {}

    def ensure_masks(
        self,
        volume_shape: Tuple[int, int, int],
        existing_masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        确保掩码列表与 volume 形状一致；若不一致则重新初始化。

        Args:
            volume_shape: (Z, H, W)
            existing_masks: 当前的掩码列表
        """
        depth, height, width = int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2])
        if len(existing_masks) != depth:
            masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(depth)]
            return masks
        return existing_masks

    def apply_brush(
        self,
        masks: List[np.ndarray],
        slice_idx: int,
        center: Tuple[int, int],
        radius: int,
    ) -> List[np.ndarray]:
        """
        在指定切片上应用画笔（圆形前景）。

        Args:
            masks: 掩码列表
            slice_idx: 当前切片索引
            center: (ix, iy) 像素坐标
            radius: 画笔半径

        Returns:
            更新后的掩码列表
        """
        if slice_idx < 0 or slice_idx >= len(masks):
            return masks

        # 如果当前切片的掩码为 None，则创建一个空的掩码
        if masks[slice_idx] is None:
            # 从其他切片获取形状信息
            for m in masks:
                if m is not None:
                    masks[slice_idx] = np.zeros_like(m, dtype=np.uint8)
                    break
            # 如果所有掩码都是 None，无法确定形状，返回原始列表
            if masks[slice_idx] is None:
                return masks

        h, w = masks[slice_idx].shape[:2]
        ix, iy = int(center[0]), int(center[1])
        r = int(radius)

        # 记录历史
        if slice_idx not in self.mask_history:
            self.mask_history[slice_idx] = []
        self.mask_history[slice_idx].append(masks[slice_idx].copy())

        # 应用画笔
        y_min = max(0, iy - r)
        y_max = min(h, iy + r + 1)
        x_min = max(0, ix - r)
        x_max = min(w, ix + r + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
        masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 255

        return masks

    def apply_eraser(
        self,
        masks: List[np.ndarray],
        slice_idx: int,
        center: Tuple[int, int],
        radius: int,
    ) -> List[np.ndarray]:
        """
        在指定切片上应用擦除笔（清除圆形区域内的掩码）。

        Args:
            masks: 掩码列表
            slice_idx: 当前切片索引
            center: (ix, iy) 像素坐标
            radius: 擦除笔半径

        Returns:
            更新后的掩码列表
        """
        if slice_idx < 0 or slice_idx >= len(masks):
            return masks

        # 如果当前切片的掩码为 None，则创建一个空的掩码
        if masks[slice_idx] is None:
            # 需要从其他切片或 volume 获取形状信息
            # 如果无法获取，则返回原始列表
            for m in masks:
                if m is not None:
                    masks[slice_idx] = np.zeros_like(m)
                    break
            # 如果所有掩码都是 None，无法确定形状，返回原始列表
            if masks[slice_idx] is None:
                return masks

        h, w = masks[slice_idx].shape[:2]
        ix, iy = int(center[0]), int(center[1])
        r = int(radius)

        # 记录历史
        if slice_idx not in self.mask_history:
            self.mask_history[slice_idx] = []
        self.mask_history[slice_idx].append(masks[slice_idx].copy())

        # 应用擦除笔（将圆形区域内的掩码设为0）
        y_min = max(0, iy - r)
        y_max = min(h, iy + r + 1)
        x_min = max(0, ix - r)
        x_max = min(w, ix + r + 1)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        mask_circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r ** 2
        masks[slice_idx][y_min:y_max, x_min:x_max][mask_circle] = 0

        return masks


