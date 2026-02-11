"""
DataManager: 统一管理应用的数据状态

职责：
- 管理 volume、masks、metadata 等数据
- 提供数据访问和修改接口
- 支持多标签分割（每个标签独立存储）
- 不依赖 GUI，便于复用和测试
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np


# 预定义标签颜色 (RGB, 0-255)
DEFAULT_LABEL_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (255, 0, 0),      # 红色
    2: (0, 100, 255),     # 蓝色
    3: (0, 200, 0),       # 绿色
    4: (255, 165, 0),     # 橙色
    5: (180, 0, 255),     # 紫色
    6: (0, 200, 200),     # 青色
    7: (255, 255, 0),     # 黄色
    8: (255, 105, 180),   # 粉色
}


class DataManager:
    """统一数据管理器"""
    
    def __init__(self):
        # 数据加载相关
        self.loaded_volume: Optional[np.ndarray] = None
        self.loaded_metadata: Optional[Dict[str, Any]] = None
        self.loaded_paths: List[str] = []
        self.selected_dir: str = ""
        self.current_format: str = "png"
        
        # 分割相关 — 多标签 mask
        # masks[i] 存储第 i 个切片的多标签 mask (dtype=uint8)
        # 像素值: 0=背景, 1=标签1, 2=标签2, ...
        self.masks: List[Optional[np.ndarray]] = []
        self.seg_mask_volume: Optional[np.ndarray] = None
        
        # 多标签管理
        self.current_label: int = 1  # 当前活跃标签 ID
        self.label_names: Dict[int, str] = {1: "Label 1"}  # 标签名称
        self.label_colors: Dict[int, Tuple[int, int, int]] = {
            1: DEFAULT_LABEL_COLORS[1]
        }
        self._next_label_id: int = 2  # 下一个可用标签 ID
        
        # 阈值相关
        try:
            from surgical_robot_app.segmentation.hsv_threshold import DEFAULT_THRESHOLDS
        except ImportError:
            try:
                from segmentation.hsv_threshold import DEFAULT_THRESHOLDS
            except ImportError:
                DEFAULT_THRESHOLDS = {
                    "h1_low": 0, "h1_high": 10,
                    "h2_low": 160, "h2_high": 180,
                    "s_low": 50, "s_high": 255,
                    "v_low": 50, "v_high": 255,
                }
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        
        # 文件列表相关
        self.list_offset: int = 0
        self.exclusion_history: List[List[str]] = []
    
    # ========== Volume 相关方法 ==========
    
    def set_volume(self, volume: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """设置加载的 volume"""
        self.loaded_volume = volume
        self.loaded_metadata = metadata or {}
        # 初始化 masks
        if volume is not None:
            depth = volume.shape[0]
            self.masks = [None] * depth
    
    def get_volume(self) -> Optional[np.ndarray]:
        """获取当前 volume"""
        return self.loaded_volume
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """获取当前 volume 的元数据"""
        return self.loaded_metadata
    
    def get_volume_shape(self) -> Optional[tuple]:
        """获取 volume 的形状"""
        if self.loaded_volume is not None:
            return self.loaded_volume.shape
        return None
    
    def clear_volume(self):
        """清除 volume 数据"""
        self.loaded_volume = None
        self.loaded_metadata = None
        self.masks = []
        self.seg_mask_volume = None
    
    # ========== 多标签管理 ==========
    
    def add_label(self, name: Optional[str] = None) -> int:
        """
        添加新标签
        
        Args:
            name: 标签名称（可选，自动生成）
        
        Returns:
            新标签的 ID
        """
        label_id = self._next_label_id
        self._next_label_id += 1
        
        if name is None:
            name = f"Label {label_id}"
        self.label_names[label_id] = name
        
        # 分配颜色（循环使用预定义颜色）
        color_idx = label_id if label_id in DEFAULT_LABEL_COLORS else (label_id % len(DEFAULT_LABEL_COLORS)) + 1
        self.label_colors[label_id] = DEFAULT_LABEL_COLORS.get(color_idx, (128, 128, 128))
        
        return label_id
    
    def remove_label(self, label_id: int):
        """
        删除标签及其在所有切片中的分割数据
        
        Args:
            label_id: 要删除的标签 ID
        """
        if label_id not in self.label_names:
            return
        
        # 从所有 mask 中移除此标签的像素
        for i, mask in enumerate(self.masks):
            if mask is not None:
                mask[mask == label_id] = 0
                # 如果 mask 全空，设为 None
                if np.count_nonzero(mask) == 0:
                    self.masks[i] = None
        
        # 移除标签信息
        del self.label_names[label_id]
        self.label_colors.pop(label_id, None)
        
        # 如果删除的是当前标签，切换到第一个可用标签
        if self.current_label == label_id:
            if self.label_names:
                self.current_label = min(self.label_names.keys())
            else:
                # 没有标签了，创建一个默认的
                self.current_label = self.add_label("Label 1")
        
        self._update_seg_mask_volume()
    
    def set_current_label(self, label_id: int):
        """设置当前活跃标签"""
        if label_id in self.label_names:
            self.current_label = label_id
    
    def get_current_label(self) -> int:
        """获取当前活跃标签 ID"""
        return self.current_label
    
    def get_label_names(self) -> Dict[int, str]:
        """获取所有标签名称"""
        return self.label_names.copy()
    
    def get_label_color(self, label_id: int) -> Tuple[int, int, int]:
        """获取标签颜色 (RGB, 0-255)"""
        return self.label_colors.get(label_id, (128, 128, 128))
    
    def get_label_color_float(self, label_id: int) -> Tuple[float, float, float]:
        """获取标签颜色 (RGB, 0.0-1.0)，适用于 VTK"""
        r, g, b = self.get_label_color(label_id)
        return (r / 255.0, g / 255.0, b / 255.0)
    
    def rename_label(self, label_id: int, new_name: str):
        """重命名标签"""
        if label_id in self.label_names:
            self.label_names[label_id] = new_name
    
    def get_all_label_ids(self) -> List[int]:
        """获取所有标签 ID（排序）"""
        return sorted(self.label_names.keys())
    
    def get_labels_in_volume(self) -> List[int]:
        """获取 seg_mask_volume 中实际存在的标签 ID"""
        if self.seg_mask_volume is None:
            return []
        unique = np.unique(self.seg_mask_volume)
        return [int(v) for v in unique if v > 0]
    
    # ========== Masks 相关方法 ==========
    
    def get_mask(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的多标签 mask（像素值为标签 ID）"""
        if 0 <= index < len(self.masks):
            return self.masks[index]
        return None
    
    def set_mask(self, index: int, mask: Optional[np.ndarray]):
        """
        设置指定索引的 mask（多标签感知）
        
        如果传入的是二值 mask（0/255），会自动转换为当前标签 ID。
        只更新当前标签对应的区域，不影响其他标签的像素。
        
        Args:
            index: 切片索引
            mask: 二值 mask (0/255) 或多标签 mask (标签ID值)
        """
        if not (0 <= index < len(self.masks)):
            return
        
        if mask is None:
            # 清除当前标签在此切片的分割
            existing = self.masks[index]
            if existing is not None:
                existing[existing == self.current_label] = 0
                if np.count_nonzero(existing) == 0:
                    self.masks[index] = None
                else:
                    self.masks[index] = existing
        else:
            # 判断是否为二值 mask (SAM2 输出的 0/255)
            is_binary = False
            unique_vals = np.unique(mask)
            if len(unique_vals) <= 2 and (255 in unique_vals or 1 in unique_vals):
                # 二值 mask，需要转换为当前标签
                is_binary = True
            
            if is_binary:
                # 获取或创建现有的多标签 mask
                existing = self.masks[index]
                if existing is None:
                    h, w = mask.shape[:2]
                    existing = np.zeros((h, w), dtype=np.uint8)
                else:
                    existing = existing.copy()
                
                # 先清除当前标签的旧区域
                existing[existing == self.current_label] = 0
                # 写入当前标签的新区域
                foreground = mask > 128 if 255 in unique_vals else mask > 0
                existing[foreground] = self.current_label
                self.masks[index] = existing
            else:
                # 已经是多标签 mask，直接存储
                self.masks[index] = mask.astype(np.uint8)
        
        # 更新 seg_mask_volume
        self._update_seg_mask_volume()
    
    def get_binary_mask_for_label(self, index: int, label_id: int) -> Optional[np.ndarray]:
        """
        获取指定切片中某个标签的二值 mask (0/255)
        
        Args:
            index: 切片索引
            label_id: 标签 ID
        
        Returns:
            二值 mask (0/255) 或 None
        """
        mask = self.get_mask(index)
        if mask is None:
            return None
        binary = (mask == label_id).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return None
        return binary
    
    def get_masks(self) -> List[Optional[np.ndarray]]:
        """获取所有 masks（多标签）"""
        return self.masks
    
    def set_masks(self, masks: List[Optional[np.ndarray]]):
        """设置所有 masks"""
        self.masks = masks
        self._update_seg_mask_volume()
    
    def ensure_masks(self, shape: tuple) -> List[Optional[np.ndarray]]:
        """确保 masks 列表大小正确"""
        depth = shape[0]
        if len(self.masks) != depth:
            self.masks = [None] * depth
        return self.masks
    
    def clear_all_masks(self):
        """清除所有 masks"""
        if self.loaded_volume is not None:
            depth = self.loaded_volume.shape[0]
            self.masks = [None] * depth
        else:
            self.masks = []
        self.seg_mask_volume = None
    
    def clear_label_masks(self, label_id: int):
        """清除某个标签的所有分割数据"""
        for i, mask in enumerate(self.masks):
            if mask is not None:
                mask[mask == label_id] = 0
                if np.count_nonzero(mask) == 0:
                    self.masks[i] = None
        self._update_seg_mask_volume()
    
    def get_seg_mask_volume(self) -> Optional[np.ndarray]:
        """获取分割掩码体数据（多标签，像素值为标签 ID）"""
        return self.seg_mask_volume
    
    def get_binary_volume_for_label(self, label_id: int) -> Optional[np.ndarray]:
        """
        获取某个标签的二值体数据 (0/255)
        
        Args:
            label_id: 标签 ID
        
        Returns:
            二值体数据 (Z, H, W) 或 None
        """
        if self.seg_mask_volume is None:
            return None
        binary = (self.seg_mask_volume == label_id).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            return None
        return binary
    
    def set_seg_mask_volume(self, seg_mask_volume: Optional[np.ndarray]):
        """设置分割掩码体数据"""
        self.seg_mask_volume = seg_mask_volume
    
    def _update_seg_mask_volume(self):
        """更新 seg_mask_volume（多标签）"""
        if not self.masks or all(m is None for m in self.masks):
            self.seg_mask_volume = None
            return
        
        # 找到第一个非 None 的 mask 来确定形状
        first_mask = next((m for m in self.masks if m is not None), None)
        if first_mask is None:
            self.seg_mask_volume = None
            return
        
        # 创建 seg_mask_volume
        depth = len(self.masks)
        h, w = first_mask.shape[:2]
        self.seg_mask_volume = np.zeros((depth, h, w), dtype=np.uint8)
        
        for i, mask in enumerate(self.masks):
            if mask is not None:
                self.seg_mask_volume[i] = mask
    
    # ========== Paths 相关方法 ==========
    
    def set_paths(self, paths: List[str]):
        """设置文件路径列表"""
        self.loaded_paths = paths
    
    def get_paths(self) -> List[str]:
        """获取文件路径列表"""
        return self.loaded_paths
    
    def exclude_paths(self, paths_to_exclude: List[str]):
        """排除指定的路径"""
        remaining = [p for p in self.loaded_paths if p not in paths_to_exclude]
        self.exclusion_history.append(list(self.loaded_paths))
        self.loaded_paths = remaining
    
    def undo_exclusion(self) -> Optional[List[str]]:
        """撤销上一次排除操作"""
        if not self.exclusion_history:
            return None
        prev_paths = self.exclusion_history.pop()
        self.loaded_paths = prev_paths
        return prev_paths
    
    # ========== Thresholds 相关方法 ==========
    
    def get_threshold(self, name: str) -> int:
        """获取阈值"""
        return self.thresholds.get(name, 0)
    
    def set_threshold(self, name: str, value: int):
        """设置阈值"""
        self.thresholds[name] = value
    
    def reset_thresholds(self):
        """重置阈值到默认值"""
        try:
            from surgical_robot_app.segmentation.hsv_threshold import DEFAULT_THRESHOLDS
        except ImportError:
            try:
                from segmentation.hsv_threshold import DEFAULT_THRESHOLDS
            except ImportError:
                DEFAULT_THRESHOLDS = {
                    "h1_low": 0, "h1_high": 10,
                    "h2_low": 160, "h2_high": 180,
                    "s_low": 50, "s_high": 255,
                    "v_low": 50, "v_high": 255,
                }
        self.thresholds = DEFAULT_THRESHOLDS.copy()

