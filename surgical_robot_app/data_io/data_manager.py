"""
DataManager: 统一管理应用的数据状态

职责：
- 管理 volume、masks、metadata 等数据
- 提供数据访问和修改接口
- 不依赖 GUI，便于复用和测试
"""

from typing import Optional, List, Dict, Any
import numpy as np


class DataManager:
    """统一数据管理器"""
    
    def __init__(self):
        # 数据加载相关
        self.loaded_volume: Optional[np.ndarray] = None
        self.loaded_metadata: Optional[Dict[str, Any]] = None
        self.loaded_paths: List[str] = []
        self.selected_dir: str = ""
        self.current_format: str = "png"
        
        # 分割相关
        self.masks: List[Optional[np.ndarray]] = []
        self.seg_mask_volume: Optional[np.ndarray] = None
        
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
    
    # ========== Masks 相关方法 ==========
    
    def get_mask(self, index: int) -> Optional[np.ndarray]:
        """获取指定索引的 mask"""
        if 0 <= index < len(self.masks):
            return self.masks[index]
        return None
    
    def set_mask(self, index: int, mask: Optional[np.ndarray]):
        """设置指定索引的 mask"""
        if 0 <= index < len(self.masks):
            self.masks[index] = mask
            # 更新 seg_mask_volume
            self._update_seg_mask_volume()
    
    def get_masks(self) -> List[Optional[np.ndarray]]:
        """获取所有 masks"""
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
    
    def get_seg_mask_volume(self) -> Optional[np.ndarray]:
        """获取分割掩码体数据"""
        return self.seg_mask_volume
    
    def set_seg_mask_volume(self, seg_mask_volume: Optional[np.ndarray]):
        """设置分割掩码体数据"""
        self.seg_mask_volume = seg_mask_volume
    
    def _update_seg_mask_volume(self):
        """更新 seg_mask_volume"""
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

