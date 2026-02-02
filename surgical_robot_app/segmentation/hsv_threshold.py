"""
HSV 阈值分割相关工具函数

职责：
- 提供单张切片的阈值分割函数
- 提供对整个 volume 进行批量阈值分割的函数

注意：
- 这里不依赖任何 Qt / VTK 组件，只做纯粹的图像处理，方便在 GUI 之外复用或单元测试。
"""

from typing import Dict, Tuple, List, Optional, Callable

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # 在无 OpenCV 环境下优雅降级
    cv2 = None  # type: ignore


# 默认 HSV 阈值配置，与 MainWindow 中原来的默认值保持一致
DEFAULT_THRESHOLDS: Dict[str, int] = {
    "h1_low": 0,
    "h1_high": 10,
    "h2_low": 160,
    "h2_high": 180,
    "s_low": 50,
    "s_high": 255,
    "v_low": 50,
    "v_high": 255,
}


def compute_threshold_mask(slice_img: np.ndarray, thresholds: Dict[str, int]) -> Optional[np.ndarray]:
    """
    对单张切片进行 HSV 阈值分割，返回二值掩码（0/255）

    Args:
        slice_img: 输入切片，形状 (H, W) 或 (H, W, C)
        thresholds: 阈值配置字典，键为 h1_low/h1_high/... 等

    Returns:
        二值掩码 (H, W)，uint8，前景为255；若 OpenCV 不可用则返回 None
    """
    if cv2 is None:
        return None

    if slice_img.ndim == 2:
        img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
    else:
        img = slice_img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h1_low = thresholds.get("h1_low", 0)
    h1_high = thresholds.get("h1_high", 10)
    h2_low = thresholds.get("h2_low", 160)
    h2_high = thresholds.get("h2_high", 180)
    s_low = thresholds.get("s_low", 50)
    s_high = thresholds.get("s_high", 255)
    v_low = thresholds.get("v_low", 50)
    v_high = thresholds.get("v_high", 255)

    lower1 = np.array([h1_low, s_low, v_low], dtype=np.uint8)
    upper1 = np.array([h1_high, s_high, v_high], dtype=np.uint8)
    lower2 = np.array([h2_low, s_low, v_low], dtype=np.uint8)
    upper2 = np.array([h2_high, s_high, v_high], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    m = cv2.bitwise_or(m1, m2)
    return m


def apply_threshold_all(
    volume: np.ndarray, 
    thresholds: Dict[str, int], 
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    对整个 volume 逐层应用 HSV 阈值分割，并做简单后处理（闭运算+开运算+最大连通域）

    Args:
        volume: 3D 体数据，形状 (Z, H, W) 或 (Z, H, W, C)
        thresholds: HSV 阈值配置
        progress_callback: 进度回调函数

    Returns:
        masks_list: 每一层的 2D 掩码列表
        seg_volume: 堆叠后的 3D 掩码 (Z, H, W)；若没有有效掩码则为 None
    """
    masks: List[np.ndarray] = []

    if volume is None:
        return masks, None

    depth = volume.shape[0]

    for i in range(depth):
        slice_img = volume[i]
        m = compute_threshold_mask(slice_img, thresholds)
        if m is None:
            # 如果 OpenCV 不可用，填充全 0 掩码
            masks.append(np.zeros(volume.shape[1:3], dtype=np.uint8))
            continue

        if cv2 is not None:
            kernel = np.ones((3, 3), np.uint8)
            # 闭运算填孔，开运算去噪
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

            # 只保留最大连通域，避免多块小噪声
            num, labels, stats, _ = cv2.connectedComponentsWithStats(m)
            if num > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                idx = 1 + int(np.argmax(areas))
                m = np.where(labels == idx, 255, 0).astype(np.uint8)

        masks.append(m)
        
        # 更新进度
        if progress_callback:
            progress_callback(int((i + 1) / depth * 100))

    seg_volume = np.stack(masks, axis=0) if masks else None
    return masks, seg_volume


