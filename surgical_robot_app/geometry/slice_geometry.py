"""
3D 体数据切片与 2D/3D 坐标几何工具

职责：
- 从 3D volume 中抽取不同平面（冠状、矢状、横断）上的 2D 切片
- 将 2D 切片上的归一化点击坐标转换为 3D 空间坐标 ([0, 100] 范围)
- 将来自两个不同切片平面的坐标合成为完整的 3D 坐标

注意：
- 本模块不依赖 Qt，仅做纯数学/数组运算，便于测试与复用。
"""

from typing import Tuple, Optional

import numpy as np


def extract_slice(
    volume: np.ndarray,
    plane_type: str,
    slice_pos_normalized: float,
) -> np.ndarray:
    """
    从 3D volume 中提取指定平面的一张 2D 切片。

    Args:
        volume: 体数据，形状 (Z, H, W) 或 (Z, H, W, C)
        plane_type: 'coronal' / 'sagittal' / 'axial'
        slice_pos_normalized: 切片位置，0-100

    Returns:
        2D 或 3D numpy 数组（单通道或多通道），不做类型转换。
    """
    depth, height, width = volume.shape[0], volume.shape[1], volume.shape[2]

    if plane_type == "coronal":
        # 冠状面：沿 Y 轴切分，显示 X-Z 维度
        y_index = int((slice_pos_normalized / 100.0) * height)
        y_index = max(0, min(y_index, height - 1))
        slice_img = volume[:, y_index, :]  # (depth, width) -> (z, x)
    elif plane_type == "sagittal":
        # 矢状面：沿 X 轴切分，显示 Y-Z 维度
        x_index = int((slice_pos_normalized / 100.0) * width)
        x_index = max(0, min(x_index, width - 1))
        slice_img = volume[:, :, x_index]  # (depth, height) -> (z, y)
    else:
        # 横断面：沿 Z 轴切分，显示 X-Y 维度
        z_index = int((slice_pos_normalized / 100.0) * depth)
        z_index = max(0, min(z_index, depth - 1))
        slice_img = volume[z_index, :, :]  # (height, width) -> (y, x)

    return slice_img


def plane_click_to_space_coord(
    plane_type: str,
    slice_pos_normalized: float,
    img_x_norm: float,
    img_y_norm: float,
) -> Tuple[Tuple[float, float], Tuple[float, float, float]]:
    """
    将 2D 切片上的点击（归一化到 0-1）转换为 3D 空间坐标 ([0, 100])。

    Args:
        plane_type: 'coronal' / 'sagittal' / 'axial'
        slice_pos_normalized: 当前切片位置（0-100）
        img_x_norm: 在当前 2D 图像中的 X 归一化坐标（0-1）
        img_y_norm: 在当前 2D 图像中的 Y 归一化坐标（0-1）

    Returns:
        (click_2d, (x, y, z))：
            click_2d: 在当前平面中的 2D 坐标（同原代码中的 click_2d）
            (x, y, z): 归一化的 3D 空间坐标
    """
    if plane_type == "coronal":
        # 冠状面：显示 X-Z 维度，Y 由切片位置确定
        x_coord = img_x_norm * 100.0
        z_coord = img_y_norm * 100.0
        y_coord = slice_pos_normalized
        click_2d = (x_coord, z_coord)
    elif plane_type == "sagittal":
        # 矢状面：显示 Y-Z 维度，X 由切片位置确定
        y_coord = img_x_norm * 100.0
        z_coord = img_y_norm * 100.0
        x_coord = slice_pos_normalized
        click_2d = (y_coord, z_coord)
    else:  # axial
        # 横断面：显示 X-Y 维度，Z 由切片位置确定
        x_coord = img_x_norm * 100.0
        y_coord = img_y_norm * 100.0
        z_coord = slice_pos_normalized
        click_2d = (x_coord, y_coord)

    return click_2d, (x_coord, y_coord, z_coord)


def merge_two_planes_to_3d(
    plane1_type: str,
    coords1: Tuple[float, float, float],
    plane2_type: str,
    coords2: Tuple[float, float, float],
) -> Optional[Tuple[float, float, float]]:
    """
    根据两个不同平面的坐标数据，计算完整的 3D 空间坐标。

    Args:
        plane1_type: 第一个平面类型
        coords1: 第一个平面的 (x, y, z)
        plane2_type: 第二个平面类型
        coords2: 第二个平面的 (x, y, z)

    Returns:
        完整的 3D 坐标 (x, y, z)，若无法组合则返回 None
    """
    x_final: Optional[float] = None
    y_final: Optional[float] = None
    z_final: Optional[float] = None

    if plane1_type == "coronal" and plane2_type == "sagittal":
        x_final = coords1[0]
        y_final = coords2[1]
        z_final = (coords1[2] + coords2[2]) / 2.0
    elif plane1_type == "coronal" and plane2_type == "axial":
        x_final = (coords1[0] + coords2[0]) / 2.0
        y_final = coords2[1]
        z_final = coords1[2]
    elif plane1_type == "sagittal" and plane2_type == "coronal":
        x_final = coords2[0]
        y_final = coords1[1]
        z_final = (coords1[2] + coords2[2]) / 2.0
    elif plane1_type == "sagittal" and plane2_type == "axial":
        x_final = coords2[0]
        y_final = (coords1[1] + coords2[1]) / 2.0
        z_final = coords1[2]
    elif plane1_type == "axial" and plane2_type == "coronal":
        x_final = (coords1[0] + coords2[0]) / 2.0
        y_final = coords1[1]
        z_final = coords2[2]
    elif plane1_type == "axial" and plane2_type == "sagittal":
        x_final = coords1[0]
        y_final = (coords1[1] + coords2[1]) / 2.0
        z_final = coords2[2]

    if x_final is None or y_final is None or z_final is None:
        return None
    return float(x_final), float(y_final), float(z_final)


