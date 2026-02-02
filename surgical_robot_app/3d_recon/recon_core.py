import numpy as np
from typing import Optional, Tuple, Callable
try:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkMarchingCubes
except Exception:
    numpy_to_vtk = None
    vtkImageData = None
    vtkPolyData = None
    vtkMarchingCubes = None


def reconstruct_3d(
    volume: np.ndarray,
    spacing: Optional[Tuple[float, float, float]] = None,
    threshold: int = 128,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[object], Optional[object]]:
    """
    3D 重建核心逻辑：从体数据生成等值面（Marching Cubes）

    Args:
        volume: 3D 体数据，形状 (Z, H, W) 或 (Z, H, W, C)
        spacing: 体素间距 (sx, sy, sz)，可选
        threshold: 等值面阈值（与原来 on_reconstruct_3d 中 128 一致）
        progress_cb: 可选进度回调，接收 0-100 的整数

    Returns:
        poly: VTK PolyData 等值面（若失败则为 None）
        image: VTK ImageData 体数据（若失败则为 None）
    """
    if numpy_to_vtk is None or vtkImageData is None or vtkMarchingCubes is None:
        return None, None
    if volume.ndim == 4 and volume.shape[-1] == 3:
        volume = volume[..., 0]
    z, h, w = volume.shape[0], volume.shape[1], volume.shape[2]
    image = vtkImageData()
    image.SetDimensions(w, h, z)
    image.SetExtent(0, w - 1, 0, h - 1, 0, z - 1)
    if spacing is not None:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    arr = numpy_to_vtk(volume.ravel(order="C"), deep=True)
    image.GetPointData().SetScalars(arr)

    mc = vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, float(threshold))

    # 如果提供了进度回调，则连接 VTK 的进度事件
    if progress_cb is not None:
        def _on_progress(caller, event):
            try:
                p = int(caller.GetProgress() * 100)
                progress_cb(p)
            except Exception:
                pass

        # 使用字符串事件名以兼容性更好
        mc.AddObserver("ProgressEvent", _on_progress)

    mc.Update()
    poly = mc.GetOutput()
    return poly, image