import numpy as np
import cv2
import os
import pydicom
import vtk
import matplotlib.pyplot as plt
from glob import glob
from skimage import measure
from datetime import datetime
from typing import Tuple, Dict, List
from tqdm import tqdm

def load_medical_images(image_dir: str, image_format: str = "dicom") -> Tuple[np.ndarray, Dict]:
    """
    加载并预处理医学影像数据（DICOM/PNG）
    :param image_dir: 影像文件所在文件夹路径
    :param image_format: 影像格式（"dicom"或"png"）
    :return: 处理后的影像数据（三维数组：[层数, 高度, 宽度]）、患者元数据（字典）
    """
    metadata = {}
    slices = []
    
    if image_format.lower() == "dicom":
        # 读取DICOM文件
        dicom_files = sorted(glob(os.path.join(image_dir, "*.dcm")))
        if not dicom_files:
            raise ValueError(f"在目录 {image_dir} 中未找到DICOM文件")
        
        # 提取元数据和图像数据
        for file in dicom_files:
            ds = pydicom.dcmread(file)
            img = ds.pixel_array
            
            # 转换为8位图像以便处理
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            slices.append(img)
            
            # 提取元数据
            if not metadata:
                metadata['PatientID'] = ds.PatientID if hasattr(ds, 'PatientID') else "Unknown"
                metadata['StudyDate'] = ds.StudyDate if hasattr(ds, 'StudyDate') else "Unknown"
                metadata['Modality'] = ds.Modality if hasattr(ds, 'Modality') else "Unknown"
                metadata['PixelSpacing'] = ds.PixelSpacing if hasattr(ds, 'PixelSpacing') else [1.0, 1.0]
                metadata['SliceThickness'] = ds.SliceThickness if hasattr(ds, 'SliceThickness') else 1.0
                metadata['Spacing'] = (
                    float(metadata['PixelSpacing'][0]),
                    float(metadata['PixelSpacing'][1]),
                    float(metadata['SliceThickness'])
                )
    
    elif image_format.lower() == "png":
        # 加载PNG文件
        slice_files = sorted(glob(os.path.join(image_dir, "*.png")))
        print("切片加载顺序:", [os.path.basename(f) for f in slice_files[:5]])
        
        if not slice_files:
            raise ValueError(f"在目录 {image_dir} 中未找到PNG文件")
        
        for file in slice_files:
            # 使用彩色模式加载PNG文件，以便能够检测红色区域
            slice_img = cv2.imread(file, cv2.IMREAD_COLOR)
            if slice_img is None:
                print(f"跳过无法读取的文件: {file}")
                continue
            # 检查是否成功加载为彩色图像
            if len(slice_img.shape) == 3:
                print(f"成功加载彩色图像: {os.path.basename(file)}, 形状: {slice_img.shape}")
            else:
                print(f"警告: {os.path.basename(file)} 不是彩色图像")
            slices.append(slice_img)
        
        # 设置元数据
        metadata['ImageFormat'] = 'PNG'
        metadata['SliceCount'] = len(slices)
        metadata['Spacing'] = (0.68359, 0.68359, 1.0)  # 默认体素间距
    
    else:
        raise ValueError(f"不支持的影像格式: {image_format}")
    
    # 堆叠为三维数组 (z, y, x)
    volume = np.stack(slices, axis=0)
    print(f"成功加载 {len(slices)} 张切片，体数据形状: {volume.shape}")
    
    return volume, metadata


def detect_red_regions(images: np.ndarray, visualize: bool = False) -> np.ndarray:
    """
    专门识别图像中的红色区域
    :param images: 输入影像数据（三维数组：[层数, 高度, 宽度, 通道数]）
    :param visualize: 是否可视化识别结果
    :return: 二值化的红色区域掩码（1表示红色区域）
    """
    # 初始化结果掩码
    masks = np.zeros((images.shape[0], images.shape[1], images.shape[2]), dtype=np.int32)
    
    # 对每层图像进行红色区域识别
    for i in tqdm(range(images.shape[0]), desc="红色区域检测"):
        img = images[i]
        
        # 确保图像是BGR格式
        if len(img.shape) == 2:  # 灰度图无法识别红色
            print(f"警告：第{i}层是灰度图，无法识别红色区域")
            continue
        
        # 转换到HSV颜色空间，更容易识别特定颜色
        # OpenCV读取的图像是BGR格式，需要转换为HSV
        if len(img.shape) == 3:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            print(f"警告：图像不是彩色格式，无法转换为HSV")
            continue
        
        # 定义红色的HSV范围（红色在HSV中有两个范围）
        # 使用更宽松的阈值来捕获更多红色区域
        lower_red1 = np.array([0, 30, 30])  # 降低饱和度和亮度阈值
        upper_red1 = np.array([30, 255, 255])  # 扩大色相范围
        lower_red2 = np.array([140, 30, 30])  # 降低饱和度和亮度阈值
        upper_red2 = np.array([180, 255, 255])
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # 应用形态学操作改善掩码质量
        kernel = np.ones((5, 5), np.uint8)
        # 先进行多次闭运算以填充较大的空洞
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  # 再开运算去除噪点
        
        # 找到连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
        
        # 按面积排序连通区域
        sizes = stats[1:, cv2.CC_STAT_AREA]
        if len(sizes) > 0:  # 确保有连通区域
            sorted_indices = np.argsort(sizes)[::-1]  # 降序排列
            
            # 创建新掩码，保留最大的两个连通区域
            new_mask = np.zeros_like(red_mask)
            
            # 保留最大的连通区域
            largest_label = sorted_indices[0] + 1  # +1 因为背景是0
            new_mask[labels == largest_label] = 255
            
            # 如果有第二大的连通区域，且其面积超过最大区域的10%，也保留它
            if len(sorted_indices) > 1:
                second_largest_area = sizes[sorted_indices[1]]
                largest_area = sizes[sorted_indices[0]]
                if second_largest_area > largest_area * 0.1:
                    second_largest_label = sorted_indices[1] + 1
                    new_mask[labels == second_largest_label] = 255
            
            red_mask = new_mask
        
        # 统计红色像素数量
        red_pixel_count = np.sum(red_mask > 0)
        print(f"切片 {i}: 检测到 {red_pixel_count} 个红色像素")
        
        # 保存掩码
        masks[i] = (red_mask > 0).astype(np.int32)
        
        # 可视化识别结果
        if visualize and i % 10 == 0:
            visualize_red_detection(img, red_mask, i)
    
    return masks

# 可视化红色区域检测结果
def visualize_red_detection(image, mask, slice_index):
    """
    可视化红色区域检测结果
    :param image: 原始图像
    :param mask: 红色区域掩码
    :param slice_index: 切片索引
    """
    # 创建结果目录
    result_dir = "red_detection_results"
    os.makedirs(result_dir, exist_ok=True)
    
    # 创建可视化图像
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = [0, 0, 255]  # 用蓝色标记红色区域（BGR格式）
    
    # 叠加原图和掩码
    alpha = 0.5
    overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)
    
    # 保存结果
    cv2.imwrite(f"{result_dir}/slice_{slice_index:03d}_detection.png", overlay)
    print(f"已保存切片 {slice_index} 的红色区域检测结果")

def calculate_structure_volume(segment_mask: np.ndarray, voxel_spacing: tuple) -> dict:
    """计算各解剖结构的体积"""
    # 计算体素体积 (mm³)
    voxel_volume = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    
    # 统计目标结构的体素数量
    structure_voxels = np.sum(segment_mask == 1)
    
    # 转换为cm³
    volume_cm3 = (structure_voxels * voxel_volume) / 1000
    
    return {"target_structure": round(volume_cm3, 2)}

def preprocess_images(images: np.ndarray) -> np.ndarray:
    """
    影像预处理（降噪、增强、层厚校准）
    :param images: 原始影像数据（三维数组）
    :return: 预处理后的影像数据
    """
    # 复制数据避免修改原始数据
    processed = np.copy(images)
    num_slices = processed.shape[0]
    
    # 1. 中值滤波降噪
    for i in range(num_slices):
        processed[i] = cv2.medianBlur(processed[i], ksize=3)
    
    # 2. 对比度增强（CLAHE）
    processed_normalized = ((processed - processed.min()) / 
                           (processed.max() - processed.min() + 1e-8) * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(num_slices):
        processed_normalized[i] = clahe.apply(processed_normalized[i])
    
    return processed_normalized

def reconstruct_3d_model(images: np.ndarray, segment_mask: np.ndarray, voxel_spacing: tuple) -> dict:
    """
    基于影像和分割掩码重建3D模型
    :param images: 预处理后的影像数据
    :param segment_mask: 优化后的分割掩码
    :param voxel_spacing: 体素间距（x, y, z，单位：mm）
    :return: 3D模型字典
    """
    # 提取目标结构的二值体数据
    target_volume = (segment_mask == 1).astype(bool)
    
    # 使用VTK进行三维重建
    class TempReconstructor:
        def __init__(self, volume, spacing):
            self.volume = volume
            self.spacing = spacing
            self.output_dir = "ct_3d_reconstruction"
            os.makedirs(self.output_dir, exist_ok=True)
        
        def create_mesh_from_volume(self, level=0.5):
            # 使用skimage生成等值面
            vertices, triangles = self._marching_cubes(level)
            
            # 转换为VTK网格
            mesh = self._numpy_to_vtk_mesh(vertices, triangles)
            
            # 应用体素间距缩放
            x_spacing, y_spacing, z_spacing = self.spacing
            transform = vtk.vtkTransform()
            transform.Scale(x_spacing, y_spacing, z_spacing)
            
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputData(mesh)
            transform_filter.SetTransform(transform)
            transform_filter.Update()
            
            # 先进行平滑处理
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputData(transform_filter.GetOutput())
            smoother.SetNumberOfIterations(15)
            smoother.SetPassBand(0.1)
            smoother.SetBoundarySmoothing(True)
            smoother.Update()
            
            # 填充孔洞
            fill_holes = vtk.vtkFillHolesFilter()
            fill_holes.SetInputData(smoother.GetOutput())
            fill_holes.SetHoleSize(20.0)  # 使用较小的值，只填充小孔洞
            fill_holes.Update()
            
            return fill_holes.GetOutput()
        
        def _marching_cubes(self, level):
            # 在进行marching cubes前，对体数据进行预处理
            volume_padded = np.pad(self.volume, ((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=False)
            
            # 使用更小的步长和更低的阈值，确保捕获更多细节
            vertices, faces, _, _ = measure.marching_cubes(
                volume_padded.astype(np.float32),
                level=level * 0.9,  # 降低阈值以包含更多区域
                spacing=self.spacing,
                step_size=1,  # 使用更小的步长获取更多细节
                allow_degenerate=True  # 允许退化三角形以确保完整性
            )
            return vertices, faces
        
        def _numpy_to_vtk_mesh(self, vertices, triangles):
            # 创建点
            vtk_points = vtk.vtkPoints()
            for vertex in vertices:
                vtk_points.InsertNextPoint(vertex)
            
            # 创建三角形
            vtk_triangles = vtk.vtkCellArray()
            for triangle in triangles:
                vtk_triangle = vtk.vtkTriangle()
                vtk_triangle.GetPointIds().SetId(0, triangle[0])
                vtk_triangle.GetPointIds().SetId(1, triangle[1])
                vtk_triangle.GetPointIds().SetId(2, triangle[2])
                vtk_triangles.InsertNextCell(vtk_triangle)
            
            # 创建网格
            mesh = vtk.vtkPolyData()
            mesh.SetPoints(vtk_points)
            mesh.SetPolys(vtk_triangles)
            
            return mesh
    
    # 处理可能的阈值问题
    min_val = target_volume.min()
    max_val = target_volume.max()
    level = 0.5 if (min_val < 0.5 < max_val) else (min_val + max_val) / 2
    
    # 生成网格模型
    reconstructor = TempReconstructor(target_volume, voxel_spacing)
    mesh = reconstructor.create_mesh_from_volume(level=level)
    
    # 设置单一颜色（白色）
    mesh_property = vtk.vtkProperty()
    mesh_property.SetColor(1.0, 1.0, 1.0)  # 白色
    mesh_property.SetOpacity(1.0)
    
    # 保存带颜色的模型
    mesh_path = os.path.join(reconstructor.output_dir, "ct_mesh.obj")
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(mesh_path)
    writer.SetInputData(mesh)
    writer.Write()
    print(f"带颜色的三维网格已保存至: {mesh_path}")
    
    # 可视化
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetProperty(mesh_property)
    
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName("3D Reconstruction (Single Color)")
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)  # 黑色背景
    
    render_window.Render()
    interactor.Initialize()
    interactor.Start()
    
    return {
        "target_structure": {
            "mesh": mesh,
            "color": (1.0, 1.0, 1.0),  # 与设置的颜色保持一致
            "mesh_path": mesh_path
        }
    }

# 使用红色区域检测的主函数
def main():
    # 1. 加载影像数据（使用彩色模式）
    image_dir = "CTimage/1504425_output/major_overlay_outputs"
    volume, metadata = load_medical_images(image_dir, image_format="png")
    
    # 2. 使用红色区域检测
    red_masks = detect_red_regions(volume, visualize=True)
    
    # 检查是否检测到足够的红色区域
    if np.sum(red_masks) < 100:
        print("警告：检测到的红色区域太少，无法进行3D重建")
        print("保存掩码数据以供后续分析...")
        np.save("detected_masks.npy", red_masks)
        return
    
    # 3. 后续处理（3D重建等）
    spacing = metadata['Spacing']
    try:
        reconstruct_3d_model(volume, red_masks, spacing)
        volume_stats = calculate_structure_volume(red_masks, spacing)
        print(f"目标区域体积统计: {volume_stats}")
    except RuntimeError as e:
        print(f"3D重建失败: {e}")
        print("保存掩码数据以供后续分析...")
        np.save("detected_masks.npy", red_masks)

if __name__ == "__main__":
    main()
