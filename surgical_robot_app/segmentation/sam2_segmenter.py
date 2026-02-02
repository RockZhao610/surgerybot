"""
SAM2分割器模块
提供基于SAM2的交互式图像分割功能
"""
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# 初始化日志系统
try:
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.segmentation.sam2_segmenter")

try:
    import torch
    import torchvision
    SAM2_AVAILABLE = True
    # 如果检测到 MPS 设备，启用 CPU 回退以支持所有操作
    # MPS 设备不支持某些操作（如 upsample_bicubic2d），需要回退到 CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
except ImportError:
    SAM2_AVAILABLE = False
    logger.warning("PyTorch not available. SAM2 segmentation will be disabled.")


class SAM2Segmenter:
    """
    SAM2分割器类
    支持交互式提示分割（点击、框选）
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "hiera_large"):
        """
        初始化SAM2分割器
        
        Args:
            model_path: SAM2模型文件路径，如果为None则尝试自动查找
            model_type: 模型类型 ('hiera_tiny', 'hiera_small', 'hiera_base', 'hiera_large')
        """
        self.model = None
        self.predictor = None
        # 设备选择：优先 CUDA，其次 MPS (Mac M1/M2)，最后 CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"  # Mac M1/M2 芯片的 GPU 加速
            # MPS 设备不支持某些操作（如 upsample_bicubic2d），需要启用 CPU 回退
            # 环境变量已在模块级别设置，这里只是确认
            if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
                logger.info("MPS device detected. CPU fallback enabled for unsupported operations.")
        else:
            self.device = "cpu"
        
        # 根据模型文件名自动推断 model_type
        if model_path:
            model_name = os.path.basename(model_path).lower()
            if 'hiera_small' in model_name or ('small' in model_name and 'hiera' in model_name):
                self.model_type = "hiera_small"
            elif 'hiera_base' in model_name or ('base' in model_name and 'hiera' in model_name):
                self.model_type = "hiera_base"
            elif 'hiera_tiny' in model_name or ('tiny' in model_name and 'hiera' in model_name):
                self.model_type = "hiera_tiny"
            elif 'hiera_large' in model_name or ('large' in model_name and 'hiera' in model_name):
                self.model_type = "hiera_large"
            else:
                self.model_type = model_type  # 使用默认值
        else:
            self.model_type = model_type
        
        self.model_path = model_path
        self.is_loaded = False
        
        if not SAM2_AVAILABLE:
            raise ImportError("PyTorch is required for SAM2. Please install: pip install torch torchvision")
        
        # 尝试加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None):
        """
        加载SAM2模型 (支持 Image 和 Video 预测器)
        
        Args:
            model_path: 模型文件路径，如果为None则使用self.model_path
        """
        if not SAM2_AVAILABLE:
            raise ImportError("PyTorch is required for SAM2")
        
        try:
            # 尝试导入SAM2
            try:
                from sam2.build_sam import build_sam2, build_sam2_video_predictor
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                logger.warning("SAM2 not installed. Please install from: git+https://github.com/facebookresearch/sam2.git")
                self.is_loaded = False
                return
            
            model_path = model_path or self.model_path
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                self.is_loaded = False
                return
            
            config_name = self._get_config_name()
            if not config_name:
                self.is_loaded = False
                return
            
            # 加载 Image Predictor (用于单层交互)
            sam2_model = build_sam2(config_name, model_path, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            
            # 加载 Video Predictor (用于跨层追踪)
            # 注意：Video Predictor 需要相同的 config 和 checkpoint
            self.video_predictor = build_sam2_video_predictor(config_name, model_path, device=self.device)
            
            self.is_loaded = True
            device_name = "Mac M1/M2 GPU (MPS)" if self.device == "mps" else self.device.upper()
            logger.info(f"[SAM2] Predictors (Image & Video) loaded successfully on {device_name}")
            
        except Exception as e:
            logger.error(f"Error loading SAM2 model: {e}", exc_info=True)
            self.is_loaded = False

    def init_video_state(self, frames_path_or_list):
        """初始化视频状态"""
        if not self.is_loaded or self.video_predictor is None:
            raise RuntimeError("SAM2 Video Predictor not loaded")
        return self.video_predictor.init_state(video_path=frames_path_or_list)

    def add_new_points_to_video(self, inference_state, frame_idx, obj_id, points, labels):
        """在视频特定帧添加提示点"""
        return self.video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )

    def propagate_video(self, inference_state, start_frame_idx=None, reverse=False):
        """
        在整个序列中传播分割
        Args:
            inference_state: 视频状态
            start_frame_idx: 起始帧索引（如果为 None，则从第一个有提示的帧开始）
            reverse: 是否反向传播（向前追踪）
        """
        return self.video_predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=start_frame_idx, 
            reverse=reverse
        )
    
    def _get_config_name(self) -> Optional[str]:
        """
        获取SAM2配置名称（Hydra格式）
        SAM2 使用 Hydra 配置系统，配置路径相对于 pkg://sam2
        格式应该是 "configs/sam2.1/sam2.1_hiera_s.yaml"
        """
        # 根据model_type选择配置名称（Hydra格式，相对于 pkg://sam2）
        config_map = {
            "hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "hiera_base": "configs/sam2.1/sam2.1_hiera_b+.yaml",  # 注意是 b+ 不是 b
            "hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        }
        
        config_name = config_map.get(self.model_type, config_map["hiera_large"])
        
        # 验证配置文件是否存在（可选，用于调试）
        try:
            import sam2
            if hasattr(sam2, '__file__'):
                sam2_path = os.path.dirname(sam2.__file__)
                # 提取配置文件名（去掉 sam2.1/ 前缀）
                config_filename = os.path.basename(config_name)
                config_path = os.path.join(sam2_path, 'configs', 'sam2.1', config_filename)
                if os.path.exists(config_path):
                    logger.debug(f"[SAM2] Config file exists at: {config_path}")
                else:
                    logger.warning(f"[SAM2] Config file not found at: {config_path}")
                    logger.info(f"[SAM2] Using Hydra config name: {config_name}")
        except Exception as e:
            logger.warning(f"[SAM2] Could not verify config file: {e}")
            logger.info(f"[SAM2] Using Hydra config name: {config_name}")
        
        return config_name
    
    def segment_with_mixed_prompts(
        self, 
        image: np.ndarray, 
        point_coords: Optional[np.ndarray] = None, 
        point_labels: Optional[np.ndarray] = None, 
        box: Optional[Tuple[int, int, int, int]] = None,
        multimask: bool = False # 新增参数
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        核心分割接口：支持框点结合的交互式分割 (B 方案)
        Args:
            multimask: 如果为 True，返回 3 个候选掩码列表；否则返回最佳单掩码
        """
        if not self.is_loaded:
            raise RuntimeError("SAM2 model not loaded.")
        
        try:
            # 1. 图像格式转换 (RGB uint8)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # 2. 设置图像上下文
            self.predictor.set_image(image)
            
            # 3. 准备提示数据
            input_box = np.array(box, dtype=np.float32)[None, :] if box is not None else None
            
            # 4. 执行预测
            masks, scores, logits = self.predictor.predict(
                point_coords=np.array(point_coords, dtype=np.float32) if point_coords is not None else None,
                point_labels=np.array(point_labels, dtype=np.int32) if point_labels is not None else None,
                box=input_box,
                multimask_output=multimask, # 这里使用参数控制
            )
            
            # 5. 返回结果
            if multimask:
                return [(m * 255).astype(np.uint8) for m in masks]
            else:
                return (masks[0] * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in SAM2 mixed prompt segmentation: {e}", exc_info=True)
            return np.zeros(image.shape[:2], dtype=np.uint8) if not multimask else []

    def segment_with_point_prompt(self, image, point_coords, point_labels):
        """向前兼容接口"""
        return self.segment_with_mixed_prompts(image, point_coords=point_coords, point_labels=point_labels)

    def segment_with_box_prompt(self, image, box):
        """向前兼容接口"""
        return self.segment_with_mixed_prompts(image, box=box)
    
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.is_loaded


def create_sam2_segmenter(model_path: Optional[str] = None, model_type: str = "hiera_large") -> Optional[SAM2Segmenter]:
    """
    创建SAM2分割器实例的便捷函数
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型
    
    Returns:
        SAM2Segmenter实例，如果失败则返回None
    """
    try:
        segmenter = SAM2Segmenter(model_path=model_path, model_type=model_type)
        if model_path:
            segmenter.load_model(model_path)
        return segmenter
    except Exception as e:
        logger.error(f"Failed to create SAM2 segmenter: {e}", exc_info=True)
        return None

