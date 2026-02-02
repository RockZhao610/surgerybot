"""
统一的导入工具模块

用于处理各种导入场景，减少重复的 try-except 导入块。
"""

import importlib
import logging
from typing import Any, Optional, Callable, List, Tuple, Dict, Union

logger = logging.getLogger(__name__)


def safe_import(
    module_paths: Union[str, List[str]],
    item_names: Optional[Union[str, List[str]]] = None,
    default: Any = None,
    fallback_func: Optional[Callable] = None,
    log_success: bool = False,
    log_failure: bool = False,
) -> Any:
    """
    统一的安全导入函数
    
    Args:
        module_paths: 尝试导入的模块路径（字符串或列表）
        item_names: 要从模块中导入的项目名称（字符串或列表），如果为 None 则导入整个模块
        default: 如果所有导入都失败，返回的默认值
        fallback_func: 如果所有导入都失败，调用的回退函数（应返回默认值或实现）
        log_success: 是否记录成功导入的日志
        log_failure: 是否记录失败导入的日志
    
    Returns:
        导入的模块、项目或默认值
    
    Examples:
        # 导入整个模块
        SequenceReader = safe_import(
            ['surgical_robot_app.data_io.sequence_reader', 'data_io.sequence_reader'],
            default=None
        )
        
        # 导入模块中的项目
        AStarPlanner, space_to_grid = safe_import(
            ['surgical_robot_app.path_planning.a_star_planner', 'path_planning.a_star_planner'],
            item_names=['AStarPlanner', 'space_to_grid'],
            default=(None, None)
        )
        
        # 导入多个项目
        items = safe_import(
            ['surgical_robot_app.segmentation.hsv_threshold', 'segmentation.hsv_threshold'],
            item_names=['DEFAULT_THRESHOLDS', 'compute_threshold_mask', 'apply_threshold_all'],
            default=(None, None, None)
        )
    """
    if isinstance(module_paths, str):
        module_paths = [module_paths]
    
    if item_names is None:
        # 导入整个模块
        for path in module_paths:
            try:
                module = importlib.import_module(path)
                if log_success:
                    logger.debug(f"Successfully imported module: {path}")
                return module
            except ImportError as e:
                if log_failure:
                    logger.debug(f"Failed to import module {path}: {e}")
                continue
    else:
        # 导入模块中的项目
        if isinstance(item_names, str):
            item_names = [item_names]
        
        for path in module_paths:
            try:
                module = importlib.import_module(path)
                items = []
                for name in item_names:
                    if hasattr(module, name):
                        items.append(getattr(module, name))
                    else:
                        raise AttributeError(f"Module {path} has no attribute {name}")
                
                if log_success:
                    logger.debug(f"Successfully imported {item_names} from {path}")
                
                # 如果只有一个项目，直接返回；否则返回元组
                return items[0] if len(items) == 1 else tuple(items)
            except (ImportError, AttributeError) as e:
                if log_failure:
                    logger.debug(f"Failed to import {item_names} from {path}: {e}")
                continue
    
    # 所有导入都失败
    if log_failure:
        logger.warning(f"All import attempts failed for {module_paths}")
    
    if fallback_func:
        return fallback_func()
    return default


def safe_import_with_aliases(
    module_paths: Union[str, List[str]],
    imports: Dict[str, Union[str, List[str]]],
    defaults: Optional[Dict[str, Any]] = None,
    fallback_funcs: Optional[Dict[str, Callable]] = None,
    log_success: bool = False,
    log_failure: bool = False,
) -> Dict[str, Any]:
    """
    导入多个项目并支持别名
    
    Args:
        module_paths: 尝试导入的模块路径（字符串或列表）
        imports: 字典，键是导入后的名称，值是要导入的项目名称（字符串或列表）
        defaults: 默认值字典
        fallback_funcs: 回退函数字典
        log_success: 是否记录成功导入的日志
        log_failure: 是否记录失败导入的日志
    
    Returns:
        包含所有导入项目的字典
    
    Example:
        result = safe_import_with_aliases(
            ['surgical_robot_app.vtk_utils.coords', 'vtk_utils.coords'],
            imports={
                'get_model_bounds': 'get_model_bounds',
                'world_to_space': 'world_to_space',
                'space_to_world': 'space_to_world',
                'estimate_radius': 'estimate_radius',
            },
            defaults={
                'get_model_bounds': None,
                'world_to_space': None,
                # ...
            }
        )
    """
    if isinstance(module_paths, str):
        module_paths = [module_paths]
    
    if defaults is None:
        defaults = {}
    if fallback_funcs is None:
        fallback_funcs = {}
    
    result = {}
    
    # 尝试从每个路径导入
    for path in module_paths:
        try:
            module = importlib.import_module(path)
            all_found = True
            
            for alias, item_name in imports.items():
                if isinstance(item_name, list):
                    # 导入多个项目
                    items = []
                    for name in item_name:
                        if hasattr(module, name):
                            items.append(getattr(module, name))
                        else:
                            all_found = False
                            break
                    if all_found:
                        result[alias] = tuple(items) if len(items) > 1 else items[0]
                else:
                    # 导入单个项目
                    if hasattr(module, item_name):
                        result[alias] = getattr(module, item_name)
                    else:
                        all_found = False
                        break
            
            if all_found:
                if log_success:
                    logger.debug(f"Successfully imported all items from {path}")
                return result
        except ImportError as e:
            if log_failure:
                logger.debug(f"Failed to import from {path}: {e}")
            continue
    
    # 所有导入都失败，使用默认值或回退函数
    if log_failure:
        logger.warning(f"All import attempts failed for {module_paths}")
    
    for alias in imports.keys():
        if alias in result:
            continue
        if alias in fallback_funcs:
            result[alias] = fallback_funcs[alias]()
        elif alias in defaults:
            result[alias] = defaults[alias]
        else:
            result[alias] = None
    
    return result


def safe_import_multiple_modules(
    imports: List[Tuple[Union[str, List[str]], Optional[Union[str, List[str]]], Any, Optional[Callable]]],
    log_success: bool = False,
    log_failure: bool = False,
) -> List[Any]:
    """
    批量导入多个模块
    
    Args:
        imports: 导入配置列表，每个元素是 (module_paths, item_names, default, fallback_func)
        log_success: 是否记录成功导入的日志
        log_failure: 是否记录失败导入的日志
    
    Returns:
        导入结果的列表
    
    Example:
        results = safe_import_multiple_modules([
            (['surgical_robot_app.data_io.sequence_reader', 'data_io.sequence_reader'], None, None, None),
            (['surgical_robot_app.path_planning.path_controller', 'path_planning.path_controller'], 'PathController', None, None),
        ])
    """
    results = []
    for config in imports:
        module_paths, item_names, default, fallback_func = config
        result = safe_import(
            module_paths,
            item_names,
            default,
            fallback_func,
            log_success,
            log_failure,
        )
        results.append(result)
    return results

