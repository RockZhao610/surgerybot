"""
DataImportController: 数据导入控制器

职责：
- 处理数据导入相关的事件（文件选择、导入、排除等）
- 管理文件列表的显示和更新
- 不依赖 Qt Widget，只接收必要的对象引用
"""

import os
from typing import Optional, Callable
from PyQt5.QtWidgets import QFileDialog, QListWidget

try:
    from surgical_robot_app.data_io.data_manager import DataManager
    from surgical_robot_app.data_io.sequence_reader import SequenceReader
except ImportError:
    try:
        from data_io.data_manager import DataManager
        from data_io.sequence_reader import SequenceReader
    except ImportError:
        DataManager = None  # type: ignore
        SequenceReader = None  # type: ignore


class DataImportController:
    """数据导入控制器"""
    
    def __init__(
        self,
        data_manager: DataManager,
        reader: SequenceReader,
        file_list: QListWidget,
        btn_undo,
        parent_widget=None,
    ):
        """
        初始化数据导入控制器
        
        Args:
            data_manager: 数据管理器
            reader: 序列读取器
            file_list: 文件列表控件
            btn_undo: 撤销按钮
            parent_widget: 父窗口（用于文件对话框）
        """
        self.data_manager = data_manager
        self.reader = reader
        self.file_list = file_list
        self.btn_undo = btn_undo
        self.parent_widget = parent_widget
        
        # 回调函数（由 MainWindow 设置）
        self.on_volume_loaded: Optional[Callable] = None
        self.on_volume_cleared: Optional[Callable] = None
        self.on_slice_slider_update: Optional[Callable] = None
    
    def handle_select_files(self, *args):
        """处理选择文件夹事件"""
        directory = QFileDialog.getExistingDirectory(
            self.parent_widget,
            "Select Image Folder",
            "",
        )
        if directory:
            self.data_manager.selected_dir = directory
            fmt = self.reader.guess_format(directory) or "png"
            self.data_manager.current_format = fmt
            paths = self.reader.get_series_paths(directory, fmt)
            self.data_manager.set_paths(paths)
            
            self.file_list.clear()
            self.data_manager.list_offset = 0
            for p in paths:
                name = os.path.basename(p)
                self.file_list.addItem(name)
            
            self.update_file_list_height()
            self.data_manager.exclusion_history = []
            self.update_undo_button()
    
    def handle_import(self, *args):
        """处理导入事件"""
        if not self.data_manager.selected_dir:
            return
        
        fmt = self.reader.guess_format(self.data_manager.selected_dir) or "png"
        self.data_manager.current_format = fmt
        volume, meta = self.reader.load_volume(self.data_manager.selected_dir, fmt)
        
        self.data_manager.set_volume(volume, meta)
        
        # 更新文件列表显示
        self.file_list.insertItem(
            0, 
            f"Series loaded | format: {fmt} | slices: {volume.shape[0]}"
        )
        self.data_manager.list_offset = 1
        
        # 更新 UI（通过回调）
        if self.on_volume_loaded:
            self.on_volume_loaded(volume, meta)
        
        self.update_file_list_height()
        self.update_undo_button()
    
    def handle_exclude_selected(self, *args):
        """处理排除选中文件事件"""
        if not self.data_manager.loaded_paths:
            return
        
        rows = sorted([self.file_list.row(i) for i in self.file_list.selectedItems()])
        if not rows:
            return
        
        rows = [r for r in rows if r >= self.data_manager.list_offset]
        if not rows:
            return
        
        files = []
        for r in rows:
            idx = r - self.data_manager.list_offset
            if 0 <= idx < len(self.data_manager.loaded_paths):
                files.append(self.data_manager.loaded_paths[idx])
        
        if not files:
            return
        
        # 排除文件
        self.data_manager.exclude_paths(files)
        
        # 如果已加载 volume，需要重新加载
        if self.data_manager.loaded_volume is not None:
            if self.data_manager.loaded_paths:
                fmt = self.data_manager.current_format
                volume, meta = self.reader.load_volume_from_paths(
                    self.data_manager.loaded_paths, fmt
                )
                self.data_manager.set_volume(volume, meta)
                
                # 更新 UI
                if self.on_volume_loaded:
                    self.on_volume_loaded(volume, meta)
            else:
                self.data_manager.clear_volume()
                if self.on_volume_cleared:
                    self.on_volume_cleared()
        
        self.refresh_file_list()
        self.update_undo_button()
    
    def handle_undo_exclusion(self, *args):
        """处理撤销排除事件"""
        if not self.data_manager.exclusion_history:
            return
        
        prev_paths = self.data_manager.undo_exclusion()
        if prev_paths is None:
            return
        
        # 如果已加载 volume，需要重新加载
        if self.data_manager.loaded_volume is not None:
            if self.data_manager.loaded_paths:
                fmt = self.data_manager.current_format
                volume, meta = self.reader.load_volume_from_paths(
                    self.data_manager.loaded_paths, fmt
                )
                self.data_manager.set_volume(volume, meta)
                
                # 更新 UI
                if self.on_volume_loaded:
                    self.on_volume_loaded(volume, meta)
            else:
                self.data_manager.clear_volume()
                if self.on_volume_cleared:
                    self.on_volume_cleared()
        
        self.refresh_file_list()
        self.update_undo_button()
    
    def refresh_file_list(self):
        """刷新文件列表显示"""
        self.file_list.clear()
        if self.data_manager.loaded_volume is not None:
            shape = self.data_manager.get_volume_shape()
            if shape:
                self.file_list.insertItem(
                    0,
                    f"Series loaded | format: {self.data_manager.current_format} | slices: {shape[0]}"
                )
                self.data_manager.list_offset = 1
            else:
                self.data_manager.list_offset = 0
        else:
            self.data_manager.list_offset = 0
        
        for p in self.data_manager.loaded_paths:
            name = os.path.basename(p)
            self.file_list.addItem(name)
        
        self.update_file_list_height()
        self.update_undo_button()
    
    def update_file_list_height(self):
        """更新文件列表高度"""
        count = self.file_list.count()
        item_height = self.file_list.sizeHintForRow(0)
        if item_height > 0:
            max_height = min(count * item_height + 10, 300)
            self.file_list.setMaximumHeight(max_height)
    
    def update_undo_button(self):
        """更新撤销按钮状态"""
        has_history = len(self.data_manager.exclusion_history) > 0
        self.btn_undo.setEnabled(has_history)

