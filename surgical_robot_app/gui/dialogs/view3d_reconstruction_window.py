"""
3D Reconstruction 独立窗口

功能：
- 只显示3D视图，不包含操作按钮
- 与主窗口的3D视图同步显示（模型、路径、坐标轴等）
- 所有操作在主窗口进行，独立窗口仅用于查看
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from typing import Optional

try:
    from vtkmodules.vtkRenderingCore import vtkRenderer
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
except Exception:
    try:
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor as QVTKWidget
    except Exception:
        QVTKWidget = None
        vtkRenderer = None

try:
    from surgical_robot_app.utils.logger import get_logger
    from surgical_robot_app.utils.error_handler import handle_errors
except ImportError:
    from utils.logger import get_logger
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = get_logger("surgical_robot_app.gui.dialogs.view3d_reconstruction_window")


class View3DReconstructionWindow(QDialog):
    """3D Reconstruction 独立窗口"""
    
    # 信号：窗口关闭时触发，用于同步状态回主窗口
    window_closing = pyqtSignal()
    
    def __init__(
        self, 
        parent=None,
        main_view=None,  # 主窗口的MainView引用
        source_renderer: Optional[vtkRenderer] = None,
        coordinate_system=None,
        path_ui_controller=None,
        view3d_manager=None,
        data_manager=None
    ):
        """
        初始化3D Reconstruction独立窗口
        
        Args:
            parent: 父窗口
            main_view: 主窗口的MainView引用（用于同步状态）
            source_renderer: 源渲染器（主窗口的renderer）
            coordinate_system: 坐标系可视化器（主窗口的）
            path_ui_controller: 路径UI控制器（主窗口的）
            view3d_manager: 3D视图管理器（主窗口的）
            data_manager: 数据管理器（主窗口的）
        """
        super().__init__(parent)
        self.setWindowTitle("3D Reconstruction - Independent Window")
        self.setMinimumSize(900, 700)
        self.resize(1200, 900)
        
        # 设置窗口标志，允许调整大小
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        # 保存主窗口引用
        self.main_view = main_view
        self.source_renderer = source_renderer
        self.source_coordinate_system = coordinate_system
        self.source_path_ui_controller = path_ui_controller
        self.view3d_manager = view3d_manager
        self.data_manager = data_manager
        
        # 独立的VTK组件
        self.vtk_renderer: Optional[vtkRenderer] = None
        self.vtk_widget = None
        
        # 独立的坐标系可视化器
        self.coordinate_system = None
        
        # 独立的路径UI控制器（用于同步路径显示）
        self.path_ui_controller = path_ui_controller
        
        # 同步状态
        self._vtk_initialized = False
        self._sync_pending = True
        
        # 存储同步的actors引用
        self.synced_actors = []
        
        self._build_ui()
    
    def _build_ui(self):
        """构建UI - 只显示3D视图"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)
        
        # 只添加VTK Widget，不添加任何按钮或控制面板
        if QVTKWidget is None:
            vtk_widget = QLabel("VTK Qt bindings not available")
            vtk_widget.setAlignment(Qt.AlignCenter)
        else:
            vtk_widget = QVTKWidget(self)
            vtk_widget.setStyleSheet("background-color: rgb(26, 26, 26);")
        self.vtk_widget = vtk_widget
        layout.addWidget(vtk_widget)
    
    
    def showEvent(self, event):
        """窗口显示时初始化VTK"""
        super().showEvent(event)
        if not self._vtk_initialized:
            # 延迟初始化，确保窗口完全显示
            QTimer.singleShot(100, self._init_vtk)
    
    def _init_vtk(self):
        """初始化VTK"""
        if self._vtk_initialized:
            return
        
        if not self.vtk_widget or not hasattr(self.vtk_widget, "GetRenderWindow"):
            logger.warning("VTK widget未正确初始化")
            return
        
        try:
            # 创建独立的renderer
            from vtkmodules.vtkRenderingCore import vtkRenderer
            self.vtk_renderer = vtkRenderer()
            
            # 设置背景色
            if self.main_view and hasattr(self.main_view, 'config'):
                config = self.main_view.config
                if config and hasattr(config, 'view3d'):
                    bg = config.view3d.background_color
                else:
                    bg = (0.1, 0.1, 0.1)
            else:
                bg = (0.1, 0.1, 0.1)
            self.vtk_renderer.SetBackground(*bg)
            
            # 添加到render window
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                rw.AddRenderer(self.vtk_renderer)
                
                # 初始化interactor
                if hasattr(self.vtk_widget, "Initialize"):
                    self.vtk_widget.Initialize()
                if hasattr(self.vtk_widget, "Start"):
                    self.vtk_widget.Start()
                if hasattr(self.vtk_widget, "GetInteractor"):
                    interactor = self.vtk_widget.GetInteractor()
                    if interactor:
                        interactor.Enable()
            
            # 创建独立的坐标系可视化器
            if self.vtk_renderer:
                try:
                    from surgical_robot_app.vtk_utils.coordinate_system import CoordinateSystemVisualizer
                    self.coordinate_system = CoordinateSystemVisualizer(self.vtk_renderer)
                except ImportError:
                    try:
                        from vtk_utils.coordinate_system import CoordinateSystemVisualizer
                        self.coordinate_system = CoordinateSystemVisualizer(self.vtk_renderer)
                    except ImportError:
                        logger.warning("CoordinateSystemVisualizer not available")
                        self.coordinate_system = None
            
            self._vtk_initialized = True
            
            # 初始同步
            if self._sync_pending:
                QTimer.singleShot(200, self.sync_from_source)
            
            logger.info("✅ 独立窗口VTK初始化完成")
        except Exception as e:
            logger.error(f"初始化独立窗口VTK时出错: {e}")
    
    def sync_from_source(self):
        """从主窗口同步所有内容到独立窗口"""
        if not self._vtk_initialized or not self.vtk_renderer:
            logger.warning("独立窗口VTK未初始化，延迟同步...")
            QTimer.singleShot(200, self.sync_from_source)
            return
        
        if not self.source_renderer:
            logger.warning("源renderer不存在")
            return
        
        try:
            # 清除旧的同步actors
            for actor in self.synced_actors:
                try:
                    self.vtk_renderer.RemoveActor(actor)
                except:
                    pass
            self.synced_actors.clear()
            
            # 同步所有actors（模型、路径等）
            actors = self.source_renderer.GetActors()
            actors.InitTraversal()
            while True:
                actor = actors.GetNextItem()
                if actor is None:
                    break
                
                # 跳过坐标系actor（我们有自己的）
                if hasattr(actor, 'GetClassName') and 'Axes' in actor.GetClassName():
                    continue
                
                # 深拷贝actor
                try:
                    copied_actor = self._deep_copy_actor(actor)
                    if copied_actor:
                        self.vtk_renderer.AddActor(copied_actor)
                        self.synced_actors.append(copied_actor)
                except Exception as e:
                    logger.warning(f"复制actor时出错: {e}")
            
            # 同步坐标系
            if self.coordinate_system and self.source_coordinate_system:
                if hasattr(self.source_coordinate_system, 'is_visible'):
                    if self.source_coordinate_system.is_visible():
                        self.coordinate_system.show_coordinate_system()
                    else:
                        self.coordinate_system.hide_coordinate_system()
            
            
            # 同步相机
            self._sync_camera_from_source()
            
            # 更新显示
            self._update_display()
            
            logger.debug("✅ 从主窗口同步完成")
        except Exception as e:
            logger.error(f"同步内容时出错: {e}")
    
    def _sync_camera_from_source(self):
        """同步相机位置从源renderer"""
        if not self.vtk_renderer or not self.source_renderer:
            return
        
        try:
            source_camera = self.source_renderer.GetActiveCamera()
            target_camera = self.vtk_renderer.GetActiveCamera()
            
            if source_camera and target_camera:
                target_camera.SetPosition(source_camera.GetPosition())
                target_camera.SetFocalPoint(source_camera.GetFocalPoint())
                target_camera.SetViewUp(source_camera.GetViewUp())
                target_camera.SetViewAngle(source_camera.GetViewAngle())
                self.vtk_renderer.ResetCameraClippingRange()
        except Exception as e:
            logger.warning(f"同步相机时出错: {e}")
    
    def _deep_copy_actor(self, source_actor):
        """深拷贝VTK actor"""
        try:
            mapper = source_actor.GetMapper()
            if not mapper:
                return None
            
            # 获取输入数据
            input_data = None
            if hasattr(mapper, 'GetInput'):
                input_data = mapper.GetInput()
            elif hasattr(mapper, 'GetInputDataObject'):
                input_data = mapper.GetInputDataObject(0, 0)
            elif hasattr(mapper, 'GetInputConnection'):
                conn = mapper.GetInputConnection(0, 0)
                if conn:
                    input_data = conn.GetProducer().GetOutputDataObject(0)
            
            if not input_data:
                return None
            
            # 创建新的mapper和actor
            from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
            new_mapper = vtkPolyDataMapper()
            new_mapper.SetInputData(input_data)
            
            # 复制 mapper 的标量可见性设置（关键：关闭后才会使用 actor 颜色）
            source_mapper = source_actor.GetMapper()
            if source_mapper:
                new_mapper.SetScalarVisibility(source_mapper.GetScalarVisibility())
            
            new_actor = vtkActor()
            new_actor.SetMapper(new_mapper)
            
            # 复制属性
            source_prop = source_actor.GetProperty()
            if source_prop:
                target_prop = new_actor.GetProperty()
                target_prop.SetColor(source_prop.GetColor())
                target_prop.SetOpacity(source_prop.GetOpacity())
                target_prop.SetRepresentation(source_prop.GetRepresentation())
                target_prop.SetSpecular(source_prop.GetSpecular())
                target_prop.SetSpecularPower(source_prop.GetSpecularPower())
            
            return new_actor
        except Exception as e:
            logger.warning(f"深拷贝actor时出错: {e}")
            return None
    
    def _update_display(self):
        """更新VTK显示"""
        if not self.vtk_widget or not self.vtk_renderer:
            return
        
        try:
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                rw.Render()
                self.vtk_widget.update()
        except Exception as e:
            logger.warning(f"更新显示时出错: {e}")
    
    def sync_to_main_view(self):
        """同步状态回主窗口（窗口关闭时调用）"""
        # 独立窗口只用于显示，所有操作都在主窗口进行
        # 所以不需要同步任何状态回主窗口
        
        logger.info("✅ 独立窗口已关闭")
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 同步状态回主窗口
        self.sync_to_main_view()
        
        # 发送关闭信号
        self.window_closing.emit()
        
        super().closeEvent(event)

