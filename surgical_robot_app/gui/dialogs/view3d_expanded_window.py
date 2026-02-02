"""
独立的3D视图窗口

功能：
- 提供可拖拽、可调整大小的独立3D视图窗口
- 与主窗口的3D视图同步显示（模型、路径、坐标轴等）
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt, pyqtSignal
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
except ImportError:
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.gui.dialogs.view3d_expanded_window")


class View3DExpandedWindow(QDialog):
    """独立的3D视图窗口"""
    
    def __init__(
        self, 
        parent=None, 
        source_renderer: Optional[vtkRenderer] = None,
        coordinate_system=None,
        path_ui_controller=None
    ):
        """
        初始化独立3D视图窗口
        
        Args:
            parent: 父窗口
            source_renderer: 源渲染器（主窗口的renderer），用于同步内容
            coordinate_system: 坐标系可视化器（主窗口的），用于同步坐标系
            path_ui_controller: 路径UI控制器（主窗口的），用于同步路径点和路径线
        """
        super().__init__(parent)
        self.setWindowTitle("3D View - Expanded")
        self.setMinimumSize(800, 600)
        self.resize(1000, 800)
        
        # 设置窗口标志，允许调整大小
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        
        self.source_renderer = source_renderer
        self.source_coordinate_system = coordinate_system
        self.source_path_ui_controller = path_ui_controller
        self.expanded_renderer: Optional[vtkRenderer] = None
        self.vtk_widget = None
        
        # 存储同步的actors引用
        self.synced_actors = []
        
        # 独立的坐标系可视化器
        self.coordinate_system = None
        
        self._build_ui()
        # 注意：不在__init__中初始化VTK，等待窗口显示后再初始化
        # 这样可以确保VTK widget完全初始化
        self._vtk_initialized = False
        self._sync_pending = True  # 标记需要同步
    
    def _build_ui(self):
        """构建UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # VTK Widget
        if QVTKWidget is None:
            from PyQt5.QtWidgets import QLabel
            vtk_widget = QLabel("VTK Qt bindings not available")
            vtk_widget.setAlignment(Qt.AlignCenter)
        else:
            vtk_widget = QVTKWidget(self)
            # 设置widget的背景色为深色，与renderer一致
            # 这样即使渲染未完成，也能看到正确的背景色
            vtk_widget.setStyleSheet("background-color: rgb(26, 26, 26);")
        self.vtk_widget = vtk_widget
        layout.addWidget(vtk_widget)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        btn_reset_cam = QPushButton("Reset Camera")
        btn_reset_cam.setObjectName("secondary_btn")
        btn_reset_cam.clicked.connect(self.reset_camera)
        button_layout.addWidget(btn_reset_cam)
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def _init_vtk(self):
        """初始化VTK"""
        if not QVTKWidget or self.vtk_widget is None:
            return
        
        try:
            # 创建独立的renderer
            if vtkRenderer:
                self.expanded_renderer = vtkRenderer()
                
                # 设置背景色（与主窗口一致，深色背景）
                self.expanded_renderer.SetBackground(0.1, 0.1, 0.1)
                # 启用深度测试
                self.expanded_renderer.SetAutomaticLightCreation(True)
                self.expanded_renderer.TwoSidedLightingOn()
            
            # 获取或创建render window
            if hasattr(self.vtk_widget, "GetRenderWindow"):
                rw = self.vtk_widget.GetRenderWindow()
                if rw:
                    # 确保renderer已添加
                    rw.AddRenderer(self.expanded_renderer)
                    # 设置render window属性
                    rw.SetNumberOfLayers(1)
                    # 确保背景色正确应用（再次设置，确保生效）
                    self.expanded_renderer.SetBackground(0.1, 0.1, 0.1)
                    logger.debug("✅ Renderer已添加到RenderWindow，背景色已设置")
            
            # 初始化interactor（必须在添加到render window之后）
            if hasattr(self.vtk_widget, "Initialize"):
                self.vtk_widget.Initialize()
            if hasattr(self.vtk_widget, "Start"):
                self.vtk_widget.Start()
            if hasattr(self.vtk_widget, "GetInteractor"):
                interactor = self.vtk_widget.GetInteractor()
                if interactor:
                    interactor.Enable()
                    logger.debug("✅ Interactor已启用")
            
            # 初始渲染（确保相机和背景色正确显示）
            self._force_initial_render()
            
            logger.info("✅ 独立3D视图窗口VTK初始化成功")
            
            # 初始化坐标系可视化器
            try:
                from surgical_robot_app.vtk_utils.coordinate_system import CoordinateSystemVisualizer
                self.coordinate_system = CoordinateSystemVisualizer(self.expanded_renderer)
            except ImportError:
                try:
                    from vtk_utils.coordinate_system import CoordinateSystemVisualizer
                    self.coordinate_system = CoordinateSystemVisualizer(self.expanded_renderer)
                except ImportError:
                    logger.warning("CoordinateSystemVisualizer not available")
                    self.coordinate_system = None
                    
        except Exception as e:
            logger.error(f"❌ VTK初始化失败: {e}", exc_info=True)
    
    def _force_initial_render(self):
        """强制初始渲染，确保背景色正确显示"""
        if not hasattr(self, 'expanded_renderer') or self.expanded_renderer is None:
            return
        
        if hasattr(self.vtk_widget, "GetRenderWindow"):
            rw = self.vtk_widget.GetRenderWindow()
            if rw and self.expanded_renderer:
                # 重置相机（即使没有内容，也要设置默认相机）
                camera = self.expanded_renderer.GetActiveCamera()
                if camera:
                    camera.SetPosition(0, 0, 1)
                    camera.SetFocalPoint(0, 0, 0)
                    camera.SetViewUp(0, 1, 0)
                self.expanded_renderer.ResetCamera()
                self.expanded_renderer.ResetCameraClippingRange()
                
                # 确保背景色设置
                self.expanded_renderer.SetBackground(0.1, 0.1, 0.1)
                
                # 多次渲染，确保背景色显示
                from PyQt5.QtWidgets import QApplication
                # 第一次渲染
                rw.Render()
                self.vtk_widget.update()
                QApplication.processEvents()
                # 再次渲染
                rw.Render()
                self.vtk_widget.update()
                QApplication.processEvents()
                # 最后一次渲染
                rw.Render()
                logger.debug("✅ 初始渲染完成（相机已重置，背景色已设置）")
    
    def sync_from_source(self):
        """从源renderer同步所有actors到当前renderer"""
        if not self.expanded_renderer:
            logger.warning("独立窗口的renderer未初始化")
            return
        
        if not self.source_renderer:
            # 如果源renderer不存在，清空独立窗口
            logger.info("源renderer不存在，清空独立窗口")
            self.clear_all()
            return
        
        # 确保VTK widget已完全初始化
        if not self.vtk_widget or not hasattr(self.vtk_widget, "GetRenderWindow"):
            logger.warning("VTK widget未正确初始化，延迟同步...")
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(200, self.sync_from_source)
            return
        
        # 确保renderer已添加到render window
        try:
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                renderers = rw.GetRenderers()
                if renderers.GetNumberOfItems() == 0:
                    logger.warning("独立窗口的renderer未添加到render window，正在添加...")
                    rw.AddRenderer(self.expanded_renderer)
        except Exception as e:
            logger.warning(f"检查renderer时出错: {e}")
        
        # 确保源renderer已经渲染
        try:
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
        except:
            pass
        
        try:
            logger.info("开始同步源renderer内容到独立窗口...")
            # 清除当前renderer
            self.expanded_renderer.RemoveAllViewProps()
            self.synced_actors.clear()
            
            # 获取源renderer中的所有actors
            actors = self.source_renderer.GetActors()
            if actors is None:
                logger.warning("无法获取源renderer的actors集合")
                return
            
            actors.InitTraversal()
            actor_count = 0
            
            while True:
                actor = actors.GetNextItem()
                if actor is None:
                    break
                actor_count += 1
                
                # 深拷贝actor（创建新的mapper和property）
                try:
                    # 获取actor的mapper
                    mapper = actor.GetMapper()
                    if not mapper:
                        continue
                    
                    # 尝试多种方式获取输入数据
                    input_data = None
                    
                    # 方法1: 尝试 GetInput() (用于 SetInputData)
                    try:
                        input_data = mapper.GetInput()
                    except:
                        pass
                    
                    # 方法2: 尝试 GetInputDataObject() (更通用的方法)
                    if input_data is None:
                        try:
                            input_data = mapper.GetInputDataObject(0, 0)
                        except:
                            pass
                    
                    # 方法3: 尝试从连接获取 (用于 SetInputConnection)
                    if input_data is None:
                        try:
                            connection = mapper.GetInputConnection(0, 0)
                            if connection:
                                connection.Update()
                                input_data = connection.GetOutputDataObject(0)
                        except:
                            pass
                    
                    if input_data is None:
                        logger.warning(f"⚠️ 无法获取actor #{actor_count} 的输入数据，跳过此actor")
                        continue
                    
                    # 检查输入数据是否有效
                    try:
                        num_points = input_data.GetNumberOfPoints()
                        if num_points == 0:
                            logger.warning(f"⚠️ Actor #{actor_count} 的输入数据没有点，跳过")
                            continue
                    except Exception as e:
                        logger.warning(f"⚠️ 检查actor #{actor_count} 输入数据时出错: {e}")
                        continue
                    
                    # 创建新的mapper和actor
                    from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
                    
                    new_mapper = vtkPolyDataMapper()
                    new_mapper.SetInputData(input_data)
                    new_mapper.ScalarVisibilityOff()
                    
                    new_actor = vtkActor()
                    new_actor.SetMapper(new_mapper)
                    
                    # 复制属性
                    prop = actor.GetProperty()
                    if prop:
                        new_prop = new_actor.GetProperty()
                        new_prop.SetColor(prop.GetColor())
                        new_prop.SetOpacity(prop.GetOpacity())
                        new_prop.SetSpecular(prop.GetSpecular())
                        new_prop.SetSpecularPower(prop.GetSpecularPower())
                        # 复制更多属性
                        try:
                            new_prop.SetLineWidth(prop.GetLineWidth())
                        except:
                            pass
                    
                    # 添加到expanded renderer
                    self.expanded_renderer.AddActor(new_actor)
                    self.synced_actors.append(new_actor)
                    logger.debug(f"✅ 成功同步actor: {actor_count}")
                except Exception as e:
                    logger.warning(f"同步actor时出错: {e}", exc_info=True)
                    continue
            
            # 同步坐标系（如果有）
            try:
                if self.source_coordinate_system and self.coordinate_system:
                    # 获取源坐标系的中心位置和大小
                    source_center = getattr(self.source_coordinate_system, 'center', (0.0, 0.0, 0.0))
                    source_size = getattr(self.source_coordinate_system, 'size', 5.0)
                    source_visible = getattr(self.source_coordinate_system, 'visible', False)
                    
                    # 如果源坐标系可见，则在独立窗口中显示
                    if source_visible:
                        # 获取缩放因子（从配置中读取）
                        scale_factor = None
                        try:
                            from surgical_robot_app.config.settings import get_config
                            config = get_config()
                            scale_factor = config.view3d.axes_actor_scale_factor
                        except Exception:
                            scale_factor = 1.0
                        self.coordinate_system.show_coordinate_system(center=source_center, size=source_size, scale_factor=scale_factor)
                        logger.info(f"✅ 已同步坐标系: center={source_center}, size={source_size}, scale_factor={scale_factor}")
                    else:
                        self.coordinate_system.hide_coordinate_system()
            except Exception as e:
                logger.warning(f"同步坐标系时出错: {e}")
            
            # 同步路径点和路径线（如果有）
            try:
                if self.source_path_ui_controller:
                    # 同步路径点标记（point_actors）
                    if hasattr(self.source_path_ui_controller, 'point_actors'):
                        for point_actor in self.source_path_ui_controller.point_actors:
                            if point_actor:
                                # 深拷贝路径点标记
                                try:
                                    mapper = point_actor.GetMapper()
                                    if mapper:
                                        input_data = mapper.GetInput()
                                        if input_data:
                                            from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
                                            
                                            new_mapper = vtkPolyDataMapper()
                                            new_mapper.SetInputData(input_data)
                                            new_mapper.ScalarVisibilityOff()
                                            
                                            new_actor = vtkActor()
                                            new_actor.SetMapper(new_mapper)
                                            
                                            # 复制属性
                                            prop = point_actor.GetProperty()
                                            if prop:
                                                new_prop = new_actor.GetProperty()
                                                new_prop.SetColor(prop.GetColor())
                                                new_prop.SetOpacity(prop.GetOpacity())
                                            
                                            self.expanded_renderer.AddActor(new_actor)
                                            self.synced_actors.append(new_actor)
                                except Exception as e:
                                    logger.warning(f"同步路径点标记时出错: {e}")
                    
                    # 同步路径线（path_actors）
                    if hasattr(self.source_path_ui_controller, 'path_actors'):
                        for path_actor in self.source_path_ui_controller.path_actors:
                            if path_actor:
                                # 深拷贝路径线
                                try:
                                    mapper = path_actor.GetMapper()
                                    if mapper:
                                        input_data = mapper.GetInput()
                                        if input_data:
                                            from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
                                            
                                            new_mapper = vtkPolyDataMapper()
                                            new_mapper.SetInputData(input_data)
                                            new_mapper.ScalarVisibilityOff()
                                            
                                            new_actor = vtkActor()
                                            new_actor.SetMapper(new_mapper)
                                            
                                            # 复制属性
                                            prop = path_actor.GetProperty()
                                            if prop:
                                                new_prop = new_actor.GetProperty()
                                                new_prop.SetColor(prop.GetColor())
                                                new_prop.SetOpacity(prop.GetOpacity())
                                                new_prop.SetLineWidth(prop.GetLineWidth())
                                            
                                            self.expanded_renderer.AddActor(new_actor)
                                            self.synced_actors.append(new_actor)
                                except Exception as e:
                                    logger.warning(f"同步路径线时出错: {e}")
                    
                    logger.info(f"✅ 已同步路径点和路径线")
            except Exception as e:
                logger.warning(f"同步路径相关actors时出错: {e}")
            
            # 同步相机（在添加所有actors之后）
            # 先重置相机，确保能看到所有内容
            try:
                # 计算可见边界
                bounds = self.expanded_renderer.ComputeVisiblePropBounds()
                if bounds and len(bounds) >= 6:
                    # 有内容，重置相机以包含所有内容
                    self.expanded_renderer.ResetCamera()
                    self.expanded_renderer.ResetCameraClippingRange()
                    logger.debug(f"✅ 相机已重置，bounds={bounds}")
                else:
                    # 没有内容，设置默认相机位置
                    camera = self.expanded_renderer.GetActiveCamera()
                    if camera:
                        camera.SetPosition(0, 0, 1)
                        camera.SetFocalPoint(0, 0, 0)
                        camera.SetViewUp(0, 1, 0)
                        self.expanded_renderer.ResetCameraClippingRange()
                        logger.debug("✅ 相机已设置为默认位置（无内容）")
                
                # 然后尝试同步源相机的视角（如果有内容）
                if bounds and len(bounds) >= 6:
                    source_camera = self.source_renderer.GetActiveCamera()
                    expanded_camera = self.expanded_renderer.GetActiveCamera()
                    
                    if source_camera and expanded_camera:
                        try:
                            expanded_camera.SetPosition(source_camera.GetPosition())
                            expanded_camera.SetFocalPoint(source_camera.GetFocalPoint())
                            expanded_camera.SetViewUp(source_camera.GetViewUp())
                            expanded_camera.SetViewAngle(source_camera.GetViewAngle())
                            expanded_camera.SetClippingRange(source_camera.GetClippingRange())
                            logger.debug("✅ 相机视角已同步")
                        except Exception as e:
                            logger.debug(f"同步相机参数时出错，使用ResetCamera: {e}")
                            # 回退到ResetCamera
                            self.expanded_renderer.ResetCamera()
                            self.expanded_renderer.ResetCameraClippingRange()
            except Exception as e:
                logger.warning(f"同步相机时出错: {e}", exc_info=True)
                # 确保至少重置了相机
                try:
                    self.expanded_renderer.ResetCamera()
                    self.expanded_renderer.ResetCameraClippingRange()
                except:
                    pass
            
            # 强制渲染（在相机设置后）
            self._update_display()
            
            # 再次确保渲染（有时需要多次渲染才能正确显示）
            try:
                if hasattr(self.vtk_widget, "GetRenderWindow"):
                    rw = self.vtk_widget.GetRenderWindow()
                    if rw:
                        # 确保renderer已添加
                        renderers = rw.GetRenderers()
                        if renderers.GetNumberOfItems() == 0:
                            logger.warning("⚠️ Renderer未添加到RenderWindow，正在添加...")
                            rw.AddRenderer(self.expanded_renderer)
                            self.expanded_renderer.SetBackground(0.1, 0.1, 0.1)
                        
                        # 多次渲染，确保内容显示
                        from PyQt5.QtWidgets import QApplication
                        for i in range(3):
                            rw.Render()
                            self.vtk_widget.update()
                            QApplication.processEvents()
                        logger.debug("✅ 已完成多次渲染")
            except Exception as e:
                logger.warning(f"最终渲染时出错: {e}", exc_info=True)
            
            # 检查bounds，确保有内容
            try:
                bounds = self.expanded_renderer.ComputeVisiblePropBounds()
                if bounds and len(bounds) >= 6:
                    logger.info(f"✅ 独立窗口bounds: {bounds}")
                else:
                    logger.warning("⚠️ 独立窗口bounds为空，可能没有内容")
            except Exception as e:
                logger.warning(f"计算bounds时出错: {e}")
            
            logger.info(f"✅ 已同步 {len(self.synced_actors)} 个actors到独立窗口（源renderer中有 {actor_count} 个actors）")
            
            # 如果源renderer是空的，确保独立窗口也是空的
            if actor_count == 0 and len(self.synced_actors) == 0:
                logger.info("源renderer为空，独立窗口已清空")
        except Exception as e:
            logger.error(f"❌ 同步源renderer内容时出错: {e}", exc_info=True)
    
    def add_actor(self, actor):
        """添加actor到独立窗口（用于实时同步）"""
        if self.expanded_renderer and actor:
            try:
                # 检查是否已存在
                if actor not in self.synced_actors:
                    self.expanded_renderer.AddActor(actor)
                    self.synced_actors.append(actor)
                    self._update_display()
            except Exception as e:
                logger.warning(f"添加actor到独立窗口时出错: {e}")
    
    def remove_actor(self, actor):
        """从独立窗口移除actor"""
        if self.expanded_renderer and actor:
            try:
                self.expanded_renderer.RemoveActor(actor)
                if actor in self.synced_actors:
                    self.synced_actors.remove(actor)
                self._update_display()
            except Exception as e:
                logger.warning(f"从独立窗口移除actor时出错: {e}")
    
    def clear_all(self):
        """清除所有actors"""
        if self.expanded_renderer:
            self.expanded_renderer.RemoveAllViewProps()
            self.synced_actors.clear()
            # 隐藏坐标系
            if self.coordinate_system:
                self.coordinate_system.hide_coordinate_system()
            self._update_display()
    
    def reset_camera(self):
        """重置相机"""
        if self.expanded_renderer:
            try:
                self.expanded_renderer.ResetCamera()
                self.expanded_renderer.ResetCameraClippingRange()
                self._update_display()
            except Exception as e:
                logger.warning(f"重置相机时出错: {e}")
    
    def _update_display(self):
        """更新显示"""
        if self.vtk_widget and hasattr(self.vtk_widget, "GetRenderWindow"):
            try:
                rw = self.vtk_widget.GetRenderWindow()
                if rw:
                    # 确保renderer已添加到render window
                    if self.expanded_renderer:
                        renderers = rw.GetRenderers()
                        if renderers.GetNumberOfItems() == 0:
                            logger.warning("独立窗口的renderer未添加到render window，正在添加...")
                            rw.AddRenderer(self.expanded_renderer)
                            # 确保背景色设置
                            self.expanded_renderer.SetBackground(0.1, 0.1, 0.1)
                    
                    # 确保相机已初始化
                    if self.expanded_renderer:
                        camera = self.expanded_renderer.GetActiveCamera()
                        if camera:
                            # 检查相机是否在有效位置
                            pos = camera.GetPosition()
                            if pos[0] == 0 and pos[1] == 0 and pos[2] == 0:
                                # 相机在原点，设置默认位置
                                camera.SetPosition(0, 0, 1)
                                camera.SetFocalPoint(0, 0, 0)
                                camera.SetViewUp(0, 1, 0)
                    
                    # 强制渲染所有视图
                    rw.Render()
                    # 更新widget
                    self.vtk_widget.update()
                    # 处理事件，确保UI更新
                    from PyQt5.QtWidgets import QApplication
                    QApplication.processEvents()
                    # 再次渲染，确保显示
                    rw.Render()
                    # 再次更新widget
                    self.vtk_widget.update()
                    logger.debug("✅ 独立窗口显示已更新")
            except Exception as e:
                logger.warning(f"更新显示时出错: {e}", exc_info=True)
    
    def showEvent(self, event):
        """窗口显示事件 - 在窗口显示后初始化VTK并同步内容"""
        super().showEvent(event)
        
        logger.info("🔵 Expand View窗口显示事件触发")
        
        # 如果VTK还未初始化，现在初始化
        if not self._vtk_initialized:
            logger.info("🔵 开始初始化VTK...")
            self._init_vtk()
            self._vtk_initialized = True
            # 初始化后立即强制渲染
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self._force_initial_render)
            logger.info("✅ VTK初始化完成")
        
        # 窗口显示后，确保VTK widget已完全初始化，然后同步内容
        if self.source_renderer:
            logger.info(f"🔵 源renderer存在，准备同步内容（actors数量: {self.source_renderer.GetActors().GetNumberOfItems() if self.source_renderer.GetActors() else 0}）")
            # 延迟一点，确保窗口完全显示
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(300, lambda: self._do_sync_after_show())
        else:
            logger.warning("⚠️ 源renderer不存在，无法同步内容")
    
    def _do_sync_after_show(self):
        """在窗口显示后执行同步"""
        logger.info("窗口已显示，开始同步内容...")
        self.sync_from_source()
        # 同步后再次强制渲染，确保背景色正确
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, self._force_initial_render)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 清理资源
        if self.expanded_renderer:
            self.expanded_renderer.RemoveAllViewProps()
        event.accept()

