from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QSpacerItem, QSizePolicy, QHBoxLayout)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

class WelcomeView(QWidget):
    enter_system_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        # Set gradient background
        self.setStyleSheet("""
            WelcomeView {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #F5F7FA, stop:1 #E8EBF0);
            }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 40)
        
        # Spacer for top
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Logo placeholder (Simplified Medical Icon)
        logo_label = QLabel("✚") # Placeholder for a real logo
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("font-size: 80px; color: #1E88E5; margin-bottom: 20px;")
        layout.addWidget(logo_label)
        
        # App Name
        title_label = QLabel("Surgical Robot 3D Reconstruction System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #333333; 
            margin-bottom: 60px;
        """)
        layout.addWidget(title_label)
        
        # Enter Button
        btn_layout = QHBoxLayout()
        self.btn_enter = QPushButton("ENTER SYSTEM")
        self.btn_enter.setFixedSize(240, 50)
        self.btn_enter.setCursor(Qt.PointingHandCursor)
        self.btn_enter.setToolTip("Click to enter patient information entry")
        self.btn_enter.setStyleSheet("""
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
                padding-top: 2px;
                padding-left: 2px;
            }
        """)
        self.btn_enter.clicked.connect(self.enter_system_clicked.emit)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_enter)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Spacer for bottom
        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Version Info
        version_label = QLabel("Version 2.1.0 | © 2026 ME5400 Surgical Tech")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #7F8C8D; font-size: 12px;")
        layout.addWidget(version_label)

