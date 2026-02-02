from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QLineEdit, QComboBox, QPushButton, QFrame, 
                             QGridLayout, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal

class PatientInfoView(QWidget):
    confirmed = pyqtSignal(dict)
    back_clicked = pyqtSignal()
    exit_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        self.setStyleSheet("""
            PatientInfoView {
                background-color: #F5F7FA;
            }
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 20, 40, 40)
        main_layout.setSpacing(20)
        
        # --- Top Header ---
        header_layout = QHBoxLayout()
        title_label = QLabel("Patient Information Registration")
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #1E88E5;")
        
        self.btn_back = QPushButton("Back")
        self.btn_exit = QPushButton("Exit")
        for btn in [self.btn_back, self.btn_exit]:
            btn.setFixedSize(80, 30)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #ECF0F1;
                    border: 1px solid #BDC3C7;
                    border-radius: 4px;
                    color: #2C3E50;
                }
                QPushButton:hover { background-color: #D5DBDB; }
            """)
        
        self.btn_back.clicked.connect(self.back_clicked.emit)
        self.btn_exit.clicked.connect(self.exit_clicked.emit)
        
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_back)
        header_layout.addWidget(self.btn_exit)
        main_layout.addLayout(header_layout)
        
        # --- Main Body (Card Style) ---
        body_layout = QHBoxLayout()
        
        # Left side: Form
        form_frame = QFrame()
        form_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #E0E0E0;
            }
        """)
        form_layout = QGridLayout(form_frame)
        form_layout.setContentsMargins(30, 40, 30, 40)
        form_layout.setVerticalSpacing(20)
        form_layout.setHorizontalSpacing(15)
        
        label_style = "font-size: 14px; font-weight: bold; color: #34495E;"
        input_style = """
            QLineEdit, QComboBox {
                padding: 8px;
                border: 1px solid #DCDCDC;
                border-radius: 4px;
                background-color: #FFFFFF;
                font-size: 14px;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #1E88E5;
            }
        """
        
        # Name
        form_layout.addWidget(QLabel("Patient Name:"), 0, 0)
        self.input_name = QLineEdit()
        self.input_name.setPlaceholderText("e.g. John Doe")
        self.input_name.setStyleSheet(input_style)
        form_layout.addWidget(self.input_name, 0, 1)
        
        # ID (Required)
        id_label = QLabel("Patient ID *:")
        id_label.setStyleSheet(label_style)
        form_layout.addWidget(id_label, 1, 0)
        self.input_id = QLineEdit()
        self.input_id.setPlaceholderText("Unique identification number")
        self.input_id.setStyleSheet(input_style)
        form_layout.addWidget(self.input_id, 1, 1)
        
        # Case No
        form_layout.addWidget(QLabel("Case Number:"), 2, 0)
        self.input_case = QLineEdit()
        self.input_case.setPlaceholderText("Format: CASE-XXXX")
        self.input_case.setStyleSheet(input_style)
        form_layout.addWidget(self.input_case, 2, 1)
        
        # Exam Type
        form_layout.addWidget(QLabel("Exam Type:"), 3, 0)
        self.combo_type = QComboBox()
        self.combo_type.addItems(["CT", "MRI", "X-Ray", "Ultrasound"])
        self.combo_type.setStyleSheet(input_style)
        form_layout.addWidget(self.combo_type, 3, 1)
        
        body_layout.addStretch(1)
        body_layout.addWidget(form_frame, 3)
        
        # Right side: Decorative Icon
        info_icon_label = QLabel("📋")
        info_icon_label.setStyleSheet("font-size: 120px; color: #F0F3F4;")
        body_layout.addWidget(info_icon_label, 2, Qt.AlignCenter)
        body_layout.addStretch(1)
        
        main_layout.addLayout(body_layout)
        
        # --- Bottom Area ---
        footer_layout = QVBoxLayout()
        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #D32F2F; font-size: 13px;")
        footer_layout.addWidget(self.error_label)
        
        self.btn_confirm = QPushButton("CONFIRM && PROCEED")
        self.btn_confirm.setFixedSize(220, 45)
        self.btn_confirm.setCursor(Qt.PointingHandCursor)
        self.btn_confirm.setStyleSheet("""
            QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #BDC3C7; }
        """)
        self.btn_confirm.clicked.connect(self.handle_confirm)
        
        btn_center_layout = QHBoxLayout()
        btn_center_layout.addStretch()
        btn_center_layout.addWidget(self.btn_confirm)
        btn_center_layout.addStretch()
        
        footer_layout.addLayout(btn_center_layout)
        main_layout.addLayout(footer_layout)
        
    def handle_confirm(self):
        p_id = self.input_id.text().strip()
        if not p_id:
            self.error_label.setText("⚠ Error: Patient ID is required!")
            self.input_id.setStyleSheet(self.input_id.styleSheet() + "border: 1px solid #D32F2F;")
            return
        
        data = {
            "name": self.input_name.text().strip() or "Anonymous",
            "patient_id": p_id,
            "case_no": self.input_case.text().strip() or "N/A",
            "exam_type": self.combo_type.currentText(),
        }
        self.error_label.setText("")
        self.confirmed.emit(data)

    def clear_inputs(self):
        self.input_name.clear()
        self.input_id.clear()
        self.input_case.clear()
        self.combo_type.setCurrentIndex(0)
        self.error_label.setText("")

