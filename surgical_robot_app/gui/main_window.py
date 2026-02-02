from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QStackedWidget,
    QVBoxLayout,
    QMessageBox,
    QApplication,
)
from PyQt5.QtCore import Qt
import logging
import os

# Import our new views
try:
    from surgical_robot_app.gui.views.welcome_view import WelcomeView
    from surgical_robot_app.gui.views.patient_info_view import PatientInfoView
    from surgical_robot_app.gui.views.main_view import MainView
    from surgical_robot_app.utils.logger import get_logger
except ImportError:
    from gui.views.welcome_view import WelcomeView
    from gui.views.patient_info_view import PatientInfoView
    from gui.views.main_view import MainView
    from utils.logger import get_logger

logger = get_logger("surgical_robot_app.gui.main_window")

class MainWindow(QMainWindow):
    """
    Main entry window for the application.
    Manages multiple pages using QStackedWidget.
    Workflow: Welcome -> Patient Info -> Image Processing
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surgical Robot Navigation System")
        self.resize(1400, 900)
        
        # Apply global medical style
        self.apply_styles()
        
        # Initialize Stacked Widget
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # Initialize Views
        self.welcome_view = WelcomeView()
        self.patient_info_view = PatientInfoView()
        self.main_view = MainView()
        
        # Add views to stack
        self.stack.addWidget(self.welcome_view)      # Index 0
        self.stack.addWidget(self.patient_info_view) # Index 1
        self.stack.addWidget(self.main_view)         # Index 2
        
        # Add Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("background-color: white; border-top: 1px solid #E0E4E8; color: #7F8C8D;")
        self.status_bar.showMessage("System Ready")
        
        # Connect Signals
        self.connect_signals()
        
        # Start at Welcome page
        self.stack.setCurrentIndex(0)

    def apply_styles(self):
        """Apply global medical style QSS"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F7FA;
            }
            QWidget {
                font-family: 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            }
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                color: #2C3E50;
            }
            QMessageBox QPushButton {
                background-color: #1E88E5;
                color: white;
                border-radius: 4px;
                padding: 5px 15px;
                min-width: 60px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1976D2;
            }
        """)

    def connect_signals(self):
        # 1. Welcome -> Patient Info
        self.welcome_view.enter_system_clicked.connect(lambda: self.stack.setCurrentIndex(1))
        
        # 2. Patient Info Navigation
        self.patient_info_view.back_clicked.connect(self.handle_back_to_welcome)
        self.patient_info_view.exit_clicked.connect(self.confirm_exit)
        self.patient_info_view.confirmed.connect(self.handle_patient_confirmed)
        
        # 3. Main View Navigation
        self.main_view.back_clicked.connect(self.handle_back_to_info)
        self.main_view.exit_clicked.connect(self.confirm_exit)

    def handle_patient_confirmed(self, patient_data):
        """Pass data to main view and switch to it"""
        self.main_view.set_patient_context(patient_data)
        self.stack.setCurrentIndex(2)
        logger.info(f"Navigated to Main View for patient: {patient_data['patient_id']}")

    def handle_back_to_welcome(self):
        reply = QMessageBox.question(self, 'Confirm Return', 
                                   "Are you sure you want to return to the welcome screen?",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stack.setCurrentIndex(0)

    def handle_back_to_info(self):
        reply = QMessageBox.question(self, 'Confirm Return', 
                                   "Return to patient info page? Unsaved progress might be lost.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stack.setCurrentIndex(1)

    def confirm_exit(self):
        reply = QMessageBox.question(self, 'Confirm Exit', 
                                   "Are you sure you want to exit the system? All unsaved data will be lost.",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            QApplication.quit()

if __name__ == "__main__":
    # Test block
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
