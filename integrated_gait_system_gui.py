"""
integrated_gait_system_gui.py - 통합 보행 분석 시스템 GUI

이 모듈은 보행 분석 시스템의 메인 윈도우를 제공합니다.
데이터 동기화, 보행 지표 계산, 비디오 검증 등 주요 기능을 통합합니다.

Author: Assistant
Date: 2025-06-14
"""

import sys
from PyQt5.QtWidgets import QApplication
from main_window import MainWindow

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 앱 스타일 설정
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f8f9fa;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QPushButton:disabled {
            background-color: #6c757d;
        }
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #e9ecef;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #007bff;
            color: white;
        }
    """)
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()