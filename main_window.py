from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from data_sync_widget import DataSynchronizationWidget
from gait_metrics_widget import GaitMetricsWidget
from video_validation_widget import VideoValidationWidget


class MainWindow(QMainWindow):
    """메인 윈도우 - 2가지 핵심 기능 통합"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("🚶 2-Function Gait Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 제목
        title_label = QLabel("🚶 보행 분석 시스템 - 2가지 핵심 기능")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 10px; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        
        # 1. 데이터 동기화 및 시각화 탭
        self.sync_widget = DataSynchronizationWidget()
        self.sync_widget.main_window = self  # 참조 설정
        self.tab_widget.addTab(self.sync_widget, "1️⃣ 데이터 동기화 & 시각화")
        
        # 2. 보행 지표 계산 탭
        self.metrics_widget = GaitMetricsWidget()
        self.tab_widget.addTab(self.metrics_widget, "2️⃣ 보행 지표 계산 & 분석")
        
        # 3. 비디오 검증 탭
        self.validation_widget = VideoValidationWidget()
        self.tab_widget.addTab(self.validation_widget, "3️⃣ 영상 & 데이터 검증")
        
        main_layout.addWidget(self.tab_widget)
        
        # 상태바
        self.statusBar().showMessage("시스템 준비 완료 - 1번 탭에서 데이터를 로드한 후 2번 탭에서 분석을 수행하세요") 