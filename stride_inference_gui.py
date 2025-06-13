#!/usr/bin/env python3
"""
Stride Inference GUI
PyQt 기반 stride length 예측 및 비교 GUI

Features:
1. 모델 선택
2. Walking 파일 선택 (자동 label 파일 매칭)
3. 추론 실행
4. 실제 결과와 비교
5. 결과 시각화

Author: Assistant
Date: 2025-01-12
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox, QProgressBar,
    QTabWidget, QGroupBox, QSplitter, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# Local imports
from stride_inference_pipeline import StrideInferencePipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """추론 작업을 백그라운드에서 실행하는 워커 쓰레드"""
    
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, model_path: str, walking_file: str, labels_file: str):
        super().__init__()
        self.model_path = model_path
        self.walking_file = walking_file
        self.labels_file = labels_file
    
    def run(self):
        """추론 실행"""
        try:
            self.progress.emit("모델 로딩 중...")
            pipeline = StrideInferencePipeline(self.model_path)
            
            self.progress.emit("추론 실행 중...")
            results = pipeline.run_inference(self.labels_file, self.walking_file)
            
            self.progress.emit("완료!")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class ComparisonPlotWidget(FigureCanvas):
    """예측 vs 실제 결과 비교 플롯"""
    
    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def plot_comparison(self, predicted_data: List[Dict], actual_data: List[Dict]):
        """예측 결과와 실제 결과 비교 플롯"""
        self.fig.clear()
        
        # 데이터 정렬 및 매칭
        pred_df = pd.DataFrame(predicted_data)
        actual_df = pd.DataFrame(actual_data)
        
        if len(pred_df) == 0 or len(actual_df) == 0:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No data to compare', ha='center', va='center', transform=ax.transAxes)
            self.draw()
            return
        
        # 서브플롯 생성
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Scatter plot: 예측 vs 실제
        ax1 = self.fig.add_subplot(gs[0, 0])
        if 'predicted_stride_length' in pred_df.columns and 'stride_length' in actual_df.columns:
            pred_values = pred_df['predicted_stride_length'].values
            actual_values = actual_df['stride_length'].values[:len(pred_values)]  # 길이 맞추기
            
            ax1.scatter(actual_values, pred_values, alpha=0.7, s=50)
            
            # 완벽한 예측 라인 (y=x)
            min_val = min(min(actual_values), min(pred_values))
            max_val = max(max(actual_values), max(pred_values))
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax1.set_xlabel('Actual Stride Length (m)')
            ax1.set_ylabel('Predicted Stride Length (m)')
            ax1.set_title('Predicted vs Actual')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 바이어스 플롯 (Residual plot)
        ax2 = self.fig.add_subplot(gs[0, 1])
        if 'predicted_stride_length' in pred_df.columns and 'stride_length' in actual_df.columns:
            residuals = pred_values - actual_values
            cycle_numbers = range(1, len(residuals) + 1)
            
            ax2.scatter(cycle_numbers, residuals, alpha=0.7, s=50)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Cycle Number')
            ax2.set_ylabel('Residual (Pred - Actual)')
            ax2.set_title('Residual Plot')
            ax2.grid(True, alpha=0.3)
        
        # 3. 시간 시리즈 비교
        ax3 = self.fig.add_subplot(gs[1, :])
        if len(pred_df) > 0 and len(actual_df) > 0:
            cycle_numbers = range(1, min(len(pred_df), len(actual_df)) + 1)
            
            if 'predicted_stride_length' in pred_df.columns:
                ax3.plot(cycle_numbers, pred_values[:len(cycle_numbers)], 'b-o', 
                        label='Predicted', markersize=6, linewidth=2)
            
            if 'stride_length' in actual_df.columns:
                ax3.plot(cycle_numbers, actual_values[:len(cycle_numbers)], 'r-s', 
                        label='Actual', markersize=6, linewidth=2)
            
            ax3.set_xlabel('Cycle Number')
            ax3.set_ylabel('Stride Length (m)')
            ax3.set_title('Stride Length Comparison Over Cycles')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        self.draw()


class StrideInferenceGUI(QMainWindow):
    """Stride Inference GUI 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stride Inference System")
        self.setGeometry(100, 100, 1400, 900)
        
        # 경로 설정
        self.base_dir = Path(__file__).parent
        self.walking_data_dir = self.base_dir / "walking_data"
        self.support_label_dir = self.base_dir / "support_label_data"
        self.models_dir = self.base_dir / "models_2"
        self.results_dir = self.base_dir / "stride_analysis_results"
        
        # 상태 변수
        self.current_walking_file = None
        self.current_labels_file = None
        self.current_model_path = None
        self.prediction_results = None
        self.actual_results = None
        
        # UI 초기화
        self.init_ui()
        
        # UI 초기화 완료 후 모델 로드
        self.load_available_models()
        
        # 스타일 설정
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def init_ui(self):
        """UI 컴포넌트 초기화"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 (Splitter 사용)
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget_layout = QVBoxLayout(central_widget)
        central_widget_layout.addWidget(main_splitter)
        
        # 왼쪽 패널 (컨트롤)
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # 오른쪽 패널 (결과)
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)
        
        # Splitter 비율 설정
        main_splitter.setSizes([400, 1000])
        
        # 메뉴바 생성
        self.create_menu_bar()
        
        # 상태바
        self.statusBar().showMessage("Ready")
        
        # 프로그레스 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_control_panel(self) -> QWidget:
        """컨트롤 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 파일 선택 그룹
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        # 모델 선택
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        file_layout.addLayout(model_layout)
        
        # Walking 파일 선택
        walking_layout = QHBoxLayout()
        walking_layout.addWidget(QLabel("Walking File:"))
        self.walking_file_edit = QLineEdit()
        self.walking_file_edit.setReadOnly(True)
        walking_layout.addWidget(self.walking_file_edit)
        
        self.browse_walking_btn = QPushButton("Browse")
        self.browse_walking_btn.clicked.connect(self.browse_walking_file)
        walking_layout.addWidget(self.browse_walking_btn)
        file_layout.addLayout(walking_layout)
        
        # Label 파일 (자동 감지)
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label File:"))
        self.label_file_edit = QLineEdit()
        self.label_file_edit.setReadOnly(True)
        label_layout.addWidget(self.label_file_edit)
        file_layout.addLayout(label_layout)
        
        layout.addWidget(file_group)
        
        # 실행 버튼
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.clicked.connect(self.run_inference)
        self.run_btn.setEnabled(False)
        layout.addWidget(self.run_btn)
        
        # 정보 표시
        info_group = QGroupBox("Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(200)
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return panel
    
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # File 메뉴
        file_menu = menubar.addMenu('File')
        
        # Export 서브메뉴
        export_menu = file_menu.addMenu('Export Results')
        
        # 전체 결과 내보내기
        export_all_action = export_menu.addAction('Export All Results to CSV')
        export_all_action.triggered.connect(self.export_all_results_to_csv)
        
        # 예측 결과만 내보내기
        export_pred_action = export_menu.addAction('Export Predictions Only')
        export_pred_action.triggered.connect(self.export_predictions_to_csv)
        
        # 비교 결과만 내보내기
        export_comp_action = export_menu.addAction('Export Comparison Only')
        export_comp_action.triggered.connect(self.export_comparison_to_csv)
        
        file_menu.addSeparator()
        
        # 종료
        exit_action = file_menu.addAction('Exit')
        exit_action.triggered.connect(self.close)
    
    def create_results_panel(self) -> QWidget:
        """결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 탭 위젯
        self.tab_widget = QTabWidget()
        
        # 1. 예측 결과 탭
        self.predictions_tab = self.create_predictions_tab()
        self.tab_widget.addTab(self.predictions_tab, "Predictions")
        
        # 2. 비교 결과 탭
        self.comparison_tab = self.create_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "Comparison")
        
        # 3. 시각화 탭
        self.visualization_tab = self.create_visualization_tab()
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        
        layout.addWidget(self.tab_widget)
        return panel
    
    def create_predictions_tab(self) -> QWidget:
        """예측 결과 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 요약 정보
        summary_group = QGroupBox("Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.subject_label = QLabel("Subject: -")
        self.cycles_label = QLabel("Total Cycles: -")
        self.mean_stride_label = QLabel("Mean Stride Length: -")
        self.mean_velocity_label = QLabel("Mean Velocity: -")
        
        summary_layout.addWidget(self.subject_label, 0, 0)
        summary_layout.addWidget(self.cycles_label, 0, 1)
        summary_layout.addWidget(self.mean_stride_label, 1, 0)
        summary_layout.addWidget(self.mean_velocity_label, 1, 1)
        
        layout.addWidget(summary_group)
        
        # 상세 결과 테이블
        self.predictions_table = QTableWidget()
        self.predictions_table.setColumnCount(8)
        self.predictions_table.setHorizontalHeaderLabels([
            "Cycle", "Foot", "Start Frame", "End Frame", 
            "Sequence Length", "Stride Time", "Predicted Length", "Velocity"
        ])
        self.predictions_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 예측 결과 테이블 헤더와 버튼
        table_header_layout = QHBoxLayout()
        table_header_layout.addWidget(QLabel("Detailed Predictions:"))
        table_header_layout.addStretch()
        
        self.export_predictions_btn = QPushButton("Export to CSV")
        self.export_predictions_btn.clicked.connect(self.export_predictions_to_csv)
        self.export_predictions_btn.setEnabled(False)
        table_header_layout.addWidget(self.export_predictions_btn)
        
        layout.addLayout(table_header_layout)
        layout.addWidget(self.predictions_table)
        
        return tab
    
    def create_comparison_tab(self) -> QWidget:
        """비교 결과 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 비교 통계
        stats_group = QGroupBox("Comparison Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.mae_label = QLabel("MAE: -")
        self.rmse_label = QLabel("RMSE: -")
        self.correlation_label = QLabel("Correlation: -")
        self.bias_label = QLabel("Bias: -")
        
        stats_layout.addWidget(self.mae_label, 0, 0)
        stats_layout.addWidget(self.rmse_label, 0, 1)
        stats_layout.addWidget(self.correlation_label, 1, 0)
        stats_layout.addWidget(self.bias_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        # 비교 테이블
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(6)
        self.comparison_table.setHorizontalHeaderLabels([
            "Cycle", "Foot", "Predicted", "Actual", "Difference", "% Error"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # 비교 결과 테이블 헤더와 버튼
        comparison_header_layout = QHBoxLayout()
        comparison_header_layout.addWidget(QLabel("Detailed Comparison:"))
        comparison_header_layout.addStretch()
        
        self.export_comparison_btn = QPushButton("Export to CSV")
        self.export_comparison_btn.clicked.connect(self.export_comparison_to_csv)
        self.export_comparison_btn.setEnabled(False)
        comparison_header_layout.addWidget(self.export_comparison_btn)
        
        layout.addLayout(comparison_header_layout)
        layout.addWidget(self.comparison_table)
        
        return tab
    
    def create_visualization_tab(self) -> QWidget:
        """시각화 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 플롯 위젯
        self.plot_widget = ComparisonPlotWidget(tab, width=12, height=8, dpi=100)
        layout.addWidget(self.plot_widget)
        
        return tab
    
    def load_available_models(self):
        """사용 가능한 모델 목록 로드"""
        try:
            if self.models_dir.exists():
                model_files = list(self.models_dir.glob("*.keras"))
                model_files.extend(self.models_dir.glob("*.h5"))
                
                self.model_combo.clear()
                for model_file in model_files:
                    self.model_combo.addItem(model_file.name, str(model_file))
                
                if len(model_files) > 0:
                    self.current_model_path = str(model_files[0])
                    self.add_info(f"Found {len(model_files)} model(s)")
                else:
                    self.add_info("No models found in models_2 directory")
            else:
                self.add_info("Models directory not found")
                
        except Exception as e:
            self.add_info(f"Error loading models: {e}")
    
    def on_model_changed(self, model_name: str):
        """모델 변경 이벤트"""
        if model_name:
            self.current_model_path = self.model_combo.currentData()
            self.add_info(f"Selected model: {model_name}")
            self.check_ready_to_run()
    
    def browse_walking_file(self):
        """Walking 파일 선택"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Walking Data File",
            str(self.walking_data_dir),
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.current_walking_file = file_path
            self.walking_file_edit.setText(file_path)
            
            # 자동으로 label 파일 찾기
            self.find_matching_label_file(file_path)
            self.check_ready_to_run()
    
    def find_matching_label_file(self, walking_file_path: str):
        """Walking 파일에 매칭되는 label 파일 찾기"""
        try:
            walking_path = Path(walking_file_path)
            base_name = walking_path.stem  # S01T01R01
            
            # support_label_data에서 매칭되는 파일 찾기
            label_pattern = f"{base_name}_support_labels.csv"
            
            # 모든 하위 디렉토리에서 검색
            for label_file in self.support_label_dir.rglob(label_pattern):
                self.current_labels_file = str(label_file)
                self.label_file_edit.setText(str(label_file))
                self.add_info(f"Found matching label file: {label_file.name}")
                return
            
            # 찾지 못한 경우
            self.current_labels_file = None
            self.label_file_edit.setText("Label file not found")
            self.add_info(f"Warning: No matching label file found for {base_name}")
            
        except Exception as e:
            self.add_info(f"Error finding label file: {e}")
            self.current_labels_file = None
    
    def check_ready_to_run(self):
        """실행 준비 상태 확인"""
        # UI가 아직 초기화되지 않은 경우 건너뛰기
        if not hasattr(self, 'run_btn') or self.run_btn is None:
            return
            
        ready = bool(
            self.current_model_path and 
            self.current_walking_file and 
            self.current_labels_file
        )
        self.run_btn.setEnabled(ready)
        
        if ready:
            self.add_info("Ready to run inference")
    
    def run_inference(self):
        """추론 실행"""
        if not all([self.current_model_path, self.current_walking_file, self.current_labels_file]):
            QMessageBox.warning(self, "Warning", "Please select all required files")
            return
        
        # UI 비활성화
        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # 워커 쓰레드 시작
        self.worker = InferenceWorker(
            self.current_model_path,
            self.current_walking_file,
            self.current_labels_file
        )
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_inference_error)
        self.worker.progress.connect(self.on_inference_progress)
        self.worker.start()
    
    def on_inference_progress(self, message: str):
        """추론 진행 상황 업데이트"""
        self.statusBar().showMessage(message)
        self.add_info(message)
    
    def on_inference_finished(self, results: Dict):
        """추론 완료 처리"""
        self.prediction_results = results
        self.display_predictions(results)
        
        # 실제 결과 로드 및 비교
        self.load_and_compare_actual_results()
        
        # UI 다시 활성화
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Inference completed successfully")
        
        self.add_info("✅ Inference completed successfully!")
        
        # CSV 내보내기 버튼 활성화
        self.export_predictions_btn.setEnabled(True)
    
    def on_inference_error(self, error_message: str):
        """추론 오류 처리"""
        QMessageBox.critical(self, "Error", f"Inference failed:\n{error_message}")
        
        # UI 다시 활성화
        self.run_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Inference failed")
        
        self.add_info(f"❌ Inference failed: {error_message}")
    
    def display_predictions(self, results: Dict):
        """예측 결과 표시"""
        if 'error' in results:
            self.add_info(f"Error in results: {results['error']}")
            return
        
        # 요약 정보 업데이트
        self.subject_label.setText(f"Subject: {results.get('subject_id', 'Unknown')}")
        self.cycles_label.setText(f"Total Cycles: {results.get('total_cycles', 0)}")
        self.mean_stride_label.setText(f"Mean Stride Length: {results.get('mean_stride_length', 0):.3f} m")
        self.mean_velocity_label.setText(f"Mean Velocity: {results.get('mean_velocity', 0):.3f} m/s")
        
        # 예측 결과 테이블 업데이트
        predictions = results.get('predictions', [])
        self.predictions_table.setRowCount(len(predictions))
        
        for i, pred in enumerate(predictions):
            self.predictions_table.setItem(i, 0, QTableWidgetItem(str(pred['cycle_number'])))
            self.predictions_table.setItem(i, 1, QTableWidgetItem(pred['foot']))
            self.predictions_table.setItem(i, 2, QTableWidgetItem(str(pred['start_frame'])))
            self.predictions_table.setItem(i, 3, QTableWidgetItem(str(pred['end_frame'])))
            self.predictions_table.setItem(i, 4, QTableWidgetItem(str(pred['sequence_length'])))
            self.predictions_table.setItem(i, 5, QTableWidgetItem(f"{pred['stride_time']:.3f}"))
            self.predictions_table.setItem(i, 6, QTableWidgetItem(f"{pred['predicted_stride_length']:.3f}"))
            self.predictions_table.setItem(i, 7, QTableWidgetItem(f"{pred['predicted_velocity']:.3f}"))
    
    def load_and_compare_actual_results(self):
        """실제 결과 로드 및 비교"""
        if not self.current_walking_file:
            return
        
        try:
            # 파일명에서 base name 추출
            walking_path = Path(self.current_walking_file)
            base_name = walking_path.stem  # S01T01R01
            
            # stride_analysis_results에서 매칭되는 파일 찾기
            self.add_info(f"Searching for actual results with pattern: {base_name}*")
            
            # JSON 파일 먼저 확인
            for result_file in self.results_dir.rglob(f"{base_name}*.json"):
                self.add_info(f"Found actual results (JSON): {result_file.name}")
                
                with open(result_file, 'r', encoding='utf-8') as f:
                    self.actual_results = json.load(f)
                
                self.compare_results()
                return
            
            # CSV 파일 확인 (stride_labels.csv 형태)
            for result_file in self.results_dir.rglob(f"{base_name}*stride_labels.csv"):
                self.add_info(f"Found actual results (CSV): {result_file.name}")
                
                df = pd.read_csv(result_file)
                # CSV 컬럼명을 영어로 매핑
                df_mapped = df.rename(columns={
                    'Stride Length(m)': 'stride_length',
                    '발': 'foot',
                    '시작프레임': 'start_frame',
                    '종료프레임': 'end_frame',
                    'Stride Time(s)': 'stride_time',
                    'Velocity(m/s)': 'velocity'
                })
                
                # CSV를 dict 형태로 변환
                self.actual_results = {
                    'cycles': df_mapped.to_dict('records')
                }
                
                self.compare_results()
                return
            
            self.add_info(f"Warning: No actual results found for {base_name}")
            
        except Exception as e:
            self.add_info(f"Error loading actual results: {e}")
    
    def compare_results(self):
        """예측 결과와 실제 결과 비교"""
        if not self.prediction_results or not self.actual_results:
            return
        
        try:
            pred_data = self.prediction_results.get('predictions', [])
            
            # 실제 데이터 구조 확인
            if 'cycles' in self.actual_results:
                actual_data = self.actual_results['cycles']
            elif isinstance(self.actual_results, list):
                actual_data = self.actual_results
            else:
                actual_data = [self.actual_results]
            
            if len(pred_data) == 0 or len(actual_data) == 0:
                self.add_info("No data available for comparison")
                return
            
            # 비교 데이터 생성
            comparison_data = []
            pred_values = []
            actual_values = []
            
            min_len = min(len(pred_data), len(actual_data))
            
            for i in range(min_len):
                pred = pred_data[i]
                actual = actual_data[i]
                
                pred_length = pred.get('predicted_stride_length', 0)
                actual_length = actual.get('stride_length', 0)
                
                if actual_length > 0:  # 유효한 실제 값만 사용
                    pred_values.append(pred_length)
                    actual_values.append(actual_length)
                    
                    diff = pred_length - actual_length
                    percent_error = (diff / actual_length) * 100 if actual_length != 0 else 0
                    
                    comparison_data.append({
                        'cycle': pred.get('cycle_number', i+1),
                        'foot': pred.get('foot', 'unknown'),
                        'predicted': pred_length,
                        'actual': actual_length,
                        'difference': diff,
                        'percent_error': percent_error
                    })
            
            # 통계 계산
            if len(pred_values) > 0:
                pred_array = np.array(pred_values)
                actual_array = np.array(actual_values)
                
                mae = np.mean(np.abs(pred_array - actual_array))
                rmse = np.sqrt(np.mean((pred_array - actual_array) ** 2))
                correlation = np.corrcoef(pred_array, actual_array)[0, 1] if len(pred_values) > 1 else 0
                bias = np.mean(pred_array - actual_array)
                
                # 통계 표시 업데이트
                self.mae_label.setText(f"MAE: {mae:.4f} m")
                self.rmse_label.setText(f"RMSE: {rmse:.4f} m")
                self.correlation_label.setText(f"Correlation: {correlation:.4f}")
                self.bias_label.setText(f"Bias: {bias:.4f} m")
                
                # 비교 테이블 업데이트
                self.comparison_table.setRowCount(len(comparison_data))
                for i, comp in enumerate(comparison_data):
                    self.comparison_table.setItem(i, 0, QTableWidgetItem(str(comp['cycle'])))
                    self.comparison_table.setItem(i, 1, QTableWidgetItem(comp['foot']))
                    self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{comp['predicted']:.3f}"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{comp['actual']:.3f}"))
                    self.comparison_table.setItem(i, 4, QTableWidgetItem(f"{comp['difference']:.3f}"))
                    self.comparison_table.setItem(i, 5, QTableWidgetItem(f"{comp['percent_error']:.1f}%"))
                
                # 시각화 업데이트
                self.plot_widget.plot_comparison(pred_data, actual_data)
                
                self.add_info(f"Comparison completed: MAE={mae:.4f}m, RMSE={rmse:.4f}m, r={correlation:.3f}")
                
                # 비교 결과 CSV 내보내기 버튼 활성화
                self.export_comparison_btn.setEnabled(True)
            
        except Exception as e:
            self.add_info(f"Error in comparison: {e}")
    
    def export_predictions_to_csv(self):
        """예측 결과 테이블을 CSV로 내보내기"""
        if self.predictions_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No prediction data to export")
            return
        
        try:
            # 파일 저장 대화상자
            default_filename = "stride_predictions.csv"
            if self.current_walking_file:
                walking_path = Path(self.current_walking_file)
                base_name = walking_path.stem
                default_filename = f"{base_name}_predictions.csv"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Predictions to CSV",
                default_filename,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # 테이블 데이터를 DataFrame으로 변환
            headers = []
            for col in range(self.predictions_table.columnCount()):
                headers.append(self.predictions_table.horizontalHeaderItem(col).text())
            
            data = []
            for row in range(self.predictions_table.rowCount()):
                row_data = []
                for col in range(self.predictions_table.columnCount()):
                    item = self.predictions_table.item(row, col)
                    row_data.append(item.text() if item else "")
                data.append(row_data)
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            self.add_info(f"✅ Predictions exported to: {file_path}")
            QMessageBox.information(self, "Success", f"Predictions exported successfully to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to export predictions: {str(e)}"
            self.add_info(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
    
    def export_comparison_to_csv(self):
        """비교 결과 테이블을 CSV로 내보내기"""
        if self.comparison_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No comparison data to export")
            return
        
        try:
            # 파일 저장 대화상자
            default_filename = "stride_comparison.csv"
            if self.current_walking_file:
                walking_path = Path(self.current_walking_file)
                base_name = walking_path.stem
                default_filename = f"{base_name}_comparison.csv"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Comparison to CSV",
                default_filename,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # 테이블 데이터를 DataFrame으로 변환
            headers = []
            for col in range(self.comparison_table.columnCount()):
                headers.append(self.comparison_table.horizontalHeaderItem(col).text())
            
            data = []
            for row in range(self.comparison_table.rowCount()):
                row_data = []
                for col in range(self.comparison_table.columnCount()):
                    item = self.comparison_table.item(row, col)
                    row_data.append(item.text() if item else "")
                data.append(row_data)
            
            # DataFrame 생성 및 저장
            df = pd.DataFrame(data, columns=headers)
            
            # 통계 정보도 함께 저장
            stats_data = {
                'Metric': ['MAE', 'RMSE', 'Correlation', 'Bias'],
                'Value': [
                    self.mae_label.text().split(': ')[1] if ': ' in self.mae_label.text() else '-',
                    self.rmse_label.text().split(': ')[1] if ': ' in self.rmse_label.text() else '-',
                    self.correlation_label.text().split(': ')[1] if ': ' in self.correlation_label.text() else '-',
                    self.bias_label.text().split(': ')[1] if ': ' in self.bias_label.text() else '-'
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            
            # 두 개의 시트로 저장 (CSV는 단일 시트이므로 구분자로 분리)
            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                f.write("# Comparison Statistics\n")
                stats_df.to_csv(f, index=False)
                f.write("\n# Detailed Comparison\n")
                df.to_csv(f, index=False)
            
            self.add_info(f"✅ Comparison results exported to: {file_path}")
            QMessageBox.information(self, "Success", f"Comparison results exported successfully to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to export comparison: {str(e)}"
            self.add_info(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
    
    def export_all_results_to_csv(self):
        """모든 결과를 하나의 CSV 파일로 내보내기"""
        if not self.prediction_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        try:
            # 파일 저장 대화상자
            default_filename = "stride_inference_results.csv"
            if self.current_walking_file:
                walking_path = Path(self.current_walking_file)
                base_name = walking_path.stem
                default_filename = f"{base_name}_inference_results.csv"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export All Results to CSV",
                default_filename,
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if not file_path:
                return
            
            # 예측 결과를 DataFrame으로 변환
            predictions = self.prediction_results.get('predictions', [])
            df = pd.DataFrame(predictions)
            
            # 컬럼 순서 정리
            if not df.empty:
                column_order = [
                    'cycle_number', 'foot', 'start_frame', 'end_frame',
                    'sequence_length', 'stride_time', 'predicted_stride_length', 'predicted_velocity'
                ]
                # 존재하는 컬럼만 선택
                available_columns = [col for col in column_order if col in df.columns]
                df = df[available_columns]
            
            # 요약 정보 추가
            summary_info = {
                'Subject ID': self.prediction_results.get('subject_id', 'Unknown'),
                'Total Cycles': self.prediction_results.get('total_cycles', 0),
                'Mean Stride Length': f"{self.prediction_results.get('mean_stride_length', 0):.3f} m",
                'Mean Velocity': f"{self.prediction_results.get('mean_velocity', 0):.3f} m/s",
                'Model Used': Path(self.current_model_path).name if self.current_model_path else 'Unknown',
                'Walking File': Path(self.current_walking_file).name if self.current_walking_file else 'Unknown',
                'Labels File': Path(self.current_labels_file).name if self.current_labels_file else 'Unknown'
            }
            
            # CSV 파일에 저장
            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                f.write("# Stride Inference Results Summary\n")
                for key, value in summary_info.items():
                    f.write(f"# {key}: {value}\n")
                f.write("\n# Detailed Predictions\n")
                df.to_csv(f, index=False)
            
            self.add_info(f"✅ All results exported to: {file_path}")
            QMessageBox.information(self, "Success", f"All results exported successfully to:\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to export results: {str(e)}"
            self.add_info(f"❌ {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)

    def add_info(self, message: str):
        """정보 텍스트에 메시지 추가"""
        # UI가 아직 초기화되지 않은 경우 건너뛰기
        if not hasattr(self, 'info_text') or self.info_text is None:
            print(f"[GUI] {message}")  # 콘솔에 출력
            return
            
        self.info_text.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        self.info_text.verticalScrollBar().setValue(
            self.info_text.verticalScrollBar().maximum()
        )


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName("Stride Inference System")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Vision Gait Lab")
    
    # 메인 윈도우 생성 및 표시
    window = StrideInferenceGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()