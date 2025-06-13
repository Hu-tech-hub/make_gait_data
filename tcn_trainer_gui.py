#!/usr/bin/env python3
"""
TCN 보폭 예측 시스템 GUI

PyQt5 기반의 사용자 친화적인 인터페이스:
- 하이퍼파라미터 설정
- 실시간 학습 진행 상황 모니터
- 결과 시각화 및 분석
- 모델 관리
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QTabWidget, QGroupBox, QLabel, QLineEdit, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QPushButton, QTextEdit, QProgressBar,
    QComboBox, QFileDialog, QMessageBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QFont, QPixmap, QIcon

# 로컬 모듈 import
from tcn_trainer import TCNTrainer

# 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

class TrainingWorker(QThread):
    """백그라운드에서 학습을 실행하는 워커 스레드"""
    
    # 시그널 정의
    progress_updated = pyqtSignal(str)  # 진행 상황 텍스트
    fold_completed = pyqtSignal(int, dict)  # fold 완료 (fold_idx, result)
    training_completed = pyqtSignal(dict)  # 전체 학습 완료 (cv_summary)
    error_occurred = pyqtSignal(str)  # 에러 발생
    
    def __init__(self, model_config: Dict, training_config: Dict, 
                 directories: Dict, options: Dict):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.directories = directories
        self.options = options
        self.trainer = None
        
    def run(self):
        """학습 실행"""
        try:
            # 트레이너 초기화
            self.progress_updated.emit("🔧 트레이너 초기화 중...")
            
            self.trainer = TCNTrainer(
                metadata_dir=self.directories['metadata_dir'],
                pkl_dir=self.directories['pkl_dir'],
                models_dir=self.directories['models_dir'],
                logs_dir=self.directories['logs_dir'],
                strict_mode=self.options['strict_mode']
            )
            
            # 메타데이터 로드
            self.progress_updated.emit("📁 메타데이터 로드 중...")
            df, cv_splits = self.trainer.load_and_validate_metadata()
            
            # 데이터 제너레이터 초기화
            self.progress_updated.emit("📊 데이터 제너레이터 초기화 중...")
            self.trainer.initialize_data_generator()
            
            # 교차검증 실행
            self.progress_updated.emit("🚀 교차검증 학습 시작...")
            
            n_folds = len(cv_splits)
            self.trainer.cv_results = []
            
            for fold_idx in range(n_folds):
                self.progress_updated.emit(f"🏃 Fold {fold_idx + 1}/{n_folds} 학습 중...")
                
                fold_result = self.trainer.train_single_fold(
                    fold_idx=fold_idx,
                    model_config=self.model_config,
                    training_config=self.training_config
                )
                
                self.trainer.cv_results.append(fold_result)
                self.fold_completed.emit(fold_idx, fold_result)
            
            # 결과 분석
            self.progress_updated.emit("📊 결과 분석 중...")
            cv_duration = sum(r['duration_minutes'] for r in self.trainer.cv_results)
            cv_summary = self.trainer._analyze_cv_results(cv_duration)
            
            # 리포트 생성
            self.progress_updated.emit("📄 리포트 생성 중...")
            self.trainer._generate_cv_report(cv_summary, self.model_config, self.training_config)
            
            # 전체 재훈련 (선택적)
            if self.options['full_retrain']:
                self.progress_updated.emit("🔄 전체 데이터로 최종 모델 재훈련 중...")
                final_model_path = self.trainer.retrain_final_model(
                    self.model_config, self.training_config
                )
                cv_summary['final_model_path'] = final_model_path
            
            self.training_completed.emit(cv_summary)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class ResultsVisualizationWidget(QWidget):
    """결과 시각화 위젯"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cv_results = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 결과 테이블
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Fold', 'Test Subject', 'Train Cycles', 'Val Cycles', 
            'Best Epoch', 'Val MAE', 'Train MAE', 'Duration (min)'
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("📊 Fold별 결과"))
        layout.addWidget(self.results_table)
        
        # 시각화 영역
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(QLabel("📈 결과 시각화"))
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def update_results(self, fold_idx: int, fold_result: Dict):
        """Fold 결과 업데이트"""
        self.cv_results.append(fold_result)
        
        # 테이블 업데이트
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        items = [
            str(fold_result['fold']),
            fold_result['test_subject'],
            str(fold_result['n_train_cycles']),
            str(fold_result['n_val_cycles']),
            str(fold_result['best_epoch']),
            f"{fold_result['best_val_mae']:.4f}",
            f"{fold_result['final_train_mae']:.4f}",
            f"{fold_result['duration_minutes']:.1f}"
        ]
        
        for col, item in enumerate(items):
            self.results_table.setItem(row, col, QTableWidgetItem(item))
        
        # 시각화 업데이트
        self.update_visualization()
    
    def update_visualization(self):
        """시각화 업데이트"""
        if not self.cv_results:
            return
        
        self.figure.clear()
        
        # 2x2 서브플롯
        ax1 = self.figure.add_subplot(2, 2, 1)
        ax2 = self.figure.add_subplot(2, 2, 2)
        ax3 = self.figure.add_subplot(2, 2, 3)
        ax4 = self.figure.add_subplot(2, 2, 4)
        
        # 데이터 준비
        folds = [r['fold'] for r in self.cv_results]
        val_maes = [r['best_val_mae'] for r in self.cv_results]
        train_maes = [r['final_train_mae'] for r in self.cv_results]
        durations = [r['duration_minutes'] for r in self.cv_results]
        subjects = [r['test_subject'] for r in self.cv_results]
        
        # 1. MAE 비교 (Train vs Val)
        x = np.arange(len(folds))
        width = 0.35
        ax1.bar(x - width/2, train_maes, width, label='Train MAE', alpha=0.8)
        ax1.bar(x + width/2, val_maes, width, label='Val MAE', alpha=0.8)
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('MAE (m)')
        ax1.set_title('Train vs Validation MAE')
        ax1.set_xticks(x)
        ax1.set_xticklabels(folds)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Validation MAE 분포
        ax2.boxplot([val_maes], labels=['Val MAE'])
        ax2.scatter([1] * len(val_maes), val_maes, alpha=0.6, s=50)
        ax2.set_ylabel('MAE (m)')
        ax2.set_title('Validation MAE Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. 피험자별 성능
        ax3.bar(range(len(subjects)), val_maes, alpha=0.8)
        ax3.set_xlabel('Test Subject')
        ax3.set_ylabel('Val MAE (m)')
        ax3.set_title('Performance by Test Subject')
        ax3.set_xticks(range(len(subjects)))
        ax3.set_xticklabels(subjects, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 학습 시간
        ax4.bar(folds, durations, alpha=0.8, color='orange')
        ax4.set_xlabel('Fold')
        ax4.set_ylabel('Duration (min)')
        ax4.set_title('Training Duration per Fold')
        ax4.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def finalize_visualization(self, cv_summary: Dict):
        """최종 결과 시각화"""
        # 통계 정보 추가
        mean_mae = cv_summary['mean_val_mae']
        std_mae = cv_summary['std_val_mae']
        
        # 제목에 통계 정보 추가
        self.figure.suptitle(
            f'TCN Cross-Validation Results\n'
            f'Mean Val MAE: {mean_mae:.4f} ± {std_mae:.4f}m',
            fontsize=14, fontweight='bold'
        )
        
        self.canvas.draw()

class TCNTrainerGUI(QMainWindow):
    """TCN 트레이너 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        self.load_default_settings()
        
    def init_ui(self):
        self.setWindowTitle('TCN 보폭 예측 시스템 - GUI')
        self.setGeometry(100, 100, 1400, 900)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 좌측 패널 (설정)
        left_panel = self.create_settings_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 우측 패널 (결과 및 로그)
        right_panel = self.create_results_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_settings_panel(self) -> QWidget:
        """설정 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 제목
        title = QLabel("🧠 TCN 트레이너 설정")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 탭 위젯
        tabs = QTabWidget()
        
        # 모델 설정 탭
        model_tab = self.create_model_settings_tab()
        tabs.addTab(model_tab, "🏗️ 모델")
        
        # 학습 설정 탭
        training_tab = self.create_training_settings_tab()
        tabs.addTab(training_tab, "🏃 학습")
        
        # 디렉토리 설정 탭
        directory_tab = self.create_directory_settings_tab()
        tabs.addTab(directory_tab, "📁 경로")
        
        layout.addWidget(tabs)
        
        # 실행 버튼들
        button_layout = QVBoxLayout()
        
        self.start_button = QPushButton("🚀 학습 시작")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ 중지")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        # 설정 저장/로드 버튼
        save_button = QPushButton("💾 설정 저장")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        load_button = QPushButton("📂 설정 로드")
        load_button.clicked.connect(self.load_settings)
        button_layout.addWidget(load_button)
        
        layout.addLayout(button_layout)
        
        # 진행 상황
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("준비됨")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        panel.setLayout(layout)
        return panel
    
    def create_model_settings_tab(self) -> QWidget:
        """모델 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # TCN 설정
        tcn_group = QGroupBox("TCN 아키텍처")
        tcn_layout = QGridLayout()
        
        tcn_layout.addWidget(QLabel("TCN Filters:"), 0, 0)
        self.tcn_filters = QSpinBox()
        self.tcn_filters.setRange(16, 256)
        self.tcn_filters.setValue(64)
        tcn_layout.addWidget(self.tcn_filters, 0, 1)
        
        tcn_layout.addWidget(QLabel("TCN Stacks:"), 1, 0)
        self.tcn_stacks = QSpinBox()
        self.tcn_stacks.setRange(1, 8)
        self.tcn_stacks.setValue(4)
        tcn_layout.addWidget(self.tcn_stacks, 1, 1)
        
        tcn_layout.addWidget(QLabel("Dropout Rate:"), 2, 0)
        self.dropout_rate = QDoubleSpinBox()
        self.dropout_rate.setRange(0.0, 0.5)
        self.dropout_rate.setSingleStep(0.1)
        self.dropout_rate.setValue(0.1)
        tcn_layout.addWidget(self.dropout_rate, 2, 1)
        
        tcn_layout.addWidget(QLabel("Dense Units:"), 3, 0)
        self.dense_units = QSpinBox()
        self.dense_units.setRange(16, 256)
        self.dense_units.setValue(64)
        tcn_layout.addWidget(self.dense_units, 3, 1)
        
        tcn_layout.addWidget(QLabel("Learning Rate:"), 4, 0)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0001, 0.1)
        self.learning_rate.setSingleStep(0.001)
        self.learning_rate.setDecimals(4)
        self.learning_rate.setValue(0.001)
        tcn_layout.addWidget(self.learning_rate, 4, 1)
        
        tcn_group.setLayout(tcn_layout)
        layout.addWidget(tcn_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_training_settings_tab(self) -> QWidget:
        """학습 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 학습 설정
        training_group = QGroupBox("학습 파라미터")
        training_layout = QGridLayout()
        
        training_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 500)
        self.epochs.setValue(100)
        training_layout.addWidget(self.epochs, 0, 1)
        
        training_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(8, 128)
        self.batch_size.setValue(32)
        training_layout.addWidget(self.batch_size, 1, 1)
        
        training_layout.addWidget(QLabel("Early Stopping Patience:"), 2, 0)
        self.patience_early = QSpinBox()
        self.patience_early.setRange(1, 50)
        self.patience_early.setValue(10)
        training_layout.addWidget(self.patience_early, 2, 1)
        
        training_layout.addWidget(QLabel("LR Reduce Patience:"), 3, 0)
        self.patience_lr = QSpinBox()
        self.patience_lr.setRange(1, 20)
        self.patience_lr.setValue(5)
        training_layout.addWidget(self.patience_lr, 3, 1)
        
        training_layout.addWidget(QLabel("LR Factor:"), 4, 0)
        self.lr_factor = QDoubleSpinBox()
        self.lr_factor.setRange(0.1, 0.9)
        self.lr_factor.setSingleStep(0.1)
        self.lr_factor.setValue(0.5)
        training_layout.addWidget(self.lr_factor, 4, 1)
        
        training_layout.addWidget(QLabel("Min LR:"), 5, 0)
        self.min_lr = QDoubleSpinBox()
        self.min_lr.setRange(1e-8, 1e-4)
        self.min_lr.setSingleStep(1e-6)
        self.min_lr.setDecimals(8)
        self.min_lr.setValue(1e-6)
        training_layout.addWidget(self.min_lr, 5, 1)
        
        training_group.setLayout(training_layout)
        layout.addWidget(training_group)
        
        # 옵션
        options_group = QGroupBox("추가 옵션")
        options_layout = QVBoxLayout()
        
        self.strict_mode = QCheckBox("엄격 모드 (데이터 무결성 검사)")
        options_layout.addWidget(self.strict_mode)
        
        self.full_retrain = QCheckBox("전체 재훈련 수행")
        options_layout.addWidget(self.full_retrain)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_directory_settings_tab(self) -> QWidget:
        """디렉토리 설정 탭"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # 디렉토리 설정
        dir_group = QGroupBox("디렉토리 경로")
        dir_layout = QGridLayout()
        
        # 메타데이터 디렉토리
        dir_layout.addWidget(QLabel("메타데이터:"), 0, 0)
        self.metadata_dir = QLineEdit("metadata")
        dir_layout.addWidget(self.metadata_dir, 0, 1)
        metadata_browse = QPushButton("찾기")
        metadata_browse.clicked.connect(lambda: self.browse_directory(self.metadata_dir))
        dir_layout.addWidget(metadata_browse, 0, 2)
        
        # PKL 디렉토리
        dir_layout.addWidget(QLabel("PKL 파일:"), 1, 0)
        self.pkl_dir = QLineEdit("stride_train_data_pkl")
        dir_layout.addWidget(self.pkl_dir, 1, 1)
        pkl_browse = QPushButton("찾기")
        pkl_browse.clicked.connect(lambda: self.browse_directory(self.pkl_dir))
        dir_layout.addWidget(pkl_browse, 1, 2)
        
        # 모델 저장 디렉토리
        dir_layout.addWidget(QLabel("모델 저장:"), 2, 0)
        self.models_dir = QLineEdit("models")
        dir_layout.addWidget(self.models_dir, 2, 1)
        models_browse = QPushButton("찾기")
        models_browse.clicked.connect(lambda: self.browse_directory(self.models_dir))
        dir_layout.addWidget(models_browse, 2, 2)
        
        # 로그 디렉토리
        dir_layout.addWidget(QLabel("로그:"), 3, 0)
        self.logs_dir = QLineEdit("logs")
        dir_layout.addWidget(self.logs_dir, 3, 1)
        logs_browse = QPushButton("찾기")
        logs_browse.clicked.connect(lambda: self.browse_directory(self.logs_dir))
        dir_layout.addWidget(logs_browse, 3, 2)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def create_results_panel(self) -> QWidget:
        """결과 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 탭 위젯
        tabs = QTabWidget()
        
        # 로그 탭
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        
        log_layout.addWidget(QLabel("📋 실시간 로그"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        log_tab.setLayout(log_layout)
        tabs.addTab(log_tab, "📋 로그")
        
        # 결과 시각화 탭
        self.results_widget = ResultsVisualizationWidget()
        tabs.addTab(self.results_widget, "📊 결과")
        
        layout.addWidget(tabs)
        panel.setLayout(layout)
        return panel
    
    def browse_directory(self, line_edit: QLineEdit):
        """디렉토리 선택"""
        directory = QFileDialog.getExistingDirectory(self, "디렉토리 선택")
        if directory:
            line_edit.setText(directory)
    
    def get_model_config(self) -> Dict:
        """모델 설정 수집"""
        return {
            'tcn_filters': self.tcn_filters.value(),
            'tcn_stacks': self.tcn_stacks.value(),
            'dropout_rate': self.dropout_rate.value(),
            'dense_units': self.dense_units.value(),
            'learning_rate': self.learning_rate.value()
        }
    
    def get_training_config(self) -> Dict:
        """학습 설정 수집"""
        return {
            'epochs': self.epochs.value(),
            'batch_size': self.batch_size.value(),
            'patience_early': self.patience_early.value(),
            'patience_lr': self.patience_lr.value(),
            'lr_factor': self.lr_factor.value(),
            'min_lr': self.min_lr.value()
        }
    
    def get_directories(self) -> Dict:
        """디렉토리 설정 수집"""
        return {
            'metadata_dir': self.metadata_dir.text(),
            'pkl_dir': self.pkl_dir.text(),
            'models_dir': self.models_dir.text(),
            'logs_dir': self.logs_dir.text()
        }
    
    def get_options(self) -> Dict:
        """옵션 설정 수집"""
        return {
            'strict_mode': self.strict_mode.isChecked(),
            'full_retrain': self.full_retrain.isChecked()
        }
    
    def start_training(self):
        """학습 시작"""
        # 설정 수집
        model_config = self.get_model_config()
        training_config = self.get_training_config()
        directories = self.get_directories()
        options = self.get_options()
        
        # 디렉토리 존재 확인
        required_dirs = ['metadata_dir', 'pkl_dir']
        for dir_key in required_dirs:
            if not Path(directories[dir_key]).exists():
                QMessageBox.warning(
                    self, "경고", 
                    f"디렉토리가 존재하지 않습니다: {directories[dir_key]}"
                )
                return
        
        # UI 상태 변경
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 무한 진행바
        self.log_text.clear()
        
        # 워커 스레드 시작
        self.worker = TrainingWorker(model_config, training_config, directories, options)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.fold_completed.connect(self.on_fold_completed)
        self.worker.training_completed.connect(self.on_training_completed)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.start()
    
    def stop_training(self):
        """학습 중지"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.reset_ui_state()
        self.update_progress("❌ 학습이 중지되었습니다.")
    
    def reset_ui_state(self):
        """UI 상태 리셋"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("준비됨")
    
    def update_progress(self, message: str):
        """진행 상황 업데이트"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.status_label.setText(message)
    
    def on_fold_completed(self, fold_idx: int, fold_result: Dict):
        """Fold 완료 처리"""
        self.update_progress(
            f"✅ Fold {fold_idx + 1} 완료 - Val MAE: {fold_result['best_val_mae']:.4f}"
        )
        self.results_widget.update_results(fold_idx, fold_result)
    
    def on_training_completed(self, cv_summary: Dict):
        """학습 완료 처리"""
        self.reset_ui_state()
        
        mean_mae = cv_summary['mean_val_mae']
        std_mae = cv_summary['std_val_mae']
        duration = cv_summary['total_duration_minutes']
        
        self.update_progress(
            f"🎉 교차검증 완료! 평균 Val MAE: {mean_mae:.4f} ± {std_mae:.4f}, "
            f"소요 시간: {duration:.1f}분"
        )
        
        self.results_widget.finalize_visualization(cv_summary)
        
        # 완료 메시지
        QMessageBox.information(
            self, "완료", 
            f"교차검증이 완료되었습니다!\n\n"
            f"평균 Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}m\n"
            f"총 소요 시간: {duration:.1f}분\n\n"
            f"결과는 models/cv_report.txt에서 확인할 수 있습니다."
        )
    
    def on_error(self, error_message: str):
        """에러 처리"""
        self.reset_ui_state()
        self.update_progress(f"❌ 에러 발생: {error_message}")
        QMessageBox.critical(self, "에러", f"학습 중 에러가 발생했습니다:\n\n{error_message}")
    
    def save_settings(self):
        """설정 저장"""
        settings = {
            'model_config': self.get_model_config(),
            'training_config': self.get_training_config(),
            'directories': self.get_directories(),
            'options': self.get_options()
        }
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "설정 저장", "tcn_settings.json", "JSON Files (*.json)"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "저장 완료", f"설정이 저장되었습니다: {filename}")
    
    def load_settings(self):
        """설정 로드"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "설정 로드", "", "JSON Files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                self.apply_settings(settings)
                QMessageBox.information(self, "로드 완료", f"설정이 로드되었습니다: {filename}")
                
            except Exception as e:
                QMessageBox.warning(self, "로드 실패", f"설정 로드에 실패했습니다:\n{str(e)}")
    
    def apply_settings(self, settings: Dict):
        """설정 적용"""
        # 모델 설정
        model_config = settings.get('model_config', {})
        self.tcn_filters.setValue(model_config.get('tcn_filters', 64))
        self.tcn_stacks.setValue(model_config.get('tcn_stacks', 4))
        self.dropout_rate.setValue(model_config.get('dropout_rate', 0.1))
        self.dense_units.setValue(model_config.get('dense_units', 64))
        self.learning_rate.setValue(model_config.get('learning_rate', 0.001))
        
        # 학습 설정
        training_config = settings.get('training_config', {})
        self.epochs.setValue(training_config.get('epochs', 100))
        self.batch_size.setValue(training_config.get('batch_size', 32))
        self.patience_early.setValue(training_config.get('patience_early', 10))
        self.patience_lr.setValue(training_config.get('patience_lr', 5))
        self.lr_factor.setValue(training_config.get('lr_factor', 0.5))
        self.min_lr.setValue(training_config.get('min_lr', 1e-6))
        
        # 디렉토리 설정
        directories = settings.get('directories', {})
        self.metadata_dir.setText(directories.get('metadata_dir', 'metadata'))
        self.pkl_dir.setText(directories.get('pkl_dir', 'stride_train_data_pkl'))
        self.models_dir.setText(directories.get('models_dir', 'models'))
        self.logs_dir.setText(directories.get('logs_dir', 'logs'))
        
        # 옵션 설정
        options = settings.get('options', {})
        self.strict_mode.setChecked(options.get('strict_mode', False))
        self.full_retrain.setChecked(options.get('full_retrain', False))
    
    def load_default_settings(self):
        """기본 설정 로드"""
        # 기본값들은 이미 위젯 생성시 설정됨
        pass

def main():
    """메인 실행 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성
    window = TCNTrainerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()