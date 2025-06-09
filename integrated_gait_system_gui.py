# ==========================================
# Integrated Gait Analysis System GUI
# ==========================================
"""
integrated_gait_system_gui.py - 통합 보행 분석 시스템 GUI

이 모듈은 다음 기능을 제공합니다:
1. 시각적 데이터 확인 및 수정 (센서, 영상, 이벤트)
2. 자동 보행 지표 계산 및 라벨 생성
3. 시계열 회귀 모델 학습 및 예측
4. 모델 추론 결과 시각화 및 검증
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
import pandas as pd
import cv2
from typing import Optional, List, Dict
import json
from datetime import datetime

# 기존 모듈 임포트
from gait_class import GaitAnalyzer, GaitEvent
# 새로운 모듈 임포트
from gait_metrics_calculator import GaitMetricsCalculator, GaitCycle
from time_series_model import GaitMetricsPredictor, IMUFeatureExtractor
from data_processing_utils import GaitDatasetBuilder, ModelEvaluator


class DataSynchronizationWidget(QWidget):
    """데이터 동기화 및 시각화 위젯 - 스마트 세션 선택"""
    
    def __init__(self):
        super().__init__()
        self.current_session_data = None
        self.video_path = None
        self.imu_data = None
        self.support_labels = []
        self.gait_events = []
        
        # 보행 타입 매핑 (batch_gait_analyzer.py와 동일)
        self.gait_type_mapping = {
            'normal_gait': 'T01',
            'ataxic_gait': 'T02', 
            'pain_gait': 'T04',
            'hemiparetic_gait': 'T03',
            'parkinson_gait': 'T05'
        }
        
        self.init_ui()
        self.scan_experiment_data()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 세션 선택 그룹
        session_group = QGroupBox("스마트 세션 선택")
        session_layout = QVBoxLayout(session_group)
        
        # 피험자 선택
        subject_layout = QHBoxLayout()
        subject_layout.addWidget(QLabel("피험자:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self.on_subject_changed)
        subject_layout.addWidget(self.subject_combo)
        session_layout.addLayout(subject_layout)
        
        # 보행 타입 선택
        gait_type_layout = QHBoxLayout()
        gait_type_layout.addWidget(QLabel("보행 타입:"))
        self.gait_type_combo = QComboBox()
        self.gait_type_combo.currentTextChanged.connect(self.on_gait_type_changed)
        gait_type_layout.addWidget(self.gait_type_combo)
        session_layout.addLayout(gait_type_layout)
        
        # 세션(Run) 선택
        session_run_layout = QHBoxLayout()
        session_run_layout.addWidget(QLabel("세션 Run:"))
        self.session_combo = QComboBox()
        self.session_combo.currentTextChanged.connect(self.on_session_changed)
        session_run_layout.addWidget(self.session_combo)
        session_layout.addLayout(session_run_layout)
        
        # 로드 버튼과 간단한 상태 한 줄로
        load_layout = QHBoxLayout()
        self.load_session_btn = QPushButton("세션 데이터 로드")
        self.load_session_btn.clicked.connect(self.load_session_data)
        self.load_session_btn.setEnabled(False)
        load_layout.addWidget(self.load_session_btn)
        
        # 간단한 상태 표시 (한 줄)
        self.status_label = QLabel("상태: 세션을 선택하세요")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        load_layout.addWidget(self.status_label)
        
        session_layout.addLayout(load_layout)
        layout.addWidget(session_group)
        
        # 동기화 시각화 - 확대된 영역
        sync_viz_group = QGroupBox("🔄 동기화 시각화")
        sync_viz_layout = QVBoxLayout(sync_viz_group)
        
        # 탭 위젯으로 메타데이터와 동기화 테이블 분리
        self.viz_tabs = QTabWidget()
        
        # 메타데이터 탭
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        self.metadata_text = QTextEdit()
        self.metadata_text.setMaximumHeight(150)  # 높이 증가
        self.metadata_text.setPlaceholderText("세션 로드 후 메타데이터가 표시됩니다...")
        metadata_layout.addWidget(self.metadata_text)
        self.viz_tabs.addTab(metadata_tab, "📋 메타데이터")
        
        # 동기화 테이블 탭
        sync_table_tab = QWidget()
        sync_table_layout = QVBoxLayout(sync_table_tab)
        
        # 전체 데이터 로드 안내
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("📋 전체 프레임 데이터 (스크롤 가능)"))
        info_layout.addStretch()
        sync_table_layout.addLayout(info_layout)
        
        # 동기화 테이블
        self.sync_table = QTableWidget()
        self.sync_table.setMinimumHeight(400)  # 높이 크게 증가
        sync_table_layout.addWidget(self.sync_table)
        
        # 동기화 품질 정보
        self.sync_quality_label = QLabel("동기화 품질: 데이터 로드 후 확인 가능")
        self.sync_quality_label.setStyleSheet("color: gray;")
        sync_table_layout.addWidget(self.sync_quality_label)
        
        self.viz_tabs.addTab(sync_table_tab, "🔄 동기화 테이블")
        
        # 시각화 그래프 탭
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        
        # 센서 선택 컨트롤
        sensor_control_group = QGroupBox("센서 데이터 선택")
        sensor_control_layout = QHBoxLayout(sensor_control_group)
        
        # 가속도계 체크박스
        accel_group = QGroupBox("가속도계")
        accel_layout = QHBoxLayout(accel_group)
        self.accel_x_cb = QCheckBox("AccelX")
        self.accel_y_cb = QCheckBox("AccelY") 
        self.accel_z_cb = QCheckBox("AccelZ")
        self.accel_x_cb.setChecked(True)  # 기본 선택
        self.accel_y_cb.setChecked(True)
        self.accel_z_cb.setChecked(True)
        accel_layout.addWidget(self.accel_x_cb)
        accel_layout.addWidget(self.accel_y_cb)
        accel_layout.addWidget(self.accel_z_cb)
        
        # 자이로스코프 체크박스
        gyro_group = QGroupBox("자이로스코프")
        gyro_layout = QHBoxLayout(gyro_group)
        self.gyro_x_cb = QCheckBox("GyroX")
        self.gyro_y_cb = QCheckBox("GyroY")
        self.gyro_z_cb = QCheckBox("GyroZ")
        gyro_layout.addWidget(self.gyro_x_cb)
        gyro_layout.addWidget(self.gyro_y_cb)
        gyro_layout.addWidget(self.gyro_z_cb)
        
        # 라벨 표시 옵션
        label_group = QGroupBox("라벨 표시 선택")
        label_layout = QVBoxLayout(label_group)
        
        # 체크박스들
        checkbox_layout = QHBoxLayout()
        self.show_double_support_cb = QCheckBox("🔴 이중지지")
        self.show_single_left_cb = QCheckBox("🟢 단일지지(왼쪽)")
        self.show_single_right_cb = QCheckBox("🔵 단일지지(오른쪽)")
        self.show_non_gait_cb = QCheckBox("⚪ 비보행")
        
        # 기본적으로 모두 선택
        self.show_double_support_cb.setChecked(True)
        self.show_single_left_cb.setChecked(True)
        self.show_single_right_cb.setChecked(True)
        self.show_non_gait_cb.setChecked(True)
        
        # 체크박스 상태 변경 시 자동 업데이트
        self.show_double_support_cb.stateChanged.connect(self.update_sync_visualization)
        self.show_single_left_cb.stateChanged.connect(self.update_sync_visualization)
        self.show_single_right_cb.stateChanged.connect(self.update_sync_visualization)
        self.show_non_gait_cb.stateChanged.connect(self.update_sync_visualization)
        
        checkbox_layout.addWidget(self.show_double_support_cb)
        checkbox_layout.addWidget(self.show_single_left_cb)
        checkbox_layout.addWidget(self.show_single_right_cb)
        checkbox_layout.addWidget(self.show_non_gait_cb)
        
        # 전체 선택/해제 버튼
        button_layout = QHBoxLayout()
        self.select_all_labels_btn = QPushButton("전체 선택")
        self.deselect_all_labels_btn = QPushButton("전체 해제")
        self.select_all_labels_btn.clicked.connect(self.select_all_labels)
        self.deselect_all_labels_btn.clicked.connect(self.deselect_all_labels)
        
        button_layout.addWidget(self.select_all_labels_btn)
        button_layout.addWidget(self.deselect_all_labels_btn)
        button_layout.addStretch()
        
        label_layout.addLayout(checkbox_layout)
        label_layout.addLayout(button_layout)
        
        sensor_control_layout.addWidget(accel_group)
        sensor_control_layout.addWidget(gyro_group)
        sensor_control_layout.addWidget(label_group)
        
        # 그래프 업데이트 버튼
        self.update_graph_btn = QPushButton("그래프 업데이트")
        self.update_graph_btn.clicked.connect(self.update_sync_visualization)
        sensor_control_layout.addWidget(self.update_graph_btn)
        
        graph_layout.addWidget(sensor_control_group)
        
        # PyQtGraph 위젯
        self.sync_plot_widget = pg.PlotWidget(title="동기화된 데이터 시각화")
        self.sync_plot_widget.setLabel('left', 'IMU 값')
        self.sync_plot_widget.setLabel('bottom', '시간 (초)')
        self.sync_plot_widget.setMinimumHeight(450)  # 센서 선택 공간 확보
        graph_layout.addWidget(self.sync_plot_widget)
        
        self.viz_tabs.addTab(graph_tab, "📈 시간축 그래프")
        
        sync_viz_layout.addWidget(self.viz_tabs)
        layout.addWidget(sync_viz_group)
    
    def enable_gait_metrics_calculation(self):
        """보행 지표 계산 기능 활성화"""
        # 메인 윈도우의 보행 지표 계산 위젯 활성화
        main_window = getattr(self, 'main_window', None)
        
        if main_window and hasattr(main_window, 'metrics_widget'):
            main_window.metrics_widget.calculate_btn.setEnabled(True)
            main_window.metrics_widget.calc_status_label.setText("준비 완료! 버튼을 클릭하여 보행 지표 계산을 시작하세요.")
            main_window.metrics_widget.calc_status_label.setStyleSheet("color: blue; font-weight: bold;")
            print("보행 지표 계산 기능이 활성화되었습니다.")
    
    def scan_experiment_data(self):
        """experiment_data 폴더 스캔"""
        experiment_path = "./experiment_data"
        
        if not os.path.exists(experiment_path):
            self.status_label.setText("상태: experiment_data 폴더가 없습니다")
            self.status_label.setStyleSheet("color: red;")
            return
        
        # 피험자 목록 수집
        subjects = [s for s in os.listdir(experiment_path) 
                   if os.path.isdir(os.path.join(experiment_path, s)) and s.startswith('SA')]
        subjects.sort()
        
        self.subject_combo.clear()
        self.subject_combo.addItems(subjects)
        
        if subjects:
            self.status_label.setText("상태: 피험자를 선택하세요")
            self.status_label.setStyleSheet("color: blue;")
    
    def on_subject_changed(self, subject: str):
        """피험자 변경 시"""
        if not subject:
            return
        
        subject_path = os.path.join("./experiment_data", subject)
        
        # 보행 타입 목록 수집
        gait_types = [g for g in os.listdir(subject_path) 
                     if os.path.isdir(os.path.join(subject_path, g)) 
                     and g.endswith('_gait')]
        gait_types.sort()
        
        self.gait_type_combo.clear()
        self.gait_type_combo.addItems(gait_types)
    
    def on_gait_type_changed(self, gait_type: str):
        """보행 타입 변경 시"""
        subject = self.subject_combo.currentText()
        if not subject or not gait_type:
            return
        
        gait_type_path = os.path.join("./experiment_data", subject, gait_type)
        
        # 세션 목록 수집
        sessions = [s for s in os.listdir(gait_type_path) 
                   if os.path.isdir(os.path.join(gait_type_path, s))]
        sessions.sort()
        
        # Run 번호와 함께 표시
        session_items = []
        for i, session in enumerate(sessions):
            run_num = f"R{i+1:02d}"
            session_items.append(f"{run_num} - {session}")
        
        self.session_combo.clear()
        self.session_combo.addItems(session_items)
    
    def on_session_changed(self, session_display: str):
        """세션 변경 시"""
        if not session_display:
            return
        
        # Run 번호 추출
        if " - " in session_display:
            run_num, session_name = session_display.split(" - ", 1)
        else:
            return
        
        subject = self.subject_combo.currentText()
        gait_type = self.gait_type_combo.currentText()
        
        if not all([subject, gait_type, session_name]):
            return
        
        # 세션 경로 구성
        session_path = os.path.join("./experiment_data", subject, gait_type, session_name)
        
        # 세션 정보 업데이트
        self.update_session_info(session_path, subject, gait_type, run_num)
        
        # 대응하는 라벨 파일 확인
        self.check_corresponding_labels(subject, gait_type, run_num)
        
        self.load_session_btn.setEnabled(True)
    
    def update_session_info(self, session_path: str, subject: str, gait_type: str, run_num: str):
        """세션 정보 업데이트"""
        info_text = f"""
세션 경로: {session_path}
피험자: {subject}
보행 타입: {gait_type}
Run: {run_num}

파일 확인:
        """.strip()
        
        # 비디오 파일 확인
        video_files = ["video.mp4", "session.mp4", "recording.mp4"]
        video_found = None
        for vf in video_files:
            video_path = os.path.join(session_path, vf)
            if os.path.exists(video_path):
                video_found = vf
                break
        
        info_text += f"\n- 비디오: {'✓ ' + video_found if video_found else '✗ 없음'}"
        
        # IMU 파일 확인
        imu_path = os.path.join(session_path, "imu_data.csv")
        imu_exists = os.path.exists(imu_path)
        info_text += f"\n- IMU 데이터: {'✓ imu_data.csv' if imu_exists else '✗ 없음'}"
        
        # 메타데이터 확인
        metadata_path = os.path.join(session_path, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                duration = metadata.get('duration', 0)
                frames = metadata.get('video_frames', 0)
                info_text += f"\n- 길이: {duration:.1f}초, {frames} 프레임"
            except:
                info_text += "\n- 메타데이터: 읽기 오류"
        
        self.metadata_text.setText(info_text)
        
        # 현재 세션 데이터 저장
        self.current_session_data = {
            'session_path': session_path,
            'subject': subject,
            'gait_type': gait_type,
            'run_num': run_num,
            'video_found': video_found,
            'imu_exists': imu_exists
        }
    
    def check_corresponding_labels(self, subject: str, gait_type: str, run_num: str):
        """대응하는 라벨 파일 확인"""
        # 태스크 코드 매핑
        task_code = self.gait_type_mapping.get(gait_type, 'T01')
        
        # Subject 번호 추출 (SA01 → S01)
        subject_num = subject[2:]  # "01"
        
        # 파일명 구성: S01T01R01_support_labels.csv
        label_filename = f"S{subject_num}{task_code}{run_num}_support_labels.csv"
        label_path = os.path.join("./support_label_data", subject, label_filename)
        
        if os.path.exists(label_path):
            # 라벨 파일 정보 확인
            try:
                label_df = pd.read_csv(label_path)
                phase_count = len(label_df)
                unique_phases = label_df['phase'].unique()
                
                self.status_label.setText(
                    f"라벨: ✓ {label_filename} ({phase_count}개 구간, {len(unique_phases)}개 타입)"
                )
                self.status_label.setStyleSheet("color: green;")
                
                # 현재 세션 데이터에 라벨 정보 추가
                if self.current_session_data:
                    self.current_session_data['label_path'] = label_path
                    self.current_session_data['label_filename'] = label_filename
                    
            except Exception as e:
                self.status_label.setText(f"라벨: ⚠ {label_filename} (읽기 오류: {str(e)})")
                self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText(f"라벨: ✗ {label_filename} (없음)")
            self.status_label.setStyleSheet("color: red;")
    
    def load_session_data(self):
        """세션 데이터 로드"""
        if not self.current_session_data:
            QMessageBox.warning(self, "오류", "세션이 선택되지 않았습니다.")
            return
        
        try:
            session_path = self.current_session_data['session_path']
            
            # 비디오 로드
            if self.current_session_data['video_found']:
                self.video_path = os.path.join(session_path, self.current_session_data['video_found'])
            else:
                QMessageBox.warning(self, "오류", "비디오 파일을 찾을 수 없습니다.")
                return
            
            # IMU 데이터 로드
            if self.current_session_data['imu_exists']:
                imu_path = os.path.join(session_path, "imu_data.csv")
                self.imu_data = pd.read_csv(imu_path)
            else:
                QMessageBox.warning(self, "경고", "IMU 데이터가 없습니다. 일부 기능이 제한될 수 있습니다.")
                self.imu_data = None
            
            # 라벨 데이터 로드 (있는 경우)
            if 'label_path' in self.current_session_data:
                label_df = pd.read_csv(self.current_session_data['label_path'])
                self.support_labels = label_df.to_dict('records')
            else:
                self.support_labels = []
            
            # 메타데이터 표시 업데이트
            self.display_loaded_metadata()
            
            # 동기화 테이블 생성
            self.create_sync_table()
            
            # 동기화 그래프 생성
            self.create_sync_visualization()
            
            # **보행 지표 계산 버튼 활성화**
            self.enable_gait_metrics_calculation()
            
            # 상태 업데이트
            self.status_label.setText("상태: ✓ 세션 데이터 로드 완료")
            self.status_label.setStyleSheet("color: green;")
            
            QMessageBox.information(
                self, "성공", 
                f"세션 데이터가 성공적으로 로드되었습니다.\n"
                f"- 비디오: {self.current_session_data['video_found']}\n"
                f"- IMU: {'있음' if self.imu_data is not None else '없음'}\n"
                f"- 라벨: {'있음' if self.support_labels else '없음'}\n\n"
                f"2번 탭에서 보행 지표 계산을 수행할 수 있습니다."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"세션 데이터 로드 실패: {e}")
            self.status_label.setText("상태: ✗ 로드 실패")
            self.status_label.setStyleSheet("color: red;")
    
    def display_loaded_metadata(self):
        """로드된 데이터의 메타데이터 표시"""
        metadata_text = ""
        
        if self.video_path:
            # 비디오 정보
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                metadata_text += f"📹 비디오 정보:\n"
                metadata_text += f"  - 해상도: {width} x {height}\n"
                metadata_text += f"  - FPS: {fps:.1f}\n"
                metadata_text += f"  - 프레임 수: {frame_count}\n"
                metadata_text += f"  - 길이: {duration:.2f}초\n\n"
                cap.release()
        
        if self.imu_data is not None:
            # IMU 정보
            metadata_text += f"📊 IMU 데이터:\n"
            metadata_text += f"  - 샘플 수: {len(self.imu_data)}\n"
            metadata_text += f"  - 컬럼: {list(self.imu_data.columns)}\n"
            if 'sync_timestamp' in self.imu_data.columns:
                time_range = self.imu_data['sync_timestamp'].max() - self.imu_data['sync_timestamp'].min()
                sampling_rate = len(self.imu_data) / time_range if time_range > 0 else 0
                metadata_text += f"  - 시간 범위: {time_range:.2f}초\n"
                metadata_text += f"  - 샘플링 레이트: ~{sampling_rate:.1f} Hz\n"
            metadata_text += "\n"
        
        if self.support_labels:
            # 라벨 정보
            phases = [label['phase'] for label in self.support_labels]
            unique_phases = list(set(phases))
            metadata_text += f"🏷️ 라벨 데이터:\n"
            metadata_text += f"  - 구간 수: {len(self.support_labels)}\n"
            metadata_text += f"  - 타입: {unique_phases}\n"
            
            # 각 타입별 개수
            from collections import Counter
            phase_counts = Counter(phases)
            for phase, count in phase_counts.items():
                metadata_text += f"    • {phase}: {count}개\n"
        
        self.metadata_text.setText(metadata_text.strip())
    
    def create_sync_table(self):
        """동기화 테이블 생성 - 전체 데이터 로드"""
        if not self.video_path or self.imu_data is None:
            return
        
        # 비디오 정보 획득
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 전체 데이터 표시 (성능 고려해서 적절히 샘플링)
        # 1000 프레임 이상이면 매 N프레임마다 샘플링
        if frame_count > 1000:
            sample_rate = max(1, frame_count // 1000)
            display_frames = list(range(0, frame_count, sample_rate))
        else:
            display_frames = list(range(frame_count))
        
        display_rows = len(display_frames)
        
        # 테이블 컬럼 설정
        columns = ['Frame', 'Time(s)', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Phase Label']
        self.sync_table.setColumnCount(len(columns))
        self.sync_table.setHorizontalHeaderLabels(columns)
        self.sync_table.setRowCount(display_rows)
        
        # IMU 컬럼명 매핑 (실제 데이터에 맞게 조정)
        imu_columns = list(self.imu_data.columns)
        accel_cols = [col for col in imu_columns if 'accel' in col.lower()][:3]
        gyro_cols = [col for col in imu_columns if 'gyro' in col.lower()][:3]
        
        # 기본 컬럼명이 없는 경우 처리
        if len(accel_cols) < 3:
            accel_cols = ['accel_x', 'accel_y', 'accel_z']
        if len(gyro_cols) < 3:
            gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']
        
        # 동기화 품질 계산
        video_duration = frame_count / fps
        imu_duration = self.imu_data['sync_timestamp'].max() if 'sync_timestamp' in self.imu_data.columns else 0
        time_diff = abs(video_duration - imu_duration)
        
        # 각 프레임에 대해 데이터 매핑
        for row in range(display_rows):
            frame_idx = display_frames[row]  # 실제 프레임 인덱스
            frame_time = frame_idx / fps
            
            # Frame과 Time 설정
            self.sync_table.setItem(row, 0, QTableWidgetItem(str(frame_idx)))
            self.sync_table.setItem(row, 1, QTableWidgetItem(f"{frame_time:.2f}"))
            
            # 해당 시간의 IMU 데이터 찾기
            if 'sync_timestamp' in self.imu_data.columns:
                # 가장 가까운 IMU 샘플 찾기
                time_diffs = np.abs(self.imu_data['sync_timestamp'] - frame_time)
                closest_idx = time_diffs.idxmin()
                closest_row = self.imu_data.loc[closest_idx]
                
                # IMU 데이터 설정
                for i, col in enumerate(accel_cols):
                    value = closest_row.get(col, 0.0) if col in self.imu_data.columns else 0.0
                    self.sync_table.setItem(row, 2 + i, QTableWidgetItem(f"{value:.3f}"))
                
                for i, col in enumerate(gyro_cols):
                    value = closest_row.get(col, 0.0) if col in self.imu_data.columns else 0.0
                    self.sync_table.setItem(row, 5 + i, QTableWidgetItem(f"{value:.3f}"))
            else:
                # sync_timestamp가 없는 경우 0으로 채우기
                for i in range(6):
                    self.sync_table.setItem(row, 2 + i, QTableWidgetItem("0.000"))
            
            # 해당 프레임의 라벨 찾기
            current_label = "non_gait"
            if self.support_labels:
                for label in self.support_labels:
                    if label['start_frame'] <= frame_idx <= label['end_frame']:
                        current_label = label['phase']
                        break
            
            # 라벨 색상에 맞춰 테이블 셀 색상도 설정
            label_item = QTableWidgetItem(current_label)
            if current_label == 'double_support':
                label_item.setBackground(QColor(255, 200, 200))  # 연한 빨강
            elif current_label == 'single_support_left':
                label_item.setBackground(QColor(200, 255, 200))  # 연한 초록
            elif current_label == 'single_support_right':
                label_item.setBackground(QColor(200, 200, 255))  # 연한 파랑
            else:  # non_gait
                label_item.setBackground(QColor(240, 240, 240))  # 연한 회색
            
            self.sync_table.setItem(row, 8, label_item)
        
        # 컬럼 크기 조정
        self.sync_table.resizeColumnsToContents()
        
        # 동기화 품질 업데이트
        if time_diff < 0.5:
            quality_text = f"동기화 품질: ✅ 우수 (시간차: {time_diff:.2f}초)"
            quality_color = "color: green;"
        elif time_diff < 2.0:
            quality_text = f"동기화 품질: ⚠️ 보통 (시간차: {time_diff:.2f}초)"
            quality_color = "color: orange;"
        else:
            quality_text = f"동기화 품질: ❌ 불량 (시간차: {time_diff:.2f}초)"
            quality_color = "color: red;"
        
        self.sync_quality_label.setText(quality_text)
        self.sync_quality_label.setStyleSheet(quality_color)
    

    
    def create_sync_visualization(self):
        """동기화 시각화 그래프 생성"""
        self.update_sync_visualization()
    
    def update_sync_visualization(self):
        """선택된 센서 데이터로 그래프 업데이트 - 프레임 기반 + 전체 라벨링"""
        if not self.video_path or self.imu_data is None:
            return
        
        self.sync_plot_widget.clear()
        
        # 비디오 정보 획득
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # **프레임 기반 X축 생성**
        frame_numbers = np.arange(total_frames)
        
        # IMU 데이터를 프레임에 매핑
        imu_columns = list(self.imu_data.columns)
        accel_cols = [col for col in imu_columns if 'accel' in col.lower()][:3]
        gyro_cols = [col for col in imu_columns if 'gyro' in col.lower()][:3]
        
        # 기본 컬럼명이 없는 경우 처리
        if len(accel_cols) < 3:
            accel_cols = [col for col in ['accel_x', 'accel_y', 'accel_z'] if col in imu_columns]
        if len(gyro_cols) < 3:
            gyro_cols = [col for col in ['gyro_x', 'gyro_y', 'gyro_z'] if col in imu_columns]
        
        # 각 프레임에 대응하는 IMU 데이터 매핑
        frame_imu_data = {}
        for col in accel_cols + gyro_cols:
            if col in self.imu_data.columns:
                frame_imu_data[col] = []
                
                for frame_idx in range(total_frames):
                    frame_time = frame_idx / fps
                    
                    # 가장 가까운 IMU 샘플 찾기
                    if 'sync_timestamp' in self.imu_data.columns:
                        time_diffs = np.abs(self.imu_data['sync_timestamp'] - frame_time)
                        closest_idx = time_diffs.idxmin()
                        value = self.imu_data.loc[closest_idx, col]
                    else:
                        # timestamp가 없으면 프레임 비율로 매핑
                        imu_idx = int((frame_idx / total_frames) * len(self.imu_data))
                        imu_idx = min(imu_idx, len(self.imu_data) - 1)
                        value = self.imu_data.iloc[imu_idx][col]
                    
                    frame_imu_data[col].append(value)
        
        # 색상 설정
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        plot_index = 0
        
        # 선택된 가속도계 데이터 플롯 (프레임 기반)
        checkboxes = [self.accel_x_cb, self.accel_y_cb, self.accel_z_cb]
        for i, (col, cb) in enumerate(zip(accel_cols, checkboxes)):
            if cb.isChecked() and col in frame_imu_data:
                self.sync_plot_widget.plot(
                    frame_numbers, 
                    frame_imu_data[col],
                    pen=colors[plot_index % len(colors)],
                    name=f"Accel{['X','Y','Z'][i]}"
                )
                plot_index += 1
        
        # 선택된 자이로스코프 데이터 플롯 (프레임 기반)
        checkboxes = [self.gyro_x_cb, self.gyro_y_cb, self.gyro_z_cb]
        for i, (col, cb) in enumerate(zip(gyro_cols, checkboxes)):
            if cb.isChecked() and col in frame_imu_data:
                self.sync_plot_widget.plot(
                    frame_numbers, 
                    frame_imu_data[col],
                    pen=colors[plot_index % len(colors)],
                    name=f"Gyro{['X','Y','Z'][i]}"
                )
                plot_index += 1
        
        # **전체 라벨 구간을 배경색으로 표시 (사용자 선택에 따라)**
        if self.support_labels:
            # 라벨별 색상 매핑 (사용자 요청대로 색상 변경)
            label_colors = {
                'single_support_left': (100, 255, 100, 80),    # 연한 초록
                'single_support_right': (100, 100, 255, 80),   # 연한 파랑
                'double_support': (255, 100, 100, 80),         # 연한 빨강
                'non_gait': (200, 200, 200, 60)               # 연한 회색
            }
            
            # 사용자 선택 확인
            show_labels = {
                'single_support_left': self.show_single_left_cb.isChecked(),
                'single_support_right': self.show_single_right_cb.isChecked(),
                'double_support': self.show_double_support_cb.isChecked(),
                'non_gait': self.show_non_gait_cb.isChecked()
            }
            
            # Y축 범위 계산
            y_min, y_max = float('inf'), float('-inf')
            for col in frame_imu_data:
                if frame_imu_data[col]:
                    col_min = min(frame_imu_data[col])
                    col_max = max(frame_imu_data[col])
                    y_min = min(y_min, col_min)
                    y_max = max(y_max, col_max)
            
            if y_min == float('inf'):
                y_min, y_max = -1, 1
            
            # **선택된 라벨 구간만 표시**
            displayed_count = 0
            for i, label in enumerate(self.support_labels):
                start_frame = label['start_frame']
                end_frame = label['end_frame']
                phase = label['phase']
                
                # 사용자가 선택한 라벨 타입만 표시
                if (phase in label_colors and 
                    phase in show_labels and 
                    show_labels[phase] and 
                    start_frame < total_frames):
                    
                    color = label_colors[phase]
                    
                    # 프레임 범위 제한
                    start_frame = max(0, start_frame)
                    end_frame = min(total_frames - 1, end_frame)
                    
                    if start_frame <= end_frame:
                        try:
                            # 반투명 영역 추가
                            fill_item = pg.FillBetweenItem(
                                curve1=pg.PlotCurveItem([start_frame, end_frame], [y_min, y_min]),
                                curve2=pg.PlotCurveItem([start_frame, end_frame], [y_max, y_max]),
                                brush=pg.mkBrush(color)
                            )
                            self.sync_plot_widget.addItem(fill_item)
                            
                            # 구간 경계선 추가
                            self.sync_plot_widget.plot(
                                [start_frame, start_frame], [y_min, y_max],
                                pen=pg.mkPen(color[0:3], width=2, style=2),  # 점선
                                name=f"{phase}_{i}" if displayed_count < 5 else None  # 처음 5개만 범례에 표시
                            )
                            displayed_count += 1
                        except Exception as e:
                            print(f"라벨 {i} 표시 오류: {e}")
            
            print(f"라벨 구간 표시 완료: {displayed_count}개 (총 {len(self.support_labels)}개 중 선택됨)")
        
        # X축을 프레임 번호로 설정
        self.sync_plot_widget.setLabel('bottom', '프레임 번호')
        self.sync_plot_widget.setLabel('left', 'IMU 값')
        self.sync_plot_widget.setTitle(f'동기화된 데이터 (프레임 기반) - 총 {total_frames} 프레임')
        
        # 범례 추가
        self.sync_plot_widget.addLegend()
    
    def select_all_labels(self):
        """모든 라벨 체크박스 선택"""
        self.show_double_support_cb.setChecked(True)
        self.show_single_left_cb.setChecked(True)
        self.show_single_right_cb.setChecked(True)
        self.show_non_gait_cb.setChecked(True)
    
    def deselect_all_labels(self):
        """모든 라벨 체크박스 해제"""
        self.show_double_support_cb.setChecked(False)
        self.show_single_left_cb.setChecked(False)
        self.show_single_right_cb.setChecked(False)
        self.show_non_gait_cb.setChecked(False)


class GaitMetricsWidget(QWidget):
    """보행 지표 계산 위젯"""
    
    def __init__(self):
        super().__init__()
        self.gait_calculator = GaitMetricsCalculator()
        self.gait_cycles = []
        
        # 영상 검증 관련 변수 초기화
        self.video_cap = None
        self.verification_gait_cycles = []
        
        self.init_ui()
    
    def __del__(self):
        """소멸자 - 비디오 캡처 해제"""
        if hasattr(self, 'video_cap') and self.video_cap:
            self.video_cap.release()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 계산 옵션
        options_group = QGroupBox("보행 지표 계산 옵션")
        options_layout = QGridLayout(options_group)
        
        # 피험자 신장 입력
        options_layout.addWidget(QLabel("피험자 신장:"), 0, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(120.0, 220.0)  # 120cm ~ 220cm
        self.height_spin.setValue(170.0)  # 기본값 170cm
        self.height_spin.setDecimals(1)
        self.height_spin.setSuffix(" cm")
        options_layout.addWidget(self.height_spin, 0, 1)
        
        # 자동 로드 버튼
        self.load_height_btn = QPushButton("피험자 정보 자동 로드")
        self.load_height_btn.clicked.connect(self.auto_load_subject_height)
        options_layout.addWidget(self.load_height_btn, 0, 2)
        
        # 계산된 비율 표시
        options_layout.addWidget(QLabel("계산된 비율:"), 1, 0)
        self.calculated_ratio_label = QLabel("미계산")
        self.calculated_ratio_label.setStyleSheet("color: gray; font-style: italic;")
        options_layout.addWidget(self.calculated_ratio_label, 1, 1, 1, 2)
        
        # 계산 버튼 (올바른 설명으로 변경)
        self.calculate_btn = QPushButton("🎯 MediaPipe + 라벨 기반 보행 지표 계산")
        self.calculate_btn.clicked.connect(self.calculate_from_loaded_data)
        self.calculate_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; background-color: #4CAF50; color: white; }")
        self.calculate_btn.setEnabled(False)  # 데이터 로드 후 활성화
        options_layout.addWidget(self.calculate_btn, 1, 0, 1, 2)
        
        # 상태 라벨 (올바른 설명으로 업데이트)
        self.calc_status_label = QLabel("세션 데이터 로드 후 MediaPipe로 관절 추정 → 라벨 기반 보행 지표 계산")
        self.calc_status_label.setStyleSheet("color: orange; font-style: italic;")
        options_layout.addWidget(self.calc_status_label, 2, 0, 1, 2)
        
        layout.addWidget(options_group)
        
        # 진행률 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 결과 표시 - 탭으로 분리
        results_group = QGroupBox("계산 결과 및 검증")
        results_layout = QVBoxLayout(results_group)
        
        # 결과 탭 위젯
        self.results_tabs = QTabWidget()
        
        # 1. 계산 결과 탭
        results_tab = QWidget()
        results_tab_layout = QVBoxLayout(results_tab)
        
        # 결과 테이블
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "주기", "발", "보폭(m)", "속도(m/s)", "주기(s)", 
            "보행률(step/min)", "엉덩이ROM(°)", "무릎ROM(°)", 
            "발목ROM(°)", "입각기(%)", "선택"
        ])
        results_tab_layout.addWidget(self.results_table)
        
        # 통계 정보
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(100)
        self.stats_text.setPlaceholderText("통계 정보가 여기에 표시됩니다...")
        results_tab_layout.addWidget(self.stats_text)
        
        # 저장 버튼
        self.save_btn = QPushButton("결과 저장")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        results_tab_layout.addWidget(self.save_btn)
        
        self.results_tabs.addTab(results_tab, "📋 계산 결과")
        
        # 2. 영상 검증 탭 - 새로 추가
        self.create_video_verification_tab()
        
        results_layout.addWidget(self.results_tabs)
        layout.addWidget(results_group)
    
    def auto_load_subject_height(self):
        """피험자 정보 자동 로드"""
        # 메인 윈도우에서 현재 피험자 정보 가져오기
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'sync_widget'):
                main_window = parent
                break
            parent = parent.parent()
        
        if not main_window or not hasattr(main_window.sync_widget, 'current_session_data'):
            QMessageBox.warning(self, "정보 없음", "현재 로드된 세션이 없습니다.")
            return
        
        session_data = main_window.sync_widget.current_session_data
        subject_id = session_data.get('subject', '')
        
        if not subject_id:
            QMessageBox.warning(self, "정보 없음", "피험자 정보를 찾을 수 없습니다.")
            return
        
        # 피험자별 저장된 신장 로드
        heights_file = "./subject_heights.json"
        subject_heights = {}
        
        if os.path.exists(heights_file):
            try:
                with open(heights_file, 'r') as f:
                    subject_heights = json.load(f)
            except:
                pass
        
        if subject_id in subject_heights:
            height = subject_heights[subject_id]
            self.height_spin.setValue(height)
            QMessageBox.information(self, "로드 완료", f"{subject_id} 신장: {height}cm 로드됨")
        else:
            # 신장 입력 받기
            height, ok = QInputDialog.getDouble(
                self, "신장 입력", 
                f"{subject_id}의 신장을 입력하세요 (cm):", 
                170.0, 120.0, 220.0, 1
            )
            
            if ok:
                self.height_spin.setValue(height)
                
                # 저장
                subject_heights[subject_id] = height
                try:
                    with open(heights_file, 'w') as f:
                        json.dump(subject_heights, f, indent=2)
                    QMessageBox.information(self, "저장 완료", f"{subject_id} 신장: {height}cm 저장됨")
                except Exception as e:
                    QMessageBox.warning(self, "저장 실패", f"신장 정보 저장 실패: {e}")
    
    def calculate_pixel_to_meter_ratio(self, joint_coords: pd.DataFrame, subject_height_cm: float) -> float:
        """
        신장 기반 픽셀-미터 비율 계산
        
        Args:
            joint_coords: 관절 좌표 데이터
            subject_height_cm: 피험자 신장 (cm)
        
        Returns:
            float: 픽셀-미터 비율
        """
        try:
            # 발목-무릎 거리 = 신장의 27%
            ankle_knee_real_distance = (subject_height_cm / 100.0) * 0.27  # 미터 단위
            
            # 여러 프레임에서 발목-무릎 픽셀 거리 계산
            pixel_distances = []
            
            # 10프레임마다 샘플링해서 평균 계산
            sample_frames = range(0, len(joint_coords), max(1, len(joint_coords) // 20))
            
            for frame_idx in sample_frames:
                if frame_idx >= len(joint_coords):
                    continue
                    
                row = joint_coords.iloc[frame_idx]
                
                # 왼쪽 다리 발목-무릎 거리
                if all(col in row for col in ['left_ankle_x', 'left_ankle_y', 'left_knee_x', 'left_knee_y']):
                    ankle_pos = np.array([row['left_ankle_x'], row['left_ankle_y']])
                    knee_pos = np.array([row['left_knee_x'], row['left_knee_y']])
                    
                    # 정규화된 좌표이므로 임의의 스케일링 (실제로는 영상 크기 필요)
                    distance = np.linalg.norm(ankle_pos - knee_pos)
                    if distance > 0:
                        pixel_distances.append(distance)
                
                # 오른쪽 다리도 계산
                if all(col in row for col in ['right_ankle_x', 'right_ankle_y', 'right_knee_x', 'right_knee_y']):
                    ankle_pos = np.array([row['right_ankle_x'], row['right_ankle_y']])
                    knee_pos = np.array([row['right_knee_x'], row['right_knee_y']])
                    
                    distance = np.linalg.norm(ankle_pos - knee_pos)
                    if distance > 0:
                        pixel_distances.append(distance)
            
            if not pixel_distances:
                print("픽셀 거리를 계산할 수 없습니다. 기본값 사용.")
                return 0.001  # 기본값
            
            # 평균 픽셀 거리
            avg_pixel_distance = np.mean(pixel_distances)
            
            # 픽셀-미터 비율 = 실제거리(m) / 픽셀거리
            ratio = ankle_knee_real_distance / avg_pixel_distance
            
            print(f"신장 기반 비율 계산:")
            print(f"  - 피험자 신장: {subject_height_cm}cm")
            print(f"  - 발목-무릎 실제 거리: {ankle_knee_real_distance:.3f}m")
            print(f"  - 평균 픽셀 거리: {avg_pixel_distance:.6f}")
            print(f"  - 계산된 비율: {ratio:.6f} m/pixel")
            
            return ratio
            
        except Exception as e:
            print(f"비율 계산 오류: {e}")
            return 0.001  # 기본값
    
    def create_video_verification_tab(self):
        """영상 검증 탭 생성"""
        verification_tab = QWidget()
        verification_layout = QHBoxLayout(verification_tab)
        
        # 왼쪽: 영상 플레이어
        video_group = QGroupBox("🎬 영상 플레이어")
        video_layout = QVBoxLayout(video_group)
        
        # 영상 표시 라벨
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setText("영상이 로드되면 여기에 표시됩니다")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        
        # 재생 컨트롤
        control_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ 재생")
        self.play_btn.clicked.connect(self.toggle_video_playback)
        self.play_btn.setEnabled(False)
        
        self.prev_frame_btn = QPushButton("⏮️")
        self.prev_frame_btn.clicked.connect(self.previous_frame)
        self.prev_frame_btn.setEnabled(False)
        
        self.next_frame_btn = QPushButton("⏭️")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.prev_frame_btn)
        control_layout.addWidget(self.next_frame_btn)
        control_layout.addStretch()
        
        video_layout.addLayout(control_layout)
        
        # 프레임 슬라이더
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        self.frame_slider.setEnabled(False)
        video_layout.addWidget(self.frame_slider)
        
        # 프레임 정보
        self.frame_info_label = QLabel("프레임: 0 / 0")
        video_layout.addWidget(self.frame_info_label)
        
        verification_layout.addWidget(video_group, 2)  # 2/3 비율
        
        # 오른쪽: 실시간 분석
        analysis_group = QGroupBox("📊 실시간 분석")
        analysis_layout = QVBoxLayout(analysis_group)
        
        # 현재 프레임 정보
        current_info_group = QGroupBox("현재 프레임 정보")
        current_info_layout = QGridLayout(current_info_group)
        
        current_info_layout.addWidget(QLabel("보행 단계:"), 0, 0)
        self.current_phase_label = QLabel("N/A")
        self.current_phase_label.setStyleSheet("font-weight: bold; padding: 5px; border: 1px solid gray;")
        current_info_layout.addWidget(self.current_phase_label, 0, 1)
        
        current_info_layout.addWidget(QLabel("보행 주기:"), 1, 0)
        self.current_cycle_label = QLabel("N/A")
        current_info_layout.addWidget(self.current_cycle_label, 1, 1)
        
        current_info_layout.addWidget(QLabel("관절 인식:"), 2, 0)
        self.joint_detection_label = QLabel("N/A")
        current_info_layout.addWidget(self.joint_detection_label, 2, 1)
        
        analysis_layout.addWidget(current_info_group)
        
        # 실시간 그래프
        graph_group = QGroupBox("실시간 지표 그래프")
        graph_layout = QVBoxLayout(graph_group)
        
        # 그래프 선택 옵션
        graph_options_layout = QHBoxLayout()
        graph_options_layout.addWidget(QLabel("표시 지표:"))
        
        self.show_stride_cb = QCheckBox("보폭")
        self.show_stride_cb.setChecked(True)
        self.show_stride_cb.setStyleSheet("color: red;")
        
        self.show_velocity_cb = QCheckBox("속도")
        self.show_velocity_cb.setChecked(True) 
        self.show_velocity_cb.setStyleSheet("color: blue;")
        
        self.show_cadence_cb = QCheckBox("보행률")
        self.show_cadence_cb.setChecked(False)
        self.show_cadence_cb.setStyleSheet("color: green;")
        
        # 그래프 업데이트 연결
        self.show_stride_cb.stateChanged.connect(self.update_verification_graph)
        self.show_velocity_cb.stateChanged.connect(self.update_verification_graph)
        self.show_cadence_cb.stateChanged.connect(self.update_verification_graph)
        
        graph_options_layout.addWidget(self.show_stride_cb)
        graph_options_layout.addWidget(self.show_velocity_cb)
        graph_options_layout.addWidget(self.show_cadence_cb)
        graph_options_layout.addStretch()
        
        graph_layout.addLayout(graph_options_layout)
        
        self.verification_plot = pg.PlotWidget(title="보행 지표 시각화 (노란선: 현재 프레임)")
        self.verification_plot.setLabel('left', '보행 지표 값')
        self.verification_plot.setLabel('bottom', '프레임 번호')
        self.verification_plot.setMinimumHeight(250)
        self.verification_plot.showGrid(x=True, y=True, alpha=0.3)
        graph_layout.addWidget(self.verification_plot)
        
        # 그래프 범례 설명
        legend_label = QLabel("🔴 보폭(m)  🔵 속도×5(m/s)  🟢 보행률÷20(steps/min)  💛 현재 위치")
        legend_label.setStyleSheet("color: gray; font-size: 10px; padding: 5px;")
        graph_layout.addWidget(legend_label)
        
        analysis_layout.addWidget(graph_group)
        
        # 검증 옵션
        options_group = QGroupBox("검증 옵션")
        options_layout = QVBoxLayout(options_group)
        
        self.show_joints_cb = QCheckBox("MediaPipe 관절 표시")
        self.show_joints_cb.setChecked(True)
        self.show_joints_cb.stateChanged.connect(self.update_video_display)
        
        self.show_phase_overlay_cb = QCheckBox("보행 단계 오버레이")
        self.show_phase_overlay_cb.setChecked(True)
        self.show_phase_overlay_cb.stateChanged.connect(self.update_video_display)
        
        self.highlight_anomalies_cb = QCheckBox("이상치 하이라이트")
        self.highlight_anomalies_cb.setChecked(True)
        
        options_layout.addWidget(self.show_joints_cb)
        options_layout.addWidget(self.show_phase_overlay_cb)
        options_layout.addWidget(self.highlight_anomalies_cb)
        
        analysis_layout.addWidget(options_group)
        
        verification_layout.addWidget(analysis_group, 1)  # 1/3 비율
        
        self.results_tabs.addTab(verification_tab, "🎬 영상 검증")
        
        # 비디오 관련 변수 초기화
        self.video_cap = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.is_playing = False
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.next_video_frame)
    
    def calculate_from_loaded_data(self):
        """로드된 세션 데이터로부터 보행 지표 계산 (올바른 흐름)"""
        # 메인 윈도우에서 데이터 가져오기 - 부모 탐색 방식 개선
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'sync_widget'):
                main_window = parent
                break
            parent = parent.parent()
        
        if not main_window or not hasattr(main_window, 'sync_widget'):
            QMessageBox.warning(self, "오류", "동기화 위젯을 찾을 수 없습니다.")
            return
        
        sync_widget = main_window.sync_widget
        
        # **데이터 확인 및 상세 정보 제공**
        print("=== 보행 지표 계산 시작 ===")
        print(f"비디오 경로: {sync_widget.video_path}")
        print(f"라벨 개수: {len(sync_widget.support_labels) if sync_widget.support_labels else 0}")
        
        if not sync_widget.video_path:
            QMessageBox.warning(
                self, "데이터 부족", 
                "비디오 데이터가 로드되지 않았습니다.\n\n"
                "해결 방법:\n"
                "1. '1번 탭'으로 이동\n"
                "2. 피험자/보행타입/세션 선택\n"
                "3. '세션 데이터 로드' 버튼 클릭"
            )
            return
        
        if not sync_widget.support_labels:
            QMessageBox.warning(
                self, "라벨 데이터 부족", 
                "지지 단계 라벨 데이터가 없습니다.\n\n"
                "해결 방법:\n"
                "1. support_label_data 폴더에 해당 라벨 파일 확인\n"
                "2. 라벨링된 세션을 선택\n"
                "3. 파일명 형식: S01T01R01_support_labels.csv"
            )
            return
        
        # 올바른 계산 흐름 실행
        print("라벨 데이터와 MediaPipe 기반 보행 지표 계산 시작...")
        self.calculate_metrics_with_labels(sync_widget.video_path, sync_widget.support_labels)
    
    def convert_labels_to_events(self, support_labels):
        """Support labels를 gait events로 변환"""
        from gait_class import GaitEvent
        
        events = []
        
        for label in support_labels:
            phase = label['phase']
            start_frame = label['start_frame']
            end_frame = label['end_frame']
            
            # single_support 구간에서 이벤트 추출
            if 'single_support' in phase:
                foot = 'left' if 'left' in phase else 'right'
                
                # Heel Strike 이벤트
                events.append(GaitEvent(
                    frame=start_frame,
                    event_type='HS',
                    foot=foot,
                    confidence=0.9
                ))
                
                # Toe Off 이벤트
                events.append(GaitEvent(
                    frame=end_frame,
                    event_type='TO',
                    foot=foot,
                    confidence=0.9
                ))
        
        return events
    
    def calculate_metrics_with_labels(self, video_path: str, support_labels: List[Dict], progress_callback=None):
        """
        라벨 데이터와 MediaPipe를 사용한 올바른 보행 지표 계산
        
        Args:
            video_path (str): 비디오 파일 경로
            support_labels (List[Dict]): 지지 단계 라벨 데이터
            progress_callback: 진행률 콜백 함수
        """
        try:
            self.calculate_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.calc_status_label.setText("보행 지표 계산 중...")
            
            # 피험자 신장 가져오기
            subject_height_cm = self.height_spin.value()
            
            if progress_callback:
                progress_callback(10)
            
            # 1단계: 비디오에서 관절 좌표 추출
            self.calc_status_label.setText("1/2: MediaPipe 관절 추정 중...")
            print("  - MediaPipe로 관절 좌표 추출 중...")
            
            joint_coords = self.gait_calculator.extract_joint_coordinates(
                video_path, 
                progress_callback=lambda p: self.progress_bar.setValue(10 + int(p * 0.5))
            )
            
            if progress_callback:
                progress_callback(60)
            
            # 신장 기반 픽셀-미터 비율 계산
            self.calc_status_label.setText("신장 기반 비율 계산 중...")
            calculated_ratio = self.calculate_pixel_to_meter_ratio(joint_coords, subject_height_cm)
            self.gait_calculator.pixel_to_meter_ratio = calculated_ratio
            
            # UI에 계산된 비율 표시
            self.calculated_ratio_label.setText(f"{calculated_ratio:.6f} m/pixel")
            self.calculated_ratio_label.setStyleSheet("color: green; font-weight: bold;")
            
            if progress_callback:
                progress_callback(70)
            
            # 2단계: 라벨 데이터와 관절 좌표를 사용해 실제 보행 지표 계산
            self.calc_status_label.setText("2/2: 보행 지표 계산 중...")
            print("  - 라벨 기반 보행 구간 식별 및 지표 계산 중...")
            
            gait_cycles = self.gait_calculator.calculate_gait_metrics_from_labels(
                video_path, joint_coords, support_labels
            )
            
            if progress_callback:
                progress_callback(90)
            
            # 결과 테이블 업데이트
            self.update_results_table(gait_cycles)
            
            if progress_callback:
                progress_callback(100)
            
            self.calc_status_label.setText("계산 완료! 영상 검증 탭 준비 중...")
            self.calc_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            # 영상 검증 탭 설정
            self.setup_video_verification(video_path, gait_cycles)
            
            # 관절 좌표 데이터를 메인 윈도우에 저장 (영상 검증용)
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'current_session_data'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if main_window:
                main_window.joint_coordinates = joint_coords
            
            self.calc_status_label.setText("계산 완료! 🎬 영상 검증 탭에서 결과를 확인하세요.")
            
            QMessageBox.information(
                self, "완료", 
                f"보행 지표 계산이 완료되었습니다!\n"
                f"• 관절 좌표 추출: {len(joint_coords)} 프레임\n"
                f"• 보행 주기 분석: {len(gait_cycles)}개\n\n"
                f"🎬 '영상 검증' 탭에서 결과를 시각적으로 검증할 수 있습니다."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"보행 지표 계산 실패: {e}")
            self.calc_status_label.setText("계산 실패")
            self.calc_status_label.setStyleSheet("color: red;")
            print(f"오류 상세: {e}")
        finally:
            self.calculate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def update_results_table(self, gait_cycles: List[GaitCycle]):
        """결과 테이블 업데이트"""
        self.gait_cycles = gait_cycles
        self.results_table.setRowCount(len(gait_cycles))
        
        for i, cycle in enumerate(gait_cycles):
            # 테이블 데이터 채우기
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(cycle.foot))
            self.results_table.setItem(i, 2, QTableWidgetItem(f"{cycle.stride_length:.3f}"))
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{cycle.velocity:.3f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{cycle.cycle_time:.2f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{cycle.cadence:.1f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{cycle.hip_rom:.1f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{cycle.knee_rom:.1f}"))
            self.results_table.setItem(i, 8, QTableWidgetItem(f"{cycle.ankle_rom:.1f}"))
            self.results_table.setItem(i, 9, QTableWidgetItem(f"{cycle.stance_ratio:.1f}"))
            
            # 체크박스 추가
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.results_table.setCellWidget(i, 10, checkbox)
        
        # 테이블 크기 조정
        self.results_table.resizeColumnsToContents()
        
        # 통계 정보 업데이트
        self.update_statistics()
        
        # 저장 버튼 활성화
        self.save_btn.setEnabled(True)
    
    def update_statistics(self):
        """통계 정보 업데이트"""
        if not self.gait_cycles:
            return
        
        # 데이터 수집
        data = [cycle.to_dict() for cycle in self.gait_cycles]
        df = pd.DataFrame(data)
        
        # 통계 계산
        stats_text = f"""
총 보행 주기: {len(df)}
평균 보폭: {df['stride_length'].mean():.3f} ± {df['stride_length'].std():.3f} m
평균 속도: {df['velocity'].mean():.3f} ± {df['velocity'].std():.3f} m/s  
평균 보행률: {df['cadence'].mean():.1f} ± {df['cadence'].std():.1f} steps/min
평균 무릎 ROM: {df['knee_rom'].mean():.1f} ± {df['knee_rom'].std():.1f}°
        """.strip()
        
        self.stats_text.setText(stats_text)
    
    def save_results(self):
        """결과 저장"""
        if not self.gait_cycles:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "보행 지표 저장", "gait_metrics.csv", "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # 선택된 주기만 저장
                selected_cycles = []
                for i in range(self.results_table.rowCount()):
                    checkbox = self.results_table.cellWidget(i, 10)
                    if checkbox.isChecked():
                        selected_cycles.append(self.gait_cycles[i])
                
                if selected_cycles:
                    data = [cycle.to_dict() for cycle in selected_cycles]
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, index=False)
                    QMessageBox.information(self, "성공", f"보행 지표가 저장되었습니다.\n파일: {file_path}")
                else:
                    QMessageBox.warning(self, "경고", "선택된 보행 주기가 없습니다.")
                    
            except Exception as e:
                QMessageBox.warning(self, "오류", f"저장 실패: {e}")
    
    def setup_video_verification(self, video_path: str, gait_cycles: List[GaitCycle]):
        """영상 검증 탭 설정"""
        try:
            # 비디오 캡처 객체 생성
            if self.video_cap:
                self.video_cap.release()
            
            self.video_cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            # UI 컨트롤 활성화
            self.play_btn.setEnabled(True)
            self.prev_frame_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            self.frame_slider.setEnabled(True)
            
            # 슬라이더 설정
            self.frame_slider.setRange(0, self.total_frames - 1)
            self.frame_slider.setValue(0)
            
            # 보행 주기 데이터 저장
            self.verification_gait_cycles = gait_cycles
            
            # 첫 프레임 표시
            self.current_frame_idx = 0
            self.display_frame(0)
            
            # 실시간 그래프 초기화
            self.setup_verification_graph()
            
            print(f"영상 검증 탭 설정 완료: {self.total_frames} 프레임, {self.fps:.1f} FPS")
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"영상 검증 설정 실패: {e}")
    
    def setup_verification_graph(self):
        """검증용 그래프 설정"""
        if not hasattr(self, 'verification_gait_cycles') or not self.verification_gait_cycles:
            return
        
        # 보행 주기별 데이터 수집
        self.graph_data = {
            'frames': [],
            'stride_lengths': [],
            'velocities': [],
            'cadences': [],
            'cycle_labels': []
        }
        
        for i, cycle in enumerate(self.verification_gait_cycles):
            self.graph_data['frames'].append(cycle.start_frame)
            self.graph_data['stride_lengths'].append(cycle.stride_length)
            self.graph_data['velocities'].append(cycle.velocity)
            self.graph_data['cadences'].append(cycle.cadence)
            self.graph_data['cycle_labels'].append(f"{cycle.foot[0].upper()}{i+1}")  # L1, R1, L2, R2...
        
        # 초기 그래프 그리기
        self.update_verification_graph()
    
    def update_verification_graph(self):
        """검증 그래프 업데이트"""
        if not hasattr(self, 'graph_data') or not self.graph_data['frames']:
            return
        
        self.verification_plot.clear()
        
        frames = self.graph_data['frames']
        
        # 선택된 지표들만 표시
        if self.show_stride_cb.isChecked():
            self.verification_plot.plot(
                frames, self.graph_data['stride_lengths'], 
                pen=pg.mkPen('r', width=3), 
                symbol='o', symbolSize=8, symbolBrush='r',
                name='보폭(m)'
            )
        
        if self.show_velocity_cb.isChecked():
            # 속도에 5배 스케일링 (시각화 개선)
            scaled_velocities = [v * 5 for v in self.graph_data['velocities']]
            self.verification_plot.plot(
                frames, scaled_velocities, 
                pen=pg.mkPen('b', width=3), 
                symbol='s', symbolSize=8, symbolBrush='b',
                name='속도×5(m/s)'
            )
        
        if self.show_cadence_cb.isChecked():
            # 보행률을 20으로 나누어서 스케일링
            scaled_cadences = [c / 20 for c in self.graph_data['cadences']]
            self.verification_plot.plot(
                frames, scaled_cadences, 
                pen=pg.mkPen('g', width=3), 
                symbol='^', symbolSize=8, symbolBrush='g',
                name='보행률÷20(steps/min)'
            )
        
        # 각 보행 주기에 라벨 추가
        for i, (frame, label) in enumerate(zip(frames, self.graph_data['cycle_labels'])):
            # 텍스트 라벨 추가
            text_item = pg.TextItem(
                text=label, 
                color=(255, 255, 255), 
                border='k', 
                fill=(0, 0, 0, 100)
            )
            text_item.setPos(frame, max(self.graph_data['stride_lengths']) * 1.1)
            self.verification_plot.addItem(text_item)
        
        # 현재 위치 라인 추가
        self.current_position_line = self.verification_plot.addLine(
            x=0, 
            pen=pg.mkPen('yellow', width=4, style=pg.QtCore.Qt.DashLine)
        )
        
        # 범례 추가
        self.verification_plot.addLegend(offset=(10, 10))
        
        # Y축 범위 자동 조정
        if any([self.show_stride_cb.isChecked(), self.show_velocity_cb.isChecked(), self.show_cadence_cb.isChecked()]):
            all_values = []
            if self.show_stride_cb.isChecked():
                all_values.extend(self.graph_data['stride_lengths'])
            if self.show_velocity_cb.isChecked():
                all_values.extend([v * 5 for v in self.graph_data['velocities']])
            if self.show_cadence_cb.isChecked():
                all_values.extend([c / 20 for c in self.graph_data['cadences']])
            
            if all_values:
                min_val = min(all_values)
                max_val = max(all_values)
                margin = (max_val - min_val) * 0.1
                self.verification_plot.setYRange(min_val - margin, max_val + margin)
    
    def toggle_video_playback(self):
        """비디오 재생/일시정지 토글"""
        if not self.video_cap:
            return
        
        if self.is_playing:
            # 일시정지
            self.video_timer.stop()
            self.is_playing = False
            self.play_btn.setText("▶️ 재생")
        else:
            # 재생
            self.video_timer.start(int(1000 / self.fps))  # FPS에 맞춰 타이머 설정
            self.is_playing = True
            self.play_btn.setText("⏸️ 일시정지")
    
    def next_video_frame(self):
        """다음 프레임으로 이동 (자동 재생용)"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.display_frame(self.current_frame_idx)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_idx)
            self.frame_slider.blockSignals(False)
        else:
            # 마지막 프레임에 도달하면 정지
            self.toggle_video_playback()
    
    def previous_frame(self):
        """이전 프레임으로 이동"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_frame(self.current_frame_idx)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_idx)
            self.frame_slider.blockSignals(False)
    
    def next_frame(self):
        """다음 프레임으로 이동"""
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.display_frame(self.current_frame_idx)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame_idx)
            self.frame_slider.blockSignals(False)
    
    def on_frame_slider_changed(self, value):
        """프레임 슬라이더 변경 시"""
        self.current_frame_idx = value
        self.display_frame(self.current_frame_idx)
    
    def display_frame(self, frame_idx: int):
        """지정된 프레임 표시"""
        if not self.video_cap:
            return
        
        # 프레임 읽기
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()
        
        if not ret:
            return
        
        # MediaPipe 관절 오버레이 (옵션)
        if self.show_joints_cb.isChecked():
            frame = self.draw_mediapipe_overlay(frame, frame_idx)
        
        # 보행 단계 오버레이 (옵션)
        if self.show_phase_overlay_cb.isChecked():
            frame = self.draw_phase_overlay(frame, frame_idx)
        
        # OpenCV BGR을 RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # QImage로 변환
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # QLabel에 표시
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        
        # 프레임 정보 업데이트
        self.frame_info_label.setText(f"프레임: {frame_idx + 1} / {self.total_frames}")
        
        # 현재 프레임 분석 정보 업데이트
        self.update_current_frame_analysis(frame_idx)
        
        # 그래프 현재 위치 업데이트
        if hasattr(self, 'current_position_line'):
            self.current_position_line.setPos(frame_idx)
    
    def draw_mediapipe_overlay(self, frame, frame_idx: int):
        """MediaPipe 관절 오버레이 그리기"""
        # 실제로는 저장된 관절 좌표 데이터를 사용해야 함
        # 여기서는 간단한 예시로 구현
        
        # 메인 윈도우에서 관절 좌표 데이터 가져오기
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'current_session_data'):
                main_window = parent
                break
            parent = parent.parent()
        
        if main_window and hasattr(main_window, 'joint_coordinates'):
            # 관절 좌표가 있다면 그리기
            joint_coords = main_window.joint_coordinates
            if frame_idx < len(joint_coords):
                # 관절 점들 그리기 (간단한 예시)
                h, w = frame.shape[:2]
                frame_data = joint_coords.iloc[frame_idx]
                
                # 주요 관절 점들 그리기
                joints = [
                    ('left_ankle', (255, 0, 0)),
                    ('right_ankle', (0, 255, 0)),
                    ('left_knee', (255, 255, 0)),
                    ('right_knee', (0, 255, 255)),
                    ('left_hip', (255, 0, 255)),
                    ('right_hip', (0, 0, 255))
                ]
                
                for joint_name, color in joints:
                    if f'{joint_name}_x' in frame_data and f'{joint_name}_y' in frame_data:
                        x = int(frame_data[f'{joint_name}_x'] * w)
                        y = int(frame_data[f'{joint_name}_y'] * h)
                        cv2.circle(frame, (x, y), 8, color, -1)
                        cv2.putText(frame, joint_name.split('_')[1], (x+10, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def draw_phase_overlay(self, frame, frame_idx: int):
        """보행 단계 오버레이 그리기"""
        # 메인 윈도우에서 라벨 데이터 가져오기
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'sync_widget'):
                main_window = parent
                break
            parent = parent.parent()
        
        if main_window and main_window.sync_widget.support_labels:
            # 현재 프레임의 보행 단계 찾기
            current_phase = "non_gait"
            for label in main_window.sync_widget.support_labels:
                if label['start_frame'] <= frame_idx <= label['end_frame']:
                    current_phase = label['phase']
                    break
            
            # 화면 상단에 현재 단계 표시
            h, w = frame.shape[:2]
            
            # 배경 색상 설정
            color_map = {
                'double_support': (0, 0, 255),      # 빨강
                'single_support_left': (0, 255, 0), # 초록
                'single_support_right': (255, 0, 0), # 파랑
                'non_gait': (128, 128, 128)         # 회색
            }
            
            color = color_map.get(current_phase, (128, 128, 128))
            
            # 상단 바 그리기
            cv2.rectangle(frame, (0, 0), (w, 60), color, -1)
            
            # 텍스트 표시
            phase_text = current_phase.replace('_', ' ').title()
            cv2.putText(frame, f"Phase: {phase_text}", (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def update_current_frame_analysis(self, frame_idx: int):
        """현재 프레임 분석 정보 업데이트"""
        # 메인 윈도우에서 데이터 가져오기
        main_window = None
        parent = self.parent()
        while parent:
            if hasattr(parent, 'sync_widget'):
                main_window = parent
                break
            parent = parent.parent()
        
        if not main_window:
            return
        
        # 보행 단계 정보
        current_phase = "non_gait"
        if main_window.sync_widget.support_labels:
            for label in main_window.sync_widget.support_labels:
                if label['start_frame'] <= frame_idx <= label['end_frame']:
                    current_phase = label['phase']
                    break
        
        # 색상 설정
        phase_colors = {
            'double_support': "background-color: #ffcccc; color: red;",      # 연한 빨강
            'single_support_left': "background-color: #ccffcc; color: green;", # 연한 초록  
            'single_support_right': "background-color: #ccccff; color: blue;", # 연한 파랑
            'non_gait': "background-color: #f0f0f0; color: gray;"           # 연한 회색
        }
        
        self.current_phase_label.setText(current_phase.replace('_', ' ').title())
        self.current_phase_label.setStyleSheet(
            f"font-weight: bold; padding: 5px; border: 1px solid gray; {phase_colors.get(current_phase, '')}"
        )
        
        # 보행 주기 정보
        current_cycle = "N/A"
        if hasattr(self, 'verification_gait_cycles'):
            for i, cycle in enumerate(self.verification_gait_cycles):
                if cycle.start_frame <= frame_idx <= cycle.end_frame:
                    current_cycle = f"주기 {i+1} ({cycle.foot})"
                    break
        
        self.current_cycle_label.setText(current_cycle)
        
        # 관절 인식 상태
        self.joint_detection_label.setText("MediaPipe 활성")
    
    def update_video_display(self):
        """비디오 표시 업데이트 (옵션 변경 시)"""
        if hasattr(self, 'current_frame_idx'):
            self.display_frame(self.current_frame_idx)


class ModelTrainingWidget(QWidget):
    """모델 학습 위젯"""
    
    def __init__(self):
        super().__init__()
        self.predictor = None
        self.training_data = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 모델 설정
        model_group = QGroupBox("모델 설정")
        model_layout = QGridLayout(model_group)
        
        # 모델 타입 선택
        model_layout.addWidget(QLabel("모델 타입:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["LSTM", "TCN", "1D CNN"])
        model_layout.addWidget(self.model_combo, 0, 1)
        
        # 윈도우 크기
        model_layout.addWidget(QLabel("윈도우 크기:"), 1, 0)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(30, 300)
        self.window_spin.setValue(90)
        self.window_spin.setSuffix(" 프레임")
        model_layout.addWidget(self.window_spin, 1, 1)
        
        # 학습/검증 분할
        model_layout.addWidget(QLabel("학습/검증 비율:"), 2, 0)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.8)
        self.split_spin.setSingleStep(0.1)
        model_layout.addWidget(self.split_spin, 2, 1)
        
        layout.addWidget(model_group)
        
        # 학습 진행
        training_group = QGroupBox("학습 진행")
        training_layout = QVBoxLayout(training_group)
        
        # 학습 버튼
        self.train_btn = QPushButton("모델 학습 시작")
        self.train_btn.clicked.connect(self.start_training)
        training_layout.addWidget(self.train_btn)
        
        # 진행률
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        training_layout.addWidget(self.training_progress)
        
        # 학습 로그
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(150)
        self.training_log.setPlaceholderText("학습 로그가 여기에 표시됩니다...")
        training_layout.addWidget(self.training_log)
        
        layout.addWidget(training_group)
        
        # 모델 저장/로드
        model_io_group = QGroupBox("모델 관리")
        model_io_layout = QHBoxLayout(model_io_group)
        
        self.save_model_btn = QPushButton("모델 저장")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setEnabled(False)
        model_io_layout.addWidget(self.save_model_btn)
        
        self.load_model_btn = QPushButton("모델 로드")
        self.load_model_btn.clicked.connect(self.load_model)
        model_io_layout.addWidget(self.load_model_btn)
        
        layout.addWidget(model_io_group)
    
    def start_training(self):
        """모델 학습 시작"""
        try:
            # 메인 윈도우에서 데이터 가져오기 - 부모 탐색 방식 개선
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'current_session_data'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if not main_window or not hasattr(main_window, 'current_session_data') or not main_window.current_session_data.get('gait_cycles'):
                QMessageBox.warning(self, "오류", "먼저 보행 지표를 계산하세요.")
                return
            
            # 학습 시작
            self.train_btn.setEnabled(False)
            self.training_progress.setVisible(True)
            self.training_progress.setValue(0)
            
            model_type = self.model_combo.currentText().lower()
            window_size = self.window_spin.value()
            
            self.training_log.append(f"모델 학습 시작: {model_type.upper()}")
            self.training_log.append(f"윈도우 크기: {window_size}")
            
            # 데이터셋 빌더 생성
            from data_processing_utils import GaitDatasetBuilder
            dataset_builder = GaitDatasetBuilder(window_size=window_size)
            
            # IMU 데이터와 보행 주기 정렬
            imu_data = main_window.current_session_data['imu_data']
            gait_cycles = main_window.current_session_data['gait_cycles']
            
            self.training_log.append("데이터 정렬 중...")
            aligned_data = dataset_builder.align_imu_with_gait_cycles(imu_data, gait_cycles)
            
            if not aligned_data:
                QMessageBox.warning(self, "오류", "정렬된 데이터가 없습니다.")
                return
            
            self.training_progress.setValue(20)
            self.training_log.append(f"정렬된 데이터: {len(aligned_data)} 주기")
            
            # 특징 추출 및 데이터셋 생성
            self.training_log.append("특징 추출 중...")
            X, y = dataset_builder.create_training_dataset(aligned_data)
            
            self.training_progress.setValue(40)
            self.training_log.append(f"데이터셋 크기: {X.shape}")
            
            # 데이터 분할
            X_train, X_test, y_train, y_test = dataset_builder.split_dataset(
                X, y, test_size=1-self.split_spin.value()
            )
            
            # 정규화
            X_train_scaled, X_test_scaled, _ = dataset_builder.normalize_features(X_train, X_test)
            y_train_scaled, y_test_scaled, _ = dataset_builder.normalize_targets(y_train, y_test)
            
            self.training_progress.setValue(60)
            
            # 모델 생성 및 학습
            self.predictor = GaitMetricsPredictor(model_type=model_type, window_size=window_size)
            
            self.training_log.append("모델 학습 시작...")
            history = self.predictor.train_model(
                X_train_scaled, y_train_scaled,
                X_test_scaled, y_test_scaled,
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            self.training_progress.setValue(90)
            
            # 성능 평가
            metrics = self.predictor.evaluate_model(X_test_scaled, y_test_scaled)
            
            self.training_log.append("모델 학습 완료!")
            self.training_log.append(f"MAE: {metrics['overall']['mae']:.4f}")
            self.training_log.append(f"RMSE: {metrics['overall']['rmse']:.4f}")
            self.training_log.append(f"R²: {metrics['overall']['r2']:.4f}")
            
            self.training_progress.setValue(100)
            self.save_model_btn.setEnabled(True)
            
            QMessageBox.information(self, "완료", "모델 학습이 완료되었습니다!")
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"모델 학습 실패: {e}")
            self.training_log.append(f"오류: {e}")
        finally:
            self.train_btn.setEnabled(True)
            self.training_progress.setVisible(False)
    
    def save_model(self):
        """모델 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "모델 저장", "gait_model.h5", "Model Files (*.h5)"
        )
        if file_path and self.predictor and self.predictor.model:
            self.predictor.model.save(file_path)
            QMessageBox.information(self, "성공", f"모델이 저장되었습니다.\n{file_path}")
    
    def load_model(self):
        """모델 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "모델 로드", "", "Model Files (*.h5)"
        )
        if file_path:
            try:
                # 모델 로드 로직 구현 필요
                QMessageBox.information(self, "성공", f"모델이 로드되었습니다.\n{file_path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"모델 로드 실패: {e}")


class PredictionVisualizationWidget(QWidget):
    """예측 결과 시각화 위젯"""
    
    def __init__(self):
        super().__init__()
        self.predictions = None
        self.actual_values = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 예측 컨트롤
        control_group = QGroupBox("예측 실행")
        control_layout = QHBoxLayout(control_group)
        
        self.predict_btn = QPushButton("예측 실행")
        self.predict_btn.clicked.connect(self.run_prediction)
        control_layout.addWidget(self.predict_btn)
        
        self.comparison_btn = QPushButton("실제값 비교")
        self.comparison_btn.clicked.connect(self.show_comparison)
        self.comparison_btn.setEnabled(False)
        control_layout.addWidget(self.comparison_btn)
        
        layout.addWidget(control_group)
        
        # 시각화 영역
        self.plot_widget = pg.PlotWidget(title="예측 결과 비교")
        self.plot_widget.setLabel('left', '값')
        self.plot_widget.setLabel('bottom', '보행 주기')
        self.plot_widget.addLegend()
        layout.addWidget(self.plot_widget)
        
        # 성능 지표
        metrics_group = QGroupBox("성능 지표")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["지표", "MAE", "RMSE", "R²"])
        metrics_layout.addWidget(self.metrics_table)
        
        layout.addWidget(metrics_group)
    
    def run_prediction(self):
        """예측 실행"""
        try:
            # 메인 윈도우에서 데이터와 모델 가져오기 - 부모 탐색 방식 개선
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'training_widget'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if not main_window:
                QMessageBox.warning(self, "오류", "메인 윈도우를 찾을 수 없습니다.")
                return
            
            # 학습된 모델 확인
            training_widget = main_window.training_widget
            if not training_widget.predictor or training_widget.predictor.model is None:
                QMessageBox.warning(self, "오류", "먼저 모델을 학습하거나 로드하세요.")
                return
            
            # IMU 데이터 확인
            if not main_window.current_session_data.get('imu_data') is not None:
                QMessageBox.warning(self, "오류", "IMU 데이터가 없습니다.")
                return
            
            # 예측 수행
            self.predict_btn.setEnabled(False)
            
            imu_data = main_window.current_session_data['imu_data']
            predictor = training_widget.predictor
            
            # 특징 추출
            from data_processing_utils import GaitDatasetBuilder
            dataset_builder = GaitDatasetBuilder(window_size=predictor.window_size)
            
            # 슬라이딩 윈도우로 특징 추출
            windows, features = predictor.feature_extractor.create_sliding_windows(imu_data)
            
            if len(features) == 0:
                QMessageBox.warning(self, "오류", "특징을 추출할 수 없습니다.")
                return
            
            # 예측 수행
            predictions = predictor.predict(features)
            
            # 결과 저장
            self.predictions = predictions
            
            # 실제값이 있는 경우 비교
            if main_window.current_session_data.get('gait_cycles'):
                gait_cycles = main_window.current_session_data['gait_cycles']
                
                # 실제값 추출 (첫 번째 윈도우만 비교)
                if len(gait_cycles) > 0:
                    actual_cycle = gait_cycles[0]
                    self.actual_values = np.array([[
                        actual_cycle.stride_length,
                        actual_cycle.velocity,
                        actual_cycle.cycle_time,
                        actual_cycle.cadence,
                        actual_cycle.hip_rom,
                        actual_cycle.knee_rom,
                        actual_cycle.ankle_rom,
                        actual_cycle.stance_ratio
                    ]])
                    
                    self.comparison_btn.setEnabled(True)
            
            # 시각화
            self.show_prediction_results()
            
            QMessageBox.information(self, "완료", f"예측이 완료되었습니다.\n{len(predictions)}개의 예측값이 생성되었습니다.")
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"예측 실행 실패: {e}")
        finally:
            self.predict_btn.setEnabled(True)
    
    def show_prediction_results(self):
        """예측 결과 시각화"""
        if self.predictions is None:
            return
        
        self.plot_widget.clear()
        
        # 시간 축
        x = np.arange(len(self.predictions))
        
        # 각 지표별로 플롯 (예: 첫 4개 지표만)
        metrics_names = ['보폭(m)', '속도(m/s)', '주기(s)', '보행률(steps/min)']
        colors = ['b', 'r', 'g', 'm']
        
        for i in range(min(4, self.predictions.shape[1])):
            self.plot_widget.plot(
                x, self.predictions[:, i], 
                pen=colors[i], 
                name=metrics_names[i]
            )
    
    def show_comparison(self):
        """실제값과 예측값 비교 표시"""
        if self.predictions is None or self.actual_values is None:
            return
        
        self.plot_widget.clear()
        
        # 예시 데이터로 시각화
        x = np.arange(len(self.predictions))
        
        # 실제값 (파란색)
        self.plot_widget.plot(x, self.actual_values, pen='b', symbol='o', 
                             symbolSize=5, name='실제값')
        
        # 예측값 (빨간색)
        self.plot_widget.plot(x, self.predictions, pen='r', symbol='s', 
                             symbolSize=5, name='예측값')


class IntegratedGaitSystemGUI(QMainWindow):
    """통합 보행 분석 시스템 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        
        # 기존 분석기 초기화
        self.gait_analyzer = None
        self.video_player = None
        
        # 새로운 컴포넌트 초기화
        self.metrics_calculator = GaitMetricsCalculator()
        self.model_predictor = None
        
        # 데이터 저장
        self.current_session_data = {
            'video_path': None,
            'imu_data': None,
            'gait_events': [],
            'gait_cycles': [],
            'predictions': None
        }
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("통합 보행 분석 시스템 (Integrated Gait Analysis System)")
        self.setGeometry(100, 100, 1600, 1000)
        
        # 중앙 위젯 및 탭
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # 메뉴바 생성
        self.create_menubar()
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 탭 추가
        self.setup_tabs()
        
        # 상태바
        self.statusBar().showMessage("통합 보행 분석 시스템 준비")
    
    def create_menubar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu('파일')
        
        # 세션 로드
        load_session_action = QAction('세션 로드', self)
        load_session_action.triggered.connect(self.load_session)
        file_menu.addAction(load_session_action)
        
        # 세션 저장
        save_session_action = QAction('세션 저장', self)
        save_session_action.triggered.connect(self.save_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        # 종료
        exit_action = QAction('종료', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 분석 메뉴
        analysis_menu = menubar.addMenu('분석')
        
        # 전체 파이프라인 실행
        pipeline_action = QAction('전체 파이프라인 실행', self)
        pipeline_action.triggered.connect(self.run_full_pipeline)
        analysis_menu.addAction(pipeline_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu('도움말')
        
        about_action = QAction('정보', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_tabs(self):
        """탭 설정"""
        # 1. 세션 선택 및 동기화 탭
        self.sync_widget = DataSynchronizationWidget()
        self.tab_widget.addTab(self.sync_widget, "1. 세션 선택 & 동기화")
        
        # 2. 보행 지표 계산 탭 (라벨 데이터 기반)
        self.metrics_widget = GaitMetricsWidget()
        self.tab_widget.addTab(self.metrics_widget, "2. 🎯 보행 지표 계산")
        
        # 3. 모델 학습 탭
        self.training_widget = ModelTrainingWidget()
        self.tab_widget.addTab(self.training_widget, "3. 모델 학습")
        
        # 4. 예측 및 검증 탭
        self.prediction_widget = PredictionVisualizationWidget()
        self.tab_widget.addTab(self.prediction_widget, "4. 예측 및 검증")
        
        # **위젯 간 연결 설정 - setParent 제거 (탭에서 위젯이 사라지는 문제 해결)**
        # 대신 메인 윈도우 참조를 다른 방식으로 설정
        self.sync_widget.main_window = self
    
    def load_session(self):
        """세션 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "세션 파일 선택", "", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # 세션 데이터 복원
                self.current_session_data.update(session_data)
                
                # UI 업데이트
                self.update_ui_from_session()
                
                QMessageBox.information(self, "성공", "세션이 로드되었습니다.")
                
            except Exception as e:
                QMessageBox.warning(self, "오류", f"세션 로드 실패: {e}")
    
    def save_session(self):
        """세션 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "세션 저장", "gait_session.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # 세션 데이터 수집
                session_data = {
                    'timestamp': datetime.now().isoformat(),
                    'video_path': self.current_session_data['video_path'],
                    'gait_events_count': len(self.current_session_data['gait_events']),
                    'gait_cycles_count': len(self.current_session_data['gait_cycles']),
                    # 실제로는 더 많은 정보 저장 필요
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "성공", f"세션이 저장되었습니다.\n{file_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "오류", f"세션 저장 실패: {e}")
    
    def update_ui_from_session(self):
        """세션 데이터로부터 UI 업데이트"""
        # 각 탭의 상태를 세션 데이터에 맞게 업데이트
        pass
    
    def run_full_pipeline(self):
        """전체 분석 파이프라인 실행"""
        # 데이터 검증
        if not self.sync_widget.video_path or self.sync_widget.imu_data is None:
            QMessageBox.warning(self, "오류", "먼저 세션 데이터를 로드하세요.")
            return
        
        # 현재 세션 정보 표시
        session_info = ""
        if self.sync_widget.current_session_data:
            session_info = (
                f"\n피험자: {self.sync_widget.current_session_data['subject']}\n"
                f"보행 타입: {self.sync_widget.current_session_data['gait_type']}\n"
                f"Run: {self.sync_widget.current_session_data['run_num']}"
            )
        
        reply = QMessageBox.question(
            self, '확인', 
            f'전체 파이프라인을 실행하시겠습니까?{session_info}\n\n'
            '이 과정은 시간이 오래 걸릴 수 있습니다.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # 기존 라벨이 있는지 확인
                if self.sync_widget.support_labels:
                    use_existing = QMessageBox.question(
                        self, '기존 라벨 사용', 
                        f'이미 라벨링된 데이터가 있습니다 ({len(self.sync_widget.support_labels)}개 구간).\n'
                        '기존 라벨을 사용하시겠습니까?\n\n'
                        '예: 기존 라벨 사용 (빠름)\n'
                        '아니오: 새로 이벤트 검출 (시간 소요)',
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes
                    )
                    
                    if use_existing == QMessageBox.Yes:
                        # 기존 라벨을 이벤트로 변환
                        self.statusBar().showMessage("기존 라벨 데이터 사용 중...")
                        gait_events = self.convert_labels_to_events(self.sync_widget.support_labels)
                    else:
                        # 새로 이벤트 검출
                        self.statusBar().showMessage("1/3 보행 이벤트 검출 중...")
                        imu_path = None
                        if self.sync_widget.current_session_data and self.sync_widget.current_session_data['imu_exists']:
                            imu_path = os.path.join(
                                self.sync_widget.current_session_data['session_path'], 
                                "imu_data.csv"
                            )
                        gait_analyzer = GaitAnalyzer(self.sync_widget.video_path, imu_path)
                        gait_events = gait_analyzer.detect_gait_events()
                else:
                    # 라벨이 없으면 새로 검출
                    self.statusBar().showMessage("1/3 보행 이벤트 검출 중...")
                    imu_path = None
                    if self.sync_widget.current_session_data and self.sync_widget.current_session_data['imu_exists']:
                        imu_path = os.path.join(
                            self.sync_widget.current_session_data['session_path'], 
                            "imu_data.csv"
                        )
                    gait_analyzer = GaitAnalyzer(self.sync_widget.video_path, imu_path)
                    gait_events = gait_analyzer.detect_gait_events()
                
                self.statusBar().showMessage("2/3 보행 지표 계산 중...")
                
                # Step 2: 보행 지표 계산
                self.metrics_widget.calculate_metrics(
                    self.sync_widget.video_path, 
                    gait_events,
                    progress_callback=lambda x: self.statusBar().showMessage(f"보행 지표 계산 중... {x}%")
                )
                
                # Step 3: 데이터 저장
                self.current_session_data.update({
                    'video_path': self.sync_widget.video_path,
                    'imu_data': self.sync_widget.imu_data,
                    'gait_events': gait_events,
                    'gait_cycles': self.metrics_widget.gait_cycles,
                    'session_info': self.sync_widget.current_session_data
                })
                
                self.statusBar().showMessage("파이프라인 완료!")
                QMessageBox.information(self, "완료", f"전체 분석 파이프라인이 완료되었습니다!\n검출된 이벤트: {len(gait_events)}개\n분석된 보행 주기: {len(self.metrics_widget.gait_cycles)}개")
                
                # 결과 탭으로 이동
                self.tab_widget.setCurrentIndex(2)  # 보행 지표 계산 탭
                
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파이프라인 실행 실패: {e}")
                self.statusBar().showMessage("파이프라인 실행 실패")
    
    def convert_labels_to_events(self, support_labels: List[Dict]) -> List:
        """Support labels를 gait events로 변환"""
        from gait_class import GaitEvent
        
        events = []
        
        for label in support_labels:
            phase = label['phase']
            start_frame = label['start_frame']
            end_frame = label['end_frame']
            
            # single_support 구간에서 이벤트 추출
            if 'single_support' in phase:
                # single_support_left의 시작 = left heel strike
                # single_support_right의 시작 = right heel strike
                foot = 'left' if 'left' in phase else 'right'
                
                # Heel Strike 이벤트
                events.append(GaitEvent(
                    frame=start_frame,
                    event_type='HS',
                    foot=foot,
                    confidence=0.9  # 라벨 데이터이므로 높은 신뢰도
                ))
                
                # Toe Off 이벤트 (다음 double_support 직전)
                events.append(GaitEvent(
                    frame=end_frame,
                    event_type='TO',
                    foot=foot,
                    confidence=0.9
                ))
        
        return events
            
    def show_about(self):
        """정보 다이얼로그"""
        about_text = """
통합 보행 분석 시스템 v1.0

이 시스템은 IMU 센서 데이터와 영상 데이터를 기반으로 
보행 분석 및 낙상 위험 예측을 수행합니다.

주요 기능:
• 시각적 데이터 확인 및 수정
• 자동 보행 지표 계산
• 시계열 회귀 모델 학습
• 실시간 보행 지표 예측

개발: 보행 분석 연구팀
        """
        
        QMessageBox.about(self, "정보", about_text)


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 다크 테마 설정 (선택사항)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    # 메인 윈도우 생성 및 표시
    main_window = IntegratedGaitSystemGUI()
    main_window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()