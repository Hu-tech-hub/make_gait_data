import sys
import os
import json
import numpy as np
import pandas as pd
import cv2
from collections import Counter

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QTabWidget, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

import pyqtgraph as pg

# 공통 유틸리티 import (중복 제거)
from gait_param_class import GaitAnalysisConfig, GaitAnalysisUtils


class DataSynchronizationWidget(QWidget):
    """데이터 동기화 및 시각화 위젯"""
    
    def __init__(self):
        super().__init__()
        self.current_session_data = None
        self.video_path = None
        self.imu_data = None
        self.support_labels = []
        self.gait_events = []
        
        self.init_ui()
        self.scan_experiment_data()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 세션 선택 그룹
        session_group = QGroupBox("🔍 스마트 세션 선택")
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
        
        # 로드 버튼과 상태
        load_layout = QHBoxLayout()
        self.load_session_btn = QPushButton("🚀 세션 데이터 로드")
        self.load_session_btn.clicked.connect(self.load_session_data)
        self.load_session_btn.setEnabled(False)
        load_layout.addWidget(self.load_session_btn)
        
        self.status_label = QLabel("상태: 세션을 선택하세요")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        load_layout.addWidget(self.status_label)
        
        session_layout.addLayout(load_layout)
        layout.addWidget(session_group)
        
        # 동기화 시각화
        sync_viz_group = QGroupBox("🔄 동기화 시각화")
        sync_viz_layout = QVBoxLayout(sync_viz_group)
        
        self.viz_tabs = QTabWidget()
        self.setup_visualization_tabs()
        sync_viz_layout.addWidget(self.viz_tabs)
        layout.addWidget(sync_viz_group)    
    def setup_visualization_tabs(self):
        """시각화 탭들 설정"""
        # 메타데이터 탭
        metadata_tab = QWidget()
        metadata_layout = QVBoxLayout(metadata_tab)
        self.metadata_text = QTextEdit()
        self.metadata_text.setMaximumHeight(150)
        self.metadata_text.setPlaceholderText("세션 로드 후 메타데이터가 표시됩니다...")
        metadata_layout.addWidget(self.metadata_text)
        self.viz_tabs.addTab(metadata_tab, "📋 메타데이터")
        
        # 동기화 테이블 탭
        self.setup_sync_table_tab()
        
        # 시각화 그래프 탭
        self.setup_graph_tab()
    
    def setup_sync_table_tab(self):
        """동기화 테이블 탭 설정"""
        sync_table_tab = QWidget()
        sync_table_layout = QVBoxLayout(sync_table_tab)
        
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("📋 전체 프레임 데이터 (스크롤 가능)"))
        info_layout.addStretch()
        sync_table_layout.addLayout(info_layout)
        
        self.sync_table = QTableWidget()
        self.sync_table.setMinimumHeight(400)
        sync_table_layout.addWidget(self.sync_table)
        
        self.sync_quality_label = QLabel("동기화 품질: 데이터 로드 후 확인 가능")
        self.sync_quality_label.setStyleSheet("color: gray;")
        sync_table_layout.addWidget(self.sync_quality_label)
        
        self.viz_tabs.addTab(sync_table_tab, "🔄 동기화 테이블")    
    def setup_graph_tab(self):
        """그래프 탭 설정"""
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        
        # 센서 선택 컨트롤
        sensor_control_group = QGroupBox("센서 데이터 선택")
        sensor_control_layout = QHBoxLayout(sensor_control_group)
        
        # 가속도계 그룹
        accel_group = QGroupBox("가속도계")
        accel_layout = QHBoxLayout(accel_group)
        self.accel_x_cb = QCheckBox("AccelX")
        self.accel_y_cb = QCheckBox("AccelY") 
        self.accel_z_cb = QCheckBox("AccelZ")
        self.accel_x_cb.setChecked(True)
        self.accel_y_cb.setChecked(True)
        self.accel_z_cb.setChecked(True)
        accel_layout.addWidget(self.accel_x_cb)
        accel_layout.addWidget(self.accel_y_cb)
        accel_layout.addWidget(self.accel_z_cb)
        
        # 자이로스코프 그룹
        gyro_group = QGroupBox("자이로스코프")
        gyro_layout = QHBoxLayout(gyro_group)
        self.gyro_x_cb = QCheckBox("GyroX")
        self.gyro_y_cb = QCheckBox("GyroY")
        self.gyro_z_cb = QCheckBox("GyroZ")
        gyro_layout.addWidget(self.gyro_x_cb)
        gyro_layout.addWidget(self.gyro_y_cb)
        gyro_layout.addWidget(self.gyro_z_cb)
        
        sensor_control_layout.addWidget(accel_group)
        sensor_control_layout.addWidget(gyro_group)
        self.setup_label_controls(sensor_control_layout)
        
        graph_layout.addWidget(sensor_control_group)
        
        # PyQtGraph 위젯
        self.sync_plot_widget = pg.PlotWidget(title="동기화된 데이터 시각화")
        self.sync_plot_widget.setLabel('left', 'IMU 값')
        self.sync_plot_widget.setLabel('bottom', '시간 (초)')
        self.sync_plot_widget.setMinimumHeight(450)
        graph_layout.addWidget(self.sync_plot_widget)
        
        self.viz_tabs.addTab(graph_tab, "📈 시간축 그래프")    
    def setup_label_controls(self, sensor_control_layout):
        """라벨 컨트롤 설정"""
        label_group = QGroupBox("라벨 표시 선택")
        label_layout = QVBoxLayout(label_group)
        
        # 체크박스들
        checkbox_layout = QHBoxLayout()
        self.show_double_support_cb = QCheckBox("🔴 이중지지")
        self.show_single_left_cb = QCheckBox("🟢 단일지지(왼쪽)")
        self.show_single_right_cb = QCheckBox("🔵 단일지지(오른쪽)")
        self.show_non_gait_cb = QCheckBox("⚪ 비보행")
        
        # 기본 선택
        for cb in [self.show_double_support_cb, self.show_single_left_cb, 
                   self.show_single_right_cb, self.show_non_gait_cb]:
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_sync_visualization)
        
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
        
        # 그래프 업데이트 버튼
        self.update_graph_btn = QPushButton("📈 그래프 업데이트")
        self.update_graph_btn.clicked.connect(self.update_sync_visualization)
        button_layout.addWidget(self.update_graph_btn)
        
        label_layout.addLayout(checkbox_layout)
        label_layout.addLayout(button_layout)
        
        sensor_control_layout.addWidget(label_group)    
    def scan_experiment_data(self):
        """experiment_data 폴더 스캔 - GaitAnalysisConfig 사용"""
        if not os.path.exists(GaitAnalysisConfig.EXPERIMENT_DATA_PATH):
            self.status_label.setText("상태: experiment_data 폴더가 없습니다")
            self.status_label.setStyleSheet("color: red;")
            return
        
        # 피험자 목록 수집
        subjects = [s for s in os.listdir(GaitAnalysisConfig.EXPERIMENT_DATA_PATH) 
                   if os.path.isdir(os.path.join(GaitAnalysisConfig.EXPERIMENT_DATA_PATH, s)) 
                   and s.startswith('SA')]
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
        
        subject_path = os.path.join(GaitAnalysisConfig.EXPERIMENT_DATA_PATH, subject)
        
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
        
        gait_type_path = os.path.join(GaitAnalysisConfig.EXPERIMENT_DATA_PATH, subject, gait_type)
        
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
        if not session_display or " - " not in session_display:
            return
        
        run_num, session_name = session_display.split(" - ", 1)
        subject = self.subject_combo.currentText()
        gait_type = self.gait_type_combo.currentText()
        
        if not all([subject, gait_type, session_name]):
            return
        
        # 세션 경로 구성
        session_path = os.path.join(GaitAnalysisConfig.EXPERIMENT_DATA_PATH, subject, gait_type, session_name)
        
        # 세션 정보 업데이트 - GaitAnalysisUtils 사용
        validation_result = GaitAnalysisUtils.validate_session_data(session_path)
        self.update_session_info(session_path, subject, gait_type, run_num, validation_result)
        
        # 대응하는 라벨 파일 확인
        self.check_corresponding_labels(subject, gait_type, run_num)
        
        self.load_session_btn.setEnabled(True)
    
    def update_session_info(self, session_path: str, subject: str, gait_type: str, 
                          run_num: str, validation_result: dict):
        """세션 정보 업데이트"""
        info_text = f"""📁 세션 경로: {session_path}
👤 피험자: {subject}
🚶 보행 타입: {gait_type}
📊 Run: {run_num}

📋 파일 확인:
📹 비디오: {'✓ ' + validation_result['video_filename'] if validation_result['video_exists'] else '✗ 없음'}
📊 IMU 데이터: {'✓ imu_data.csv' if validation_result['imu_exists'] else '✗ 없음'}"""
        
        # 메타데이터 정보 추가
        if validation_result.get('metadata'):
            metadata = validation_result['metadata']
            duration = metadata.get('duration', 0)
            frames = metadata.get('video_frames', 0)
            info_text += f"\n⏱️ 길이: {duration:.1f}초, {frames} 프레임"
        
        self.metadata_text.setText(info_text)
        
        # 현재 세션 데이터 저장
        self.current_session_data = {
            'session_path': session_path,
            'subject': subject,
            'gait_type': gait_type,
            'run_num': run_num,
            'validation_result': validation_result
        }    
    def check_corresponding_labels(self, subject: str, gait_type: str, run_num: str):
        """대응하는 라벨 파일 확인 - GaitAnalysisUtils 사용"""
        # 라벨 파일명 생성
        label_filename = GaitAnalysisUtils.build_label_filename(subject, gait_type, run_num)
        label_path = os.path.join(GaitAnalysisConfig.SUPPORT_LABEL_DATA_PATH, subject, label_filename)
        
        if os.path.exists(label_path):
            try:
                label_df = pd.read_csv(label_path)
                phase_count = len(label_df)
                unique_phases = label_df['phase'].unique()
                
                self.status_label.setText(
                    f"🏷️ 라벨: ✓ {label_filename} ({phase_count}개 구간, {len(unique_phases)}개 타입)"
                )
                self.status_label.setStyleSheet("color: green;")
                
                # 현재 세션 데이터에 라벨 정보 추가
                if self.current_session_data:
                    self.current_session_data['label_path'] = label_path
                    self.current_session_data['label_filename'] = label_filename
                    
            except Exception as e:
                self.status_label.setText(f"🏷️ 라벨: ⚠ {label_filename} (읽기 오류: {str(e)})")
                self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText(f"🏷️ 라벨: ✗ {label_filename} (없음)")
            self.status_label.setStyleSheet("color: red;")
    
    def load_session_data(self):
        """세션 데이터 로드"""
        if not self.current_session_data:
            QMessageBox.warning(self, "오류", "세션이 선택되지 않았습니다.")
            return
        
        try:
            session_path = self.current_session_data['session_path']
            validation_result = self.current_session_data['validation_result']
            
            # 비디오 로드
            if validation_result['video_exists']:
                self.video_path = os.path.join(session_path, validation_result['video_filename'])
            else:
                QMessageBox.warning(self, "오류", "비디오 파일을 찾을 수 없습니다.")
                return
            
            # IMU 데이터 로드
            if validation_result['imu_exists']:
                imu_path = os.path.join(session_path, GaitAnalysisConfig.IMU_FILENAME)
                self.imu_data = pd.read_csv(imu_path)
            else:
                QMessageBox.warning(self, "경고", "IMU 데이터가 없습니다. 일부 기능이 제한될 수 있습니다.")
                self.imu_data = None            
            # 라벨 데이터 로드 (있는 경우)
            if 'label_path' in self.current_session_data:
                self.support_labels = GaitAnalysisUtils.load_support_labels(
                    self.current_session_data['label_path']
                )
            else:
                self.support_labels = []
            
            # 메타데이터 표시 업데이트
            self.display_loaded_metadata()
            
            # 동기화 테이블 생성
            self.create_sync_table()
            
            # 동기화 그래프 생성
            self.create_sync_visualization()
            
            # 보행 지표 계산 버튼 활성화
            self.enable_gait_metrics_calculation()
            
            # 상태 업데이트
            self.status_label.setText("상태: ✅ 세션 데이터 로드 완료")
            self.status_label.setStyleSheet("color: green;")
            
            QMessageBox.information(
                self, "성공", 
                f"세션 데이터가 성공적으로 로드되었습니다.\n"
                f"📹 비디오: {validation_result['video_filename']}\n"
                f"📊 IMU: {'있음' if self.imu_data is not None else '없음'}\n"
                f"🏷️ 라벨: {'있음' if self.support_labels else '없음'}\n\n"
                f"🎯 2번 탭에서 보행 지표 계산을 수행할 수 있습니다."
            )
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"세션 데이터 로드 실패: {e}")
            self.status_label.setText("상태: ✗ 로드 실패")
            self.status_label.setStyleSheet("color: red;")    
    def enable_gait_metrics_calculation(self):
        """보행 지표 계산 기능 활성화"""
        main_window = getattr(self, 'main_window', None)
        
        if main_window and hasattr(main_window, 'metrics_widget'):
            main_window.metrics_widget.calculate_btn.setEnabled(True)
            main_window.metrics_widget.calc_status_label.setText(
                "✅ 준비 완료! 버튼을 클릭하여 보행 지표 계산을 시작하세요."
            )
            main_window.metrics_widget.calc_status_label.setStyleSheet("color: blue; font-weight: bold;")
            
            # 세션 데이터 전달
            main_window.metrics_widget.set_session_data(
                self.video_path, self.imu_data, self.support_labels, self.current_session_data
            )
            
            # 3번째 탭(비디오 검증)에 지지 라벨 데이터 전달
            if hasattr(main_window, 'validation_widget'):
                main_window.validation_widget.set_support_labels(self.support_labels)
                print("✅ 3번째 탭으로 지지 라벨 전달 완료")
            
            print("✅ 보행 지표 계산 기능이 활성화되었습니다.")
    
    def display_loaded_metadata(self):
        """로드된 데이터의 메타데이터 표시 - GaitAnalysisUtils 사용"""
        metadata_text = ""
        
        if self.video_path:
            video_info = GaitAnalysisUtils.get_video_info(self.video_path)
            if video_info:
                metadata_text += f"📹 비디오 정보:\n"
                metadata_text += f"  • 해상도: {video_info['width']} x {video_info['height']}\n"
                metadata_text += f"  • FPS: {video_info['fps']:.1f}\n"
                metadata_text += f"  • 프레임 수: {video_info['frame_count']}\n"
                metadata_text += f"  • 길이: {video_info['duration']:.2f}초\n\n"
        
        if self.imu_data is not None:
            metadata_text += f"📊 IMU 데이터:\n"
            metadata_text += f"  • 샘플 수: {len(self.imu_data)}\n"
            metadata_text += f"  • 컬럼: {list(self.imu_data.columns)}\n"
            if 'sync_timestamp' in self.imu_data.columns:
                time_range = self.imu_data['sync_timestamp'].max() - self.imu_data['sync_timestamp'].min()
                sampling_rate = len(self.imu_data) / time_range if time_range > 0 else 0
                metadata_text += f"  • 시간 범위: {time_range:.2f}초\n"
                metadata_text += f"  • 샘플링 레이트: ~{sampling_rate:.1f} Hz\n"
            metadata_text += "\n"
        
        if self.support_labels:
            phases = [label['phase'] for label in self.support_labels]
            unique_phases = list(set(phases))
            metadata_text += f"🏷️ 라벨 데이터:\n"
            metadata_text += f"  • 구간 수: {len(self.support_labels)}\n"
            metadata_text += f"  • 타입: {unique_phases}\n"
            
            phase_counts = Counter(phases)
            for phase, count in phase_counts.items():
                metadata_text += f"    ▶ {phase}: {count}개\n"
        
        self.metadata_text.setText(metadata_text.strip())    
    def create_sync_table(self):
        """동기화 테이블 생성"""
        if not self.video_path or self.imu_data is None:
            return
        
        video_info = GaitAnalysisUtils.get_video_info(self.video_path)
        if not video_info:
            return
        
        fps = video_info['fps']
        frame_count = video_info['frame_count']
        
        # 적절한 샘플링으로 성능 최적화
        if frame_count > 1000:
            sample_rate = max(1, frame_count // 1000)
            display_frames = list(range(0, frame_count, sample_rate))
        else:
            display_frames = list(range(frame_count))
        
        # 테이블 설정
        columns = ['Frame', 'Time(s)', 'AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'Phase Label']
        self.sync_table.setColumnCount(len(columns))
        self.sync_table.setHorizontalHeaderLabels(columns)
        self.sync_table.setRowCount(len(display_frames))
        
        # IMU 컬럼 매핑
        imu_columns = list(self.imu_data.columns)
        accel_cols = [col for col in imu_columns if 'accel' in col.lower()][:3]
        gyro_cols = [col for col in imu_columns if 'gyro' in col.lower()][:3]
        
        # 데이터 채우기
        for row, frame_idx in enumerate(display_frames):
            frame_time = frame_idx / fps
            
            self.sync_table.setItem(row, 0, QTableWidgetItem(str(frame_idx)))
            self.sync_table.setItem(row, 1, QTableWidgetItem(f"{frame_time:.2f}"))
            
            # IMU 데이터 매핑
            if 'sync_timestamp' in self.imu_data.columns:
                time_diffs = np.abs(self.imu_data['sync_timestamp'] - frame_time)
                closest_idx = time_diffs.idxmin()
                closest_row = self.imu_data.loc[closest_idx]
                
                # 가속도계 데이터
                for i, col in enumerate(accel_cols[:3]):
                    value = closest_row.get(col, 0.0) if col in self.imu_data.columns else 0.0
                    self.sync_table.setItem(row, 2 + i, QTableWidgetItem(f"{value:.3f}"))
                
                # 자이로스코프 데이터
                for i, col in enumerate(gyro_cols[:3]):
                    value = closest_row.get(col, 0.0) if col in self.imu_data.columns else 0.0
                    self.sync_table.setItem(row, 5 + i, QTableWidgetItem(f"{value:.3f}"))
            else:
                # sync_timestamp가 없는 경우
                for i in range(6):
                    self.sync_table.setItem(row, 2 + i, QTableWidgetItem("0.000"))
            
            # 라벨 정보
            current_label = self.get_label_for_frame(frame_idx)
            label_item = QTableWidgetItem(current_label)
            self.set_label_color(label_item, current_label)
            self.sync_table.setItem(row, 8, label_item)
        
        self.sync_table.resizeColumnsToContents()
        self.update_sync_quality_display(video_info)    
    def get_label_for_frame(self, frame_idx):
        """특정 프레임의 라벨 반환"""
        if not self.support_labels:
            return "non_gait"
        
        for label in self.support_labels:
            if label['start_frame'] <= frame_idx <= label['end_frame']:
                return label['phase']
        return "non_gait"
    
    def set_label_color(self, item, label):
        """라벨에 따른 색상 설정 - GaitAnalysisConfig 사용"""
        if label == 'double_support':
            item.setBackground(QColor(255, 200, 200))  # 연한 빨강
        elif label == 'single_support_left':
            item.setBackground(QColor(200, 255, 200))  # 연한 초록
        elif label == 'single_support_right':
            item.setBackground(QColor(200, 200, 255))  # 연한 파랑
        else:  # non_gait
            item.setBackground(QColor(240, 240, 240))  # 연한 회색
    
    def update_sync_quality_display(self, video_info):
        """동기화 품질 표시 업데이트"""
        if self.imu_data is not None and 'sync_timestamp' in self.imu_data.columns:
            video_duration = video_info['duration']
            imu_duration = self.imu_data['sync_timestamp'].max()
            quality, color = GaitAnalysisUtils.calculate_sync_quality(video_duration, imu_duration)
            
            time_diff = abs(video_duration - imu_duration)
            quality_text = f"동기화 품질: {'✅' if quality == '우수' else '⚠️' if quality == '보통' else '❌'} {quality} (시간차: {time_diff:.2f}초)"
        else:
            quality_text = "동기화 품질: 정보 없음"
            color = "gray"
        
        self.sync_quality_label.setText(quality_text)
        self.sync_quality_label.setStyleSheet(f"color: {color};")
    
    def create_sync_visualization(self):
        """동기화 시각화 그래프 생성"""
        self.update_sync_visualization()
    
    def update_sync_visualization(self):
        """선택된 센서 데이터로 그래프 업데이트"""
        if not self.video_path or self.imu_data is None:
            return
        
        self.sync_plot_widget.clear()
        
        video_info = GaitAnalysisUtils.get_video_info(self.video_path)
        if not video_info:
            return
        
        fps = video_info['fps']
        total_frames = video_info['frame_count']
        frame_numbers = np.arange(total_frames)
        
        # IMU 데이터 매핑
        frame_imu_data = self.map_imu_to_frames(total_frames, fps)
        
        # 그래프 플롯
        self.plot_sensor_data(frame_numbers, frame_imu_data)
        
        # 라벨 배경 표시
        self.plot_label_backgrounds(total_frames, frame_imu_data)
        
        # 축 설정
        self.sync_plot_widget.setLabel('bottom', '프레임 번호')
        self.sync_plot_widget.setLabel('left', 'IMU 값')
        self.sync_plot_widget.setTitle(f'동기화된 데이터 (프레임 기반) - 총 {total_frames} 프레임')
        self.sync_plot_widget.addLegend()    
    def map_imu_to_frames(self, total_frames, fps):
        """IMU 데이터를 프레임에 매핑"""
        imu_columns = list(self.imu_data.columns)
        accel_cols = [col for col in imu_columns if 'accel' in col.lower()][:3]
        gyro_cols = [col for col in imu_columns if 'gyro' in col.lower()][:3]
        
        frame_imu_data = {}
        for col in accel_cols + gyro_cols:
            if col in self.imu_data.columns:
                frame_imu_data[col] = []
                
                for frame_idx in range(total_frames):
                    frame_time = frame_idx / fps
                    
                    if 'sync_timestamp' in self.imu_data.columns:
                        time_diffs = np.abs(self.imu_data['sync_timestamp'] - frame_time)
                        closest_idx = time_diffs.idxmin()
                        value = self.imu_data.loc[closest_idx, col]
                    else:
                        imu_idx = int((frame_idx / total_frames) * len(self.imu_data))
                        imu_idx = min(imu_idx, len(self.imu_data) - 1)
                        value = self.imu_data.iloc[imu_idx][col]
                    
                    frame_imu_data[col].append(value)
        
        return frame_imu_data
    
    def plot_sensor_data(self, frame_numbers, frame_imu_data):
        """센서 데이터 플롯"""
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        plot_index = 0
        
        # IMU 컬럼명 정확히 가져오기
        imu_columns = list(frame_imu_data.keys())
        accel_cols = [col for col in imu_columns if 'accel' in col.lower()]
        gyro_cols = [col for col in imu_columns if 'gyro' in col.lower()]
        
        print(f"🔍 사용 가능한 IMU 컬럼: {imu_columns}")
        print(f"🔍 가속도계 컬럼: {accel_cols}")
        print(f"🔍 자이로스코프 컬럼: {gyro_cols}")
        
        # 가속도계 데이터
        accel_checkboxes = [self.accel_x_cb, self.accel_y_cb, self.accel_z_cb]
        accel_labels = ['AccelX', 'AccelY', 'AccelZ']
        
        for i, (cb, label) in enumerate(zip(accel_checkboxes, accel_labels)):
            if cb.isChecked() and i < len(accel_cols):
                col_name = accel_cols[i]
                if col_name in frame_imu_data and len(frame_imu_data[col_name]) > 0:
                    self.sync_plot_widget.plot(
                        frame_numbers, frame_imu_data[col_name],
                        pen=colors[plot_index % len(colors)], name=f"{label} ({col_name})"
                    )
                    plot_index += 1
                    print(f"✅ {label} 플롯 완료: {col_name}")
                else:
                    print(f"❌ {label} 데이터 없음: {col_name}")
        
        # 자이로스코프 데이터
        gyro_checkboxes = [self.gyro_x_cb, self.gyro_y_cb, self.gyro_z_cb]
        gyro_labels = ['GyroX', 'GyroY', 'GyroZ']
        
        for i, (cb, label) in enumerate(zip(gyro_checkboxes, gyro_labels)):
            if cb.isChecked() and i < len(gyro_cols):
                col_name = gyro_cols[i]
                if col_name in frame_imu_data and len(frame_imu_data[col_name]) > 0:
                    self.sync_plot_widget.plot(
                        frame_numbers, frame_imu_data[col_name],
                        pen=colors[plot_index % len(colors)], name=f"{label} ({col_name})"
                    )
                    plot_index += 1
                    print(f"✅ {label} 플롯 완료: {col_name}")
                else:
                    print(f"❌ {label} 데이터 없음: {col_name}")    
    def plot_label_backgrounds(self, total_frames, frame_imu_data):
        """라벨 배경 표시 - GaitAnalysisConfig 사용"""
        if not self.support_labels:
            return
        
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
        
        # 사용자 선택 확인
        show_labels = {
            'single_support_left': self.show_single_left_cb.isChecked(),
            'single_support_right': self.show_single_right_cb.isChecked(),
            'double_support': self.show_double_support_cb.isChecked(),
            'non_gait': self.show_non_gait_cb.isChecked()
        }
        
        # 선택된 라벨 구간만 표시
        for i, label in enumerate(self.support_labels):
            start_frame = max(0, label['start_frame'])
            end_frame = min(total_frames - 1, label['end_frame'])
            phase = label['phase']
            
            if (phase in GaitAnalysisConfig.LABEL_COLORS and 
                phase in show_labels and show_labels[phase] and 
                start_frame <= end_frame):
                
                color = GaitAnalysisConfig.LABEL_COLORS[phase]
                
                try:
                    # 반투명 영역 추가
                    fill_item = pg.FillBetweenItem(
                        curve1=pg.PlotCurveItem([start_frame, end_frame], [y_min, y_min]),
                        curve2=pg.PlotCurveItem([start_frame, end_frame], [y_max, y_max]),
                        brush=pg.mkBrush(color)
                    )
                    self.sync_plot_widget.addItem(fill_item)
                    
                    # 구간 경계선
                    self.sync_plot_widget.plot(
                        [start_frame, start_frame], [y_min, y_max],
                        pen=pg.mkPen(color[0:3], width=2, style=2),
                        name=f"{phase}_{i}" if i < 5 else None
                    )
                except Exception as e:
                    print(f"라벨 {i} 표시 오류: {e}")
    
    def select_all_labels(self):
        """모든 라벨 체크박스 선택"""
        for cb in [self.show_double_support_cb, self.show_single_left_cb, 
                   self.show_single_right_cb, self.show_non_gait_cb]:
            cb.setChecked(True)
    
    def deselect_all_labels(self):
        """모든 라벨 체크박스 해제"""
        for cb in [self.show_double_support_cb, self.show_single_left_cb, 
                   self.show_single_right_cb, self.show_non_gait_cb]:
            cb.setChecked(False)