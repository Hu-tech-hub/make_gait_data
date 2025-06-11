import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QMessageBox, QProgressBar,
    QTabWidget, QCheckBox, QComboBox, QSplitter, QFileDialog, QLineEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

# MediaPipe 임포트 (보행 지표 계산용)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe가 설치되지 않았습니다. 보행 지표 계산 기능이 제한됩니다.")

# 새로운 보행 계산 엔진 임포트
try:
    from gait_calculation_engine import GaitCalculationEngine, create_gait_engine
    CALCULATION_ENGINE_AVAILABLE = True
except ImportError:
    CALCULATION_ENGINE_AVAILABLE = False
    print("gait_calculation_engine를 찾을 수 없습니다. 고급 계산 기능이 제한됩니다.")


class GaitMetricsCalculationWorker(QThread):
    """보행 지표 계산 작업 스레드"""
    progress_updated = pyqtSignal(int, str)
    calculation_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, imu_data, support_labels, session_data, selected_parameters=None, user_height=1.7):
        super().__init__()
        self.video_path = video_path
        self.imu_data = imu_data
        self.support_labels = support_labels
        self.session_data = session_data
        self.user_height = user_height


    def run(self):
        """보행 지표 계산 실행"""
        try:
            # 1. 비디오 분석 시작
            self.progress_updated.emit(10, "비디오 파일 로딩 중...")
            
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception("비디오 파일을 열 수 없습니다.")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.progress_updated.emit(20, f"비디오 정보: {total_frames} 프레임, {fps:.1f} FPS")
            
            # 2. 보행 지표 계산 실행
            self.progress_updated.emit(30, "보행 지표 계산 엔진 실행...")
            results = self.run_gait_calculation(cap, fps, total_frames)
            
            cap.release()
            
            self.progress_updated.emit(100, "계산 완료!")
            self.calculation_finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

    def run_gait_calculation(self, cap, fps, total_frames):
        """보행 지표 계산 (단순화된 단일 방식)"""
        # 1. MediaPipe 초기화 (정밀도 낮추고 속도 향상)
        if MEDIAPIPE_AVAILABLE:
            self.progress_updated.emit(40, "MediaPipe 관절 추정 초기화...")
            mp_pose = mp.solutions.pose
            # 속도 우선 설정: 낮은 정밀도, 높은 속도
            pose = mp_pose.Pose(
                static_image_mode=False,          # 실시간 스트리밍 모드 (속도↑)
                model_complexity=1,              # 중간 복잡도 (속도-정확도 균형)
                enable_segmentation=False,       # 불필요한 기능 비활성화 (속도↑)
                min_detection_confidence=0.5,    # 적당한 감지 신뢰도 (정확도↑)
                min_tracking_confidence=0.5      # 적당한 추적 신뢰도 (정확도↑)
            )
        else:
            raise Exception("MediaPipe가 필요합니다.")
        
        # 2. 모든 프레임에서 관절 데이터 추출
        self.progress_updated.emit(50, "모든 프레임 관절 데이터 추출 중...")
        
        joint_data_list = []
        timestamps = []
        
        # 비디오 크기는 gait_calculation_engine에서 처리
        # (중복 제거)
        
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            timestamp = frame_idx / fps
            timestamps.append(timestamp)
            
            # MediaPipe 관절 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # DEBUG: MediaPipe 원본 좌표 확인 (첫 번째 프레임만)
                if frame_idx == 0:
                    left_ankle_raw = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
                    right_ankle_raw = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
                    print(f"🔍 DEBUG - MediaPipe 원본 좌표 (첫 프레임):")
                    print(f"   왼발목: x={left_ankle_raw.x:.6f}, y={left_ankle_raw.y:.6f}")
                    print(f"   오른발목: x={right_ankle_raw.x:.6f}, y={right_ankle_raw.y:.6f}")
                    print(f"   예상: 정규화 좌표라면 0~1 사이여야 함")
                
                # 정규화 좌표 그대로 저장 (엔진에서 픽셀 변환 처리)
                joints = {
                    'left_ankle': {
                        'x': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x,
                        'y': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].y,
                        'z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].z
                    },
                    'right_ankle': {
                        'x': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x,
                        'y': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].y,
                        'z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].z
                    },
                    'left_knee': {
                        'x': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x,
                        'y': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].y,
                        'z': landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE].z
                    },
                    'right_knee': {
                        'x': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x,
                        'y': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].y,
                        'z': landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].z
                    }
                }
                joint_data_list.append(joints)
            else:
                joint_data_list.append(None)
            
            # 진행률 업데이트 (50~80%)
            if frame_idx % 10 == 0:
                progress = 50 + int((frame_idx / total_frames) * 30)
                self.progress_updated.emit(progress, f"관절 데이터 추출: {frame_idx}/{total_frames}")
        
        # 3. 보행 계산 엔진으로 3개 파라미터 계산
        self.progress_updated.emit(80, "보행 파라미터 계산 중...")
        
        # 사용자 키와 비디오 경로와 함께 엔진 생성
        engine = GaitCalculationEngine(fps, self.user_height, "forward", self.video_path)  # 비디오에서 실제 크기 가져오기
        calculation_results = engine.calculate_gait_parameters(
            joint_data_list, timestamps, self.support_labels
        )
        
        # 4. 결과 구성
        self.progress_updated.emit(90, "결과 정리 중...")
        
        results = {
            'session_info': self.session_data,
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': total_frames / fps
            },
            'calculation_method': 'gait_engine',
            'engine_results': calculation_results,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return results


class GaitMetricsWidget(QWidget):
    """보행 지표 계산 및 결과 표시 위젯"""
    
    def __init__(self):
        super().__init__()
        self.session_data = None
        self.calculation_results = None
        self.worker = None
        
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 상단 스플리터 (설정과 계산 버튼)
        top_splitter = QSplitter(Qt.Horizontal)
        
        # 왼쪽: 파라미터 선택 및 설정
        settings_group = QGroupBox("⚙️ 계산 설정")
        settings_layout = QVBoxLayout(settings_group)
        
        # 계산 엔진 정보
        engine_info = QLabel("🔧 계산 엔진: MediaPipe + 보행 파라미터 계산")
        engine_info.setStyleSheet("color: #2E7D32; font-weight: bold;")
        settings_layout.addWidget(engine_info)
        
        # 사용자 키 입력
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("사용자 키:"))
        self.height_input = QLineEdit("170")
        self.height_input.setMaximumWidth(100)
        self.height_input.setPlaceholderText("예: 170")
        height_layout.addWidget(self.height_input)
        height_layout.addWidget(QLabel("cm"))
        height_layout.addStretch()
        settings_layout.addLayout(height_layout)
        
        # 계산 파라미터 정보
        param_label = QLabel("📊 계산되는 파라미터:")
        param_label.setFont(QFont("", 10, QFont.Bold))
        settings_layout.addWidget(param_label)
        
        # 파라미터 정보 표시
        param_info = QLabel(
            "• Stride Time: 동일한 발의 두 HS 사이 시간 간격\n"
            "• Stride Length: 보행 방향 투영 거리 (실측값)\n"
            "• Velocity: 보폭/시간 비율 (m/s)"
        )
        param_info.setWordWrap(True)
        param_info.setStyleSheet("color: #666; margin-left: 10px;")
        settings_layout.addWidget(param_info)
        
        top_splitter.addWidget(settings_group)
        
        # 오른쪽: 계산 시작 그룹
        calc_group = QGroupBox("🚀 보행 지표 계산")
        calc_layout = QVBoxLayout(calc_group)
        
        # 상태 라벨
        self.calc_status_label = QLabel("상태: 1번 탭에서 세션 데이터를 먼저 로드하세요")
        self.calc_status_label.setStyleSheet("color: orange;")
        self.calc_status_label.setWordWrap(True)
        calc_layout.addWidget(self.calc_status_label)
        
        # 계산 버튼
        button_layout = QVBoxLayout()
        
        self.calculate_btn = QPushButton("🚀 보행 파라미터 계산")
        self.calculate_btn.clicked.connect(self.start_calculation)
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setMinimumHeight(50)
        button_layout.addWidget(self.calculate_btn)
        
        calc_layout.addLayout(button_layout)
        
        # 진행 상황
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        calc_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        calc_layout.addWidget(self.progress_label)
        
        top_splitter.addWidget(calc_group)
        
        # 스플리터 비율 설정
        top_splitter.setSizes([300, 400])
        
        layout.addWidget(top_splitter)
        
        # 결과 표시 탭
        self.results_tabs = QTabWidget()
        
        # 1. 요약 통계 탭
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.summary_text = QTextEdit()
        self.summary_text.setPlaceholderText("계산 완료 후 요약 통계가 표시됩니다...")
        summary_layout.addWidget(self.summary_text)
        
        self.results_tabs.addTab(summary_tab, "📊 요약 통계")
        
        # 2. 상세 결과 테이블 탭
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        
        self.details_table = QTableWidget()
        details_layout.addWidget(self.details_table)
        
        self.results_tabs.addTab(details_tab, "📋 상세 결과")
        
        # 3. 시각화 그래프 탭
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)
        
        # Matplotlib 캔버스
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # 그래프 옵션
        graph_options = QHBoxLayout()
        self.refresh_graph_btn = QPushButton("📈 그래프 새로고침")
        self.refresh_graph_btn.clicked.connect(self.update_visualization)
        graph_options.addWidget(self.refresh_graph_btn)
        graph_options.addStretch()
        
        viz_layout.addLayout(graph_options)
        
        self.results_tabs.addTab(viz_tab, "📈 시각화")
        
        layout.addWidget(self.results_tabs)
        
        # 결과 내보내기
        export_group = QGroupBox("💾 결과 내보내기")
        export_layout = QHBoxLayout(export_group)
        
        self.export_csv_btn = QPushButton("CSV 내보내기")
        self.export_csv_btn.clicked.connect(self.export_to_csv)
        self.export_csv_btn.setEnabled(False)
        
        self.export_json_btn = QPushButton("JSON 내보내기")
        self.export_json_btn.clicked.connect(self.export_to_json)
        self.export_json_btn.setEnabled(False)
        
        export_layout.addWidget(self.export_csv_btn)
        export_layout.addWidget(self.export_json_btn)
        export_layout.addStretch()
        
        layout.addWidget(export_group)
    
    def get_selected_parameters(self):
        """선택된 파라미터 목록 반환 (항상 3개 파라미터)"""
        return ['stride_time', 'stride_length', 'velocity']
    
    def set_session_data(self, video_path, imu_data, support_labels, session_data):
        """세션 데이터 설정"""
        self.video_path = video_path
        self.imu_data = imu_data
        self.support_labels = support_labels
        self.session_data = session_data
        
        # 세션 정보 업데이트
        if session_data:
            info = f"✅ 세션 준비: {session_data['subject']} - {session_data['gait_type']} - {session_data['run_num']}"
            self.calc_status_label.setText(info)
            self.calc_status_label.setStyleSheet("color: green;")
            
            # 버튼 활성화
            self.calculate_btn.setEnabled(True)
    
    def start_calculation(self, custom_parameters=None):
        """보행 지표 계산 시작"""
        if not all([self.video_path, self.imu_data is not None, self.support_labels]):
            QMessageBox.warning(self, "오류", "필요한 데이터가 부족합니다.")
            return
        
        # UI 상태 변경
        self.calculate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 상태 메시지
        self.calc_status_label.setText("계산 중: Stride Time, Stride Length, Velocity")
        self.calc_status_label.setStyleSheet("color: blue;")
        
        # 사용자 키 가져오기
        try:
            user_height_cm = float(self.height_input.text())
            user_height_m = user_height_cm / 100.0
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "올바른 키를 입력하세요 (숫자만)")
            self.calculate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            return
        
        # 워커 스레드 시작
        self.worker = GaitMetricsCalculationWorker(
            self.video_path, self.imu_data, self.support_labels, 
            self.session_data, None, user_height_m
        )
        
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.calculation_finished.connect(self.on_calculation_finished)
        self.worker.error_occurred.connect(self.on_calculation_error)
        
        self.worker.start()
    
    def update_progress(self, progress, message):
        """진행 상황 업데이트"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
    

    
    def on_calculation_finished(self, results):
        """계산 완료 처리"""
        self.calculation_results = results
        
        # UI 상태 복원
        self.calculate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        # 상태 라벨 업데이트
        self.calc_status_label.setText(
            f"✅ 계산 완료: Stride Time, Stride Length, Velocity | {results.get('analysis_time', '')}"
        )
        self.calc_status_label.setStyleSheet("color: green;")
        
        # 결과 표시
        self.display_results()
        
        # 내보내기 버튼 활성화
        self.export_csv_btn.setEnabled(True)
        self.export_json_btn.setEnabled(True)
        
        # 결과에 따른 완료 메시지
        engine_results = results.get('engine_results', {})
        total_events = engine_results.get('total_events', 0)
        total_frames = engine_results.get('total_frames', 0)
        
        if results.get('calculation_method') == 'advanced_engine':
            engine_type = "고급 엔진"
        else:
            engine_type = "기본 모드"
            
        message = (f"{engine_type} 보행 지표 계산이 완료되었습니다!\n\n"
                  f"📊 분석 프레임: {total_frames}개\n"
                  f"🎯 검출 이벤트: {total_events}개\n"
                  f"⚙️ 계산 파라미터: Stride Time, Stride Length, Velocity")
        
        QMessageBox.information(self, "계산 완료", message)
    
    def on_calculation_error(self, error_message):
        """계산 오류 처리"""
        self.calculate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        
        self.calc_status_label.setText("❌ 계산 오류 발생")
        self.calc_status_label.setStyleSheet("color: red;")
        
        QMessageBox.critical(self, "오류", f"계산 중 오류가 발생했습니다:\n{error_message}")
    
    def display_results(self):
        """결과 표시"""
        if not self.calculation_results:
            return
        
        results = self.calculation_results
        
        # 1. 요약 통계 표시
        summary_text = self.generate_summary_text(results)
        self.summary_text.setText(summary_text)
        
        # 2. 상세 결과 테이블 표시
        self.display_details_table(results)
        
        # 3. 시각화 업데이트
        self.update_visualization()
    
    def generate_summary_text(self, results):
        """요약 텍스트 생성"""
        session_info = results['session_info']
        video_info = results['video_info']
        calculation_method = results.get('calculation_method', 'unknown')
        
        summary = f"""
🎯 보행 지표 계산 결과 요약

📋 세션 정보:
  • 피험자: {session_info['subject']}
  • 보행 타입: {session_info['gait_type']}
  • Run: {session_info['run_num']}
  • 분석 시간: {results['analysis_time']}

📹 비디오 정보:
  • 총 프레임: {video_info['total_frames']}
  • FPS: {video_info['fps']:.1f}
  • 길이: {video_info['duration']:.2f}초

⚙️ 계산 정보:
  • 사용 기술: MediaPipe 관절 추정 + 라벨링 데이터
  • 계산 파라미터: Stride Time, Stride Length, Velocity
        """.strip()
        
        # 보행 계산 엔진 결과
        engine_results = results.get('engine_results', {})
        
        summary += f"\n\n🚶 분석 결과:\n"
        summary += f"  • 총 프레임: {engine_results.get('total_frames', 0)}개\n"
        
        parameters = engine_results.get('parameters', {})
        if parameters:
            summary += "\n\n📊 보행 지표 통계:\n"
            
            # Stride Time
            stride_time = parameters.get('stride_time', {})
            if stride_time.get('count', 0) > 0:
                summary += f"\n  🔹 Stride Time (초):\n"
                summary += f"     평균: {stride_time['mean']:.3f}\n"
                summary += f"     측정수: {stride_time['count']}회\n"
            
            # Stride Length
            stride_length = parameters.get('stride_length', {})
            if stride_length.get('count', 0) > 0:
                summary += f"\n  🔹 Stride Length (m):\n"
                summary += f"     평균: {stride_length['mean']:.3f}\n"
                summary += f"     측정수: {stride_length['count']}회\n"
            
            # Velocity
            velocity = parameters.get('velocity', {})
            if velocity.get('count', 0) > 0:
                summary += f"\n  🔹 Velocity (m/s):\n"
                summary += f"     평균: {velocity['mean']:.3f}\n"
                summary += f"     측정수: {velocity['count']}회\n"
        
        return summary
    
    def display_details_table(self, results):
        """상세 결과 테이블 표시"""
        engine_results = results.get('engine_results', {})
        details = engine_results.get('details', [])
        
        if not details:
            return
        
        # 테이블 설정
        columns = ['번호', '발', '시작프레임', '종료프레임', '시작시간(s)', '종료시간(s)',
                  'Stride Time(s)', 'Stride Length(m)', 'Velocity(m/s)']
        
        self.details_table.setColumnCount(len(columns))
        self.details_table.setHorizontalHeaderLabels(columns)
        self.details_table.setRowCount(len(details))
        
        # 데이터 입력
        for i, detail in enumerate(details):
            self.details_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.details_table.setItem(i, 1, QTableWidgetItem(detail.get('foot', '')))
            self.details_table.setItem(i, 2, QTableWidgetItem(str(detail.get('start_frame', 0))))
            self.details_table.setItem(i, 3, QTableWidgetItem(str(detail.get('end_frame', 0))))
            self.details_table.setItem(i, 4, QTableWidgetItem(f"{detail.get('start_time', 0):.3f}"))
            self.details_table.setItem(i, 5, QTableWidgetItem(f"{detail.get('end_time', 0):.3f}"))
            self.details_table.setItem(i, 6, QTableWidgetItem(f"{detail.get('stride_time', 0):.3f}"))
            self.details_table.setItem(i, 7, QTableWidgetItem(f"{detail.get('stride_length', 0):.3f}"))
            self.details_table.setItem(i, 8, QTableWidgetItem(f"{detail.get('velocity', 0):.3f}"))
        
        # 컬럼 크기 조정
        self.details_table.resizeColumnsToContents()
    
    def update_visualization(self):
        """시각화 그래프 업데이트 (X축: 누적 이동 거리, 발자국 마커)"""
        if not self.calculation_results:
            return
        
        self.figure.clear()
        engine_results = self.calculation_results.get('engine_results', {})
        details = engine_results.get('details', [])
        
        if not details:
            return
        
        # 1x3 서브플롯 생성 (3개 파라미터)
        axes = self.figure.subplots(1, 3)
        
        # 데이터 준비: 누적 이동 거리 계산
        cumulative_distances = [0]  # 시작점은 0
        for detail in details:
            cumulative_distances.append(cumulative_distances[-1] + detail.get('stride_length', 0))
        cumulative_distances = cumulative_distances[1:]  # 첫 번째 0 제거
        
        # 지표별 데이터 추출
        stride_times = [detail.get('stride_time', 0) for detail in details]
        stride_lengths = [detail.get('stride_length', 0) for detail in details]
        velocities = [detail.get('velocity', 0) for detail in details]
        foot_types = [detail.get('foot', '') for detail in details]
        
        # 왼발/오른발 데이터 분리
        left_distances = [cumulative_distances[i] for i, foot in enumerate(foot_types) if foot == 'left']
        left_stride_times = [stride_times[i] for i, foot in enumerate(foot_types) if foot == 'left']
        left_stride_lengths = [stride_lengths[i] for i, foot in enumerate(foot_types) if foot == 'left']
        left_velocities = [velocities[i] for i, foot in enumerate(foot_types) if foot == 'left']
        
        right_distances = [cumulative_distances[i] for i, foot in enumerate(foot_types) if foot == 'right']
        right_stride_times = [stride_times[i] for i, foot in enumerate(foot_types) if foot == 'right']
        right_stride_lengths = [stride_lengths[i] for i, foot in enumerate(foot_types) if foot == 'right']
        right_velocities = [velocities[i] for i, foot in enumerate(foot_types) if foot == 'right']
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Stride Time 그래프
        ax = axes[0]
        if left_distances:
            ax.scatter(left_distances, left_stride_times, c='#e74c3c', s=80, marker='<', 
                      label='Left Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        if right_distances:
            ax.scatter(right_distances, right_stride_times, c='#3498db', s=80, marker='>', 
                      label='Right Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 연결선 그리기
        if cumulative_distances and stride_times:
            ax.plot(cumulative_distances, stride_times, '-', color='gray', alpha=0.3, linewidth=1)
        
        ax.set_title('Stride Time vs Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cumulative Distance (m)')
        ax.set_ylabel('Stride Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if stride_times:
            mean_val = np.mean(stride_times)
            ax.axhline(y=mean_val, color=colors[0], linestyle='--', alpha=0.5, 
                      label=f'Mean: {mean_val:.3f}s')
        
        # Stride Length 그래프
        ax = axes[1]
        if left_distances:
            ax.scatter(left_distances, left_stride_lengths, c='#e74c3c', s=80, marker='<', 
                      label='Left Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        if right_distances:
            ax.scatter(right_distances, right_stride_lengths, c='#3498db', s=80, marker='>', 
                      label='Right Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 연결선 그리기
        if cumulative_distances and stride_lengths:
            ax.plot(cumulative_distances, stride_lengths, '-', color='gray', alpha=0.3, linewidth=1)
        
        ax.set_title('Stride Length vs Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cumulative Distance (m)')
        ax.set_ylabel('Stride Length (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if stride_lengths:
            mean_val = np.mean(stride_lengths)
            ax.axhline(y=mean_val, color=colors[1], linestyle='--', alpha=0.5, 
                      label=f'Mean: {mean_val:.3f}m')
        
        # Velocity 그래프
        ax = axes[2]
        if left_distances:
            ax.scatter(left_distances, left_velocities, c='#e74c3c', s=80, marker='<', 
                      label='Left Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        if right_distances:
            ax.scatter(right_distances, right_velocities, c='#3498db', s=80, marker='>', 
                      label='Right Foot', alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # 연결선 그리기
        if cumulative_distances and velocities:
            ax.plot(cumulative_distances, velocities, '-', color='gray', alpha=0.3, linewidth=1)
        
        ax.set_title('Velocity vs Distance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cumulative Distance (m)')
        ax.set_ylabel('Velocity (m/s)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if velocities:
            mean_val = np.mean(velocities)
            ax.axhline(y=mean_val, color=colors[2], linestyle='--', alpha=0.5, 
                      label=f'Mean: {mean_val:.3f}m/s')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def export_to_csv(self):
        """CSV로 내보내기"""
        if not self.calculation_results:
            QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "CSV 파일 저장", 
            f"gait_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                import pandas as pd
                
                # 결과 데이터를 DataFrame으로 변환
                engine_results = self.calculation_results.get('engine_results', {})
                details = engine_results.get('details', [])
                
                df = pd.DataFrame([
                    {
                        '번호': i + 1,
                        '발': detail.get('foot', ''),
                        '시작프레임': detail.get('start_frame', 0),
                        '종료프레임': detail.get('end_frame', 0),
                        '시작시간(s)': detail.get('start_time', 0),
                        '종료시간(s)': detail.get('end_time', 0),
                        'Stride Time(s)': detail.get('stride_time', 0),
                        'Stride Length(m)': detail.get('stride_length', 0),
                        'Velocity(m/s)': detail.get('velocity', 0)
                    }
                    for i, detail in enumerate(details)
                ])
                
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "성공", f"CSV 파일이 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"CSV 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def export_to_json(self):
        """JSON으로 내보내기"""
        if not self.calculation_results:
            QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "JSON 파일 저장", 
            f"gait_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                import json
                
                # JSON 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.calculation_results, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "성공", f"JSON 파일이 저장되었습니다:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "오류", f"JSON 저장 중 오류가 발생했습니다:\n{str(e)}") 