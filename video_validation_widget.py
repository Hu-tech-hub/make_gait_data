import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as patches

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QSlider, QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QFrame, QCheckBox, QSpinBox, QComboBox, QProgressBar, QTabWidget
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen

# MediaPipe 임포트
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# 보행 계산 엔진 임포트
try:
    from gait_calculation_engine import GaitCalculationEngine
    GAIT_ENGINE_AVAILABLE = True
except ImportError:
    GAIT_ENGINE_AVAILABLE = False


class SessionManager:
    """세션 간 데이터 관리 클래스"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """모든 세션 데이터 초기화"""
        # 공통 데이터
        self.video_path = None
        self.video_fps = 30.0
        self.total_frames = 0
        
        # 세션 1 결과: 라벨링 데이터
        self.session1_completed = False
        self.support_labels = None
        self.joint_data = None
        self.timestamps = None
        
        # 세션 2 결과: 보행 파라미터
        self.session2_completed = False  
        self.gait_parameters = None
        self.stride_details = None
        
        # 세션 3 상태: 시각화
        self.session3_ready = False
        
        print("🔄 세션 매니저 초기화 완료")
    
    def set_video_info(self, video_path, fps, total_frames):
        """비디오 정보 설정"""
        self.video_path = video_path
        self.video_fps = fps
        self.total_frames = total_frames
        print(f"📹 비디오 정보 설정: {video_path}, {fps:.1f}fps, {total_frames}프레임")
    
    def complete_session1(self, support_labels, joint_data, timestamps):
        """세션 1 완료 및 데이터 저장"""
        self.support_labels = support_labels
        self.joint_data = joint_data
        self.timestamps = timestamps
        self.session1_completed = True
        print(f"✅ 세션 1 완료: {len(support_labels) if support_labels else 0}개 라벨, {len(joint_data) if joint_data else 0}개 관절 데이터")
    
    def complete_session2(self, gait_parameters, stride_details):
        """세션 2 완료 및 데이터 저장"""
        self.gait_parameters = gait_parameters
        self.stride_details = stride_details
        self.session2_completed = True
        self.session3_ready = True
        print(f"✅ 세션 2 완료: {len(stride_details) if stride_details else 0}개 stride 계산됨")
    
    def get_session_status(self):
        """현재 세션 진행 상태 반환"""
        if not self.session1_completed:
            return 1, "세션 1: 라벨링 데이터 생성 필요"
        elif not self.session2_completed:
            return 2, "세션 2: 보행 파라미터 계산 필요"
        elif self.session3_ready:
            return 3, "세션 3: 시각화 준비 완료"
        else:
            return 0, "세션 진행 불가"


class VideoPlayer(QLabel):
    """비디오 플레이어 위젯 (세션 통합형)"""
    
    def __init__(self, session_manager):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid gray;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("비디오를 로드하세요")
        
        # 세션 매니저 참조
        self.session_manager = session_manager
        
        # 비디오 관련 변수들
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        
        # MediaPipe 초기화
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # 오버레이 설정
        self.show_joints = True
        self.show_support_labels = True
        self.show_stride_info = True
        
    def load_video(self, video_path):
        """비디오 로드"""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            QMessageBox.warning(self, "오류", "비디오 파일을 열 수 없습니다.")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # 세션 매니저에 비디오 정보 저장
        self.session_manager.set_video_info(video_path, self.fps, self.total_frames)
        
        print(f"📹 비디오 로드 완료: {self.total_frames} 프레임, {self.fps:.1f} FPS")
        return True
    
    def set_frame(self, frame_number):
        """특정 프레임으로 이동"""
        if not self.cap or frame_number < 0 or frame_number >= self.total_frames:
            return
            
        self.current_frame = frame_number
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if not ret:
            return
            
        # 프레임 처리 및 오버레이
        frame = self.process_frame(frame)
        
        # Qt 이미지로 변환 및 표시
        self.display_frame(frame)
    
    def process_frame(self, frame):
        """프레임 처리 및 오버레이 추가"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        # 1. MediaPipe 관절 추정 및 그리기
        if self.show_joints and MEDIAPIPE_AVAILABLE:
            results = self.pose.process(frame_rgb)
            if results.pose_landmarks:
                # 관절 그리기
                self.mp_drawing.draw_landmarks(
                    frame_rgb, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
        
        # 2. 지지 라벨 정보 오버레이 (세션 1 결과 사용)
        if self.show_support_labels and hasattr(self.session_manager, 'support_labels') and self.session_manager.support_labels:
            frame_rgb = self.draw_support_labels(frame_rgb)
        
        # 3. 보폭 정보 오버레이 (세션 2 결과 사용)
        if self.show_stride_info and hasattr(self.session_manager, 'stride_details') and self.session_manager.stride_details:
            frame_rgb = self.draw_stride_info(frame_rgb)
        
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    def draw_support_labels(self, frame_rgb):
        """지지 라벨 정보 그리기 (세션 1 데이터 사용)"""
        current_time = self.current_frame / self.fps
        
        # 현재 시간에 해당하는 라벨 찾기
        current_phase = "unknown"
        
        # support_labels가 존재하는지 확인
        if not hasattr(self.session_manager, 'support_labels') or not self.session_manager.support_labels:
            # 라벨이 없는 경우 기본 정보만 표시
            height, width = frame_rgb.shape[:2]
            cv2.rectangle(frame_rgb, (10, 10), (350, 60), (128, 128, 128), -1)
            cv2.rectangle(frame_rgb, (10, 10), (350, 60), (0, 0, 0), 2)
            cv2.putText(frame_rgb, "Support: No Labels", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_rgb, f"Frame: {self.current_frame}", (20, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return frame_rgb
        
        for label in self.session_manager.support_labels:
            start_time = label.get('start_frame', 0) / self.fps
            end_time = label.get('end_frame', 0) / self.fps
            
            if start_time <= current_time <= end_time:
                current_phase = label.get('phase', 'unknown')
                break
        
        # 라벨에 따른 색상 설정
        color_map = {
            'double_support': (255, 100, 100),  # 빨간색 계열
            'single_support_left': (100, 255, 100),  # 초록색 계열
            'left_support': (100, 255, 100),
            'single_support_right': (100, 100, 255),  # 파란색 계열
            'right_support': (100, 100, 255),
            'double_stance': (255, 100, 100),
            'left_stance': (100, 255, 100),
            'right_stance': (100, 100, 255),
            'unknown': (128, 128, 128)  # 회색
        }
        
        color = color_map.get(current_phase, (128, 128, 128))
        
        # 상단에 라벨 정보 표시
        height, width = frame_rgb.shape[:2]
        cv2.rectangle(frame_rgb, (10, 10), (350, 60), color, -1)
        cv2.rectangle(frame_rgb, (10, 10), (350, 60), (0, 0, 0), 2)
        
        label_text = current_phase.replace('_', ' ').title()
        cv2.putText(frame_rgb, f"Support: {label_text}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_rgb, f"Frame: {self.current_frame}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_rgb
    
    def draw_stride_info(self, frame_rgb):
        """보폭 정보 그리기 (세션 2 데이터 사용)"""
        current_time = self.current_frame / self.fps
        height, width = frame_rgb.shape[:2]
        
        # stride_details가 존재하는지 확인
        if not hasattr(self.session_manager, 'stride_details') or not self.session_manager.stride_details:
            # stride 정보가 없는 경우 기본 정보만 표시
            info_x = width - 400
            info_y = 10
            
            cv2.rectangle(frame_rgb, (info_x, info_y), (width - 10, info_y + 80), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame_rgb, (info_x, info_y), (width - 10, info_y + 80), 
                         (255, 255, 255), 2)
            
            cv2.putText(frame_rgb, "No Stride Data", (info_x + 10, info_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_rgb, "Complete Session 2", (info_x + 10, info_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # 현재 시간에 해당하는 stride 정보 찾기
            current_stride = None
            for stride in self.session_manager.stride_details:
                start_time = stride.get('start_time', 0)
                end_time = stride.get('end_time', 0)
                
                if start_time <= current_time <= end_time:
                    current_stride = stride
                    break
            
            if current_stride is not None:
                # 우측 상단에 stride 정보 표시
                info_x = width - 400
                info_y = 10
                
                # 배경 박스
                cv2.rectangle(frame_rgb, (info_x, info_y), (width - 10, info_y + 140), 
                             (50, 50, 50), -1)
                cv2.rectangle(frame_rgb, (info_x, info_y), (width - 10, info_y + 140), 
                             (255, 255, 255), 2)
                
                # 정보 텍스트
                foot = current_stride.get('foot', 'unknown')
                stride_time = current_stride.get('stride_time', 0)
                stride_length = current_stride.get('stride_length', 0)
                velocity = current_stride.get('velocity', 0)
                
                # gait_parameters가 존재하는지 확인
                method = 'unknown'
                if hasattr(self.session_manager, 'gait_parameters') and self.session_manager.gait_parameters:
                    method = self.session_manager.gait_parameters.get('calculation_method', 'unknown')
                
                texts = [
                    f"Active Stride: {foot.upper()} foot",
                    f"Time: {stride_time:.3f} s",
                    f"Length: {stride_length:.3f} m", 
                    f"Velocity: {velocity:.3f} m/s",
                    f"Method: {method.replace('_', ' ').title()}"
                ]
                
                for i, text in enumerate(texts):
                    cv2.putText(frame_rgb, text, (info_x + 10, info_y + 25 + i * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 하단에 전체 진행 상황 표시 (항상 표시)
        progress_y = height - 40
        cv2.rectangle(frame_rgb, (10, progress_y), (width - 10, height - 10),
                     (0, 0, 0), -1)
        
        # 진행 바
        progress = self.current_frame / max(1, self.total_frames - 1)
        progress_width = int((width - 40) * progress)
        cv2.rectangle(frame_rgb, (20, progress_y + 10), (20 + progress_width, height - 20),
                     (0, 255, 0), -1)
        
        # 시간 정보
        current_time_str = f"{current_time:.2f}s"
        total_time = self.total_frames / self.fps
        total_time_str = f"{total_time:.2f}s"
        
        cv2.putText(frame_rgb, f"{current_time_str} / {total_time_str}", 
                   (20, progress_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (255, 255, 255), 1)
        
        return frame_rgb
    
    def display_frame(self, frame):
        """프레임을 화면에 표시"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # 위젯 크기에 맞게 스케일링
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.setPixmap(scaled_pixmap)


class VideoValidationWidget(QWidget):
    """비디오 검증 메인 위젯 (세션 통합형)"""
    
    # 세션 완료 시그널
    session_completed = pyqtSignal(int)  # 세션 번호 전달
    
    def __init__(self):
        super().__init__()
        
        # 세션 매니저 초기화
        self.session_manager = SessionManager()
        
        self.init_ui()
        
        # 타이머 설정 (자동 재생용)
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.is_playing = False
        
        # 초기 상태 업데이트
        self.update_session_status()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout(self)
        
        # 상단: 세션 진행 상태
        session_group = QGroupBox("🎯 세션 진행 상태")
        session_layout = QVBoxLayout(session_group)
        
        # 세션 상태 표시
        status_layout = QHBoxLayout()
        
        self.session_status_label = QLabel("세션 1: 라벨링 데이터 생성 필요")
        self.session_status_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.session_progress = QProgressBar()
        self.session_progress.setRange(0, 3)
        self.session_progress.setValue(0)
        
        status_layout.addWidget(self.session_status_label)
        status_layout.addWidget(self.session_progress)
        
        session_layout.addLayout(status_layout)
        
        # 세션 제어 버튼
        session_btn_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("📹 비디오 로드")
        self.load_video_btn.clicked.connect(self.load_video)
        
        self.process_session_btn = QPushButton("▶️ 다음 세션 진행")
        self.process_session_btn.clicked.connect(self.process_current_session)
        self.process_session_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("🔄 세션 초기화")
        self.reset_btn.clicked.connect(self.reset_sessions)
        
        session_btn_layout.addWidget(self.load_video_btn)
        session_btn_layout.addWidget(self.process_session_btn)
        session_btn_layout.addWidget(self.reset_btn)
        session_btn_layout.addStretch()
        
        session_layout.addLayout(session_btn_layout)
        
        layout.addWidget(session_group)
        
        # 중앙: 메인 컨텐츠 (좌측 비디오, 우측 그래프)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 좌측: 비디오 플레이어
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        self.video_player = VideoPlayer(self.session_manager)
        video_layout.addWidget(self.video_player)
        
        # 비디오 컨트롤
        control_group = QGroupBox("🎮 재생 컨트롤")
        control_layout = QVBoxLayout(control_group)
        
        # 재생 버튼들
        button_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶️ 재생")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.prev_btn = QPushButton("⏮️ 이전")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        
        self.next_btn = QPushButton("⏭️ 다음")  
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        
        button_layout.addWidget(self.prev_btn)
        button_layout.addWidget(self.play_btn)
        button_layout.addWidget(self.next_btn)
        button_layout.addStretch()
        
        control_layout.addLayout(button_layout)
        
        # 프레임 슬라이더
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.slider_changed)
        self.frame_slider.setEnabled(False)
        control_layout.addWidget(self.frame_slider)
        
        # 표시 옵션
        option_layout = QHBoxLayout()
        
        self.joints_check = QCheckBox("관절 표시")
        self.joints_check.setChecked(True)
        self.joints_check.toggled.connect(self.update_display_options)
        
        self.labels_check = QCheckBox("지지 라벨")
        self.labels_check.setChecked(True)
        self.labels_check.toggled.connect(self.update_display_options)
        
        self.stride_check = QCheckBox("보폭 정보")
        self.stride_check.setChecked(True)
        self.stride_check.toggled.connect(self.update_display_options)
        
        option_layout.addWidget(self.joints_check)
        option_layout.addWidget(self.labels_check)
        option_layout.addWidget(self.stride_check)
        option_layout.addStretch()
        
        control_layout.addLayout(option_layout)
        
        video_layout.addWidget(control_group)
        
        # 우측: 실시간 그래프
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        
        # Matplotlib 캔버스
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        graph_layout.addWidget(self.canvas)
        
        # 스플리터에 추가
        main_splitter.addWidget(video_widget)
        main_splitter.addWidget(graph_widget)
        main_splitter.setSizes([600, 400])
        
        layout.addWidget(main_splitter)
        
        # 하단: 상태 정보
        status_group = QGroupBox("📊 상태 정보")
        status_layout = QHBoxLayout(status_group)
        
        self.status_label = QLabel("비디오를 로드하고 세션을 진행하세요")
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
    
    def load_video(self):
        """비디오 파일 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "비디오 파일 선택", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            if self.video_player.load_video(file_path):
                # 슬라이더 설정
                self.frame_slider.setRange(0, self.video_player.total_frames - 1)
                self.frame_slider.setValue(0)
                
                # 첫 프레임 표시
                self.video_player.set_frame(0)
                
                # 비디오 컨트롤 활성화
                self.play_btn.setEnabled(True)
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
                self.frame_slider.setEnabled(True)
                
                self.update_session_status()
    
    def process_current_session(self):
        """현재 세션 진행"""
        current_session, _ = self.session_manager.get_session_status()
        
        if current_session == 1:
            self.process_session1()
        elif current_session == 2:
            self.process_session2()
        elif current_session == 3:
            self.enable_session3()
    
    def process_session1(self):
        """세션 1: 라벨링 데이터 생성 및 관절 추출"""
        if not self.session_manager.video_path:
            QMessageBox.warning(self, "오류", "먼저 비디오를 로드하세요.")
            return
        
        if not MEDIAPIPE_AVAILABLE:
            QMessageBox.warning(self, "오류", "MediaPipe가 설치되어 있지 않습니다.")
            return
        
        self.status_label.setText("세션 1 진행 중: 관절 추출 및 라벨링...")
        
        try:
            # MediaPipe로 관절 데이터 추출
            joint_data, timestamps = self.extract_joint_data()
            
            # 간단한 지지 라벨 생성 (더미 데이터 - 실제로는 더 정교한 알고리즘 필요)
            support_labels = self.generate_support_labels(len(joint_data))
            
            # 세션 1 완료
            self.session_manager.complete_session1(support_labels, joint_data, timestamps)
            
            self.status_label.setText("✅ 세션 1 완료: 관절 데이터 및 지지 라벨 생성됨")
            self.update_session_status()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"세션 1 처리 중 오류 발생: {e}")
            self.status_label.setText("❌ 세션 1 처리 실패")
    
    def process_session2(self):
        """세션 2: 보행 파라미터 계산"""
        if not GAIT_ENGINE_AVAILABLE:
            QMessageBox.warning(self, "오류", "Gait Calculation Engine이 사용할 수 없습니다.")
            return
        
        self.status_label.setText("세션 2 진행 중: 보행 파라미터 계산...")
        
        try:
            # 보행 계산 엔진 생성
            engine = GaitCalculationEngine(
                fps=self.session_manager.video_fps,
                user_height=1.7,  # 기본 키 설정
                video_path=self.session_manager.video_path
            )
            
            # Phase 기반 보행 파라미터 계산
            gait_results = engine.calculate_gait_parameters(
                joint_data_list=self.session_manager.joint_data,
                timestamps=self.session_manager.timestamps,
                support_labels=self.session_manager.support_labels,
                use_phase_method=True
            )
            
            # 세션 2 완료
            stride_details = gait_results.get('details', [])
            self.session_manager.complete_session2(gait_results, stride_details)
            
            # 그래프 업데이트
            self.update_graphs()
            
            self.status_label.setText(f"✅ 세션 2 완료: {len(stride_details)}개 stride 계산됨")
            self.update_session_status()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"세션 2 처리 중 오류 발생: {e}")
            self.status_label.setText("❌ 세션 2 처리 실패")
            import traceback
            traceback.print_exc()
    
    def enable_session3(self):
        """세션 3: 시각화 모드 활성화"""
        self.status_label.setText("🎬 세션 3: 시각화 모드 활성화됨 - 비디오 재생으로 결과를 확인하세요!")
        
        # 모든 표시 옵션 활성화
        self.joints_check.setChecked(True)
        self.labels_check.setChecked(True)
        self.stride_check.setChecked(True)
        
        # 재생 컨트롤 활성화
        self.play_btn.setEnabled(True)
        self.prev_btn.setEnabled(True)  
        self.next_btn.setEnabled(True)
        self.frame_slider.setEnabled(True)
        
        # 첫 프레임으로 이동하여 모든 오버레이 표시
        self.video_player.set_frame(0)
        self.frame_slider.setValue(0)
    
    def extract_joint_data(self):
        """비디오에서 관절 데이터 추출"""
        joint_data_list = []
        timestamps = []
        
        cap = cv2.VideoCapture(self.session_manager.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # MediaPipe 초기화
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print(f"🔍 관절 데이터 추출 시작: {total_frames} 프레임")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 처리
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            timestamp = frame_count / fps
            timestamps.append(timestamp)
            
            # 관절 데이터 추출
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                joint_data = {
                    'left_ankle': {
                        'x': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                        'y': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                        'z': landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z
                    },
                    'right_ankle': {
                        'x': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                        'y': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                        'z': landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z
                    },
                    'left_knee': {
                        'x': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                        'y': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                        'z': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z
                    },
                    'right_knee': {
                        'x': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                        'y': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                        'z': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z
                    }
                }
            else:
                # 관절을 감지하지 못한 경우 None 추가
                joint_data = None
            
            joint_data_list.append(joint_data)
            frame_count += 1
            
            # 진행 상황 출력 (100프레임마다)
            if frame_count % 100 == 0:
                print(f"  진행: {frame_count}/{total_frames} 프레임")
        
        cap.release()
        pose.close()
        
        print(f"✅ 관절 데이터 추출 완료: {len(joint_data_list)} 프레임")
        return joint_data_list, timestamps
    
    def generate_support_labels(self, total_frames):
        """지지 라벨 생성 (더미 구현 - 실제로는 더 정교한 알고리즘 필요)"""
        support_labels = []
        
        # 간단한 패턴으로 지지 라벨 생성
        # 실제로는 발 위치, 속도 등을 분석하여 더 정확하게 생성해야 함
        
        frames_per_cycle = 60  # 2초 주기 (30fps 기준)
        current_frame = 0
        
        while current_frame < total_frames:
            # Double stance
            if current_frame + 15 < total_frames:
                support_labels.append({
                    'phase': 'double_stance',
                    'start_frame': current_frame,
                    'end_frame': current_frame + 15
                })
                current_frame += 16
            
            # Right stance
            if current_frame + 20 < total_frames:
                support_labels.append({
                    'phase': 'right_stance', 
                    'start_frame': current_frame,
                    'end_frame': current_frame + 20
                })
                current_frame += 21
            
            # Double stance
            if current_frame + 10 < total_frames:
                support_labels.append({
                    'phase': 'double_stance',
                    'start_frame': current_frame,
                    'end_frame': current_frame + 10
                })
                current_frame += 11
            
            # Left stance
            if current_frame + 20 < total_frames:
                support_labels.append({
                    'phase': 'left_stance',
                    'start_frame': current_frame,
                    'end_frame': current_frame + 20
                })
                current_frame += 21
            
            if current_frame >= total_frames:
                break
        
        print(f"🦶 지지 라벨 생성 완료: {len(support_labels)} 개")
        return support_labels

    def reset_sessions(self):
        """모든 세션 초기화"""
        self.session_manager.reset()
        
        # UI 상태 초기화
        self.play_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.frame_slider.setEnabled(False)
        
        # 그래프 초기화
        self.figure.clear()
        self.canvas.draw()
        
        self.status_label.setText("🔄 세션이 초기화되었습니다. 비디오를 로드하세요.")
        self.update_session_status()
    
    def update_session_status(self):
        """세션 진행 상태 업데이트"""
        current_session, message = self.session_manager.get_session_status()
        
        self.session_status_label.setText(message)
        self.session_progress.setValue(current_session)
        
        # 세션별 버튼 상태 업데이트
        if current_session == 0:
            self.process_session_btn.setEnabled(False)
            self.process_session_btn.setText("비디오를 먼저 로드하세요")
        elif current_session == 1 and self.session_manager.video_path:
            self.process_session_btn.setEnabled(True)
            self.process_session_btn.setText("▶️ 세션 1 시작 (관절 추출)")
        elif current_session == 2:
            self.process_session_btn.setEnabled(True)
            self.process_session_btn.setText("▶️ 세션 2 시작 (보행 계산)")
        elif current_session == 3:
            self.process_session_btn.setEnabled(True)
            self.process_session_btn.setText("🎬 세션 3 활성화 (시각화)")
        else:
            self.process_session_btn.setEnabled(False)
            self.process_session_btn.setText("모든 세션 완료")
    
    def toggle_play(self):
        """재생/일시정지 토글"""
        if not hasattr(self.session_manager, 'video_path') or not self.session_manager.video_path:
            return
            
        if self.is_playing:
            self.timer.stop()
            self.play_btn.setText("▶️ 재생")
            self.is_playing = False
        else:
            # FPS에 맞춰 타이머 설정 (밀리초)
            interval = int(1000 / self.video_player.fps)
            self.timer.start(interval)
            self.play_btn.setText("⏸️ 일시정지")
            self.is_playing = True
    
    def prev_frame(self):
        """이전 프레임"""
        if self.video_player.current_frame > 0:
            new_frame = self.video_player.current_frame - 1
            self.video_player.set_frame(new_frame)
            self.frame_slider.setValue(new_frame)
            self.update_graph_marker()
    
    def next_frame(self):
        """다음 프레임"""
        if self.video_player.current_frame < self.video_player.total_frames - 1:
            new_frame = self.video_player.current_frame + 1
            self.video_player.set_frame(new_frame)
            self.frame_slider.setValue(new_frame)
            self.update_graph_marker()
        else:
            # 끝에 도달하면 재생 중지
            if self.is_playing:
                self.toggle_play()
    
    def slider_changed(self, value):
        """슬라이더 값 변경"""
        self.video_player.set_frame(value)
        self.update_graph_marker()
    
    def update_display_options(self):
        """표시 옵션 업데이트"""
        self.video_player.show_joints = self.joints_check.isChecked()
        self.video_player.show_support_labels = self.labels_check.isChecked()
        self.video_player.show_stride_info = self.stride_check.isChecked()
        
        # 현재 프레임 다시 그리기
        self.video_player.set_frame(self.video_player.current_frame)
    
    def update_graphs(self):
        """그래프 업데이트"""
        if not hasattr(self.session_manager, 'stride_details') or not self.session_manager.stride_details:
            return
            
        self.figure.clear()
        
        # 3개 서브플롯 생성
        ax1 = self.figure.add_subplot(3, 1, 1)
        ax2 = self.figure.add_subplot(3, 1, 2)
        ax3 = self.figure.add_subplot(3, 1, 3)
        
        df = pd.DataFrame(self.session_manager.stride_details)
        
        # 발별로 분리
        left_data = df[df['foot'] == 'left']
        right_data = df[df['foot'] == 'right']
        
        # Stride Time
        if not left_data.empty:
            ax1.plot(left_data['start_time'], left_data['stride_time'], 
                    'o-', color='red', label='Left Foot', markersize=8)
        if not right_data.empty:
            ax1.plot(right_data['start_time'], right_data['stride_time'], 
                    's-', color='blue', label='Right Foot', markersize=8)
        ax1.set_ylabel('Stride Time (s)')
        ax1.set_title('Stride Time vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Stride Length
        if not left_data.empty:
            ax2.plot(left_data['start_time'], left_data['stride_length'], 
                    'o-', color='red', label='Left Foot', markersize=8)
        if not right_data.empty:
            ax2.plot(right_data['start_time'], right_data['stride_length'], 
                    's-', color='blue', label='Right Foot', markersize=8)
        ax2.set_ylabel('Stride Length (m)')
        ax2.set_title('Stride Length vs Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Velocity
        if not left_data.empty:
            ax3.plot(left_data['start_time'], left_data['velocity'], 
                    'o-', color='red', label='Left Foot', markersize=8)
        if not right_data.empty:
            ax3.plot(right_data['start_time'], right_data['velocity'], 
                    's-', color='blue', label='Right Foot', markersize=8)
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title('Velocity vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 현재 시간 마커 저장용
        self.time_lines = [
            ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7),
            ax2.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7),
            ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ]
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_graph_marker(self):
        """그래프의 현재 시간 마커 업데이트"""
        if hasattr(self, 'time_lines') and self.video_player.total_frames > 0:
            current_time = self.video_player.current_frame / self.video_player.fps
            
            for line in self.time_lines:
                line.set_xdata([current_time, current_time])
            
            self.canvas.draw_idle()
    
    def set_support_labels(self, support_labels):
        """지지 라벨 데이터 설정 (외부 호출용 - 기존 호환성 유지)"""
        print(f"🦶 외부에서 지지 라벨 설정: {len(support_labels) if support_labels else 0}개")
        
        # 세션 매니저에 직접 설정하고 세션 1 완료 처리
        if support_labels:
            # 더미 관절 데이터와 타임스탬프 생성 (실제 데이터가 없는 경우)
            if not hasattr(self.session_manager, 'joint_data') or not self.session_manager.joint_data:
                # 지지 라벨 수에 맞는 더미 관절 데이터 생성
                dummy_joint_data = []
                dummy_timestamps = []
                
                for i, label in enumerate(support_labels):
                    # 간단한 더미 관절 데이터
                    dummy_joint_data.append({
                        'left_ankle': {'x': 0.4, 'y': 0.8, 'z': 0.5},
                        'right_ankle': {'x': 0.6, 'y': 0.8, 'z': 0.5},
                        'left_knee': {'x': 0.4, 'y': 0.6, 'z': 0.5},
                        'right_knee': {'x': 0.6, 'y': 0.6, 'z': 0.5}
                    })
                    dummy_timestamps.append(i * (1/30.0))  # 30fps 가정
                
                # 세션 1 완료 처리
                self.session_manager.complete_session1(support_labels, dummy_joint_data, dummy_timestamps)
            else:
                # 기존 관절 데이터가 있는 경우 라벨만 업데이트
                self.session_manager.support_labels = support_labels
                self.session_manager.session1_completed = True
            
            # UI 상태 업데이트
            self.update_session_status()
            self.status_label.setText(f"✅ 외부 라벨 데이터 로드 완료: {len(support_labels)}개 라벨")
            
            print(f"✅ 세션 1 상태 업데이트 완료 (외부 라벨 사용)")
        else:
            print("⚠️ 빈 라벨 데이터가 전달됨")
    
    def load_gait_data(self, csv_path):
        """보행 데이터 로드 (기존 호환성 유지)"""
        try:
            import pandas as pd
            gait_data = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            # CSV 데이터를 stride_details 형식으로 변환
            stride_details = []
            for _, row in gait_data.iterrows():
                stride_detail = {
                    'foot': row.get('발', 'unknown'),
                    'start_time': row.get('시작시간(s)', 0),
                    'end_time': row.get('종료시간(s)', 0),
                    'stride_time': row.get('Stride Time(s)', 0),
                    'stride_length': row.get('Stride Length(m)', 0),
                    'velocity': row.get('Velocity(m/s)', 0),
                    'start_frame': int(row.get('시작시간(s)', 0) * 30),  # 30fps 가정
                    'end_frame': int(row.get('종료시간(s)', 0) * 30)
                }
                stride_details.append(stride_detail)
            
            # 더미 gait_parameters 생성
            gait_parameters = {
                'calculation_method': 'csv_loaded',
                'total_frames': len(stride_details),
                'parameters': {
                    'stride_time': {'mean': gait_data['Stride Time(s)'].mean() if 'Stride Time(s)' in gait_data.columns else 0},
                    'stride_length': {'mean': gait_data['Stride Length(m)'].mean() if 'Stride Length(m)' in gait_data.columns else 0},
                    'velocity': {'mean': gait_data['Velocity(m/s)'].mean() if 'Velocity(m/s)' in gait_data.columns else 0}
                }
            }
            
            # 세션 2 완료 처리
            self.session_manager.complete_session2(gait_parameters, stride_details)
            
            # 그래프 업데이트
            self.update_graphs()
            
            # UI 상태 업데이트
            self.update_session_status()
            self.status_label.setText(f"✅ CSV 데이터 로드 완료: {len(stride_details)}개 stride")
            
            print(f"📊 CSV 보행 데이터 로드 완료: {len(stride_details)}개 stride")
            return True
            
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            return False