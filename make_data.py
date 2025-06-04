#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비디오와 IMU 센서 데이터의 동기화된 수집 시스템
"""

import sys
import json
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import socket

# 전역 설정
VIDEO_FPS = 30
IMU_PORT = 5000
CAMERA_INDEX = 1
OUTPUT_DIR = "experiment_data"


class IMUReceiver(QThread):
    """IMU 데이터 수신 스레드"""
    data_received = pyqtSignal(dict)
    connection_status = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.server_socket = None
        self.client_socket = None
        self.data_buffer = []
        self.recording = False
        self.sync_start_time = None
        self.incomplete_line = ""
        self.total_received = 0
        self.total_errors = 0
        self.first_imu_timestamp = None
        
    def run(self):
        self.running = True
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('', IMU_PORT))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            
            self.connection_status.emit(f"IMU 서버 대기중... (포트: {IMU_PORT})")
            
            while self.running:
                try:
                    if not self.client_socket:
                        self.client_socket, addr = self.server_socket.accept()
                        self.client_socket.settimeout(0.1)
                        self.connection_status.emit(f"IMU 연결됨: {addr[0]}")
                    
                    try:
                        data = self.client_socket.recv(8192).decode('utf-8')
                        if data:
                            if self.incomplete_line:
                                data = self.incomplete_line + data
                                self.incomplete_line = ""
                            
                            lines = data.strip().split('\n')
                            
                            for i, line in enumerate(lines):
                                if line:
                                    if i == len(lines) - 1 and not data.endswith('\n'):
                                        self.incomplete_line = line
                                        continue
                                        
                                    try:
                                        imu_data = json.loads(line)
                                        self.total_received += 1
                                        
                                        if self.recording and self.sync_start_time:
                                            original_timestamp = imu_data.get('timestamp', 0)
                                            
                                            if self.first_imu_timestamp is None:
                                                self.first_imu_timestamp = original_timestamp
                                                imu_data['sync_timestamp'] = 0.0
                                            else:
                                                relative_time = original_timestamp - self.first_imu_timestamp
                                                imu_data['sync_timestamp'] = relative_time
                                            
                                            self.data_buffer.append(imu_data)
                                        
                                        self.data_received.emit(imu_data)
                                    except json.JSONDecodeError:
                                        self.total_errors += 1
                                        if self.total_errors <= 10:
                                            self.connection_status.emit(f"JSON 파싱 오류 #{self.total_errors}: {line[:50]}...")
                        else:
                            self.client_socket = None
                            self.connection_status.emit(f"IMU 연결 끊김 (수신: {self.total_received}, 오류: {self.total_errors})")
                    except socket.timeout:
                        pass
                        
                except socket.timeout:
                    pass
                except Exception as e:
                    if self.client_socket:
                        self.client_socket.close()
                        self.client_socket = None
                    self.connection_status.emit(f"IMU 오류: {str(e)}")
                    
        except Exception as e:
            self.connection_status.emit(f"IMU 서버 오류: {str(e)}")
        finally:
            self.cleanup()
    
    def start_recording(self, sync_time):
        self.sync_start_time = sync_time
        self.recording = True
        self.data_buffer.clear()
        self.total_received = 0
        self.total_errors = 0
        self.first_imu_timestamp = None
        self.connection_status.emit("IMU 녹화 시작")
    
    def stop_recording(self, actual_duration=None):
        self.recording = False
        
        if self.data_buffer and actual_duration is not None:
            self._interpolate_to_30hz(actual_duration)
        
        buffer_count = len(self.data_buffer)
        loss_rate = (self.total_errors / max(self.total_received + self.total_errors, 1)) * 100
        self.connection_status.emit(
            f"IMU 녹화 종료 - 총수신: {self.total_received}, 저장: {buffer_count}, "
            f"오류: {self.total_errors} ({loss_rate:.1f}% 손실)"
        )
        
        return self.data_buffer.copy()
    
    def _interpolate_to_30hz(self, pc_duration):
        if not self.data_buffer:
            return
        
        # 유효 데이터 필터링
        valid_data = [data for data in self.data_buffer if data['sync_timestamp'] <= pc_duration]
        if not valid_data:
            self.data_buffer.clear()
            return
        
        # 30Hz 보간
        downsampled_data = []
        total_frames = int(pc_duration * VIDEO_FPS)
        
        for frame_idx in range(total_frames):
            frame_time = frame_idx / VIDEO_FPS
            
            # 전후 데이터 찾기
            before_data = None
            after_data = None
            
            for data in valid_data:
                if data['sync_timestamp'] <= frame_time:
                    if before_data is None or data['sync_timestamp'] > before_data['sync_timestamp']:
                        before_data = data
                elif data['sync_timestamp'] > frame_time:
                    if after_data is None or data['sync_timestamp'] < after_data['sync_timestamp']:
                        after_data = data
            
            if before_data and after_data:
                # 선형 보간
                t1, t2 = before_data['sync_timestamp'], after_data['sync_timestamp']
                ratio = (frame_time - t1) / (t2 - t1) if t2 != t1 else 0
                
                interpolated_data = {
                    'frame_number': frame_idx,
                    'sync_timestamp': frame_time,
                    'accel': {
                        'x': before_data['accel']['x'] + ratio * (after_data['accel']['x'] - before_data['accel']['x']),
                        'y': before_data['accel']['y'] + ratio * (after_data['accel']['y'] - before_data['accel']['y']),
                        'z': before_data['accel']['z'] + ratio * (after_data['accel']['z'] - before_data['accel']['z'])
                    },
                    'gyro': {
                        'x': before_data['gyro']['x'] + ratio * (after_data['gyro']['x'] - before_data['gyro']['x']),
                        'y': before_data['gyro']['y'] + ratio * (after_data['gyro']['y'] - before_data['gyro']['y']),
                        'z': before_data['gyro']['z'] + ratio * (after_data['gyro']['z'] - before_data['gyro']['z'])
                    }
                }
                downsampled_data.append(interpolated_data)
            elif before_data:
                sample_data = before_data.copy()
                sample_data['frame_number'] = frame_idx
                sample_data['sync_timestamp'] = frame_time
                downsampled_data.append(sample_data)
            elif after_data:
                sample_data = after_data.copy()
                sample_data['frame_number'] = frame_idx
                sample_data['sync_timestamp'] = frame_time
                downsampled_data.append(sample_data)
        
        self.data_buffer.clear()
        self.data_buffer.extend(downsampled_data)
        
        removed_count = len(valid_data) - len(downsampled_data)
        self.connection_status.emit(
            f"IMU 데이터 동기화: {len(valid_data)}개 → {len(downsampled_data)}개 (30Hz 선형보간)"
        )
    
    def stop(self):
        """스레드 중지"""
        self.running = False
        self.cleanup()
        
    def cleanup(self):
        """리소스 정리"""
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


class VideoCapture(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = False
        self.recording = False
        self.video_writer = None
        self.frame_buffer = []
        self.sync_start_time = None
        self.output_path = None
        
    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.cap.isOpened():
            self.status_update.emit("카메라 연결 실패")
            return
            
        self.status_update.emit("카메라 연결됨")
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.status_update.emit(f"비디오: {width}x{height} @ {actual_fps:.1f}fps")
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 프레임 방출 (프리뷰용)
                self.frame_ready.emit(frame)
                
                # 녹화 중이면 버퍼에 저장
                if self.recording:
                    timestamp = time.time() - self.sync_start_time
                    self.frame_buffer.append({
                        'frame': frame.copy(),
                        'timestamp': timestamp
                    })
            else:
                self.status_update.emit("카메라 프레임 읽기 실패")
                break
                
        self.cleanup()
    
    def start_recording(self, sync_time, output_path):
        """녹화 시작"""
        self.sync_start_time = sync_time
        self.output_path = output_path
        self.recording = True
        self.frame_buffer.clear()
        
    def stop_recording(self):
        """녹화 중지 및 저장"""
        self.recording = False
        
        if not self.frame_buffer:
            return None
            
        # 비디오 저장
        first_frame = self.frame_buffer[0]['frame']
        height, width = first_frame.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, VIDEO_FPS, (width, height))
        
        for item in self.frame_buffer:
            out.write(item['frame'])
            
        out.release()
        
        # 타임스탬프 정보 반환
        timestamps = [item['timestamp'] for item in self.frame_buffer]
        self.frame_buffer.clear()
        
        return timestamps
    
    def stop(self):
        """스레드 중지"""
        self.running = False
        
    def cleanup(self):
        """리소스 정리"""
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()


class ExperimentControlGUI(QMainWindow):
    """메인 GUI 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        self.imu_receiver = IMUReceiver()
        self.video_capture = VideoCapture()
        self.sync_start_time = None
        self.sync_end_time = None
        self.is_recording = False
        self.session_count = 0
        self.session_timestamp_str = None
        self.current_session_dir = None
        
        self.init_ui()
        self.setup_connections()
        self.start_threads()
        
    def init_ui(self):
        self.setWindowTitle("실험 데이터 수집 시스템")
        self.setGeometry(100, 100, 1200, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QHBoxLayout(central_widget)
        
        # 왼쪽: 비디오 프리뷰
        left_panel = QVBoxLayout()
        
        # 비디오 디스플레이
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setScaledContents(True)
        left_panel.addWidget(QLabel("비디오 프리뷰"))
        left_panel.addWidget(self.video_label)
        
        # 비디오 상태
        self.video_status = QLabel("비디오: 대기중...")
        left_panel.addWidget(self.video_status)
        
        main_layout.addLayout(left_panel, 2)
        
        # 오른쪽: 제어 패널
        right_panel = QVBoxLayout()
        
        # IMU 데이터 디스플레이
        imu_group = QGroupBox("IMU 데이터")
        imu_layout = QVBoxLayout()
        
        self.imu_status = QLabel("IMU: 연결 대기중...")
        imu_layout.addWidget(self.imu_status)
        
        # IMU 실시간 값 표시
        self.imu_display = QTextEdit()
        self.imu_display.setReadOnly(True)
        self.imu_display.setMaximumHeight(200)
        imu_layout.addWidget(self.imu_display)
        
        imu_group.setLayout(imu_layout)
        right_panel.addWidget(imu_group)
        
        # 세션 제어
        control_group = QGroupBox("실험 제어")
        control_layout = QVBoxLayout()
        
        # 세션 정보
        self.session_info = QLabel("세션: 대기중")
        control_layout.addWidget(self.session_info)
        
        # 타이머
        self.timer_label = QLabel("경과 시간: 00:00:00")
        self.timer_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        control_layout.addWidget(self.timer_label)
        
        # 제어 버튼
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("시작 (F5)")
        self.start_button.setStyleSheet("background-color: green; color: white; font-size: 16px; padding: 10px;")
        self.start_button.clicked.connect(self.start_recording)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("종료 (F6)")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-size: 16px; padding: 10px;")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        button_layout.addWidget(self.stop_button)
        
        control_layout.addLayout(button_layout)
        
        # 저장 설정
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("저장 폴더:"))
        self.save_path_edit = QLineEdit(OUTPUT_DIR)
        save_layout.addWidget(self.save_path_edit)
        self.browse_button = QPushButton("찾아보기...")
        self.browse_button.clicked.connect(self.browse_folder)
        save_layout.addWidget(self.browse_button)
        control_layout.addLayout(save_layout)
        
        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)
        
        # 로그
        log_group = QGroupBox("시스템 로그")
        log_layout = QVBoxLayout()
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)
        
        # 타이머 설정
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(100)  # 100ms 간격
        
        # 단축키 설정
        QShortcut(QKeySequence("F5"), self, self.start_recording)
        QShortcut(QKeySequence("F6"), self, self.stop_recording)
        
        self.log("시스템 초기화 완료")
        
    def setup_connections(self):
        """신호 연결 설정"""
        # IMU 연결
        self.imu_receiver.data_received.connect(self.update_imu_display)
        self.imu_receiver.connection_status.connect(self.update_imu_status)
        
        # 비디오 연결
        self.video_capture.frame_ready.connect(self.update_video_display)
        self.video_capture.status_update.connect(self.update_video_status)
        
    def start_threads(self):
        """백그라운드 스레드 시작"""
        self.imu_receiver.start()
        self.video_capture.start()
        
    def start_recording(self):
        """녹화 시작"""
        if self.is_recording:
            return
            
        # 동기화 시작 시간 기록
        self.sync_start_time = time.time()
        self.is_recording = True
        self.session_count += 1
        
        # 저장 경로 생성
        output_dir = Path(self.save_path_edit.text())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = output_dir / f"session_{timestamp_str}"
        session_dir.mkdir(exist_ok=True)
        
        # 비디오 녹화 시작
        video_path = str(session_dir / "video.mp4")
        self.video_capture.start_recording(self.sync_start_time, video_path)
        
        # IMU 녹화 시작
        self.imu_receiver.start_recording(self.sync_start_time)
        
        # UI 업데이트
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.session_info.setText(f"세션 #{self.session_count}: 녹화중...")
        
        self.session_timestamp_str = timestamp_str
        self.current_session_dir = session_dir
        
        self.log(f"녹화 시작 - 세션 #{self.session_count}")
        self.log(f"저장 경로: {session_dir}")
        
    def stop_recording(self):
        """녹화 종료"""
        if not self.is_recording:
            return
            
        # 동기화 종료 시간 기록
        self.sync_end_time = time.time()
        self.is_recording = False
        
        # 실제 녹화 시간 계산
        actual_duration = self.sync_end_time - self.sync_start_time
        
        # 데이터 수집 중지
        video_timestamps = self.video_capture.stop_recording()
        imu_data = self.imu_receiver.stop_recording(actual_duration)  # 실제 duration 전달
        
        # 저장 경로 (시작할 때 생성한 세션 디렉토리 사용)
        session_dir = self.current_session_dir
        if not session_dir or not session_dir.exists():
            # 세션 디렉토리가 없으면 다시 생성
            output_dir = Path(self.save_path_edit.text())
            output_dir.mkdir(parents=True, exist_ok=True)
            session_dir = output_dir / f"session_{self.session_timestamp_str}"
            session_dir.mkdir(exist_ok=True)
            self.log(f"세션 디렉토리 재생성: {session_dir}")
        
        # IMU 데이터 저장
        if imu_data:
            imu_path = session_dir / "imu_data.csv"
            with open(imu_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'frame_number',    # 비디오 프레임 번호 (0부터 시작)
                    'sync_timestamp',  # 동기화된 타임스탬프
                    'accel_x', 'accel_y', 'accel_z',
                    'gyro_x', 'gyro_y', 'gyro_z'
                ])
                writer.writeheader()
                
                for data in imu_data:
                    writer.writerow({
                        'frame_number': data.get('frame_number', 0),               # 프레임 번호
                        'sync_timestamp': f"{data.get('sync_timestamp', 0):.3f}", # 30fps: 0.033초 단위
                        'accel_x': f"{data['accel']['x']:.3f}",                   # 가속도: 소수점 셋째 자리까지
                        'accel_y': f"{data['accel']['y']:.3f}",
                        'accel_z': f"{data['accel']['z']:.3f}",
                        'gyro_x': f"{data['gyro']['x']:.5f}",                     # 각속도: 소수점 다섯째 자리까지
                        'gyro_y': f"{data['gyro']['y']:.5f}",
                        'gyro_z': f"{data['gyro']['z']:.5f}"
                    })
            self.log(f"IMU 데이터 저장: {len(imu_data)} 샘플 (30Hz 선형보간, 프레임 동기화)")
        
        # 메타데이터 저장
        metadata = {
            'session_id': self.session_count,
            'sync_start_time_seoul': datetime.fromtimestamp(self.sync_start_time).strftime('%Y-%m-%d %H:%M:%S'),  # 서울 시간으로 변환
            'sync_end_time_seoul': datetime.fromtimestamp(self.sync_end_time).strftime('%Y-%m-%d %H:%M:%S'),    # 서울 시간으로 변환
            'duration': self.sync_end_time - self.sync_start_time,  # 유지: 실험 지속 시간 (중요)
            'video_fps': VIDEO_FPS,
            'video_frames': len(video_timestamps) if video_timestamps else 0,
            'imu_hz': VIDEO_FPS,  # IMU도 30Hz로 동기화
            'imu_samples': len(imu_data),
            'sync_method': 'frame_sync_interpolation',  # 프레임 단위 선형보간 동기화
            'timestamp': datetime.now().isoformat()       # 유지: 실험 날짜/시간 기록용
        }
        
        metadata_path = session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # UI 업데이트
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.session_info.setText(f"세션 #{self.session_count}: 저장 완료")
        
        self.log(f"녹화 종료 - 세션 #{self.session_count}")
        self.log(f"총 시간: {metadata['duration']:.2f}초")
        self.log(f"비디오 프레임: {metadata['video_frames']}")
        self.log(f"IMU 샘플: {metadata['imu_samples']}")
        self.log("데이터 저장 완료!")
        
        # 세션 변수 초기화
        self.session_timestamp_str = None
        self.current_session_dir = None
        
    def update_display(self):
        """디스플레이 업데이트"""
        if self.is_recording and self.sync_start_time:
            elapsed = time.time() - self.sync_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f"경과 시간: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
    def update_video_display(self, frame):
        """비디오 프레임 업데이트"""
        # OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # QImage로 변환
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # QPixmap으로 변환하여 표시
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_video_status(self, status):
        """비디오 상태 업데이트"""
        self.video_status.setText(f"비디오: {status}")
        self.log(f"비디오: {status}")
        
    def update_imu_display(self, data):
        """IMU 데이터 디스플레이 업데이트"""
        display_text = f"가속도 (m/s²):\n"
        display_text += f"  X: {data['accel']['x']:.3f}\n"
        display_text += f"  Y: {data['accel']['y']:.3f}\n"
        display_text += f"  Z: {data['accel']['z']:.3f}\n\n"
        display_text += f"자이로 (°/s):\n"
        display_text += f"  X: {data['gyro']['x']:.3f}\n"
        display_text += f"  Y: {data['gyro']['y']:.3f}\n"
        display_text += f"  Z: {data['gyro']['z']:.3f}"
        
        self.imu_display.setText(display_text)
        
    def update_imu_status(self, status):
        """IMU 상태 업데이트"""
        self.imu_status.setText(f"IMU: {status}")
        self.log(f"IMU: {status}")
        
    def browse_folder(self):
        """저장 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(self, "저장 폴더 선택")
        if folder:
            self.save_path_edit.setText(folder)
            
    def log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        
    def closeEvent(self, event):
        """종료 이벤트"""
        # 녹화 중이면 중지
        if self.is_recording:
            self.stop_recording()
            
        # 스레드 정지
        self.imu_receiver.stop()
        self.video_capture.stop()
        
        # 스레드 종료 대기
        self.imu_receiver.wait()
        self.video_capture.wait()
        
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성
    window = ExperimentControlGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()