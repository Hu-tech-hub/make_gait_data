# ======================
# 2. gait_analyzer_gui.py
# ======================
"""
gait_analyzer_gui.py - PyQt5 기반 보행 분석 GUI

이 모듈은 다음 기능을 제공합니다:
1. 비디오 재생 및 프레임별 이벤트 표시
2. 발목 X좌표 시계열 그래프 시각화
3. HS/TO 이벤트 마커 표시
4. 실시간 분석 및 결과 저장
"""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import numpy as np
import cv2
from typing import Optional

# gait_class 모듈 임포트
from gait_class import GaitAnalyzer, GaitEvent


class VideoPlayer:
    """비디오 플레이어 (재생 기능 없음, 시크만 지원)"""
    
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 30
        
    def load_video(self, video_path: str):
        """비디오 로드"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """특정 프레임 가져오기"""
        if self.cap and 0 <= frame_idx < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def release(self):
        """리소스 해제"""
        if self.cap:
            self.cap.release()
            self.cap = None


class GaitAnalyzerGUI(QMainWindow):
    """보행 분석 GUI 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.video_player = VideoPlayer()
        self.current_frame = 0
        
        # UI 설정
        self.setWindowTitle("보행 분석 시스템 (Gait Analysis System)")
        self.setGeometry(100, 100, 1400, 900)
        
        # 메인 위젯 및 레이아웃
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 툴바 생성
        self.create_toolbar()
        
        # 콘텐츠 영역 (비디오 + 그래프)
        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)
        
        # 좌측: 비디오 및 컨트롤
        video_widget = self.create_video_widget()
        content_layout.addWidget(video_widget, 1)
        
        # 우측: 그래프 및 분석 옵션
        analysis_widget = self.create_analysis_widget()
        content_layout.addWidget(analysis_widget, 1)
        
        # 상태바
        self.statusBar().showMessage("준비")
    
    def create_toolbar(self):
        """툴바 생성"""
        toolbar = self.addToolBar("Main")
        
        # 파일 열기
        open_action = QAction(QIcon.fromTheme("document-open"), "비디오 열기", self)
        open_action.triggered.connect(self.open_video)
        toolbar.addAction(open_action)
        
        # 분석 시작
        analyze_action = QAction(QIcon.fromTheme("system-run"), "분석 시작", self)
        analyze_action.triggered.connect(self.start_analysis)
        toolbar.addAction(analyze_action)
        
        # 결과 저장
        save_action = QAction(QIcon.fromTheme("document-save"), "결과 저장", self)
        save_action.triggered.connect(self.save_results)
        toolbar.addAction(save_action)
        
        # 재생 기능 제거 (시크바를 통한 수동 탐색만 지원)
    
    def create_video_widget(self) -> QWidget:
        """비디오 위젯 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 비디오 디스플레이 (고정 크기)
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)  # 고정 크기로 설정
        self.video_label.setStyleSheet("border: 1px solid black; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)  # 크기 조정 방지
        layout.addWidget(self.video_label)
        
        # 로딩바 (초기에는 숨김)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 시크바
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.valueChanged.connect(self.seek_video)
        layout.addWidget(self.video_slider)
        
        # 프레임 정보
        self.frame_info_label = QLabel("프레임: 0 / 0")
        layout.addWidget(self.frame_info_label)
        
        return widget
    
    def create_analysis_widget(self) -> QWidget:
        """분석 위젯 생성"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 분석 옵션 그룹
        options_group = QGroupBox("분석 옵션")
        options_layout = QVBoxLayout(options_group)
        
        # 체크박스들
        self.show_left_ankle_cb = QCheckBox("좌측 발목 표시")
        self.show_left_ankle_cb.setChecked(True)
        self.show_left_ankle_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_left_ankle_cb)
        
        self.show_right_ankle_cb = QCheckBox("우측 발목 표시")
        self.show_right_ankle_cb.setChecked(True)
        self.show_right_ankle_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_right_ankle_cb)
        
        self.show_hs_cb = QCheckBox("Heel Strike (HS) 표시")
        self.show_hs_cb.setChecked(True)
        self.show_hs_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_hs_cb)
        
        self.show_to_cb = QCheckBox("Toe Off (TO) 표시")
        self.show_to_cb.setChecked(True)
        self.show_to_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_to_cb)
        
        self.show_filtered_cb = QCheckBox("필터링된 신호 표시")
        self.show_filtered_cb.setChecked(True)
        self.show_filtered_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_filtered_cb)
        
        self.show_phases_cb = QCheckBox("보행 단계 표시")
        self.show_phases_cb.setChecked(True)
        self.show_phases_cb.stateChanged.connect(self.update_graph)
        options_layout.addWidget(self.show_phases_cb)
        
        layout.addWidget(options_group)
        
        # 그래프 위젯
        self.graph_widget = pg.PlotWidget(title="발목 X좌표 시계열")
        self.graph_widget.setLabel('left', 'X 좌표 (정규화)')
        self.graph_widget.setLabel('bottom', '프레임')
        self.graph_widget.addLegend()
        layout.addWidget(self.graph_widget)
        
        # 보행 단계 범례
        legend_group = QGroupBox("보행 단계 범례")
        legend_layout = QVBoxLayout(legend_group)
        
        legend_items = [
            ("이중지지 (Double Support)", "background-color: rgba(255, 200, 200, 120);"),
            ("단일지지 좌측 (Single Support L)", "background-color: rgba(200, 200, 255, 120);"),
            ("단일지지 우측 (Single Support R)", "background-color: rgba(200, 255, 200, 120);"),
            ("비보행 (Non-gait)", "background-color: rgba(200, 200, 200, 120);")
        ]
        
        for text, style in legend_items:
            label = QLabel(text)
            label.setStyleSheet(f"padding: 2px; {style}")
            legend_layout.addWidget(label)
        
        layout.addWidget(legend_group)
        
        # 현재 프레임 라인
        self.current_frame_line = pg.InfiniteLine(
            pos=0, 
            angle=90, 
            pen=pg.mkPen('y', width=2, style=Qt.DashLine)
        )
        self.graph_widget.addItem(self.current_frame_line)
        
        # 분석 정보
        info_group = QGroupBox("분석 정보")
        info_layout = QFormLayout(info_group)
        
        self.direction_label = QLabel("-")
        info_layout.addRow("보행 방향:", self.direction_label)
        
        self.events_label = QLabel("-")
        info_layout.addRow("검출된 이벤트:", self.events_label)
        
        self.phases_label = QLabel("-")
        info_layout.addRow("보행 단계:", self.phases_label)
        
        layout.addWidget(info_group)
        
        return widget
    
    def open_video(self):
        """비디오 파일 열기"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "비디오 파일 선택", 
            "", 
            "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if file_path:
            # IMU 파일 확인
            dir_path = os.path.dirname(file_path)
            imu_path = os.path.join(dir_path, "imu_data.csv")
            
            if not os.path.exists(imu_path):
                imu_path = None
                QMessageBox.warning(self, "경고", "IMU 데이터 파일을 찾을 수 없습니다.")
            
            # 분석기 초기화
            self.analyzer = GaitAnalyzer(file_path, imu_path)
            
            # 비디오 로드
            self.video_player.load_video(file_path)
            self.video_slider.setMaximum(self.analyzer.total_frames - 1)
            
            # 첫 프레임 표시
            self.seek_video(0)
            
            self.statusBar().showMessage(f"비디오 로드 완료: {os.path.basename(file_path)}")
    
    def start_analysis(self):
        """분석 시작"""
        if not self.analyzer:
            QMessageBox.warning(self, "경고", "먼저 비디오를 열어주세요.")
            return
        
        # 로딩바 표시 및 초기화
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        
        try:
            # 1. 보행 방향 감지 (10%)
            self.progress_bar.setValue(10)
            self.progress_bar.setFormat("방향 감지 중... (%p%)")
            self.statusBar().showMessage("보행 방향 감지 중...")
            QApplication.processEvents()
            
            direction = self.analyzer.detect_walking_direction()
            self.direction_label.setText(f"{direction} (→)" if direction == "forward" else f"{direction} (←)")
            
            # 2. 포즈 추출 (20% -> 70%)
            self.progress_bar.setValue(20)
            self.progress_bar.setFormat("포즈 추출 중... (0/0)")
            self.statusBar().showMessage("발목 좌표 추출 중...")
            QApplication.processEvents()
            
            # 진행률 콜백 함수 정의
            def update_pose_progress(current, total):
                # 20%부터 70%까지 매핑 (50% 범위)
                progress = 20 + int((current / total) * 50)
                self.progress_bar.setValue(progress)
                self.progress_bar.setFormat(f"포즈 추출 중... ({current}/{total} 프레임)")
                QApplication.processEvents()
            
            self.analyzer.extract_pose_landmarks(progress_callback=update_pose_progress)
            
            # 3. 보행 이벤트 검출 (70% -> 90%)
            self.progress_bar.setValue(70)
            self.progress_bar.setFormat("이벤트 검출 중... (%p%)")
            self.statusBar().showMessage("HS/TO 이벤트 검출 중...")
            QApplication.processEvents()
            
            events = self.analyzer.detect_gait_events()
            self.events_label.setText(f"{len(events)}개")
            
            # 보행 단계 분석
            phases = self.analyzer.analyze_gait_phases()
            ds_count = sum(1 for p in phases if p['phase'] == 'double_support')
            ss_count = sum(1 for p in phases if 'single_support' in p['phase'])
            non_count = sum(1 for p in phases if p['phase'] == 'non_gait')
            self.phases_label.setText(f"DS:{ds_count}, SS:{ss_count}, Non:{non_count}")
            
            # 4. 그래프 업데이트 (90% -> 100%)
            self.progress_bar.setValue(90)
            self.progress_bar.setFormat("그래프 생성 중... (%p%)")
            self.statusBar().showMessage("그래프 업데이트 중...")
            QApplication.processEvents()
            
            self.update_graph()
            
            # 완료
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("완료! (%p%)")
            self.statusBar().showMessage("분석 완료")
            
            # 2초 후 로딩바 숨김
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "오류", f"분석 중 오류 발생: {str(e)}")
            self.statusBar().showMessage("분석 실패")
    
    def update_graph(self):
        """그래프 업데이트"""
        if not self.analyzer or len(self.analyzer.landmarks_data) == 0:
            return
        
        # 그래프 초기화
        self.graph_widget.clear()
        self.graph_widget.addItem(self.current_frame_line)
        
        frames = np.arange(len(self.analyzer.landmarks_data))
        
        # 발목 X좌표 데이터
        if self.show_filtered_cb.isChecked() and len(self.analyzer.gait_events) > 0:
            # 필터링된 데이터
            left_x, right_x = self.analyzer.get_filtered_ankle_data()
        else:
            # 원본 데이터
            left_x = self.analyzer.landmarks_data['left_ankle_x'].values
            right_x = self.analyzer.landmarks_data['right_ankle_x'].values
        
        # 좌측 발목
        if self.show_left_ankle_cb.isChecked():
            self.graph_widget.plot(
                frames, left_x, 
                pen=pg.mkPen('b', width=2), 
                name='Left Ankle'
            )
        
        # 우측 발목
        if self.show_right_ankle_cb.isChecked():
            self.graph_widget.plot(
                frames, right_x, 
                pen=pg.mkPen('r', width=2), 
                name='Right Ankle'
            )
        
        # 보행 단계 배경색 표시
        if len(self.analyzer.gait_events) > 0 and self.show_phases_cb.isChecked():
            phases = self.analyzer.analyze_gait_phases()
            y_min = min(left_x.min(), right_x.min()) - 0.02
            y_max = max(left_x.max(), right_x.max()) + 0.02
            
            for phase_info in phases:
                start_frame = phase_info['start_frame']
                end_frame = phase_info['end_frame']
                phase = phase_info['phase']
                
                # 단계별 색상 정의
                if phase == 'double_support':
                    color = (255, 200, 200, 80)  # 연한 빨강 (RGBA)
                elif phase == 'single_support_left':
                    color = (200, 200, 255, 80)  # 연한 파랑
                elif phase == 'single_support_right':
                    color = (200, 255, 200, 80)  # 연한 초록
                else:  # non_gait
                    color = (200, 200, 200, 80)  # 연한 회색
                
                # 영역 표시를 위한 LinearRegionItem 추가
                region = pg.LinearRegionItem(
                    values=(start_frame, end_frame),
                    orientation='vertical',
                    brush=pg.mkBrush(color),
                    movable=False
                )
                self.graph_widget.addItem(region)
        
        # 이벤트 마커
        if len(self.analyzer.gait_events) > 0:
            for event in self.analyzer.gait_events:
                if event.event_type == "HS" and self.show_hs_cb.isChecked():
                    # Heel Strike - 원형 마커
                    color = 'b' if event.foot == "left" else 'r'
                    y_val = left_x[event.frame_idx] if event.foot == "left" else right_x[event.frame_idx]
                    self.graph_widget.plot(
                        [event.frame_idx], [y_val],
                        pen=None,
                        symbol='o',
                        symbolBrush=color,
                        symbolSize=10
                    )
                elif event.event_type == "TO" and self.show_to_cb.isChecked():
                    # Toe Off - 삼각형 마커
                    color = 'c' if event.foot == "left" else 'm'
                    y_val = left_x[event.frame_idx] if event.foot == "left" else right_x[event.frame_idx]
                    self.graph_widget.plot(
                        [event.frame_idx], [y_val],
                        pen=None,
                        symbol='t',
                        symbolBrush=color,
                        symbolSize=10
                    )
    
    def seek_video(self, value):
        """비디오 시크 (수동 탐색)"""
        if not self.video_player.cap:
            return
        
        self.current_frame = value
        frame = self.video_player.get_frame(value)
        if frame is not None:
            self.update_frame(frame, value)
            self.update_position(value)
    
    def update_frame(self, frame: np.ndarray, frame_idx: int):
        """프레임 업데이트"""
        self.current_frame = frame_idx
        
        # OpenCV BGR을 Qt RGB로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # 발목 관절에 HS/TO 이벤트 시각화
        if self.analyzer and len(self.analyzer.gait_events) > 0:
            # 현재 프레임의 발목 좌표 가져오기
            if frame_idx < len(self.analyzer.landmarks_data):
                landmarks_row = self.analyzer.landmarks_data.iloc[frame_idx]
                
                # 발목 좌표 (정규화된 좌표를 픽셀 좌표로 변환)
                left_ankle_x = int(landmarks_row['left_ankle_x'] * w)
                left_ankle_y = int(0.8 * h)  # 발목 추정 위치 (화면 하단 80%)
                right_ankle_x = int(landmarks_row['right_ankle_x'] * w)
                right_ankle_y = int(0.8 * h)
                
                # 현재 프레임에서 발생하는 이벤트 확인
                current_events = [e for e in self.analyzer.gait_events if e.frame_idx == frame_idx]
                
                for event in current_events:
                    if event.foot == "left":
                        x, y = left_ankle_x, left_ankle_y
                    else:
                        x, y = right_ankle_x, right_ankle_y
                    
                    # HS는 빨간색 원, TO는 파란색 원
                    if event.event_type == "HS":
                        color = (255, 0, 0)  # 빨간색 (RGB)
                        text = "HS"
                    else:  # TO
                        color = (0, 0, 255)  # 파란색 (RGB)
                        text = "TO"
                    
                    # 원 그리기
                    cv2.circle(frame_rgb, (x, y), 15, color, 3)
                    
                    # 텍스트 표시
                    cv2.putText(
                        frame_rgb, text,
                        (x - 15, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2
                    )
                    
                    # 발 표시 (L/R)
                    foot_text = "L" if event.foot == "left" else "R"
                    cv2.putText(
                        frame_rgb, foot_text,
                        (x - 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1
                    )
                
                # 근처 프레임 이벤트도 표시 (약간 투명하게)
                nearby_events = [e for e in self.analyzer.gait_events 
                               if 1 <= abs(e.frame_idx - frame_idx) <= 3]
                
                for event in nearby_events:
                    if event.foot == "left":
                        x, y = left_ankle_x, left_ankle_y
                    else:
                        x, y = right_ankle_x, right_ankle_y
                    
                    # 거리에 따른 투명도 조절
                    distance = abs(event.frame_idx - frame_idx)
                    alpha = 0.3 / distance  # 거리가 멀수록 투명
                    
                    if event.event_type == "HS":
                        color = (int(255 * alpha), 0, 0)
                    else:
                        color = (0, 0, int(255 * alpha))
                    
                    # 작은 원 그리기
                    cv2.circle(frame_rgb, (x, y), 8, color, 2)
        
        # QImage로 변환
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # QLabel에 표시 (고정 크기 유지)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            640, 480,  # 고정 크기 사용
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_position(self, frame_idx: int):
        """위치 업데이트"""
        self.video_slider.blockSignals(True)
        self.video_slider.setValue(frame_idx)
        self.video_slider.blockSignals(False)
        
        if self.analyzer:
            total = self.analyzer.total_frames
            self.frame_info_label.setText(f"프레임: {frame_idx} / {total}")
            
            # 그래프의 현재 프레임 라인 업데이트
            self.current_frame_line.setPos(frame_idx)
    
    def save_results(self):
        """결과 저장"""
        if not self.analyzer or len(self.analyzer.gait_events) == 0:
            QMessageBox.warning(self, "경고", "저장할 분석 결과가 없습니다.")
            return
        
        # 저장 디렉토리 선택
        dir_path = QFileDialog.getExistingDirectory(self, "저장 디렉토리 선택")
        
        if dir_path:
            try:
                self.analyzer.save_results(dir_path)
                
                # IMU 데이터 포함 여부에 따른 메시지
                files_list = [
                    "- gait_events.csv (이벤트 목록)",
                    "- gait_phases.csv (보행 단계 정보)",
                    "- event_timeline.csv (프레임별 타임라인)",
                    "- analysis_summary.csv (분석 요약)"
                ]
                
                if self.analyzer.imu_data is not None:
                    files_list.append("- imu_data_labeled.csv (라벨링된 IMU 데이터)")
                
                QMessageBox.information(
                    self, 
                    "완료", 
                    f"결과가 CSV 형식으로 저장되었습니다:\n{dir_path}\n\n"
                    f"저장된 파일:\n" + "\n".join(files_list)
                )
            except Exception as e:
                QMessageBox.critical(self, "오류", f"저장 중 오류 발생: {str(e)}")
    
    def closeEvent(self, event):
        """종료 이벤트"""
        self.video_player.release()
        event.accept()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    # 메인 윈도우 생성 및 표시
    window = GaitAnalyzerGUI()
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
