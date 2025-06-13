"""
batch_stride_analyzer.py - 배치 Stride 분석 GUI

📊 데이터 전처리 파이프라인 Step 2: 보폭 분석

gait_calculation_engine.py를 사용하여 여러 세션을 일괄 처리해서
각 세션마다 stride 분석 결과 CSV를 생성합니다.

주요 기능:
- 비디오에서 MediaPipe로 관절 좌표 추출
- Support labels와 자동 매칭 및 동기화
- Phase 기반 stride cycle 분석
- 피험자별 신장 정보 활용한 보폭 계산

입력: experiment_data/SA01/gait_type/session_xxx/video.mp4 + support_label_data/SA01/S01T01R01_support_labels.csv
출력: stride_analysis_results/S01T01R01_stride_labels.csv
"""

import sys
import os
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from typing import List, Dict
import pandas as pd
from pathlib import Path
from datetime import datetime
import cv2
import mediapipe as mp
import numpy as np

# 로컬 모듈
from gait_calculation_engine import GaitCalculationEngine
from gait_param_class import GaitAnalysisUtils, GaitAnalysisConfig


class BatchStrideWorker(QThread):
    """배치 Stride 분석 워커 스레드"""
    
    # 시그널 정의
    progress_updated = pyqtSignal(int, int)  # current, total
    session_started = pyqtSignal(str)        # session_path
    session_completed = pyqtSignal(str, bool, str, dict)  # session_path, success, message, stats
    log_message = pyqtSignal(str)            # log message
    batch_completed = pyqtSignal(str, dict)  # output_folder, final_stats
    
    def __init__(self, selected_sessions: List[str], output_folder: str):
        super().__init__()
        self.selected_sessions = selected_sessions
        self.output_folder = output_folder
        self.is_running = True
        
        # 통계 추적
        self.total_strides = 0
        self.processed_sessions = 0
        self.failed_sessions = 0
    
    def stop(self):
        """처리 중단"""
        self.is_running = False
    
    def run(self):
        """배치 Stride 분석 실행"""
        total_sessions = len(self.selected_sessions)
        
        self.log_message.emit(f"🔧 배치 Stride 분석 시작: {total_sessions}개 세션")
        
        for i, session_path in enumerate(self.selected_sessions):
            if not self.is_running:
                self.log_message.emit("사용자에 의해 처리가 중단되었습니다.")
                break
            
            self.session_started.emit(session_path)
            self.progress_updated.emit(i, total_sessions)
            
            try:
                # 세션 처리
                success, message, stats = self.process_session(session_path)
                
                if success:
                    self.total_strides += stats.get('strides', 0)
                    self.processed_sessions += 1
                else:
                    self.failed_sessions += 1
                
                self.session_completed.emit(session_path, success, message, stats)
                
            except Exception as e:
                error_msg = f"처리 중 오류 발생: {str(e)}"
                self.session_completed.emit(session_path, False, error_msg, {})
                self.failed_sessions += 1
        
        self.progress_updated.emit(total_sessions, total_sessions)
        
        # 최종 통계
        final_stats = {
            'total_sessions': total_sessions,
            'processed_sessions': self.processed_sessions,
            'failed_sessions': self.failed_sessions,
            'total_strides': self.total_strides
        }
        
        self.batch_completed.emit(self.output_folder, final_stats)
        self.log_message.emit("✅ 배치 Stride 분석 완료")
    
    def process_session(self, session_path: str) -> tuple:
        """개별 세션에서 Stride 분석"""
        try:
            # 1. 비디오 파일 찾기
            video_filename = GaitAnalysisUtils.find_video_file(session_path)
            if not video_filename:
                return False, "비디오 파일을 찾을 수 없습니다", {}
            
            video_path = os.path.join(session_path, video_filename)
            
            # 2. 세션 경로에서 라벨 파일 매칭
            support_file_path = self.find_matching_support_labels(session_path)
            if not support_file_path:
                return False, "매칭되는 support_labels 파일을 찾을 수 없습니다", {}
            
            # 3. 비디오에서 pose 추출
            joint_data, timestamps = self.extract_pose_from_video(video_path)
            if not joint_data:
                return False, "비디오에서 pose 추출 실패", {}
            
            # 4. Support labels 로드
            support_labels = self.load_support_labels(support_file_path)
            if not support_labels:
                return False, "Support labels 로드 실패", {}
            
            # 5. GaitCalculationEngine으로 분석
            engine = GaitCalculationEngine(fps=30.0, video_path=video_path)
            
            results = engine.calculate_gait_parameters(
                joint_data_list=joint_data,
                timestamps=timestamps,
                support_labels=support_labels,
                use_phase_method=True  # Phase 기반 방법 사용
            )
            
            if 'error' in results:
                return False, f"분석 실패: {results['error']}", {}
            
            # 6. 결과를 CSV로 저장
            csv_path = self.save_results_to_csv(session_path, results)
            
            # 통계 생성
            stats = {
                'strides': results['parameters']['stride_time']['count'],
                'csv_path': csv_path,
                'mean_stride_time': results['parameters']['stride_time']['mean'],
                'mean_stride_length': results['parameters']['stride_length']['mean'],
                'mean_velocity': results['parameters']['velocity']['mean']
            }
            
            return True, f"성공: {stats['strides']}개 stride 분석", stats
            
        except Exception as e:
            return False, f"오류: {str(e)}", {}
    
    def find_matching_support_labels(self, session_path: str) -> str:
        """세션 경로에서 매칭되는 support_labels 파일 찾기"""
        try:
            # 세션 경로 파싱: experiment_data/SA01/normal_gait/session_20250604_210219
            path_parts = session_path.replace('\\', '/').split('/')
            
            # Subject ID (SA01) 찾기
            subject_id = None
            gait_type = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('SA'):
                    subject_id = part  # SA01
                    if i + 1 < len(path_parts):
                        gait_type = path_parts[i + 1]  # normal_gait
                    break
            
            if not subject_id or not gait_type:
                return None
            
            # Run number 계산 (같은 보행 타입의 세션들 중 순서)
            parent_dir = os.path.dirname(session_path)
            if os.path.exists(parent_dir):
                all_sessions = [s for s in os.listdir(parent_dir) 
                               if os.path.isdir(os.path.join(parent_dir, s))]
                all_sessions.sort()  # 타임스탬프 순 정렬
                
                current_session = os.path.basename(session_path)
                try:
                    session_index = all_sessions.index(current_session)
                    run_num = f"R{session_index + 1:02d}"  # R01, R02, ...
                except ValueError:
                    run_num = "R01"  # 기본값
            else:
                run_num = "R01"
            
            # 라벨 파일명 생성
            label_filename = GaitAnalysisUtils.build_label_filename(subject_id, gait_type, run_num)
            
            # support_label_data 폴더에서 파일 찾기
            label_path = os.path.join(GaitAnalysisConfig.SUPPORT_LABEL_DATA_PATH, subject_id, label_filename)
            
            if os.path.exists(label_path):
                return label_path
            else:
                # 혹시 다른 Run 번호로 찾아보기
                support_dir = os.path.join(GaitAnalysisConfig.SUPPORT_LABEL_DATA_PATH, subject_id)
                if os.path.exists(support_dir):
                    subject_code = GaitAnalysisUtils.get_subject_code(subject_id)
                    task_code = GaitAnalysisUtils.get_task_code(gait_type)
                    pattern = f"{subject_code}{task_code}R*_support_labels.csv"
                    
                    import glob
                    matches = glob.glob(os.path.join(support_dir, pattern))
                    if matches:
                        matches.sort()
                        # 세션 인덱스에 맞는 파일 선택
                        if session_index < len(matches):
                            return matches[session_index]
                        else:
                            return matches[0]  # 첫 번째 파일
                
                return None
                
        except Exception as e:
            print(f"Support labels 매칭 실패: {e}")
            return None
    
    def extract_pose_from_video(self, video_path: str):
        """비디오에서 MediaPipe로 pose 추출"""
        try:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, None
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            joint_data = []
            timestamps = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB로 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Pose 추출
                results = pose.process(rgb_frame)
                
                timestamp = frame_count / fps
                frame_joints = {}
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 필요한 관절만 추출 (발목, 무릎 - 픽셀-미터 비율 계산용)
                    joint_mapping = {
                        'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
                        'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
                        'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
                        'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE
                    }
                    
                    for joint_name, landmark_id in joint_mapping.items():
                        landmark = landmarks[landmark_id.value]
                        frame_joints[joint_name] = {
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        }
                
                joint_data.append(frame_joints)
                timestamps.append(timestamp)
                frame_count += 1
                
                # 진행상황 출력 (선택적)
                if frame_count % 100 == 0:
                    print(f"  Pose 추출 중... {frame_count} frames")
            
            cap.release()
            pose.close()
            
            print(f"  Pose 추출 완료: {frame_count} frames")
            return joint_data, timestamps
            
        except Exception as e:
            print(f"Pose 추출 실패: {e}")
            return None, None
    
    def load_support_labels(self, support_file: str):
        """support labels 데이터 로드"""
        try:
            if support_file.endswith('.json'):
                with open(support_file, 'r') as f:
                    return json.load(f)
            
            elif support_file.endswith('.csv'):
                df = pd.read_csv(support_file)
                
                # CSV를 dict 리스트로 변환
                labels = []
                for _, row in df.iterrows():
                    label = {
                        'phase': row.get('phase', row.get('Phase', '')),
                        'start_frame': int(row.get('start_frame', row.get('Start_Frame', 0))),
                        'end_frame': int(row.get('end_frame', row.get('End_Frame', 0)))
                    }
                    labels.append(label)
                
                return labels
                
        except Exception as e:
            print(f"Support labels 로드 실패: {e}")
            return None
    
    def save_results_to_csv(self, session_path: str, results: dict) -> str:
        """분석 결과를 CSV로 저장 (라벨링 데이터 이름 기반)"""
        # 피험자별 키 정보 (cm)
        subject_heights = {
            'SA01': 175,
            'SA02': 170,
            'SA03': 180,
            'SA04': 160,
            'SA05': 160
        }
        
        # 세션 경로에서 피험자 ID 추출
        path_parts = session_path.replace('\\', '/').split('/')
        subject_id = None
        for part in path_parts:
            if part.startswith('SA'):
                subject_id = part
                break
        
        # 해당 피험자의 키 가져오기
        subject_height = subject_heights.get(subject_id, 170) if subject_id else 170  # 기본값 170cm
        
        # 매칭되는 support_labels 파일 경로 가져오기
        support_file_path = self.find_matching_support_labels(session_path)
        
        if support_file_path:
            # support_labels 파일명에서 _support_labels를 _stride_labels로 변경
            support_filename = os.path.basename(support_file_path)
            csv_filename = support_filename.replace('_support_labels.csv', '_stride_labels.csv')
        else:
            # 매칭 실패시 기본 명명 규칙 사용
            session_name = os.path.basename(session_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"stride_analysis_{timestamp}.csv"
        
        csv_path = os.path.join(self.output_folder, csv_filename)
        
        # CSV 데이터 준비
        csv_data = []
        
        for i, detail in enumerate(results['details']):
            row = {
                '번호': i + 1,
                '피험자ID': subject_id,
                '키(cm)': subject_height,
                '발': detail['foot'],
                '시작프레임': detail['start_frame'],
                '종료프레임': detail['end_frame'],
                '시작시간(s)': detail['start_time'],
                '종료시간(s)': detail['end_time'],
                'Stride Time(s)': detail['stride_time'],
                'Stride Length(m)': detail['stride_length'],
                'Velocity(m/s)': detail['velocity']
            }
            csv_data.append(row)
        
        # CSV 저장
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        return csv_path


class SessionTreeWidget(QTreeWidget):
    """세션 선택 트리 위젯 (support_labels가 있는 세션만 표시)"""
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("분석 가능한 세션")
        
        # 컬럼 설정
        self.setColumnCount(2)
        self.setHeaderLabels(["세션", "상태"])
        
        # 체크박스 관련 연결을 로드 후에 설정
        self.load_experiment_data()
        
        # 시그널 연결 (로드 후에)
        self.itemChanged.connect(self.on_item_changed)
        
        # 업데이트 콜백 설정
        self.update_callback = None
        
        print(f"DEBUG: TreeWidget 초기화 완료, 아이템 수: {self.topLevelItemCount()}")
    
    def set_update_callback(self, callback):
        """선택 정보 업데이트 콜백 설정"""
        self.update_callback = callback
    
    def load_experiment_data(self):
        """experiment_data 폴더 스캔 (비디오가 있고 매칭되는 support_labels가 있는 세션만)"""
        self.clear()
        
        experiment_data_path = GaitAnalysisConfig.EXPERIMENT_DATA_PATH
        
        if not os.path.exists(experiment_data_path):
            no_data_item = QTreeWidgetItem(["experiment_data 폴더를 찾을 수 없습니다", "❌"])
            self.addTopLevelItem(no_data_item)
            return
        
        valid_sessions = []
        
        # 모든 세션 폴더 검색
        for root, dirs, files in os.walk(experiment_data_path):
            # 비디오 파일이 있는지 확인
            has_video = GaitAnalysisUtils.find_video_file(root) is not None
            
            if has_video:
                # 매칭되는 support_labels 파일이 있는지 확인
                if self.check_support_labels_exist(root):
                    valid_sessions.append(root)
        
        if not valid_sessions:
            no_data_item = QTreeWidgetItem(["분석 가능한 세션이 없습니다", "❌"])
            self.addTopLevelItem(no_data_item)
            return
        
        print(f"🔍 분석 가능한 세션: {len(valid_sessions)}개")
        
        # 피험자별로 그룹화
        subjects = {}
        for session_path in valid_sessions:
            path_parts = session_path.replace('\\', '/').split('/')
            subject_id = None
            for part in path_parts:
                if part.startswith('SA'):
                    subject_id = part
                    break
            
            if subject_id:
                if subject_id not in subjects:
                    subjects[subject_id] = []
                subjects[subject_id].append(session_path)
        
        # 트리 구성 (batch_gait_analyzer.py 방식 사용)
        for subject_id, sessions in subjects.items():
            print(f"DEBUG: 피험자 {subject_id} 생성 중, 세션 수: {len(sessions)}")
            
            # 피험자 노드는 체크박스 없음 (라벨만)
            subject_item = QTreeWidgetItem([f"{subject_id} ({len(sessions)}개 세션)", "📁"])
            # Qt.ItemIsTristate 제거하여 부모-자식 연동 방지
            self.addTopLevelItem(subject_item)
            
            for session_path in sessions:
                gait_type = Path(session_path).parent.name
                session_name = Path(session_path).name
                
                # 세션 노드만 체크박스 있음
                session_item = QTreeWidgetItem([f"{gait_type}/{session_name}", "✅"])
                session_item.setFlags(session_item.flags() | Qt.ItemIsUserCheckable)
                session_item.setCheckState(0, Qt.Unchecked)
                session_item.setData(0, Qt.UserRole, session_path)
                subject_item.addChild(session_item)
                
                print(f"DEBUG: 세션 추가됨: {gait_type}/{session_name}")
        
        self.expandAll()
        print(f"DEBUG: 트리 구성 완료, 최상위 아이템 수: {self.topLevelItemCount()}")
    
    def check_support_labels_exist(self, session_path: str) -> bool:
        """세션에 매칭되는 support_labels 파일이 있는지 확인"""
        try:
            # 세션 경로 파싱
            path_parts = session_path.replace('\\', '/').split('/')
            
            subject_id = None
            gait_type = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('SA'):
                    subject_id = part
                    if i + 1 < len(path_parts):
                        gait_type = path_parts[i + 1]
                    break
            
            if not subject_id or not gait_type:
                return False
            
            # support_label_data 폴더에서 매칭되는 파일 찾기
            support_dir = os.path.join(GaitAnalysisConfig.SUPPORT_LABEL_DATA_PATH, subject_id)
            if not os.path.exists(support_dir):
                return False
            
            subject_code = GaitAnalysisUtils.get_subject_code(subject_id)
            task_code = GaitAnalysisUtils.get_task_code(gait_type)
            pattern = f"{subject_code}{task_code}R*_support_labels.csv"
            
            import glob
            matches = glob.glob(os.path.join(support_dir, pattern))
            return len(matches) > 0
            
        except:
            return False
    
    def on_item_changed(self, item, column):
        """아이템 체크 상태 변경 (batch_gait_analyzer.py 방식 - 개별 세션만)"""
        if column == 0:
            # 선택 정보 업데이트 (부모-자식 연동 제거)
            if self.update_callback:
                self.update_callback()
    
    def get_selected_sessions(self) -> List[str]:
        """선택된 세션 경로들 반환"""
        selected_sessions = []
        
        def collect_sessions(item):
            # 리프 노드이고 체크된 경우
            if item.childCount() == 0 and item.checkState(0) == Qt.Checked:
                session_path = item.data(0, Qt.UserRole)
                if session_path:
                    selected_sessions.append(session_path)
            
            # 자식 노드들 재귀 처리
            for i in range(item.childCount()):
                collect_sessions(item.child(i))
        
        # 모든 최상위 아이템부터 시작
        for i in range(self.topLevelItemCount()):
            collect_sessions(self.topLevelItem(i))
        
        return selected_sessions


class BatchStrideAnalyzerGUI(QMainWindow):
    """배치 Stride 분석 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("배치 Stride 분석 시스템 (Batch Stride Analysis)")
        self.setGeometry(100, 100, 1000, 700)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 좌측: 세션 선택
        left_panel = self.create_session_panel()
        layout.addWidget(left_panel, 1)
        
        # 우측: 진행상황 및 로그
        right_panel = self.create_progress_panel()
        layout.addWidget(right_panel, 1)
        
        # 상태바
        self.statusBar().showMessage("준비")
    
    def create_session_panel(self) -> QWidget:
        """세션 선택 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title = QLabel("📊 Stride 분석할 세션 선택")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 설명
        desc = QLabel("비디오 파일이 있고 매칭되는 support_labels가 있는 세션만 표시됩니다")
        desc.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(desc)
        
        # 세션 트리
        self.session_tree = SessionTreeWidget()
        self.session_tree.set_update_callback(self.update_selected_info)
        layout.addWidget(self.session_tree)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 새로고침")
        refresh_btn.clicked.connect(self.refresh_sessions)
        button_layout.addWidget(refresh_btn)
        
        select_all_btn = QPushButton("✅ 전체 선택")
        select_all_btn.clicked.connect(self.select_all_sessions)
        button_layout.addWidget(select_all_btn)
        
        clear_btn = QPushButton("❌ 선택 해제")
        clear_btn.clicked.connect(self.clear_all_sessions)
        button_layout.addWidget(clear_btn)
        
        layout.addLayout(button_layout)
        
        # 출력 폴더 설정
        output_group = QGroupBox("💾 출력 폴더")
        output_layout = QVBoxLayout(output_group)
        
        folder_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit("./stride_analysis_results")
        folder_layout.addWidget(self.output_path_edit)
        
        browse_btn = QPushButton("📁")
        browse_btn.clicked.connect(self.browse_output_folder)
        folder_layout.addWidget(browse_btn)
        
        output_layout.addLayout(folder_layout)
        layout.addWidget(output_group)
        
        # 선택 정보
        self.selection_info = QLabel("선택된 세션: 0개")
        layout.addWidget(self.selection_info)
        
        # 트리 변경 감지
        self.session_tree.itemChanged.connect(self.update_selected_info)
        
        return panel
    
    def create_progress_panel(self) -> QWidget:
        """진행상황 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title = QLabel("📈 처리 진행상황")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 현재 처리 중인 세션
        self.current_session_label = QLabel("대기 중...")
        layout.addWidget(self.current_session_label)
        
        # 실행 버튼들
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🚀 분석 시작")
        self.start_btn.clicked.connect(self.start_batch_analysis)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ 중단")
        self.stop_btn.clicked.connect(self.stop_batch_analysis)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # 로그 영역
        log_label = QLabel("📋 처리 로그:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # 결과 요약
        self.result_summary = QLabel("결과: 처리 대기 중")
        layout.addWidget(self.result_summary)
        
        return panel
    
    def refresh_sessions(self):
        """세션 목록 새로고침"""
        self.session_tree.load_experiment_data()
        self.update_selected_info()
    
    def select_all_sessions(self):
        """모든 세션 선택 (batch_gait_analyzer.py 방식 - 개별 세션만)"""
        # 시그널 일시 차단
        self.session_tree.blockSignals(True)
        
        for i in range(self.session_tree.topLevelItemCount()):
            subject_item = self.session_tree.topLevelItem(i)
            # 부모는 체크하지 않고 자식 세션들만 체크
            
            # 자식 세션들만 개별적으로 선택
            for j in range(subject_item.childCount()):
                session_item = subject_item.child(j)
                session_item.setCheckState(0, Qt.Checked)
        
        # 시그널 재활성화
        self.session_tree.blockSignals(False)
        self.update_selected_info()
    
    def clear_all_sessions(self):
        """모든 세션 선택 해제 (batch_gait_analyzer.py 방식 - 개별 세션만)"""
        # 시그널 일시 차단
        self.session_tree.blockSignals(True)
        
        for i in range(self.session_tree.topLevelItemCount()):
            subject_item = self.session_tree.topLevelItem(i)
            # 부모는 체크하지 않고 자식 세션들만 해제
            
            # 자식 세션들만 개별적으로 해제
            for j in range(subject_item.childCount()):
                session_item = subject_item.child(j)
                session_item.setCheckState(0, Qt.Unchecked)
        
        # 시그널 재활성화
        self.session_tree.blockSignals(False)
        self.update_selected_info()
    
    def update_selected_info(self):
        """선택된 세션 정보 업데이트"""
        selected_sessions = self.session_tree.get_selected_sessions()
        self.selection_info.setText(f"선택된 세션: {len(selected_sessions)}개")
    
    def browse_output_folder(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "출력 폴더 선택", self.output_path_edit.text()
        )
        if folder:
            self.output_path_edit.setText(folder)
    
    def start_batch_analysis(self):
        """배치 분석 시작"""
        selected_sessions = self.session_tree.get_selected_sessions()
        output_folder = self.output_path_edit.text().strip()
        
        if not selected_sessions:
            QMessageBox.warning(self, "경고", "분석할 세션을 선택해주세요.")
            return
        
        if not output_folder:
            QMessageBox.warning(self, "경고", "출력 폴더를 설정해주세요.")
            return
        
        # 출력 폴더 생성
        os.makedirs(output_folder, exist_ok=True)
        
        # 확인 대화상자
        reply = QMessageBox.question(
            self, "확인", 
            f"{len(selected_sessions)}개 세션을 분석하시겠습니까?\n"
            f"출력 폴더: {output_folder}\n"
            "• 각 비디오에서 MediaPipe로 pose 추출\n"
            "• support_label_data에서 매칭되는 라벨 사용\n"
            "• 각 세션마다 gait_analysis_YYYYMMDD_HHMMSS.csv 파일 생성",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(selected_sessions))
        self.log_text.clear()
        self.current_session_label.setText("분석 시작...")
        
        # 워커 스레드 시작
        self.worker = BatchStrideWorker(selected_sessions, output_folder)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.session_started.connect(self.on_session_started)
        self.worker.session_completed.connect(self.on_session_completed)
        self.worker.log_message.connect(self.add_log_message)
        self.worker.batch_completed.connect(self.on_batch_completed)
        self.worker.start()
        
        self.statusBar().showMessage("배치 분석 중...")
    
    def stop_batch_analysis(self):
        """배치 분석 중지"""
        if self.worker:
            self.worker.stop()
            self.add_log_message("분석 중단 요청...")
    
    def update_progress(self, current: int, total: int):
        """진행률 업데이트"""
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({current/total*100:.1f}%)")
    
    def on_session_started(self, session_path: str):
        """세션 분석 시작"""
        session_name = os.path.basename(session_path)
        self.current_session_label.setText(f"분석 중: {session_name}")
        self.add_log_message(f"🔍 시작: {session_name}")
    
    def on_session_completed(self, session_path: str, success: bool, message: str, stats: dict):
        """세션 분석 완료"""
        session_name = os.path.basename(session_path)
        
        if success:
            csv_filename = os.path.basename(stats.get('csv_path', ''))
            log_msg = f"✅ {session_name}: {message} → {csv_filename}"
            
            # 통계 정보 추가
            if stats:
                log_msg += f"\n   ⏱️ 평균 Stride Time: {stats.get('mean_stride_time', 0):.3f}s"
                log_msg += f"\n   📏 평균 Stride Length: {stats.get('mean_stride_length', 0):.3f}m"
                log_msg += f"\n   🚀 평균 Velocity: {stats.get('mean_velocity', 0):.3f}m/s"
        else:
            log_msg = f"❌ {session_name}: {message}"
        
        self.add_log_message(log_msg)
    
    def add_log_message(self, message: str):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_batch_completed(self, output_folder: str, final_stats: dict):
        """배치 분석 완료"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.current_session_label.setText("분석 완료")
        self.statusBar().showMessage("배치 분석 완료")
        
        # 결과 요약
        total = final_stats['total_sessions']
        processed = final_stats['processed_sessions']
        failed = final_stats['failed_sessions']
        total_strides = final_stats['total_strides']
        
        self.result_summary.setText(
            f"결과: {processed}/{total} 세션 처리 완료, "
            f"{failed}개 실패, 총 {total_strides}개 stride 분석"
        )
        
        # 완료 메시지
        message = f"""
🎉 배치 Stride 분석 완료!

📊 처리 결과:
• 총 세션: {total}개
• 성공: {processed}개  
• 실패: {failed}개
• 총 stride: {total_strides}개

💾 출력 폴더: {output_folder}
"""
        
        QMessageBox.information(self, "완료", message)


def main():
    app = QApplication(sys.argv)
    
    # 현재 작업 디렉토리 확인
    if not os.path.exists("./experiment_data"):
        QMessageBox.critical(
            None, "오류", 
            "experiment_data 폴더를 찾을 수 없습니다.\n"
            "이 프로그램을 experiment_data 폴더가 있는 디렉토리에서 실행해주세요."
        )
        sys.exit(1)
    
    window = BatchStrideAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()