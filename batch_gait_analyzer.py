"""
batch_gait_analyzer.py - 배치 보행 분석 GUI

📊 데이터 전처리 파이프라인 Step 1: 보행 단계 라벨 생성

이 모듈은 experiment_data 폴더의 여러 세션을 선택하여 
일괄적으로 support_labels.csv를 생성하는 GUI를 제공합니다.

주요 기능:
- MediaPipe 기반 포즈 추출 및 보행 이벤트 검출
- 여러 세션 배치 처리 (GUI 기반)
- 자동 파일명 매핑 (SA01/normal_gait → S01T01R01_support_labels.csv)
- 실시간 진행상황 모니터링

입력: experiment_data/SA01/gait_type/session_xxx/video.mp4
출력: support_label_data/SA01/S01T01R01_support_labels.csv
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

# gait_class에서 필요한 클래스 import
from gait_class import GaitAnalyzer


class BatchProcessWorker(QThread):
    """배치 처리 워커 스레드"""
    
    # 시그널 정의
    progress_updated = pyqtSignal(int, int)  # current, total
    session_started = pyqtSignal(str)        # session_path
    session_completed = pyqtSignal(str, bool, str)  # session_path, success, message
    log_message = pyqtSignal(str)            # log message
    batch_completed = pyqtSignal()
    
    def __init__(self, selected_sessions: List[str], save_in_session: bool = True, output_folder: str = None):
        super().__init__()
        self.selected_sessions = selected_sessions
        self.save_in_session = save_in_session
        self.output_folder = output_folder
        self.is_running = True
    
    def stop(self):
        """처리 중단"""
        self.is_running = False
    
    def run(self):
        """배치 처리 실행"""
        total_sessions = len(self.selected_sessions)
        
        self.log_message.emit(f"배치 처리 시작: {total_sessions}개 세션")
        
        for i, session_path in enumerate(self.selected_sessions):
            if not self.is_running:
                self.log_message.emit("사용자에 의해 처리가 중단되었습니다.")
                break
            
            self.session_started.emit(session_path)
            self.progress_updated.emit(i, total_sessions)
            
            try:
                # 세션 처리
                success, message = self.process_session(session_path)
                self.session_completed.emit(session_path, success, message)
                
            except Exception as e:
                error_msg = f"처리 중 오류 발생: {str(e)}"
                self.session_completed.emit(session_path, False, error_msg)
        
        self.progress_updated.emit(total_sessions, total_sessions)
        self.batch_completed.emit()
        self.log_message.emit("배치 처리 완료")
    
    def process_session(self, session_path: str) -> tuple:
        """개별 세션 처리"""
        try:
            # 비디오 파일 찾기 (video.mp4 또는 session.mp4)
            video_files = [
                os.path.join(session_path, "video.mp4"),
                os.path.join(session_path, "session.mp4"),
                os.path.join(session_path, "recording.mp4")
            ]
            
            video_path = None
            for vf in video_files:
                if os.path.exists(vf):
                    video_path = vf
                    break
            
            if not video_path:
                return False, "비디오 파일을 찾을 수 없습니다"
            
            # IMU 데이터 확인
            imu_path = os.path.join(session_path, "imu_data.csv")
            if not os.path.exists(imu_path):
                imu_path = None
            
            # 분석기 초기화 및 실행
            analyzer = GaitAnalyzer(video_path, imu_path)
            
            # 방향 감지
            direction = analyzer.detect_walking_direction()
            
            # 포즈 추출
            analyzer.extract_pose_landmarks()
            
            # 이벤트 검출
            events = analyzer.detect_gait_events()
            
            # 결과 저장 (새로운 명명 규칙으로 support_labels.csv 저장)
            self.save_gait_phases_only(analyzer, session_path)
            
            return True, f"성공: {len(events)}개 이벤트 검출, 방향: {direction}"
            
        except Exception as e:
            return False, f"오류: {str(e)}"
    
    def save_gait_phases_only(self, analyzer, session_path: str):
        """support_labels.csv로 새로운 명명 규칙에 따라 저장"""
        import pandas as pd
        
        # 보행 단계 분석
        phases = analyzer.analyze_gait_phases()
        
        # 세션 경로에서 정보 추출
        # session_path 예: ./experiment_data/SA01/ataxic_gait/session_20250604_213127
        path_parts = session_path.replace('\\', '/').split('/')
        
        # SA01 추출
        subject_id = None
        gait_type = None
        session_name = None
        
        for i, part in enumerate(path_parts):
            if part.startswith('SA'):
                subject_id = part  # SA01
                if i + 1 < len(path_parts):
                    gait_type = path_parts[i + 1]  # ataxic_gait
                if i + 2 < len(path_parts):
                    session_name = path_parts[i + 2]  # session_20250604_213127
                break
        
        if not subject_id or not gait_type:
            raise ValueError(f"세션 경로에서 정보를 추출할 수 없습니다: {session_path}")
        
        # 보행 타입을 Task 코드로 매핑
        gait_type_mapping = {
            'normal_gait': 'T01',
            'ataxic_gait': 'T02', 
            'pain_gait': 'T04',
            'hemiparetic_gait': 'T03',
            'parkinson_gait': 'T05'
        }
        
        task_code = gait_type_mapping.get(gait_type)
        if not task_code:
            raise ValueError(f"알 수 없는 보행 타입: {gait_type}")
        
        # Subject 번호 추출 (SA01 → S01)
        subject_num = subject_id[2:]  # "01"
        
        # 같은 조건의 다른 세션들과 비교하여 R 번호 결정
        # 현재는 타임스탬프 기준으로 정렬하여 순서 결정
        parent_dir = os.path.dirname(session_path)
        if os.path.exists(parent_dir):
            all_sessions = [s for s in os.listdir(parent_dir) 
                           if os.path.isdir(os.path.join(parent_dir, s))]
            all_sessions.sort()  # 타임스탬프 순 정렬
            
            # 현재 세션의 인덱스 찾기
            current_session = os.path.basename(session_path)
            try:
                session_index = all_sessions.index(current_session)
                run_number = f"R{session_index + 1:02d}"  # R01, R02, ...
            except ValueError:
                run_number = "R01"  # 기본값
        else:
            run_number = "R01"
        
        # 파일명 생성: S01T02R01_support_labels.csv
        filename = f"S{subject_num}{task_code}{run_number}_support_labels.csv"
        
        # 출력 폴더 구조 생성: support_label_data/SA01/
        if self.save_in_session:
            # 세션 폴더에 직접 저장 (기존 방식)
            output_dir = session_path
            csv_path = os.path.join(output_dir, filename)
        else:
            # support_label_data 구조로 저장
            output_dir = os.path.join(self.output_folder, subject_id)
            os.makedirs(output_dir, exist_ok=True)
            csv_path = os.path.join(output_dir, filename)
        
        # CSV로 저장
        phases_df = pd.DataFrame(phases)
        phases_df.to_csv(csv_path, index=False)


class SessionTreeWidget(QTreeWidget):
    """세션 선택 트리 위젯"""
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("실험 데이터 세션")
        self.setRootIsDecorated(True)
        self.load_experiment_data()
    
    def load_experiment_data(self):
        """experiment_data 폴더 구조 로드 (새 구조: SA01/gait_type/session)"""
        experiment_data_path = "./experiment_data"
        
        if not os.path.exists(experiment_data_path):
            self.addTopLevelItem(QTreeWidgetItem(["experiment_data 폴더를 찾을 수 없습니다"]))
            return
        
        # 각 피험자별로 처리 (SA01, SA02, SA03...)
        subjects = [s for s in os.listdir(experiment_data_path) 
                   if os.path.isdir(os.path.join(experiment_data_path, s)) and s.startswith('SA')]
        subjects.sort()
        
        for subject in subjects:
            subject_path = os.path.join(experiment_data_path, subject)
            
            # 피험자 노드
            subject_item = QTreeWidgetItem([subject])
            subject_item.setFlags(subject_item.flags() | Qt.ItemIsTristate)
            
            # 각 보행 타입별로 처리
            gait_types = [g for g in os.listdir(subject_path) 
                         if os.path.isdir(os.path.join(subject_path, g)) 
                         and g.endswith('_gait')]
            gait_types.sort()
            
            for gait_type in gait_types:
                gait_type_path = os.path.join(subject_path, gait_type)
                
                # 보행 타입 노드
                gait_type_item = QTreeWidgetItem([gait_type])
                gait_type_item.setFlags(gait_type_item.flags() | Qt.ItemIsTristate)
                
                # 세션들 추가
                sessions = [s for s in os.listdir(gait_type_path) 
                           if os.path.isdir(os.path.join(gait_type_path, s))]
                sessions.sort()
                
                for session in sessions:
                    session_path = os.path.join(gait_type_path, session)
                    session_item = QTreeWidgetItem([session])
                    session_item.setFlags(session_item.flags() | Qt.ItemIsUserCheckable)
                    session_item.setCheckState(0, Qt.Unchecked)
                    session_item.setData(0, Qt.UserRole, session_path)  # 전체 경로 저장
                    
                    # 세션 정보 표시 (metadata.json이 있다면)
                    metadata_path = os.path.join(session_path, "metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata = json.load(f)
                            duration = metadata.get('duration', 0)
                            frames = metadata.get('video_frames', 0)
                            info_text = f"{session} ({duration:.1f}s, {frames}frames)"
                            session_item.setText(0, info_text)
                        except:
                            pass
                    
                    gait_type_item.addChild(session_item)
                
                if gait_type_item.childCount() > 0:  # 세션이 있는 경우만 추가
                    subject_item.addChild(gait_type_item)
            
            if subject_item.childCount() > 0:  # 보행 타입이 있는 경우만 추가
                self.addTopLevelItem(subject_item)
        
        self.expandAll()
    
    def get_selected_sessions(self) -> List[str]:
        """선택된 세션들의 경로 반환 (새 구조: Subject → GaitType → Session)"""
        selected = []
        
        # Subject 레벨 (SA01, SA02, ...)
        for i in range(self.topLevelItemCount()):
            subject_item = self.topLevelItem(i)
            
            # GaitType 레벨 (ataxic_gait, normal_gait, ...)
            for j in range(subject_item.childCount()):
                gait_type_item = subject_item.child(j)
                
                # Session 레벨 (session_20250604_213127, ...)
                for k in range(gait_type_item.childCount()):
                    session_item = gait_type_item.child(k)
                    
                    if session_item.checkState(0) == Qt.Checked:
                        session_path = session_item.data(0, Qt.UserRole)
                        if session_path:
                            selected.append(session_path)
        
        return selected


class BatchGaitAnalyzerGUI(QMainWindow):
    """배치 보행 분석 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("배치 보행 분석 시스템 (Batch Gait Analysis)")
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
        title = QLabel("세션 선택")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 세션 트리
        self.session_tree = SessionTreeWidget()
        layout.addWidget(self.session_tree)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("전체 선택")
        self.select_all_btn.clicked.connect(self.select_all_sessions)
        button_layout.addWidget(self.select_all_btn)
        
        self.clear_all_btn = QPushButton("전체 해제")
        self.clear_all_btn.clicked.connect(self.clear_all_sessions)
        button_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(button_layout)
        
        # 저장 위치 설정
        save_group = QGroupBox("저장 위치 설정")
        save_layout = QVBoxLayout(save_group)
        
        # 저장 방식 선택
        self.save_in_session_rb = QRadioButton("각 세션 폴더에 저장")
        self.save_in_session_rb.setChecked(True)
        save_layout.addWidget(self.save_in_session_rb)
        
        self.save_in_output_rb = QRadioButton("출력 폴더에 저장")
        save_layout.addWidget(self.save_in_output_rb)
        
        # 출력 폴더 선택
        output_layout = QHBoxLayout()
        self.output_path_label = QLabel("./support_label_data")
        self.output_path_label.setStyleSheet("border: 1px solid gray; padding: 5px;")
        output_layout.addWidget(self.output_path_label)
        
        self.browse_btn = QPushButton("찾아보기")
        self.browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.browse_btn)
        
        save_layout.addLayout(output_layout)
        
        # 라디오 버튼 연결
        self.save_in_output_rb.toggled.connect(self.on_save_option_changed)
        self.on_save_option_changed()  # 초기 상태 설정
        
        layout.addWidget(save_group)
        
        # 선택된 세션 정보
        self.selected_info = QLabel("선택된 세션: 0개")
        layout.addWidget(self.selected_info)
        
        # 세션 선택 변경 시 업데이트
        self.session_tree.itemChanged.connect(self.update_selected_info)
        
        return panel
    
    def create_progress_panel(self) -> QWidget:
        """진행상황 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 제목
        title = QLabel("처리 진행상황")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # 진행률 바
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # 현재 처리 중인 세션
        self.current_session_label = QLabel("대기 중...")
        layout.addWidget(self.current_session_label)
        
        # 시작/중지 버튼
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("처리 시작")
        self.start_btn.clicked.connect(self.start_batch_processing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("처리 중지")
        self.stop_btn.clicked.connect(self.stop_batch_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        
        # 로그 영역
        log_label = QLabel("처리 로그:")
        layout.addWidget(log_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        layout.addWidget(self.log_text)
        
        # 결과 요약
        self.result_summary = QLabel("결과: 처리 대기 중")
        layout.addWidget(self.result_summary)
        
        return panel
    
    def select_all_sessions(self):
        """모든 세션 선택"""
        for i in range(self.session_tree.topLevelItemCount()):
            subject_item = self.session_tree.topLevelItem(i)
            
            for j in range(subject_item.childCount()):
                gait_type_item = subject_item.child(j)
                
                for k in range(gait_type_item.childCount()):
                    session_item = gait_type_item.child(k)
                    session_item.setCheckState(0, Qt.Checked)
    
    def clear_all_sessions(self):
        """모든 세션 선택 해제"""
        for i in range(self.session_tree.topLevelItemCount()):
            subject_item = self.session_tree.topLevelItem(i)
            
            for j in range(subject_item.childCount()):
                gait_type_item = subject_item.child(j)
                
                for k in range(gait_type_item.childCount()):
                    session_item = gait_type_item.child(k)
                    session_item.setCheckState(0, Qt.Unchecked)
    
    def update_selected_info(self):
        """선택된 세션 정보 업데이트"""
        selected_sessions = self.session_tree.get_selected_sessions()
        self.selected_info.setText(f"선택된 세션: {len(selected_sessions)}개")
    
    def browse_output_folder(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "출력 폴더 선택", self.output_path_label.text()
        )
        if folder:
            self.output_path_label.setText(folder)
    
    def on_save_option_changed(self):
        """저장 옵션 변경 시 호출"""
        enable_output_controls = self.save_in_output_rb.isChecked()
        self.output_path_label.setEnabled(enable_output_controls)
        self.browse_btn.setEnabled(enable_output_controls)
    
    def start_batch_processing(self):
        """배치 처리 시작"""
        selected_sessions = self.session_tree.get_selected_sessions()
        
        if not selected_sessions:
            QMessageBox.warning(self, "경고", "처리할 세션을 선택해주세요.")
            return
        
        # 확인 대화상자
        save_location = "각 세션 폴더" if self.save_in_session_rb.isChecked() else self.output_path_label.text()
        reply = QMessageBox.question(
            self, "확인", 
            f"{len(selected_sessions)}개 세션을 처리하시겠습니까?\n"
            f"저장 위치: {save_location}\n"
            "파일명: S01T01R01_support_labels.csv 형식으로 생성됩니다.",
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
        self.current_session_label.setText("처리 시작...")
        
        # 저장 설정 준비
        save_in_session = self.save_in_session_rb.isChecked()
        output_folder = self.output_path_label.text() if not save_in_session else None
        
        # 워커 스레드 시작
        self.worker = BatchProcessWorker(selected_sessions, save_in_session, output_folder)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.session_started.connect(self.on_session_started)
        self.worker.session_completed.connect(self.on_session_completed)
        self.worker.log_message.connect(self.add_log_message)
        self.worker.batch_completed.connect(self.on_batch_completed)
        self.worker.start()
        
        self.statusBar().showMessage("배치 처리 중...")
    
    def stop_batch_processing(self):
        """배치 처리 중지"""
        if self.worker:
            self.worker.stop()
            self.add_log_message("처리 중단 요청...")
    
    def update_progress(self, current: int, total: int):
        """진행률 업데이트"""
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat(f"{current}/{total} ({current/total*100:.1f}%)")
    
    def on_session_started(self, session_path: str):
        """세션 처리 시작"""
        session_name = os.path.basename(session_path)
        self.current_session_label.setText(f"처리 중: {session_name}")
        self.add_log_message(f"시작: {session_name}")
    
    def on_session_completed(self, session_path: str, success: bool, message: str):
        """세션 처리 완료"""
        session_name = os.path.basename(session_path)
        status = "✓" if success else "✗"
        log_msg = f"{status} {session_name}: {message}"
        self.add_log_message(log_msg)
    
    def add_log_message(self, message: str):
        """로그 메시지 추가"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def on_batch_completed(self):
        """배치 처리 완료"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.current_session_label.setText("처리 완료")
        self.statusBar().showMessage("배치 처리 완료")
        
        # 결과 요약
        total = self.progress_bar.maximum()
        completed = self.progress_bar.value()
        self.result_summary.setText(f"결과: {completed}/{total} 세션 처리 완료")
        
        QMessageBox.information(self, "완료", "배치 처리가 완료되었습니다!")


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
    
    window = BatchGaitAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 