#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stride Dataset Generator GUI
stride_analysis_results와 walking_data를 결합하여 JSON 데이터셋 생성

Author: Assistant
Date: 2025-01-12
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QTextEdit,
    QFileDialog, QGroupBox, QTreeWidget, QTreeWidgetItem,
    QCheckBox, QMessageBox, QSplitter, QGridLayout, QFrame
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QIcon


class StrideDatasetEngine:
    """stride 데이터셋 생성 핵심 엔진"""
    
    def __init__(self):
        self.subject_heights = {
            'SA01': 175,  # 175cm
            'SA02': 170,  # 170cm  
            'SA03': 180,  # 180cm
            'SA04': 160,   # 160cm
            'SA05': 160
        }
    
    def find_matching_files(self, stride_results_dir: str, walking_data_dir: str) -> List[Tuple[str, str]]:
        """매칭되는 stride_labels와 walking_data 파일 찾기"""
        matches = []
        
        try:
            # stride_analysis_results 파일들 스캔
            stride_files = []
            if os.path.exists(stride_results_dir):
                for file in os.listdir(stride_results_dir):
                    if file.endswith('_stride_labels.csv'):
                        stride_files.append(file)
            
            for stride_file in stride_files:
                # 파일명에서 패턴 추출: S01T01R01_stride_labels.csv → S01T01R01
                base_name = stride_file.replace('_stride_labels.csv', '')
                
                # 피험자 ID 추출: S01 → SA01
                subject_id = 'SA' + base_name[1:3]
                
                # walking_data에서 해당 파일 찾기
                walking_file = f"{base_name}.csv"  # S01T01R01.csv
                walking_path = os.path.join(walking_data_dir, subject_id, walking_file)
                
                if os.path.exists(walking_path):
                    stride_path = os.path.join(stride_results_dir, stride_file)
                    matches.append((stride_path, walking_path))
            
            return matches
            
        except Exception as e:
            print(f"파일 매칭 실패: {e}")
            return []
    
    def load_stride_labels(self, stride_file: str) -> List[Dict]:
        """stride_labels CSV 로드"""
        try:
            df = pd.read_csv(stride_file)
            cycles = []
            
            for _, row in df.iterrows():
                cycle = {
                    'cycle_number': int(row['번호']),
                    'subject_id': row['피험자ID'],
                    'height': int(row['키(cm)']),
                    'foot': row['발'],
                    'start_frame': int(row['시작프레임']),
                    'end_frame': int(row['종료프레임']),
                    'start_time': float(row['시작시간(s)']),
                    'end_time': float(row['종료시간(s)']),
                    'stride_time': float(row['Stride Time(s)']),
                    'stride_length': float(row['Stride Length(m)']),
                    'velocity': float(row['Velocity(m/s)'])
                }
                cycles.append(cycle)
            
            return cycles
            
        except Exception as e:
            print(f"Stride labels 로드 실패 ({stride_file}): {e}")
            return []
    
    def load_walking_data(self, walking_file: str) -> pd.DataFrame:
        """walking_data CSV 로드"""
        try:
            df = pd.read_csv(walking_file)
            
            # 필요한 컬럼 확인
            required_cols = ['frame_number', 'sync_timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"누락된 컬럼: {missing_cols}")
                return pd.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Walking data 로드 실패 ({walking_file}): {e}")
            return pd.DataFrame()
    
    def extract_cycle_sequence(self, walking_df: pd.DataFrame, start_frame: int, end_frame: int) -> List[List[float]]:
        """stride cycle 구간의 센서 데이터 추출"""
        try:
            # 프레임 범위 필터링
            cycle_data = walking_df[
                (walking_df['frame_number'] >= start_frame) & 
                (walking_df['frame_number'] <= end_frame)
            ].copy()
            
            if cycle_data.empty:
                print(f"프레임 범위 {start_frame}-{end_frame}에 데이터 없음")
                return []
            
            # 센서 데이터 추출 (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)
            sequence = []
            for _, row in cycle_data.iterrows():
                frame_data = [
                    round(float(row['accel_x']), 3),
                    round(float(row['accel_y']), 3),
                    round(float(row['accel_z']), 3),
                    round(float(row['gyro_x']), 5),
                    round(float(row['gyro_y']), 5),
                    round(float(row['gyro_z']), 5)
                ]
                sequence.append(frame_data)
            
            return sequence
            
        except Exception as e:
            print(f"Cycle sequence 추출 실패: {e}")
            return []
    
    def generate_session_json(self, stride_file: str, walking_file: str) -> Dict:
        """하나의 세션에 대한 JSON 데이터 생성"""
        try:
            # 1. stride labels 로드
            cycles = self.load_stride_labels(stride_file)
            if not cycles:
                return {'error': 'Stride labels 로드 실패'}
            
            # 2. walking data 로드
            walking_df = self.load_walking_data(walking_file)
            if walking_df.empty:
                return {'error': 'Walking data 로드 실패'}
            
            # 3. 각 cycle별로 sequence 추출
            session_cycles = []
            
            for cycle in cycles:
                sequence = self.extract_cycle_sequence(
                    walking_df, 
                    cycle['start_frame'], 
                    cycle['end_frame']
                )
                
                if sequence:  # 빈 sequence 제외
                    cycle_json = {
                        'sequence': sequence,
                        'height': cycle['height'],
                        'stride_time': round(cycle['stride_time'], 3),
                        'stride_length': round(cycle['stride_length'], 3),
                        'velocity': round(cycle['velocity'], 3),
                        'foot': cycle['foot'],
                        'cycle_number': cycle['cycle_number']
                    }
                    session_cycles.append(cycle_json)
            
            if not session_cycles:
                return {'error': '유효한 cycle 없음'}
            
            # 세션 정보 추출
            base_name = os.path.basename(stride_file).replace('_stride_labels.csv', '')
            
            result = {
                'session_id': base_name,
                'subject_id': cycles[0]['subject_id'],
                'height': cycles[0]['height'],
                'total_cycles': len(session_cycles),
                'cycles': session_cycles
            }
            
            return result
            
        except Exception as e:
            return {'error': f'JSON 생성 실패: {str(e)}'}
    
    def save_session_json(self, session_data: Dict, output_dir: str) -> str:
        """세션 JSON 파일 저장"""
        try:
            if 'error' in session_data:
                return session_data['error']
            
            session_id = session_data['session_id']
            
            # 출력 폴더 생성
            session_dir = os.path.join(output_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # JSON 파일 저장
            json_file = os.path.join(session_dir, f"{session_id}_Cycles.json")
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(session_data['cycles'], f, indent=2, ensure_ascii=False)
            
            return json_file
            
        except Exception as e:
            return f"저장 실패: {str(e)}"


class DatasetGeneratorWorker(QThread):
    """배치 처리 워커 스레드"""
    
    # 시그널 정의
    progress_updated = pyqtSignal(int, int)  # current, total
    file_started = pyqtSignal(str)           # file_pair
    file_completed = pyqtSignal(str, bool, str, dict)  # file_pair, success, message, stats
    log_message = pyqtSignal(str)            # log message
    batch_completed = pyqtSignal(str, dict)  # output_dir, final_stats
    
    def __init__(self, file_pairs: List[Tuple[str, str]], output_dir: str):
        super().__init__()
        self.file_pairs = file_pairs
        self.output_dir = output_dir
        self.engine = StrideDatasetEngine()
        self.is_stopped = False
    
    def stop(self):
        """워커 중지"""
        self.is_stopped = True
    
    def run(self):
        """배치 처리 실행"""
        try:
            total_files = len(self.file_pairs)
            processed = 0
            success_count = 0
            error_count = 0
            total_cycles = 0
            
            self.log_message.emit(f"총 {total_files}개 파일 쌍 처리 시작...")
            
            for i, (stride_file, walking_file) in enumerate(self.file_pairs):
                if self.is_stopped:
                    break
                
                # 진행상황 업데이트
                self.progress_updated.emit(i, total_files)
                
                # 파일 쌍 이름
                stride_name = os.path.basename(stride_file)
                walking_name = os.path.basename(walking_file)
                file_pair_name = f"{stride_name} + {walking_name}"
                
                self.file_started.emit(file_pair_name)
                
                try:
                    # JSON 생성
                    session_data = self.engine.generate_session_json(stride_file, walking_file)
                    
                    if 'error' in session_data:
                        # 실패
                        error_msg = session_data['error']
                        self.file_completed.emit(file_pair_name, False, error_msg, {})
                        error_count += 1
                    else:
                        # 저장
                        json_path = self.engine.save_session_json(session_data, self.output_dir)
                        
                        if json_path.startswith("저장 실패"):
                            self.file_completed.emit(file_pair_name, False, json_path, {})
                            error_count += 1
                        else:
                            # 성공
                            stats = {
                                'session_id': session_data['session_id'],
                                'cycles': session_data['total_cycles'],
                                'json_path': json_path
                            }
                            success_msg = f"성공: {session_data['total_cycles']}개 cycle 생성"
                            self.file_completed.emit(file_pair_name, True, success_msg, stats)
                            success_count += 1
                            total_cycles += session_data['total_cycles']
                
                except Exception as e:
                    error_msg = f"처리 실패: {str(e)}"
                    self.file_completed.emit(file_pair_name, False, error_msg, {})
                    error_count += 1
                
                processed += 1
            
            # 최종 진행상황
            self.progress_updated.emit(total_files, total_files)
            
            # 완료 통계
            final_stats = {
                'total_files': total_files,
                'success_count': success_count,
                'error_count': error_count,
                'total_cycles': total_cycles
            }
            
            self.batch_completed.emit(self.output_dir, final_stats)
            
        except Exception as e:
            self.log_message.emit(f"배치 처리 오류: {str(e)}")


class FileMatchingWidget(QWidget):
    """파일 매칭 표시 위젯"""
    
    def __init__(self):
        super().__init__()
        self.engine = StrideDatasetEngine()
        self.file_pairs = []
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 폴더 선택 섹션
        folder_group = QGroupBox("폴더 선택")
        folder_layout = QGridLayout()
        
        # stride_analysis_results 폴더
        folder_layout.addWidget(QLabel("Stride Results:"), 0, 0)
        self.stride_folder_edit = QLineEdit()
        self.stride_folder_edit.setText("C:/vision_gait/stride_analysis_results")
        folder_layout.addWidget(self.stride_folder_edit, 0, 1)
        
        stride_browse_btn = QPushButton("찾아보기")
        stride_browse_btn.clicked.connect(self.browse_stride_folder)
        folder_layout.addWidget(stride_browse_btn, 0, 2)
        
        # walking_data 폴더
        folder_layout.addWidget(QLabel("Walking Data:"), 1, 0)
        self.walking_folder_edit = QLineEdit()
        self.walking_folder_edit.setText("C:/vision_gait/walking_data")
        folder_layout.addWidget(self.walking_folder_edit, 1, 1)
        
        walking_browse_btn = QPushButton("찾아보기")
        walking_browse_btn.clicked.connect(self.browse_walking_folder)
        folder_layout.addWidget(walking_browse_btn, 1, 2)
        
        # 스캔 버튼
        scan_btn = QPushButton("파일 매칭 스캔")
        scan_btn.clicked.connect(self.scan_files)
        folder_layout.addWidget(scan_btn, 2, 1)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # 매칭 결과 표시
        result_group = QGroupBox("매칭 결과")
        result_layout = QVBoxLayout()
        
        self.result_tree = QTreeWidget()
        self.result_tree.setHeaderLabels(["Stride File", "Walking File", "Status"])
        result_layout.addWidget(self.result_tree)
        
        self.match_info_label = QLabel("매칭 파일: 0개")
        result_layout.addWidget(self.match_info_label)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.setLayout(layout)
    
    def browse_stride_folder(self):
        """stride_analysis_results 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "Stride Analysis Results 폴더 선택",
            self.stride_folder_edit.text()
        )
        if folder:
            self.stride_folder_edit.setText(folder)
    
    def browse_walking_folder(self):
        """walking_data 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "Walking Data 폴더 선택",
            self.walking_folder_edit.text()
        )
        if folder:
            self.walking_folder_edit.setText(folder)
    
    def scan_files(self):
        """파일 매칭 스캔"""
        stride_dir = self.stride_folder_edit.text()
        walking_dir = self.walking_folder_edit.text()
        
        if not os.path.exists(stride_dir):
            QMessageBox.warning(self, "경고", "Stride Results 폴더가 존재하지 않습니다.")
            return
        
        if not os.path.exists(walking_dir):
            QMessageBox.warning(self, "경고", "Walking Data 폴더가 존재하지 않습니다.")
            return
        
        # 파일 매칭
        self.file_pairs = self.engine.find_matching_files(stride_dir, walking_dir)
        
        # 결과 표시
        self.result_tree.clear()
        
        for stride_file, walking_file in self.file_pairs:
            item = QTreeWidgetItem()
            item.setText(0, os.path.basename(stride_file))
            item.setText(1, os.path.basename(walking_file))
            item.setText(2, "✓ 매칭됨")
            self.result_tree.addTopLevelItem(item)
        
        self.match_info_label.setText(f"매칭 파일: {len(self.file_pairs)}개")
        
        if len(self.file_pairs) == 0:
            QMessageBox.information(self, "알림", "매칭되는 파일이 없습니다.")
        else:
            QMessageBox.information(self, "완료", f"{len(self.file_pairs)}개 파일 쌍이 매칭되었습니다.")
    
    def get_file_pairs(self) -> List[Tuple[str, str]]:
        """매칭된 파일 쌍 반환"""
        return self.file_pairs


class StrideDatasetGeneratorGUI(QMainWindow):
    """Stride Dataset Generator 메인 GUI"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Stride Dataset Generator")
        self.setGeometry(100, 100, 1000, 700)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 상단: 파일 매칭
        self.file_matching_widget = FileMatchingWidget()
        main_layout.addWidget(self.file_matching_widget)
        
        # 중간: 출력 설정
        output_group = QGroupBox("출력 설정")
        output_layout = QHBoxLayout()
        
        output_layout.addWidget(QLabel("출력 폴더:"))
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setText("C:/vision_gait/stride_train_data")
        output_layout.addWidget(self.output_folder_edit)
        
        output_browse_btn = QPushButton("찾아보기")
        output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(output_browse_btn)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # 하단: 처리 컨트롤
        control_group = QGroupBox("처리 제어")
        control_layout = QVBoxLayout()
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("배치 처리 시작")
        self.start_btn.clicked.connect(self.start_batch_processing)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_batch_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        control_layout.addLayout(button_layout)
        
        # 진행상황
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("준비")
        control_layout.addWidget(self.status_label)
        
        # 로그
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        control_layout.addWidget(self.log_text)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
    
    def browse_output_folder(self):
        """출력 폴더 선택"""
        folder = QFileDialog.getExistingDirectory(
            self, "출력 폴더 선택",
            self.output_folder_edit.text()
        )
        if folder:
            self.output_folder_edit.setText(folder)
    
    def start_batch_processing(self):
        """배치 처리 시작"""
        file_pairs = self.file_matching_widget.get_file_pairs()
        
        if not file_pairs:
            QMessageBox.warning(self, "경고", "매칭된 파일이 없습니다. 먼저 파일 스캔을 실행하세요.")
            return
        
        output_dir = self.output_folder_edit.text()
        if not output_dir:
            QMessageBox.warning(self, "경고", "출력 폴더를 선택하세요.")
            return
        
        # 출력 폴더 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 워커 시작
        self.worker = DatasetGeneratorWorker(file_pairs, output_dir)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.log_message.connect(self.add_log_message)
        self.worker.batch_completed.connect(self.on_batch_completed)
        
        self.worker.start()
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.add_log_message("배치 처리 시작...")
    
    def stop_batch_processing(self):
        """배치 처리 중지"""
        if self.worker:
            self.worker.stop()
            self.add_log_message("처리 중지 요청...")
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def update_progress(self, current: int, total: int):
        """진행상황 업데이트"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"진행: {current}/{total}")
    
    def on_file_started(self, file_pair: str):
        """파일 처리 시작"""
        self.add_log_message(f"처리 중: {file_pair}")
    
    def on_file_completed(self, file_pair: str, success: bool, message: str, stats: dict):
        """파일 처리 완료"""
        if success:
            cycles = stats.get('cycles', 0)
            self.add_log_message(f"✓ {file_pair} - {cycles}개 cycle")
        else:
            self.add_log_message(f"✗ {file_pair} - {message}")
    
    def on_batch_completed(self, output_dir: str, final_stats: dict):
        """배치 처리 완료"""
        total = final_stats['total_files']
        success = final_stats['success_count']
        error = final_stats['error_count']
        cycles = final_stats['total_cycles']
        
        self.add_log_message("=" * 50)
        self.add_log_message(f"배치 처리 완료!")
        self.add_log_message(f"총 파일: {total}개")
        self.add_log_message(f"성공: {success}개")
        self.add_log_message(f"실패: {error}개")
        self.add_log_message(f"총 cycle: {cycles}개")
        self.add_log_message(f"출력 폴더: {output_dir}")
        
        # UI 상태 복원
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("완료")
        
        # 완료 메시지
        QMessageBox.information(
            self, "완료", 
            f"배치 처리가 완료되었습니다!\n\n"
            f"성공: {success}/{total}개\n"
            f"총 cycle: {cycles}개\n"
            f"출력: {output_dir}"
        )
    
    def add_log_message(self, message: str):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
        # 자동 스크롤
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def main():
    app = QApplication(sys.argv)
    
    # GUI 실행
    window = StrideDatasetGeneratorGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()