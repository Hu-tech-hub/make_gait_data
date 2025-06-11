"""
보행 분석 시스템 - 공통 유틸리티 및 설정
"""

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

# MediaPipe 가용성 확인
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe가 설치되지 않았습니다. 보행 지표 계산 기능이 제한됩니다.")


class GaitAnalysisConfig:
    """보행 분석 시스템 설정"""
    
    # 보행 타입 매핑
    GAIT_TYPE_MAPPING = {
            'normal_gait': 'T01',
            'ataxic_gait': 'T02', 
            'pain_gait': 'T04',
            'hemiparetic_gait': 'T03',
            'parkinson_gait': 'T05'
        }
        
    # 기본 경로
    EXPERIMENT_DATA_PATH = "./experiment_data"
    SUPPORT_LABEL_DATA_PATH = "./support_label_data"
    
    # 라벨 색상 매핑
    LABEL_COLORS = {
        'single_support_left': (100, 255, 100, 80),    # 연한 초록
        'single_support_right': (100, 100, 255, 80),   # 연한 파랑
        'double_support': (255, 100, 100, 80),         # 연한 빨강
        'non_gait': (200, 200, 200, 60)               # 연한 회색
    }
    
    # 기본 비디오 파일명
    VIDEO_FILENAMES = ["video.mp4", "session.mp4", "recording.mp4"]
    
    # 기본 IMU 파일명
    IMU_FILENAME = "imu_data.csv"
    
    # 기본 메타데이터 파일명
    METADATA_FILENAME = "metadata.json"


class GaitAnalysisUtils:
    """보행 분석 유틸리티 함수들"""
    
    @staticmethod
    def get_subject_code(subject_name):
        """피험자명에서 코드 추출 (SA01 -> S01)"""
        if subject_name.startswith('SA'):
            return f"S{subject_name[2:]}"
        return subject_name
    
    @staticmethod
    def get_task_code(gait_type):
        """보행 타입에서 태스크 코드 추출"""
        return GaitAnalysisConfig.GAIT_TYPE_MAPPING.get(gait_type, 'T01')
    
    @staticmethod
    def build_label_filename(subject, gait_type, run_num):
        """라벨 파일명 생성"""
        subject_code = GaitAnalysisUtils.get_subject_code(subject)
        task_code = GaitAnalysisUtils.get_task_code(gait_type)
        return f"{subject_code}{task_code}{run_num}_support_labels.csv"
    
    @staticmethod
    def find_video_file(session_path):
        """세션 경로에서 비디오 파일 찾기"""
        for filename in GaitAnalysisConfig.VIDEO_FILENAMES:
            video_path = os.path.join(session_path, filename)
            if os.path.exists(video_path):
                return filename
        return None
    
    @staticmethod
    def get_video_info(video_path):
        """비디오 파일 정보 추출"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        }
    
    @staticmethod
    def validate_session_data(session_path):
        """세션 데이터 유효성 검사"""
        results = {}
        
        # 비디오 파일 확인
        video_file = GaitAnalysisUtils.find_video_file(session_path)
        results['video_exists'] = video_file is not None
        results['video_filename'] = video_file
        
        # IMU 데이터 확인
        imu_path = os.path.join(session_path, GaitAnalysisConfig.IMU_FILENAME)
        results['imu_exists'] = os.path.exists(imu_path)
        
        # 메타데이터 확인
        metadata_path = os.path.join(session_path, GaitAnalysisConfig.METADATA_FILENAME)
        results['metadata_exists'] = os.path.exists(metadata_path)
        
        if results['metadata_exists']:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                results['metadata'] = metadata
            except:
                results['metadata'] = None
        
        return results
    
    @staticmethod
    def load_support_labels(label_path):
        """지지 라벨 데이터 로드"""
        try:
            label_df = pd.read_csv(label_path)
            return label_df.to_dict('records')
        except Exception as e:
            print(f"라벨 파일 로드 오류: {e}")
            return []
    
    @staticmethod
    def calculate_sync_quality(video_duration, imu_duration):
        """동기화 품질 계산"""
        time_diff = abs(video_duration - imu_duration)
        
        if time_diff < 0.5:
            return "우수", "green"
        elif time_diff < 2.0:
            return "보통", "orange"
        else:
            return "불량", "red"


def apply_application_style():
    """애플리케이션 스타일 적용"""
    return """
        QMainWindow {
            background-color: #f8f9fa;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #0056b3;
        }
        QPushButton:disabled {
            background-color: #6c757d;
        }
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
        QTabBar::tab {
            background-color: #e9ecef;
            padding: 8px 16px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #007bff;
            color: white;
        }
    """


# 버전 정보
__version__ = "2.0.0"
__author__ = "Gait Analysis Team"
__description__ = "보행 분석 시스템 - 데이터 동기화 및 보행 지표 계산"