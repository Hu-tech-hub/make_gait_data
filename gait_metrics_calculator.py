# ====================================
# Gait Metrics Calculator Module
# ====================================
"""
gait_metrics_calculator.py - 보행 지표 계산 모듈

이 모듈은 다음 기능을 제공합니다:
1. MediaPipe 기반 관절 추정
2. 공간적 보행 지표 계산 (보폭, 속도, 주기, 보행률, ROM)
3. IMU 데이터와 보행 지표 매핑
4. 보행 주기별 특징 추출
"""

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import json
from scipy.signal import find_peaks, butter, filtfilt
from scipy.spatial.distance import euclidean
import math


@dataclass
class GaitCycle:
    """보행 주기 데이터 클래스"""
    start_frame: int
    end_frame: int
    foot: str  # "left" or "right"
    stride_length: float  # 보폭 (미터)
    velocity: float       # 속도 (m/s)
    cycle_time: float     # 주기 (초)
    cadence: float        # 보행률 (steps/min)
    hip_rom: float        # 엉덩이 관절 가동 범위 (도)
    knee_rom: float       # 무릎 관절 가동 범위 (도)
    ankle_rom: float      # 발목 관절 가동 범위 (도)
    stance_ratio: float   # 입각기 비율 (%)
    
    def to_dict(self):
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'foot': self.foot,
            'stride_length': self.stride_length,
            'velocity': self.velocity,
            'cycle_time': self.cycle_time,
            'cadence': self.cadence,
            'hip_rom': self.hip_rom,
            'knee_rom': self.knee_rom,
            'ankle_rom': self.ankle_rom,
            'stance_ratio': self.stance_ratio
        }
@dataclass
class JointAngles:
    """관절 각도 데이터 클래스"""
    frame: int
    hip_angle: float    # 엉덩이 관절 각도
    knee_angle: float   # 무릎 관절 각도
    ankle_angle: float  # 발목 관절 각도


class GaitMetricsCalculator:
    """보행 지표 계산 클래스"""
    
    def __init__(self):
        """초기화"""
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 픽셀-미터 변환 비율 (실제 환경에 맞게 조정 필요)
        self.pixel_to_meter_ratio = 0.001  # 1픽셀 = 1mm 가정        

    def extract_joint_coordinates(self, video_path: str, progress_callback=None) -> pd.DataFrame:
        """
        비디오에서 관절 좌표 추출
        
        Args:
            video_path (str): 비디오 파일 경로
            progress_callback: 진행률 콜백 함수
            
        Returns:
            pd.DataFrame: 프레임별 관절 좌표 데이터
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        joint_data = []
        frame_idx = 0
        
        print(f"관절 좌표 추출 시작... 총 {total_frames} 프레임")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # RGB 변환 및 포즈 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 주요 관절 좌표 추출
                data = {
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    
                    # 발목 좌표
                    'left_ankle_x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                    'left_ankle_y': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                    'left_ankle_z': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z,
                    'right_ankle_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                    'right_ankle_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                    'right_ankle_z': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z,
                    
                    # 무릎 좌표
                    'left_knee_x': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                    'left_knee_y': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                    'left_knee_z': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].z,
                    'right_knee_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                    'right_knee_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y,
                    'right_knee_z': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].z,
                    
                    # 엉덩이 좌표
                    'left_hip_x': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                    'left_hip_y': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                    'left_hip_z': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].z,
                    'right_hip_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                    'right_hip_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                    'right_hip_z': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].z,
                }
                
                joint_data.append(data)
            
            frame_idx += 1
            
            # 진행률 업데이트
            if progress_callback and frame_idx % 10 == 0:
                progress = (frame_idx / total_frames) * 100
                progress_callback(progress)
        
        cap.release()
        
        if not joint_data:
            raise ValueError("관절 좌표를 추출할 수 없습니다.")
        
        return pd.DataFrame(joint_data)
    
    def calculate_joint_angles(self, joint_coords: pd.DataFrame) -> List[JointAngles]:
        """
        관절 각도 계산
        
        Args:
            joint_coords (pd.DataFrame): 관절 좌표 데이터
            
        Returns:
            List[JointAngles]: 프레임별 관절 각도 리스트
        """
        angles_list = []
        
        for _, row in joint_coords.iterrows():
            # 왼쪽 다리 관절 각도 계산
            left_hip = np.array([row['left_hip_x'], row['left_hip_y']])
            left_knee = np.array([row['left_knee_x'], row['left_knee_y']])
            left_ankle = np.array([row['left_ankle_x'], row['left_ankle_y']])
            
            # 무릎 관절 각도 (대퇴-하퇴 사이 각도)
            v1 = left_hip - left_knee
            v2 = left_ankle - left_knee
            knee_angle = self._calculate_angle(v1, v2)
            
            # 엉덩이 관절 각도 (몸통-대퇴 사이 각도, 수직선 기준)
            vertical = np.array([0, -1])  # 수직 아래 방향
            hip_angle = self._calculate_angle(vertical, left_knee - left_hip)
            
            # 발목 관절 각도 (하퇴-발 사이 각도)
            ankle_angle = 90  # 간단화: 실제로는 발가락 좌표 필요
            
            angles = JointAngles(
                frame=int(row['frame']),
                hip_angle=hip_angle,
                knee_angle=knee_angle,
                ankle_angle=ankle_angle
            )
            angles_list.append(angles)
        
        return angles_list
    
    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 사이의 각도 계산 (도 단위)"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 수치 오차 방지
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def identify_gait_cycles_from_labels(self, joint_coords: pd.DataFrame, 
                                       support_labels: List[Dict]) -> List[Dict]:
        """
        라벨 데이터를 사용해 보행 주기 구간 식별
        
        Args:
            joint_coords (pd.DataFrame): 관절 좌표 데이터
            support_labels (List[Dict]): 지지 단계 라벨 데이터
            
        Returns:
            List[Dict]: 보행 주기별 데이터 (관절 좌표 포함)
        """
        gait_cycles = []
        
        # single_support 구간들을 발별로 그룹화
        left_cycles = []
        right_cycles = []
        
        for label in support_labels:
            if label['phase'] == 'single_support_left':
                left_cycles.append(label)
            elif label['phase'] == 'single_support_right':
                right_cycles.append(label)
        
        # 각 발의 연속된 single_support 구간으로 보행 주기 정의
        for foot, cycles in [('left', left_cycles), ('right', right_cycles)]:
            cycles.sort(key=lambda x: x['start_frame'])
            
            for i in range(len(cycles) - 1):
                start_frame = cycles[i]['start_frame']
                end_frame = cycles[i + 1]['end_frame']
                
                # 해당 구간의 관절 좌표 추출
                cycle_joint_data = joint_coords[
                    (joint_coords['frame'] >= start_frame) & 
                    (joint_coords['frame'] <= end_frame)
                ].copy()
                
                if len(cycle_joint_data) < 10:  # 최소 10프레임 필요
                    continue
                
                gait_cycles.append({
                    'foot': foot,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'joint_data': cycle_joint_data,
                    'cycle_labels': cycles[i:i+2]  # 현재와 다음 single_support 구간
                })
        
        return gait_cycles
    
    def calculate_stride_length_from_trajectory(self, joint_data: pd.DataFrame, foot: str) -> float:
        """
        발목 궤적을 기반으로 실제 보폭 계산
        
        Args:
            joint_data (pd.DataFrame): 관절 좌표 데이터
            foot (str): 발 구분 ("left" or "right")
            
        Returns:
            float: 보폭 (미터)
        """
        ankle_x_col = f'{foot}_ankle_x'
        ankle_y_col = f'{foot}_ankle_y'
        
        if ankle_x_col not in joint_data.columns:
            return 0.0
        
        # 발목의 수평 이동 거리 계산
        start_pos = np.array([joint_data[ankle_x_col].iloc[0], joint_data[ankle_y_col].iloc[0]])
        end_pos = np.array([joint_data[ankle_x_col].iloc[-1], joint_data[ankle_y_col].iloc[-1]])
        
        # 유클리드 거리를 미터로 변환
        pixel_distance = euclidean(start_pos, end_pos)
        stride_length = pixel_distance * self.pixel_to_meter_ratio
        
        return stride_length

    def calculate_velocity_from_movement(self, joint_data: pd.DataFrame, foot: str, fps: float) -> float:
        """
        관절 움직임을 기반으로 실제 속도 계산
        
        Args:
            joint_data (pd.DataFrame): 관절 좌표 데이터
            foot (str): 발 구분
            fps (float): 비디오 프레임률
            
        Returns:
            float: 속도 (m/s)
        """
        stride_length = self.calculate_stride_length_from_trajectory(joint_data, foot)
        cycle_time = len(joint_data) / fps
        
        if cycle_time > 0:
            return stride_length / cycle_time
        return 0.0

    def calculate_actual_ankle_angle(self, joint_data: pd.DataFrame, foot: str) -> float:
        """
        실제 발목 관절 각도 계산 (고정값 대신)
        
        Args:
            joint_data (pd.DataFrame): 관절 좌표 데이터
            foot (str): 발 구분
            
        Returns:
            float: 발목 각도 범위 (도)
        """
        knee_x_col = f'{foot}_knee_x'
        knee_y_col = f'{foot}_knee_y'
        ankle_x_col = f'{foot}_ankle_x'
        ankle_y_col = f'{foot}_ankle_y'
        
        if not all(col in joint_data.columns for col in [knee_x_col, knee_y_col, ankle_x_col, ankle_y_col]):
            return 0.0
        
        angles = []
        for _, row in joint_data.iterrows():
            knee_pos = np.array([row[knee_x_col], row[knee_y_col]])
            ankle_pos = np.array([row[ankle_x_col], row[ankle_y_col]])
            
            # 하퇴 벡터 (무릎 → 발목)
            shank_vector = ankle_pos - knee_pos
            
            # 수직 벡터와의 각도 계산 (발목 굴곡/신전)
            vertical_vector = np.array([0, 1])  # 아래 방향
            
            if np.linalg.norm(shank_vector) > 0:
                angle = self._calculate_angle(vertical_vector, shank_vector)
                angles.append(angle)
        
        if angles:
            return max(angles) - min(angles)  # 각도 범위
        return 0.0

    def calculate_actual_stance_ratio(self, joint_data: pd.DataFrame, cycle_labels: List[Dict], fps: float) -> float:
        """
        실제 입각기 비율 계산 (고정값 대신)
        
        Args:
            joint_data (pd.DataFrame): 관절 좌표 데이터
            cycle_labels (List[Dict]): 해당 주기의 라벨 정보
            fps (float): 프레임률
            
        Returns:
            float: 입각기 비율 (%)
        """
        if not cycle_labels:
            return 60.0  # 기본값
        
        total_frames = len(joint_data)
        stance_frames = 0
        
        # single_support 구간을 입각기로 간주
        for label in cycle_labels:
            if 'single_support' in label['phase']:
                label_start = max(0, label['start_frame'] - joint_data['frame'].iloc[0])
                label_end = min(total_frames, label['end_frame'] - joint_data['frame'].iloc[0])
                stance_frames += max(0, label_end - label_start)
        
        if total_frames > 0:
            stance_ratio = (stance_frames / total_frames) * 100
            return min(100.0, max(0.0, stance_ratio))  # 0-100% 범위로 제한
        
        return 60.0  # 기본값

    def calculate_gait_metrics_from_labels(self, video_path: str, joint_coords: pd.DataFrame, 
                                         support_labels: List[Dict]) -> List[GaitCycle]:
        """
        라벨 데이터와 관절 좌표를 사용해 보행 지표 계산 (메인 함수)
        
        Args:
            video_path (str): 비디오 파일 경로
            joint_coords (pd.DataFrame): 관절 좌표 데이터
            support_labels (List[Dict]): 지지 단계 라벨 데이터
            
        Returns:
            List[GaitCycle]: 보행 주기별 지표 리스트
        """
        # 비디오 정보 획득
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 라벨 기반 보행 주기 식별
        gait_cycles_data = self.identify_gait_cycles_from_labels(joint_coords, support_labels)
        
        gait_cycles = []
        
        for cycle_data in gait_cycles_data:
            joint_data = cycle_data['joint_data']
            foot = cycle_data['foot']
            
            # 실제 보행 지표 계산
            stride_length = self.calculate_stride_length_from_trajectory(joint_data, foot)
            velocity = self.calculate_velocity_from_movement(joint_data, foot, fps)
            cycle_time = len(joint_data) / fps
            cadence = 60 / cycle_time if cycle_time > 0 else 0
            
            # 관절 각도 계산
            joint_angles = self.calculate_joint_angles(joint_data)
            
            if joint_angles:
                hip_angles = [ja.hip_angle for ja in joint_angles]
                knee_angles = [ja.knee_angle for ja in joint_angles]
                
                hip_rom = max(hip_angles) - min(hip_angles) if hip_angles else 0
                knee_rom = max(knee_angles) - min(knee_angles) if knee_angles else 0
            else:
                hip_rom = knee_rom = 0
            
            # 실제 발목 각도 계산
            ankle_rom = self.calculate_actual_ankle_angle(joint_data, foot)
            
            # 실제 입각기 비율 계산
            stance_ratio = self.calculate_actual_stance_ratio(joint_data, cycle_data['cycle_labels'], fps)
        
            gait_cycle = GaitCycle(
                start_frame=cycle_data['start_frame'],
                end_frame=cycle_data['end_frame'],
                foot=foot,
                stride_length=stride_length,
                velocity=velocity,
                cycle_time=cycle_time,
                cadence=cadence,
                hip_rom=hip_rom,
                knee_rom=knee_rom,
                ankle_rom=ankle_rom,
                stance_ratio=stance_ratio
            )
            
            gait_cycles.append(gait_cycle)
        
        return gait_cycles