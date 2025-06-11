# 보행 분석 시스템 (Gait Analysis System)
# ==========================================
# 이 시스템은 MediaPipe Pose 기반 관절 추정과 IMU 데이터 동기화를 통해 
# HS(Heel Strike), TO(Toe Off) 이벤트를 검출하고 시각화합니다.

# ======================
# 1. gait_class.py
# ======================
"""
gait_class.py - 보행 분석 핵심 로직 모듈

이 모듈은 다음 기능을 제공합니다:
1. 보행 방향 감지 (forward/backward)
2. 발목 좌표 노이즈 제거
3. HS/TO 이벤트 검출
4. 데이터 동기화 및 라벨링
"""

import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from scipy.signal import find_peaks, butter, filtfilt, medfilt
from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple, Dict, Optional
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GaitEvent:
    """보행 이벤트 데이터 클래스"""
    frame_idx: int
    event_type: str  # "HS" or "TO"
    foot: str        # "left" or "right"
    timestamp: float # 초 단위
    
    def to_dict(self):
        return {
            'frame': self.frame_idx,
            'event': self.event_type,
            'foot': self.foot,
            'timestamp': self.timestamp
        }


class GaitAnalyzer:
    """보행 분석 메인 클래스"""
    
    def __init__(self, video_path: str, imu_path: Optional[str] = None):
        """
        초기화 함수
        
        Args:
            video_path (str): 보행 영상 파일 경로
            imu_path (str, optional): IMU 데이터 CSV 파일 경로
        """
        self.video_path = video_path
        self.imu_path = imu_path
        
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 비디오 정보
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 데이터 저장용 변수
        self.landmarks_data = []
        self.walking_direction = None
        self.gait_events = []
        
        # IMU 데이터 로드
        if imu_path:
            self.imu_data = pd.read_csv(imu_path)
        else:
            self.imu_data = None
    
    def extract_initial_landmarks_for_direction(self, frames_count: int = 15) -> pd.DataFrame:
        """
        방향 감지용 초기 프레임 랜드마크 추출 (발목 Z축만)
        
        Args:
            frames_count (int): 추출할 초기 프레임 수 (15프레임으로 축소)
            
        Returns:
            pd.DataFrame: 초기 프레임의 발목 Z축 좌표 데이터
        """
        print(f"방향 감지용 초기 {frames_count}프레임 처리 중...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        initial_landmarks = []
        
        while frame_idx < frames_count:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # RGB 변환 및 포즈 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
                
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 발목 Z축만 추출 (방향 감지용 - 발목만으로 충분)
                data = {
                    'left_ankle_z': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z,
                    'right_ankle_z': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].z,
                }
                
                initial_landmarks.append(data)
                
            frame_idx += 1
                
        return pd.DataFrame(initial_landmarks)

    def extract_pose_landmarks(self, progress_callback=None) -> pd.DataFrame:
        """
        전체 비디오에서 발목 X좌표 추출 (이벤트 검출용)
        
        Args:
            progress_callback: 진행률 콜백 함수 (current, total)
            
        Returns:
            pd.DataFrame: 프레임별 발목 X좌표 데이터
        """
        print("이벤트 검출용 발목 좌표 추출 시작...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        landmarks_list = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # RGB 변환 및 포즈 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
                
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 발목 X좌표만 추출 (이벤트 검출용)
                data = {
                    'frame': frame_idx,
                    'timestamp': round(frame_idx / self.fps, 2),  # 소수점 2자리로 제한
                    'left_ankle_x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                    'right_ankle_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                }
                
                landmarks_list.append(data)
                
            frame_idx += 1
            
            # 진행률 콜백 호출 (매 프레임마다)
            if progress_callback:
                progress_callback(frame_idx, self.total_frames)
            
            # 진행률 표시 (60프레임마다 - 콘솔용)
            if frame_idx % 60 == 0:
                print(f"처리 중: {frame_idx}/{self.total_frames} 프레임")
        
        self.landmarks_data = pd.DataFrame(landmarks_list)
        print(f"총 {len(self.landmarks_data)} 프레임 처리 완료")
        return self.landmarks_data

    def extract_full_landmarks_for_analysis(self, progress_callback=None) -> pd.DataFrame:
        """
        전체 비디오에서 모든 관절 좌표 추출 (보행 지표 계산용)
        
        Args:
            progress_callback: 진행률 콜백 함수 (current, total)
            
        Returns:
            pd.DataFrame: 프레임별 전체 관절 좌표 데이터
        """
        print("보행 지표 계산용 전체 관절 좌표 추출 시작...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        landmarks_list = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # RGB 변환 및 포즈 추정
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
                
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 보행 분석에 필요한 모든 관절 좌표 추출
                data = {
                    'frame': frame_idx,
                    'timestamp': round(frame_idx / self.fps, 3),
                    
                    # 왼쪽 다리
                    'left_hip_x': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                    'left_hip_y': landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                    'left_knee_x': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                    'left_knee_y': landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                    'left_ankle_x': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                    'left_ankle_y': landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                    
                    # 오른쪽 다리
                    'right_hip_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                    'right_hip_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y,
                    'right_knee_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].x,
                    'right_knee_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE].y,
                    'right_ankle_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                    'right_ankle_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                    
                    # 추가 관절 (필요시)
                    'left_shoulder_x': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    'left_shoulder_y': landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    'right_shoulder_x': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    'right_shoulder_y': landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                }
                
                landmarks_list.append(data)
                
            frame_idx += 1
            
            # 진행률 콜백 호출 (매 프레임마다)
            if progress_callback:
                progress_callback(frame_idx, self.total_frames)
            
            # 진행률 표시 (30프레임마다 - 콘솔용)
            if frame_idx % 30 == 0:
                print(f"관절 좌표 추출 중: {frame_idx}/{self.total_frames} 프레임")
        
        self.landmarks_data = pd.DataFrame(landmarks_list)
        print(f"✅ 전체 관절 좌표 추출 완료: {len(self.landmarks_data)} 프레임")
        return self.landmarks_data
    
    def detect_walking_direction(self, initial_frames: int = 15) -> str:
        """
        보행 방향 감지 (발목만 사용, 프레임 수 축소)
        
        Args:
            initial_frames (int): 분석할 초기 프레임 수 (15프레임으로 축소)
            
        Returns:
            str: "forward" 또는 "backward"
        """
        print("\n보행 방향 감지 시작...")
        
        # 초기 프레임에서 발목 Z축 좌표만 추출
        initial_data = self.extract_initial_landmarks_for_direction(initial_frames)
        
        if len(initial_data) == 0:
            print("경고: 방향 감지용 데이터를 추출할 수 없습니다. 기본값 'forward' 사용")
            self.walking_direction = "forward"
            return self.walking_direction
        
        # 발목만으로 방향 판별 (훨씬 단순)
        left_avg_z = initial_data['left_ankle_z'].mean()
        right_avg_z = initial_data['right_ankle_z'].mean()
        
        # Z값 차이로 방향 판별
        delta_z = right_avg_z - left_avg_z
        
        if delta_z < 0:
            self.walking_direction = "forward"  # 오른쪽이 앞
            print(f"보행 방향: Forward (→) [delta_z = {delta_z:.3f}]")  # 소수점 3자리로 제한
        else:
            self.walking_direction = "backward"  # 왼쪽이 앞
            print(f"보행 방향: Backward (←) [delta_z = {delta_z:.3f}]")
        
        return self.walking_direction
    
    def apply_enhanced_noise_reduction(self, signal: np.ndarray) -> np.ndarray:
        """
        효율적인 노이즈 제거 파이프라인 (4단계로 간소화)
        
        Args:
            signal (np.ndarray): 원본 시계열 신호
            
        Returns:
            np.ndarray: 노이즈가 제거된 신호
        """
        signal = pd.Series(signal)
        
        # 1. 결측치 보간 (선형 보간으로 단순화)
        signal = signal.interpolate(method='linear').ffill().bfill()
        
        # 2. 스파이크 제거 (median filter)
        signal = pd.Series(medfilt(signal.values, kernel_size=5))
        
        # 3. Butterworth 저역통과 필터
        nyquist_freq = self.fps / 2
        cutoff_freq = 3.0
        normalized_cutoff = cutoff_freq / nyquist_freq
        if normalized_cutoff < 1.0:
            b, a = butter(4, normalized_cutoff, btype='low')
            signal = filtfilt(b, a, signal.values)
        
        # 4. 가우시안 스무딩
        signal = gaussian_filter1d(signal, sigma=1.5)
        
        return signal
    
    def detect_gait_events(self) -> List[GaitEvent]:
        """
        HS/TO 이벤트 검출 (Step 3)
        
        Returns:
            List[GaitEvent]: 검출된 보행 이벤트 리스트
        """
        print("\n보행 이벤트 검출 시작...")
        
        # 발목 X좌표 추출 및 노이즈 제거
        left_ankle_x = self.apply_enhanced_noise_reduction(
            self.landmarks_data['left_ankle_x'].values
        )
        right_ankle_x = self.apply_enhanced_noise_reduction(
            self.landmarks_data['right_ankle_x'].values
        )
        
        # 보행 방향에 따른 피크 검출 (강화된 파라미터)
        # prominence: 피크의 현저성 (높을수록 더 명확한 피크만 검출)
        # distance: 최소 피크 간격 (프레임 단위, 높을수록 중복 검출 방지)
        # height: 최소 높이 기준 (정규화된 값 기준)
        
        prominence_threshold = 0.015  # 0.015에서 0.035로 증가 (더 높은 기준)
        min_distance = 15  # 10에서 15로 증가 (약 0.5초 간격, 30fps 기준)
        min_height = 0.02   # 추가: 최소 높이 기준
        
        # 디버깅 정보
        print(f"신호 범위 - 좌측 발목: [{left_ankle_x.min():.3f}, {left_ankle_x.max():.3f}]")
        print(f"신호 범위 - 우측 발목: [{right_ankle_x.min():.3f}, {right_ankle_x.max():.3f}]")
        print(f"검출 파라미터 - prominence: {prominence_threshold}, distance: {min_distance}, height: {min_height}")
        
        if self.walking_direction == "forward":
            # Forward: HS=최댓값, TO=최솟값
            print("Forward 방향: HS=최댓값, TO=최솟값")
            
            # HS 검출 (최댓값)
            hs_left, _ = find_peaks(left_ankle_x, 
                                   prominence=prominence_threshold, 
                                   distance=min_distance)
            hs_right, _ = find_peaks(right_ankle_x, 
                                    prominence=prominence_threshold, 
                                    distance=min_distance)
            
            # TO 검출 (최솟값, 반전 신호의 최댓값)
            to_left, _ = find_peaks(-left_ankle_x, 
                    prominence=prominence_threshold,
                                   distance=min_distance)
            to_right, _ = find_peaks(-right_ankle_x, 
                                    prominence=prominence_threshold, 
                                    distance=min_distance)
        else:
            # Backward: HS=최솟값, TO=최댓값
            print("Backward 방향: HS=최솟값, TO=최댓값")
            
            # HS 검출 (최솟값, 반전 신호의 최댓값)
            hs_left, _ = find_peaks(-left_ankle_x, 
                    prominence=prominence_threshold,
                                   distance=min_distance)
            hs_right, _ = find_peaks(-right_ankle_x, 
                                    prominence=prominence_threshold, 
                                    distance=min_distance)
            
            # TO 검출 (최댓값)
            to_left, _ = find_peaks(left_ankle_x, 
                                   prominence=prominence_threshold, 
                                   distance=min_distance)
            to_right, _ = find_peaks(right_ankle_x, 
                                    prominence=prominence_threshold, 
                                    distance=min_distance)
        
        # 검출 결과 디버깅
        print(f"검출된 피크 수:")
        print(f"  HS_left: {len(hs_left)}, HS_right: {len(hs_right)}")
        print(f"  TO_left: {len(to_left)}, TO_right: {len(to_right)}")
        
        # 이벤트 생성 및 병합
        events = []
        
        # 이벤트 생성 (timestamp 정밀도 제한)
        for idx in hs_left:
            events.append(GaitEvent(
                frame_idx=idx,
                event_type="HS",
                foot="left",
                timestamp=round(idx / self.fps, 2)  # 소수점 2자리로 제한
            ))
        
        for idx in to_left:
            events.append(GaitEvent(
                frame_idx=idx,
                event_type="TO",
                foot="left",
                timestamp=round(idx / self.fps, 2)
            ))
        
        for idx in hs_right:
            events.append(GaitEvent(
                frame_idx=idx,
                event_type="HS",
                foot="right",
                timestamp=round(idx / self.fps, 2)
            ))
        
        for idx in to_right:
            events.append(GaitEvent(
                frame_idx=idx,
                event_type="TO",
                foot="right",
                timestamp=round(idx / self.fps, 2)
            ))
        
        # 시간순 정렬만 수행 (정제 과정 제거)
        events.sort(key=lambda x: x.frame_idx)
        
        self.gait_events = events
        print(f"총 {len(events)} 개의 보행 이벤트 검출 완료")
        
        # 이벤트 요약 출력
        hs_count = sum(1 for e in events if e.event_type == "HS")
        to_count = sum(1 for e in events if e.event_type == "TO")
        left_events = sum(1 for e in events if e.foot == "left")
        right_events = sum(1 for e in events if e.foot == "right")
        
        print(f"  - Heel Strike (HS): {hs_count}개")
        print(f"  - Toe Off (TO): {to_count}개")
        print(f"  - 좌측 발: {left_events}개, 우측 발: {right_events}개")
        
        return events
    
    def analyze_gait_phases(self) -> List[Dict]:
        """
        HS/TO 이벤트를 기반으로 보행 단계 분석 (브루탈 이중지지 패턴 인식)
        
        HS_left->TO_right 또는 HS_right->TO_left 패턴을 무조건 이중지지로 분류
        
        Returns:
            List[Dict]: 각 구간의 보행 단계 정보
            [{'start_frame': int, 'end_frame': int, 'phase': str}]
            phase: 'double_support', 'single_support_left', 'single_support_right', 'non_gait'
        """
        if len(self.gait_events) == 0:
            return [{'start_frame': 0, 'end_frame': self.total_frames-1, 'phase': 'non_gait'}]
        
        # 이벤트를 시간순으로 정렬
        sorted_events = sorted(self.gait_events, key=lambda x: x.frame_idx)
        
        # 브루탈 접근법: HS->TO 패턴을 이중지지로 강제 분류
        double_support_ranges = []
        
        for i in range(len(sorted_events) - 1):
            current_event = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # HS_left -> TO_right 또는 HS_right -> TO_left 패턴 찾기
            if (current_event.event_type == "HS" and next_event.event_type == "TO" and 
                current_event.foot != next_event.foot):
                
                double_support_ranges.append({
                    'start_frame': current_event.frame_idx,
                    'end_frame': next_event.frame_idx,
                    'phase': 'double_support'
                })
        
        # 동시 이벤트도 이중지지로 처리 (중복 방지)
        processed_frames = set()
        for event in sorted_events:
            if event.frame_idx not in processed_frames:
                same_frame_events = [e for e in sorted_events if e.frame_idx == event.frame_idx]
                if len(same_frame_events) > 1:
                    has_hs = any(e.event_type == "HS" for e in same_frame_events)
                    has_to = any(e.event_type == "TO" for e in same_frame_events)
                    if has_hs and has_to:
                        double_support_ranges.append({
                            'start_frame': event.frame_idx,
                            'end_frame': event.frame_idx,
                            'phase': 'double_support'
                        })
                processed_frames.add(event.frame_idx)
        
        # 전체 프레임 범위에서 단계 분류
        phases = []
        covered_frames = set()
        
        # 1. 이중지지 구간 먼저 추가
        for ds_range in double_support_ranges:
            phases.append(ds_range)
            for frame in range(ds_range['start_frame'], ds_range['end_frame'] + 1):
                covered_frames.add(frame)
        
        # 2. 나머지 구간 분류
        current_left_contact = False
        current_right_contact = False
        
        # 첫 이벤트로 초기 상태 설정
        first_event = sorted_events[0]
        if first_event.event_type == "TO":
            if first_event.foot == "left":
                current_left_contact = True
            else:
                current_right_contact = True
        elif first_event.event_type == "HS":
            if first_event.foot == "left":
                current_right_contact = True
            else:
                current_left_contact = True
        
        # 각 이벤트 구간 처리
        last_covered_frame = -1
        
        for i, event in enumerate(sorted_events):
            # 현재 이벤트로 상태 업데이트
            if event.event_type == "HS":
                if event.foot == "left":
                    current_left_contact = True
                else:
                    current_right_contact = True
            elif event.event_type == "TO":
                if event.foot == "left":
                    current_left_contact = False
                else:
                    current_right_contact = False
            
            # 다음 이벤트까지의 구간 결정
            if i < len(sorted_events) - 1:
                next_frame = sorted_events[i + 1].frame_idx
                end_frame = next_frame - 1
            else:
                end_frame = self.total_frames - 1
            
            # 이미 이중지지로 분류된 프레임들은 건너뛰기
            start_frame = event.frame_idx
            if start_frame in covered_frames:
                # 이중지지 구간 다음부터 시작
                start_frame = start_frame + 1
                while start_frame <= end_frame and start_frame in covered_frames:
                    start_frame += 1
            
            if start_frame <= end_frame:
                # 현재 접촉 상태에 따른 단계 결정
                if current_left_contact and current_right_contact:
                    phase = 'double_support'
                elif current_left_contact and not current_right_contact:
                    phase = 'single_support_left'
                elif not current_left_contact and current_right_contact:
                    phase = 'single_support_right'
                else:
                    phase = 'non_gait'
                
                phases.append({
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'phase': phase
                })
                
                for frame in range(start_frame, end_frame + 1):
                    covered_frames.add(frame)
                    
                last_covered_frame = end_frame
        
        # 3. 첫 번째 이벤트 전까지 non_gait 추가
        if sorted_events[0].frame_idx > 0:
            first_uncovered = 0
            last_uncovered = sorted_events[0].frame_idx - 1
            
            # 이미 커버된 프레임들 제외
            while first_uncovered <= last_uncovered and first_uncovered in covered_frames:
                first_uncovered += 1
            
            if first_uncovered <= last_uncovered:
                phases.append({
                    'start_frame': first_uncovered,
                    'end_frame': last_uncovered,
                    'phase': 'non_gait'
                })
        
        # 프레임 순서대로 정렬
        phases.sort(key=lambda x: x['start_frame'])
        
        return phases
    
    def save_results(self, output_dir: str = "./results"):
        """
        분석 결과 저장
        
        Args:
            output_dir (str): 출력 디렉토리 경로
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 이벤트 CSV 저장
        events_data = []
        for event in self.gait_events:
            events_data.append({
                'frame': event.frame_idx,
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'foot': event.foot
            })
        
        events_df = pd.DataFrame(events_data)
        events_path = os.path.join(output_dir, "gait_events.csv")
        events_df.to_csv(events_path, index=False)
        
        # 2. 보행 단계 CSV 저장
        phases = self.analyze_gait_phases()
        phases_df = pd.DataFrame(phases)
        phases_path = os.path.join(output_dir, "gait_phases.csv")
        phases_df.to_csv(phases_path, index=False)
        

        
        # 3. 분석 요약 정보 CSV 저장
        summary_data = {
            'video_path': [self.video_path],
            'total_frames': [self.total_frames],
            'fps': [self.fps],
            'walking_direction': [self.walking_direction],
            'total_events': [len(self.gait_events)],
            'hs_events': [sum(1 for e in self.gait_events if e.event_type == "HS")],
            'to_events': [sum(1 for e in self.gait_events if e.event_type == "TO")],
            'analysis_date': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "analysis_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # 4. 통합 이벤트 타임라인 CSV 저장 (IMU 데이터와 매칭용)
        timeline_data = []
        for i in range(self.total_frames):
            row = {
                'frame': i,
                'timestamp': i / self.fps,
                'hs_left': 0,
                'hs_right': 0,
                'to_left': 0,
                'to_right': 0
            }
            
            # 해당 프레임의 이벤트 확인
            for event in self.gait_events:
                if event.frame_idx == i:
                    if event.event_type == "HS" and event.foot == "left":
                        row['hs_left'] = 1
                    elif event.event_type == "HS" and event.foot == "right":
                        row['hs_right'] = 1
                    elif event.event_type == "TO" and event.foot == "left":
                        row['to_left'] = 1
                    elif event.event_type == "TO" and event.foot == "right":
                        row['to_right'] = 1
            
            timeline_data.append(row)
        
        timeline_df = pd.DataFrame(timeline_data)
        timeline_path = os.path.join(output_dir, "event_timeline.csv")
        timeline_df.to_csv(timeline_path, index=False)
        
        # 5. IMU 데이터와 이벤트 라벨 병합 (IMU 데이터가 있는 경우)
        if self.imu_data is not None:
            print("IMU 데이터와 이벤트 라벨 병합 중...")
            
            # IMU 데이터 복사
            labeled_imu = self.imu_data.copy()
            
            # 이벤트 컬럼 추가
            labeled_imu['hs_left'] = 0
            labeled_imu['hs_right'] = 0
            labeled_imu['to_left'] = 0
            labeled_imu['to_right'] = 0
            
            # 프레임 기준으로 이벤트 매핑
            if 'frame' in labeled_imu.columns:
                for event in self.gait_events:
                    frame_idx = event.frame_idx
                    if frame_idx < len(labeled_imu):
                        if event.event_type == "HS" and event.foot == "left":
                            labeled_imu.loc[labeled_imu['frame'] == frame_idx, 'hs_left'] = 1
                        elif event.event_type == "HS" and event.foot == "right":
                            labeled_imu.loc[labeled_imu['frame'] == frame_idx, 'hs_right'] = 1
                        elif event.event_type == "TO" and event.foot == "left":
                            labeled_imu.loc[labeled_imu['frame'] == frame_idx, 'to_left'] = 1
                        elif event.event_type == "TO" and event.foot == "right":
                            labeled_imu.loc[labeled_imu['frame'] == frame_idx, 'to_right'] = 1
            
            # 라벨링된 IMU 데이터 저장
            labeled_imu_path = os.path.join(output_dir, "imu_data_labeled.csv")
            labeled_imu.to_csv(labeled_imu_path, index=False)
            print(f"  - imu_data_labeled.csv: 라벨링된 IMU 데이터")
        
        print(f"\n결과 저장 완료: {output_dir}")
        print(f"  - gait_events.csv: 이벤트 목록")
        print(f"  - gait_phases.csv: 보행 단계 정보")
        print(f"  - analysis_summary.csv: 분석 요약")
        print(f"  - event_timeline.csv: 프레임별 이벤트 타임라인")
    
    def get_filtered_ankle_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """필터링된 발목 X좌표 데이터 반환"""
        left_ankle_x = self.apply_enhanced_noise_reduction(
            self.landmarks_data['left_ankle_x'].values
        )
        right_ankle_x = self.apply_enhanced_noise_reduction(
            self.landmarks_data['right_ankle_x'].values
        )
        return left_ankle_x, right_ankle_x
    