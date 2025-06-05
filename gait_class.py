"""
보행 분석 연구 지원 코드
MediaPipe Pose 기반 관절 데이터 분석 및 보행 이벤트(HS/TO) 검출

권장 파일명: gait_classes.py 또는 gait_analyzer_core.py

주요 클래스:
- GaitAnalyzer: 보행 분석을 위한 통합 클래스
  * step1_prepare_video_data(): 비디오 데이터 준비
  * step2_extract_joint_signals(): 관절 시계열 신호 추출  
  * step3_detect_gait_events(): 보행 이벤트 검출
  * step4_visualize_and_export(): 시각화 및 결과 내보내기

필요한 의존성:
- OpenCV (cv2): 비디오 처리
- MediaPipe: 포즈 추정
- NumPy: 수치 연산
- Pandas: 데이터 조작
- SciPy: 신호 처리 및 필터링
- Matplotlib: 시각화
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from typing import Dict, List, Tuple, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 주요 관절 인덱스 정의 (6개 관절만 사용: 엉덩이 2개, 발목 2개, 무릎 2개)
JOINT_INDICES = {
    'left_ankle': 27,
    'right_ankle': 28,
    'left_knee': 25,
    'right_knee': 26,
    'left_hip': 23,
    'right_hip': 24
}

class GaitAnalyzer:
    """보행 분석을 위한 통합 클래스"""
    
    def __init__(self, video_path: str, output_dir: str = "./output", enable_fast_mode: bool = True):
        self.video_path = video_path
        self.output_dir = output_dir
        self.enable_fast_mode = enable_fast_mode  # 고속 연산 모드 (기본값: True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 저장용 변수
        self.frame_data = []
        self.joint_coordinates = {}
        self.joint_angles = {}
        self.joint_distances = {}
        self.events = []
        
        # 고속 연산 모드 설정
        if self.enable_fast_mode:
            self.coord_precision = 3  # 좌표 정밀도
            self.angle_precision = 5  # 각도 정밀도
            self.distance_precision = 3  # 거리 정밀도
            self.numpy_dtype = np.float32  # 메모리 절약 및 연산 속도 향상
            logger.info("고속 연산 모드 활성화: 정밀도 제한 및 float32 사용")
        else:
            self.coord_precision = 6
            self.angle_precision = 8
            self.distance_precision = 6
            self.numpy_dtype = np.float64
            logger.info("일반 연산 모드: 높은 정밀도 사용")
        
    def round_coords(self, value: float, precision: int = None) -> float:
        """좌표값을 지정된 소수점 자리수로 제한하여 연산 속도 향상"""
        if precision is None:
            precision = self.coord_precision
        return round(float(value), precision)
    
    def round_angle(self, value: float) -> float:
        """각도값을 지정된 소수점 자리수로 제한"""
        return round(float(value), self.angle_precision)
    
    def round_distance(self, value: float) -> float:
        """거리값을 지정된 소수점 자리수로 제한"""
        return round(float(value), self.distance_precision)
    
    def optimize_array(self, arr: np.ndarray) -> np.ndarray:
        """배열을 고속 연산용으로 최적화"""
        if self.enable_fast_mode:
            return np.round(arr.astype(self.numpy_dtype), self.coord_precision)
        return arr
    
    def optimize_coordinates(self, landmarks, joint_indices: dict) -> dict:
        """관절 좌표를 벡터화하여 한번에 최적화"""
        coords = {}
        for joint_name, idx in joint_indices.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                coords[joint_name] = {
                    'x': self.round_coords(landmark.x),
                    'y': self.round_coords(landmark.y), 
                    'z': self.round_coords(landmark.z)
                }
            else:
                coords[joint_name] = {'x': np.nan, 'y': np.nan, 'z': np.nan}
        return coords
    
    def step1_prepare_video_data(self) -> pd.DataFrame:
        """
        Step 1: 데이터 준비 및 비디오 전처리
        비디오를 프레임 단위로 분해하고 타임스탬프 매핑 테이블 생성
        """
        logger.info("Step 1: 비디오 데이터 준비 시작")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"비디오 정보: FPS={fps}, 총 프레임 수={total_frames}")
        
        # 프레임-타임스탬프 매핑 생성
        frame_timestamp_mapping = []
        
        for frame_idx in range(total_frames):
            timestamp = frame_idx / fps
            frame_timestamp_mapping.append({
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'timestamp_ms': int(timestamp * 1000)
            })
        
        cap.release()
        
        # DataFrame으로 저장
        df_mapping = pd.DataFrame(frame_timestamp_mapping)
        mapping_path = os.path.join(self.output_dir, 'frame_timestamp_mapping.csv')
        df_mapping.to_csv(mapping_path, index=False)
        
        logger.info(f"프레임-타임스탬프 매핑 저장 완료: {mapping_path}")
        
        return df_mapping
    
    def extract_pose_landmarks(self, image):
        """MediaPipe를 사용하여 관절 좌표 추출"""
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return results
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """세 점 사이의 각도 계산 (p2가 꼭짓점)"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return self.round_angle(np.degrees(angle))
    
    def calculate_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """두 점 사이의 유클리드 거리 계산"""
        return self.round_distance(np.linalg.norm(p1 - p2))
    
    def step2_extract_joint_signals(self) -> Dict:
        """
        Step 2: 관심 관절의 시계열 신호 생성 및 전처리
        """
        logger.info("Step 2: 관절 시계열 신호 추출 시작")
        
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        
        # 시계열 데이터 저장용 딕셔너리
        time_series_data = {
            'frame_idx': [],
            'timestamp': []
        }
        
        # 관절별 좌표 초기화
        for joint_name in JOINT_INDICES.keys():
            for coord in ['x', 'y', 'z']:
                time_series_data[f'{joint_name}_{coord}'] = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe로 관절 추출
            results = self.extract_pose_landmarks(frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # 프레임 정보 저장
                time_series_data['frame_idx'].append(frame_idx)
                time_series_data['timestamp'].append(frame_idx / cap.get(cv2.CAP_PROP_FPS))
                
                # 각 관절 좌표 저장
                for joint_name, idx in JOINT_INDICES.items():
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        time_series_data[f'{joint_name}_x'].append(self.round_coords(landmark.x))
                        time_series_data[f'{joint_name}_y'].append(self.round_coords(landmark.y))
                        time_series_data[f'{joint_name}_z'].append(self.round_coords(landmark.z))
                    else:
                        # 누락된 경우 NaN 처리
                        time_series_data[f'{joint_name}_x'].append(np.nan)
                        time_series_data[f'{joint_name}_y'].append(np.nan)
                        time_series_data[f'{joint_name}_z'].append(np.nan)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"처리 중: {frame_idx} 프레임")
        
        cap.release()
        
        # DataFrame 변환
        df = pd.DataFrame(time_series_data)
        
        # 좌표 컬럼을 3자리로 반올림, 각도 컬럼을 5자리로 반올림 (속도 최적화)
        coord_columns = [col for col in df.columns if any(coord in col for coord in ['_x', '_y', '_z'])]
        for col in coord_columns:
            df[col] = df[col].round(3)  # 좌표는 3자리로 변경
        
        # 관절 간 거리 계산 (벡터화)
        logger.info("관절 간 거리 계산 중 (벡터화 연산)...")
        
        # 양 발목 간 거리 - 벡터화
        left_ankle = self.optimize_array(df[['left_ankle_x', 'left_ankle_y']].values)
        right_ankle = self.optimize_array(df[['right_ankle_x', 'right_ankle_y']].values)
        ankle_distances = np.linalg.norm(left_ankle - right_ankle, axis=1)
        df['ankle_distance'] = np.round(ankle_distances, self.distance_precision)
        
        # 무릎-엉덩이 거리 (좌/우) - 벡터화
        for side in ['left', 'right']:
            knee_coords = self.optimize_array(df[[f'{side}_knee_x', f'{side}_knee_y']].values)
            hip_coords = self.optimize_array(df[[f'{side}_hip_x', f'{side}_hip_y']].values)
            distances = np.linalg.norm(knee_coords - hip_coords, axis=1)
            df[f'{side}_knee_hip_distance'] = np.round(distances, self.distance_precision)
        
        # 관절 각도 계산 (벡터화)
        logger.info("관절 각도 계산 중 (벡터화 연산)...")
        
        # 무릎 각도 (엉덩이-무릎-발목) - 벡터화
        for side in ['left', 'right']:
            hip_coords = self.optimize_array(df[[f'{side}_hip_x', f'{side}_hip_y']].values)
            knee_coords = self.optimize_array(df[[f'{side}_knee_x', f'{side}_knee_y']].values)
            ankle_coords = self.optimize_array(df[[f'{side}_ankle_x', f'{side}_ankle_y']].values)
            
            # 벡터 계산
            v1 = hip_coords - knee_coords
            v2 = ankle_coords - knee_coords
            
            # 내적과 노름 계산
            dot_products = np.sum(v1 * v2, axis=1)
            norms_v1 = np.linalg.norm(v1, axis=1) + 1e-8
            norms_v2 = np.linalg.norm(v2, axis=1) + 1e-8
            
            # 각도 계산
            cos_angles = np.clip(dot_products / (norms_v1 * norms_v2), -1.0, 1.0)
            angles = np.degrees(np.arccos(cos_angles))
            df[f'{side}_knee_angle'] = np.round(angles, self.angle_precision)
        
        # 시계열 신호 필터링 (고속 모드 최적화)
        logger.info("시계열 신호 필터링 중 (고속 모드 최적화)...")
        
        # Savitzky-Golay 필터 파라미터 (고속 모드에서 조정)
        window_length = 9 if self.enable_fast_mode else 11  # 홀수여야 함
        polyorder = 2 if self.enable_fast_mode else 3
        
        filtered_columns = []
        original_columns_to_remove = []
        
        for col in df.columns:
            if col not in ['frame_idx', 'timestamp'] and df[col].notna().sum() > window_length:
                try:
                    # 결측치 보간 (고속 모드에서는 간단한 보간)
                    if self.enable_fast_mode:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    else:
                        df[col] = df[col].interpolate(method='spline', order=2, limit_direction='both')
                    
                    # 필터링
                    filtered_values = savgol_filter(df[col].fillna(method='ffill').fillna(method='bfill'), 
                                                   window_length, polyorder)
                    
                    # 고속 모드에서는 필터링 결과도 즉시 반올림
                    if self.enable_fast_mode:
                        if 'angle' in col:
                            filtered_values = np.round(filtered_values, self.angle_precision)
                        elif 'distance' in col:
                            filtered_values = np.round(filtered_values, self.distance_precision)
                        else:  # 좌표
                            filtered_values = np.round(filtered_values, self.coord_precision)
                    
                    filtered_col = f'{col}_filtered'
                    df[filtered_col] = filtered_values.astype(self.numpy_dtype if self.enable_fast_mode else np.float64)
                    filtered_columns.append(filtered_col)
                    original_columns_to_remove.append(col)
                except Exception as e:
                    logger.warning(f"필터링 실패: {col}, 오류: {e}")
        
        # 원본 데이터 컬럼 제거 (메모리 절약)
        df = df.drop(columns=original_columns_to_remove)
        logger.info(f"원본 노이즈 데이터 {len(original_columns_to_remove)}개 컬럼 제거 (메모리 절약)")
        
        # 필터링된 컬럼명에서 '_filtered' 접미사 제거
        rename_dict = {}
        for col in filtered_columns:
            clean_name = col.replace('_filtered', '')
            rename_dict[col] = clean_name
        
        df = df.rename(columns=rename_dict)
        logger.info(f"필터링된 데이터 {len(filtered_columns)}개 컬럼의 접미사 '_filtered' 제거")
        
        # 고속 모드에서는 추가 반올림 과정 생략 (이미 처리됨)
        if not self.enable_fast_mode:
            # 일반 모드에서만 추가 반올림 수행
            angle_columns = [col for col in df.columns if 'angle' in col]
            distance_columns = [col for col in df.columns if 'distance' in col]
            
            for col in angle_columns:
                df[col] = df[col].round(self.angle_precision)
            
            for col in distance_columns:
                df[col] = df[col].round(self.distance_precision)
        
        logger.info("데이터 정밀도 최적화 완료")
        
        # 최종 저장 전 모든 수치 데이터 반올림 (CSV 출력 형식 통일)
        coord_columns = [col for col in df.columns if any(coord in col for coord in ['_x', '_y', '_z'])]
        angle_columns = [col for col in df.columns if 'angle' in col]
        distance_columns = [col for col in df.columns if 'distance' in col]
        
        # 좌표는 3자리로 반올림
        for col in coord_columns:
            df[col] = df[col].round(3)
        
        # 각도는 5자리로 반올림  
        for col in angle_columns:
            df[col] = df[col].round(5)
            
        # 거리는 3자리로 반올림
        for col in distance_columns:
            df[col] = df[col].round(3)
        
        # 결과 저장 (필터링된 데이터만 포함)
        output_path = os.path.join(self.output_dir, 'joint_time_series.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"관절 시계열 데이터 저장 완료 (필터링된 데이터만): {output_path}")
        
        self.time_series_df = df
        return df
    
    def step3_detect_gait_events(self) -> pd.DataFrame:
        """
        Step 3: 규칙 기반 시계열 분석에 의한 보행 이벤트(HS, TO) 검출
        
        논문 방법론에 따라 발목의 x축(전후 방향) 변위 신호를 사용:
        - HS (Heel Strike): x축 변위의 피크(최대값) - 발이 앞으로 최대한 나아간 시점
        - TO (Toe Off): x축 변위의 계곡(최소값) - 발이 뒤로 최대한 당겨진 시점
        """
        logger.info("Step 3: 보행 이벤트 검출 시작 (x축 변위 기반)")
        
        if not hasattr(self, 'time_series_df'):
            raise ValueError("먼저 step2_extract_joint_signals()를 실행하세요.")
        
        df = self.time_series_df
        events = []
        
        # 좌/우 발목 x좌표 시계열 사용 (논문 방법론에 따라)
        for side in ['left', 'right']:
            ankle_x = df[f'{side}_ankle_x'].values  # '_filtered' 접미사 제거됨
            
            # HS (Heel Strike) 검출 - x축 변위의 피크(최대값)
            # 발이 앞으로 최대한 나아간 지점에서 지면에 닿음
            hs_indices, hs_properties = find_peaks(ankle_x,  # 직접 피크 검출
                                                   prominence=0.03,
                                                   distance=15)  # 최소 15프레임 간격
            
            # TO (Toe Off) 검출 - x축 변위의 계곡(최소값)
            # 발이 뒤로 최대한 당겨진 지점에서 지면에서 떨어짐
            to_indices, to_properties = find_peaks(-ankle_x,  # 음수로 변환하여 최소값 검출
                                                   prominence=0.03,
                                                   distance=15)
            
            # 이벤트 저장
            for idx in hs_indices:
                events.append({
                    'frame_idx': int(df.iloc[idx]['frame_idx']),
                    'timestamp': df.iloc[idx]['timestamp'],
                    'event_type': f'HS_{side}',
                    'ankle_x': ankle_x[idx],  # x축 값 저장
                    'ankle_y': df.iloc[idx][f'{side}_ankle_y']  # '_filtered' 접미사 제거됨
                })
            
            for idx in to_indices:
                events.append({
                    'frame_idx': int(df.iloc[idx]['frame_idx']),
                    'timestamp': df.iloc[idx]['timestamp'],
                    'event_type': f'TO_{side}',
                    'ankle_x': ankle_x[idx],  # x축 값 저장
                    'ankle_y': df.iloc[idx][f'{side}_ankle_y']  # '_filtered' 접미사 제거됨
                })
        
        # 이벤트 정렬
        events_df = pd.DataFrame(events).sort_values('frame_idx')
        
        # 이벤트 수치 데이터 반올림 (좌표는 3자리)
        numeric_columns = ['ankle_x', 'ankle_y']
        for col in numeric_columns:
            if col in events_df.columns:
                events_df[col] = events_df[col].round(3)
        
        # 이벤트 검증 (시각화)
        self.visualize_events(df, events_df)
        
        # 이벤트 저장
        events_path = os.path.join(self.output_dir, 'gait_events.csv')
        events_df.to_csv(events_path, index=False)
        logger.info(f"보행 이벤트 저장 완료: {events_path}")
        
        self.events_df = events_df
        return events_df
    
    def visualize_events(self, df: pd.DataFrame, events_df: pd.DataFrame):
        """
        이벤트 검출 결과 시각화 (x축 변위 + 무릎 관절 각도)
        상단: x축 변위와 HS/TO 이벤트
        하단: 무릎 관절 각도 변화
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        for i, side in enumerate(['left', 'right']):
            # 상단: x축 변위와 이벤트 (메인 분석)
            ax_x = axes[0, i]
            
            # 발목 x좌표 시계열 (논문 방법론)
            ax_x.plot(df['timestamp'], df[f'{side}_ankle_x'], 
                     label=f'{side} ankle x', alpha=0.7, color='green')
            
            # HS 이벤트 (x축 피크)
            hs_events = events_df[events_df['event_type'] == f'HS_{side}']
            if not hs_events.empty:
                ax_x.scatter(hs_events['timestamp'], hs_events['ankle_x'], 
                          color='red', s=100, label=f'HS {side} (Peak)', zorder=5)
            
            # TO 이벤트 (x축 계곡)
            to_events = events_df[events_df['event_type'] == f'TO_{side}']
            if not to_events.empty:
                ax_x.scatter(to_events['timestamp'], to_events['ankle_x'], 
                          color='blue', s=100, label=f'TO {side} (Valley)', zorder=5)
            
            ax_x.set_xlabel('Time (s)')
            ax_x.set_ylabel('Ankle X Position (Anterior-Posterior)')
            ax_x.set_title(f'{side.capitalize()} Ankle X-axis - Gait Events')
            ax_x.legend()
            ax_x.grid(True, alpha=0.3)
            
            # 하단: 무릎 관절 각도 (보행 분석에 중요한 지표)
            ax_knee = axes[1, i]
            
            # 무릎 관절 각도 시계열
            knee_angle_col = f'{side}_knee_angle'
            if knee_angle_col in df.columns:
                ax_knee.plot(df['timestamp'], df[knee_angle_col], 
                           label=f'{side} knee angle', alpha=0.7, color='purple')
                
                # 이벤트 시점에서의 무릎 각도 표시
                if not hs_events.empty:
                    hs_knee_angles = []
                    for _, event in hs_events.iterrows():
                        frame_idx = int(event['frame_idx'])
                        if frame_idx < len(df):
                            knee_angle = df.iloc[frame_idx][knee_angle_col]
                            hs_knee_angles.append(knee_angle)
                    
                    if hs_knee_angles:
                        ax_knee.scatter(hs_events['timestamp'], hs_knee_angles, 
                                      color='red', s=100, label=f'HS {side} knee angle', 
                                      zorder=5, alpha=0.8)
                
                if not to_events.empty:
                    to_knee_angles = []
                    for _, event in to_events.iterrows():
                        frame_idx = int(event['frame_idx'])
                        if frame_idx < len(df):
                            knee_angle = df.iloc[frame_idx][knee_angle_col]
                            to_knee_angles.append(knee_angle)
                    
                    if to_knee_angles:
                        ax_knee.scatter(to_events['timestamp'], to_knee_angles, 
                                      color='blue', s=100, label=f'TO {side} knee angle', 
                                      zorder=5, alpha=0.8)
            else:
                # 해당 관절 각도 데이터가 없는 경우
                ax_knee.text(0.5, 0.5, f'No {side} knee angle data', 
                           transform=ax_knee.transAxes, ha='center', va='center')
            
            ax_knee.set_xlabel('Time (s)')
            ax_knee.set_ylabel('Knee Angle (degrees)')
            ax_knee.set_title(f'{side.capitalize()} Knee Joint Angle')
            ax_knee.legend()
            ax_knee.grid(True, alpha=0.3)
            
            # 정상 보행 범위 표시 (참고선)
            ax_knee.axhline(y=160, color='gray', linestyle='--', alpha=0.5, label='Normal range')
            ax_knee.axhline(y=180, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'gait_events_plot.png'), dpi=300)
        plt.close()
    
    def step4_visualize_and_export(self):
        """
        Step 4: 이벤트 시각화 및 데이터 구조화
        """
        logger.info("Step 4: 이벤트 시각화 및 데이터 구조화 시작")
        
        if not hasattr(self, 'events_df'):
            raise ValueError("먼저 step3_detect_gait_events()를 실행하세요.")
        
        # 비디오에 스켈레톤 및 이벤트 오버레이
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 출력 비디오 설정
        output_video_path = os.path.join(self.output_dir, 'gait_analysis_overlay.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # 이벤트를 프레임 인덱스로 매핑
        event_map = {}
        for _, event in self.events_df.iterrows():
            frame_idx = int(event['frame_idx'])
            if frame_idx not in event_map:
                event_map[frame_idx] = []
            event_map[frame_idx].append(event['event_type'])
        
        frame_idx = 0
        
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # MediaPipe 처리
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # 스켈레톤 그리기
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # 이벤트 표시
                    if frame_idx in event_map:
                        events = event_map[frame_idx]
                        y_offset = 50
                        
                        for event in events:
                            color = (0, 0, 255) if 'HS' in event else (255, 0, 0)  # HS: 빨강, TO: 파랑
                            text = event.replace('_', ' ').upper()
                            
                            # 이벤트 텍스트 표시
                            cv2.putText(frame, text, (50, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                            
                            # 발목 위치에 원 그리기
                            if 'left' in event:
                                ankle_idx = JOINT_INDICES['left_ankle']
                            else:
                                ankle_idx = JOINT_INDICES['right_ankle']
                            
                            landmark = results.pose_landmarks.landmark[ankle_idx]
                            x = int(landmark.x * width)
                            y = int(landmark.y * height)
                            cv2.circle(frame, (x, y), 15, color, -1)
                            
                            y_offset += 50
                
                out.write(frame)
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"처리 중: {frame_idx} 프레임")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        logger.info(f"오버레이 비디오 저장 완료: {output_video_path}")
        
        # 전체 데이터 통합 및 구조화
        self.export_structured_data()
    
    def export_structured_data(self):
        """모든 데이터를 구조화된 형식으로 저장"""
        # 시계열 데이터와 이벤트 데이터 병합
        df = self.time_series_df.copy()
        
        # 각 프레임에 이벤트 정보 추가
        df['event_type'] = ''
        
        for _, event in self.events_df.iterrows():
            frame_idx = int(event['frame_idx'])
            if frame_idx < len(df):
                if df.loc[frame_idx, 'event_type']:
                    df.loc[frame_idx, 'event_type'] += ',' + event['event_type']
                else:
                    df.loc[frame_idx, 'event_type'] = event['event_type']
        
        # 최종 저장 전 모든 수치 데이터 반올림 (CSV 출력 형식 통일)
        coord_columns = [col for col in df.columns if any(coord in col for coord in ['_x', '_y', '_z'])]
        angle_columns = [col for col in df.columns if 'angle' in col]
        distance_columns = [col for col in df.columns if 'distance' in col]
        
        # 좌표는 3자리로 반올림
        for col in coord_columns:
            df[col] = df[col].round(3)
        
        # 각도는 5자리로 반올림  
        for col in angle_columns:
            df[col] = df[col].round(5)
            
        # 거리는 3자리로 반올림
        for col in distance_columns:
            df[col] = df[col].round(3)
        
        # 최종 구조화 데이터 저장
        final_output_path = os.path.join(self.output_dir, 'gait_analysis_complete.csv')
        df.to_csv(final_output_path, index=False)
        
        # 요약 통계 생성
        summary = {
            'total_frames': len(df),
            'total_duration_seconds': df['timestamp'].max(),
            'total_HS_left': len(self.events_df[self.events_df['event_type'] == 'HS_left']),
            'total_HS_right': len(self.events_df[self.events_df['event_type'] == 'HS_right']),
            'total_TO_left': len(self.events_df[self.events_df['event_type'] == 'TO_left']),
            'total_TO_right': len(self.events_df[self.events_df['event_type'] == 'TO_right']),
            'fps': len(df) / df['timestamp'].max() if df['timestamp'].max() > 0 else 0
        }
        
        # 보행 주기 분석
        for side in ['left', 'right']:
            hs_events = self.events_df[self.events_df['event_type'] == f'HS_{side}']['timestamp'].values
            if len(hs_events) > 1:
                stride_times = np.diff(hs_events)
                summary[f'mean_stride_time_{side}'] = np.mean(stride_times)
                summary[f'std_stride_time_{side}'] = np.std(stride_times)
                summary[f'cadence_{side}'] = 60 / np.mean(stride_times) if np.mean(stride_times) > 0 else 0
        
        # 요약 저장
        summary_path = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"분석 완료. 결과 저장 위치: {self.output_dir}")
        logger.info(f"요약 통계: {summary}")