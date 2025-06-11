"""
보행 파라미터 계산 엔진 (Gait Calculation Engine)
단순화된 3개 파라미터 계산: stride_time, stride_length, velocity

파라미터 정의:
1. Stride Time: 동일한 발의 두 HS(Heel Strike) 사이 시간 간격
2. Stride Length: dot(P2 - P1, walking_direction_unit_vector)
3. Velocity: stride_length / stride_time
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import math
import cv2


@dataclass
class JointPosition:
    """3D 관절 위치 데이터"""
    x: float
    y: float
    z: float
    
    def to_vector(self) -> np.ndarray:
        """벡터로 변환"""
        return np.array([self.x, self.y, self.z])


@dataclass
class FrameData:
    """프레임별 관절 데이터"""
    frame_number: int
    timestamp: float
    joints: Dict[str, JointPosition]
    
    def get_joint(self, joint_name: str) -> Optional[JointPosition]:
        """관절 위치 반환"""
        return self.joints.get(joint_name)
class GeometryUtils:
    """기하학적 계산 유틸리티"""
    
    @staticmethod
    def calculate_vector(point1: JointPosition, point2: JointPosition) -> np.ndarray:
        """두 점 사이의 벡터 계산"""
        return point2.to_vector() - point1.to_vector()
    
    @staticmethod
    def calculate_distance(point1: JointPosition, point2: JointPosition) -> float:
        """두 점 사이의 거리 계산"""
        return np.linalg.norm(GeometryUtils.calculate_vector(point1, point2))
    
    @staticmethod
    def project_vector_to_direction(vector: np.ndarray, direction: np.ndarray) -> float:
        """벡터를 특정 방향으로 투영"""
        direction_unit = direction / np.linalg.norm(direction)
        return np.dot(vector, direction_unit)
    
    @staticmethod
    def estimate_walking_direction(ankle_positions: List[JointPosition]) -> np.ndarray:
        """발목 위치들로부터 보행 방향 추정"""
        if len(ankle_positions) < 2:
            return np.array([1.0, 0.0, 0.0])  # 기본값: X축 방향
        
        # 전체 이동 벡터의 평균으로 보행 방향 추정
        total_movement = np.array([0.0, 0.0, 0.0])
        
        for i in range(1, len(ankle_positions)):
            movement = GeometryUtils.calculate_vector(ankle_positions[i-1], ankle_positions[i])
            total_movement += movement
        
        if np.linalg.norm(total_movement) == 0:
            return np.array([1.0, 0.0, 0.0])
        
        return total_movement / np.linalg.norm(total_movement)


class HeelStrikeDetector:
    """보행 라벨 데이터를 활용한 Heel Strike 검출"""
    
    def __init__(self, fps: float = 30.0):
        self.fps = fps    
    def detect_heel_strikes_from_labels(self, frame_data: List[FrameData], 
                                      support_labels: List[Dict]) -> List[Dict]:
        """
        라벨링 데이터를 활용한 Heel Strike 검출
        
        Args:
            frame_data: 관절 데이터
            support_labels: 지지 라벨 데이터 (이중지지, 좌발지지, 우발지지)
            
        Returns:
            검출된 HS 이벤트 정보
        """
        heel_strikes = []
        
        if not support_labels:
            print("⚠️ 지지 라벨 데이터가 없습니다.")
            return heel_strikes
        
        print(f"🔍 라벨 데이터 확인: {len(support_labels)}개 라벨")
        for i, label in enumerate(support_labels[:5]):  # 처음 5개만 출력
            print(f"  라벨 {i}: phase='{label.get('phase')}', frames={label.get('start_frame')}-{label.get('end_frame')}")
        
        # 라벨 변화 지점에서 HS 추정
        prev_phase = None
        
        for i, label in enumerate(support_labels):
            current_phase = label.get('phase', '')
            start_frame = label.get('start_frame', 0)
            end_frame = label.get('end_frame', 0)
            
            print(f"라벨 {i}: '{prev_phase}' -> '{current_phase}' (frame {start_frame})")
            
            # 이중지지에서 단일지지로 변할 때 = HS 발생
            if prev_phase == 'double_support':
                if current_phase in ['single_support_left', 'left_support']:
                    # 왼발 HS
                    heel_strikes.append({
                        'foot': 'left',
                        'frame': start_frame,
                        'time': start_frame / self.fps,
                        'type': 'heel_strike'
                    })
                    print(f"✅ 왼발 HS 검출: frame {start_frame}")
                elif current_phase in ['single_support_right', 'right_support']:
                    # 오른발 HS
                    heel_strikes.append({
                        'foot': 'right',
                        'frame': start_frame,
                        'time': start_frame / self.fps,
                        'type': 'heel_strike'
                    })
                    print(f"✅ 오른발 HS 검출: frame {start_frame}")
            
            prev_phase = current_phase
        
        print(f"🎯 총 {len(heel_strikes)}개의 Heel Strike 검출됨")
        return heel_strikes


class GaitCalculationEngine:
    """보행 계산 엔진 (3개 파라미터만)"""
    
    def __init__(self, fps: float = 30.0, user_height: float = 1.7, walking_direction: str = "forward", video_path: str = None):
        self.fps = fps
        self.hs_detector = HeelStrikeDetector(fps)
        self.user_height = user_height  # 사용자 키 (미터)
        self.walking_direction = walking_direction  # "forward" or "backward"
        self.pixel_to_meter_ratio = None  # 계산될 픽셀-미터 비율
        self.frame_width = 640  # 기본 비디오 너비
        self.frame_height = 480  # 기본 비디오 높이
        
        # 비디오 파일에서 실제 크기 가져오기
        if video_path:
            self._get_video_dimensions(video_path)
    
    def _get_video_dimensions(self, video_path: str):
        """비디오 파일에서 실제 가로세로 크기 가져오기"""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                raw_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                raw_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                self.frame_width = int(raw_width)
                self.frame_height = int(raw_height)
                print(f"📺 비디오에서 가져온 원본 크기 정보: width={raw_width}, height={raw_height}")
                print(f"📺 변환된 정수 크기: {self.frame_width} x {self.frame_height}")
                
                cap.release()
            else:
                print(f"⚠️ 비디오 파일을 열 수 없습니다: {video_path}")
                print(f"📺 기본 크기 사용: {self.frame_width} x {self.frame_height}")
        except Exception as e:
            print(f"⚠️ 비디오 크기 획득 중 오류: {e}")
            print(f"📺 기본 크기 사용: {self.frame_width} x {self.frame_height}")
    def process_joint_data(self, joint_data_list: List[Dict], timestamps: List[float]) -> List[FrameData]:
        """
        관절 데이터를 FrameData 형태로 변환
        
        Args:
            joint_data_list: MediaPipe 관절 데이터 리스트
            timestamps: 각 프레임의 타임스탬프
            
        Returns:
            변환된 FrameData 리스트
        """
        frame_data_list = []
        
        for i, (joint_data, timestamp) in enumerate(zip(joint_data_list, timestamps)):
            if joint_data is None:
                continue
            
            joints = {}
            for joint_name, joint_pos in joint_data.items():
                if joint_pos and isinstance(joint_pos, dict) and 'x' in joint_pos:
                    joints[joint_name] = JointPosition(
                        x=joint_pos['x'],
                        y=joint_pos['y'], 
                        z=joint_pos['z']
                    )
            
            frame_data = FrameData(
                frame_number=i,
                timestamp=timestamp,
                joints=joints
            )
            frame_data_list.append(frame_data)
        
        return frame_data_list
    
    def detect_walking_direction(self, frame_data: List[FrameData], initial_frames: int = 15) -> str:
        """
        보행 방향 감지 (발목만 사용, gait_class.py 방식 차용)
        
        Args:
            frame_data: 프레임 데이터 리스트
            initial_frames: 분석할 초기 프레임 수
            
        Returns:
            str: "forward" 또는 "backward"
        """
        print("\n🔍 보행 방향 감지 시작...")
        
        if len(frame_data) < initial_frames:
            initial_frames = len(frame_data)
        
        left_z_values = []
        right_z_values = []
        
        # 초기 프레임에서 발목 Z축 좌표 추출
        for i in range(min(initial_frames, len(frame_data))):
            frame = frame_data[i]
            
            left_ankle = frame.get_joint('left_ankle')
            right_ankle = frame.get_joint('right_ankle')
            
            if left_ankle and right_ankle:
                left_z_values.append(left_ankle.z)
                right_z_values.append(right_ankle.z)
        
        if not left_z_values or not right_z_values:
            print("⚠️ 방향 감지용 데이터를 추출할 수 없습니다. 기본값 'forward' 사용")
            self.walking_direction = "forward"
            return self.walking_direction
        
        # 평균 Z값으로 방향 판별
        left_avg_z = np.mean(left_z_values)
        right_avg_z = np.mean(right_z_values)
        delta_z = right_avg_z - left_avg_z
        
        if delta_z < 0:
            self.walking_direction = "forward"  # 오른쪽이 앞
            print(f"✅ 보행 방향: Forward (→) [delta_z = {delta_z:.3f}]")
        else:
            self.walking_direction = "backward"  # 왼쪽이 앞
            print(f"✅ 보행 방향: Backward (←) [delta_z = {delta_z:.3f}]")
        
        return self.walking_direction
    
    def calculate_pixel_to_meter_ratio(self, frame_data: List[FrameData]) -> float:
        """
        픽셀-미터 비율 계산 (사용자 키의 28%를 ankle to knee로 사용)
        
        Args:
            frame_data: 프레임 데이터 리스트
            
        Returns:
            float: 픽셀당 미터 비율
        """
        print(f"\n📺 비디오 크기: {self.frame_width} x {self.frame_height}")
        print(f"📏 픽셀-미터 비율 계산 시작... (사용자 키: {self.user_height:.2f}m)")
        
        # DEBUG: 첫 번째 프레임의 관절 좌표 확인
        if frame_data:
            first_frame = frame_data[0]
            left_ankle = first_frame.get_joint('left_ankle')
            right_ankle = first_frame.get_joint('right_ankle')
            if left_ankle and right_ankle:
                print(f"🔍 DEBUG - 엔진에서 받은 첫 번째 프레임 좌표:")
                print(f"   왼발목: x={left_ankle.x:.6f}, y={left_ankle.y:.6f}")
                print(f"   오른발목: x={right_ankle.x:.6f}, y={right_ankle.y:.6f}")
                print(f"   이 값들이 0~1이면 정규화, 큰 값이면 이미 픽셀")
        
        # 키의 28%를 ankle to knee 길이로 가정
        expected_ankle_knee_length = self.user_height * 0.28
        print(f"🦵 예상 ankle-knee 길이: {expected_ankle_knee_length:.3f}m")
        
        ankle_knee_distances_px = []
        
        # 여러 프레임에서 ankle-knee 거리 측정
        sample_frames = min(50, len(frame_data))  # 최대 50프레임 샘플링
        
        for i in range(0, len(frame_data), max(1, len(frame_data) // sample_frames)):
            frame = frame_data[i]
            
            # 왼발과 오른발 모두 측정
            for foot in ['left', 'right']:
                ankle = frame.get_joint(f'{foot}_ankle')
                knee = frame.get_joint(f'{foot}_knee')
                
                if ankle and knee:
                    # 정규화 좌표를 픽셀 좌표로 변환
                    ankle_px = np.array([ankle.x * self.frame_width, ankle.y * self.frame_height])
                    knee_px = np.array([knee.x * self.frame_width, knee.y * self.frame_height])
                    
                    # 픽셀 거리 계산
                    distance_px = np.linalg.norm(ankle_px - knee_px)
                    ankle_knee_distances_px.append(distance_px)
        
        if not ankle_knee_distances_px:
            print("⚠️ ankle-knee 거리를 측정할 수 없습니다. 기본 비율 사용")
            self.pixel_to_meter_ratio = 0.001  # 기본값
            print(f"📏 사용 중인 픽셀-미터 비율: {self.pixel_to_meter_ratio:.6f} m/pixel")
            return self.pixel_to_meter_ratio
        
        # 평균 픽셀 거리 계산
        avg_distance_px = np.mean(ankle_knee_distances_px)
        
        # 픽셀-미터 비율 계산
        self.pixel_to_meter_ratio = expected_ankle_knee_length / avg_distance_px
        
        print(f"📊 평균 ankle-knee 거리: {avg_distance_px:.2f} pixels")
        print(f"✅ 계산된 픽셀-미터 비율: {self.pixel_to_meter_ratio:.6f} m/pixel")
        print(f"📏 사용 중인 픽셀-미터 비율: {self.pixel_to_meter_ratio:.6f} m/pixel")
        
        return self.pixel_to_meter_ratio    
    def calculate_stride_parameters(self, frame_data: List[FrameData], 
                                  support_labels: List[Dict]) -> Dict[str, Any]:
        """
        3개 보행 파라미터 계산: stride_time, stride_length, velocity
        
        Args:
            frame_data: 관절 위치 데이터
            support_labels: 지지 라벨 데이터
            
        Returns:
            계산된 보행 파라미터
        """
        results = {
            'stride_times': [],
            'stride_lengths': [],
            'velocities': [],
            'mean_stride_time': 0.0,
            'mean_stride_length': 0.0,
            'mean_velocity': 0.0,
            'details': []
        }
        
        # 1. 라벨 데이터에서 Heel Strike 검출
        heel_strikes = self.hs_detector.detect_heel_strikes_from_labels(frame_data, support_labels)
        
        if len(heel_strikes) < 2:
            return results
        
        # 2. 발별로 그룹화
        left_hs = [hs for hs in heel_strikes if hs['foot'] == 'left']
        right_hs = [hs for hs in heel_strikes if hs['foot'] == 'right']
        
        print(f"\n🦶 보행 주기 계산 시작... (픽셀-미터 비율: {self.pixel_to_meter_ratio:.6f} m/pixel)")
        
        # 3. LEFT발 stride 계산 (새로운 양발목 간격 방식)
        if len(left_hs) >= 2:
            print(f"👣 LEFT발 stride 계산 시작... ({len(left_hs)}개 HS)")
            
            for i in range(len(left_hs) - 1):
                stride_num = i + 1
                hs1 = left_hs[i]
                hs2 = left_hs[i + 1]
                
                print(f"📏 Stride #{stride_num} (left발):")
                print(f"   🎯 HS1: frame {hs1['frame']} (t={hs1['time']:.3f}s)")
                print(f"   🎯 HS2: frame {hs2['frame']} (t={hs2['time']:.3f}s)")
                
                # Stride Time 계산
                stride_time = hs2['time'] - hs1['time']
                print(f"   ⏱️ Stride Time: {stride_time:.3f}s")
                
                # HS1에서 양발목 위치 가져오기
                left_ankle_hs1 = self._get_ankle_position_at_frame(frame_data, hs1['frame'], 'left')
                right_ankle_hs1 = self._get_ankle_position_at_frame(frame_data, hs1['frame'], 'right')
                
                # HS2에서 양발목 위치 가져오기
                left_ankle_hs2 = self._get_ankle_position_at_frame(frame_data, hs2['frame'], 'left')
                right_ankle_hs2 = self._get_ankle_position_at_frame(frame_data, hs2['frame'], 'right')
                
                if left_ankle_hs1 and right_ankle_hs1 and left_ankle_hs2 and right_ankle_hs2:
                    # 정규화 좌표를 픽셀 좌표로 변환
                    left_px_hs1 = (left_ankle_hs1.x * self.frame_width, left_ankle_hs1.y * self.frame_height)
                    right_px_hs1 = (right_ankle_hs1.x * self.frame_width, right_ankle_hs1.y * self.frame_height)
                    left_px_hs2 = (left_ankle_hs2.x * self.frame_width, left_ankle_hs2.y * self.frame_height)
                    right_px_hs2 = (right_ankle_hs2.x * self.frame_width, right_ankle_hs2.y * self.frame_height)
                    
                    print(f"   📍 HS1 왼발목: ({left_px_hs1[0]:.1f}, {left_px_hs1[1]:.1f})")
                    print(f"   📍 HS1 오른발목: ({right_px_hs1[0]:.1f}, {right_px_hs1[1]:.1f})")
                    
                    # HS1에서 양발목 간 거리 계산
                    hs1_distance_px = np.sqrt((left_px_hs1[0] - right_px_hs1[0])**2 + (left_px_hs1[1] - right_px_hs1[1])**2)
                    hs1_distance_m = hs1_distance_px * self.pixel_to_meter_ratio
                    print(f"   📏 HS1 양발목 간 거리: {hs1_distance_px:.1f} px → {hs1_distance_m:.3f} m")
                    
                    print(f"   📍 HS2 왼발목: ({left_px_hs2[0]:.1f}, {left_px_hs2[1]:.1f})")
                    print(f"   📍 HS2 오른발목: ({right_px_hs2[0]:.1f}, {right_px_hs2[1]:.1f})")
                    
                    # HS2에서 양발목 간 거리 계산
                    hs2_distance_px = np.sqrt((left_px_hs2[0] - right_px_hs2[0])**2 + (left_px_hs2[1] - right_px_hs2[1])**2)
                    hs2_distance_m = hs2_distance_px * self.pixel_to_meter_ratio
                    print(f"   📏 HS2 양발목 간 거리: {hs2_distance_px:.1f} px → {hs2_distance_m:.3f} m")
                    
                    # Stride Length = HS1 거리 + HS2 거리
                    stride_length = hs1_distance_m + hs2_distance_m
                    print(f"   📐 Stride Length: {hs1_distance_m:.3f}m + {hs2_distance_m:.3f}m = {stride_length:.3f} m")
                    
                    # Velocity 계산
                    velocity = stride_length / stride_time if stride_time > 0 else 0
                    print(f"   🚀 Velocity: {stride_length:.3f}m / {stride_time:.3f}s = {velocity:.3f} m/s")
                    
                    # 결과 저장
                    results['stride_times'].append(stride_time)
                    results['stride_lengths'].append(stride_length)
                    results['velocities'].append(velocity)
                    
                    results['details'].append({
                        'foot': 'left',
                        'start_frame': hs1['frame'],
                        'end_frame': hs2['frame'],
                        'start_time': hs1['time'],
                        'end_time': hs2['time'],
                        'stride_time': stride_time,
                        'stride_length': stride_length,
                        'velocity': velocity
                    })
                    print(f"   ✅ Stride #{stride_num} 계산 완료")
                else:
                    print(f"   ❌ Stride #{stride_num}: 발목 위치를 찾을 수 없음")
        
        # 4. RIGHT발 stride 계산 (새로운 양발목 간격 방식)
        if len(right_hs) >= 2:
            print(f"👣 RIGHT발 stride 계산 시작... ({len(right_hs)}개 HS)")
            
            for i in range(len(right_hs) - 1):
                stride_num = i + 1
                hs1 = right_hs[i]
                hs2 = right_hs[i + 1]
                
                print(f"📏 Stride #{stride_num} (right발):")
                print(f"   🎯 HS1: frame {hs1['frame']} (t={hs1['time']:.3f}s)")
                print(f"   🎯 HS2: frame {hs2['frame']} (t={hs2['time']:.3f}s)")
                
                # Stride Time 계산
                stride_time = hs2['time'] - hs1['time']
                print(f"   ⏱️ Stride Time: {stride_time:.3f}s")
                
                # HS1에서 양발목 위치 가져오기
                left_ankle_hs1 = self._get_ankle_position_at_frame(frame_data, hs1['frame'], 'left')
                right_ankle_hs1 = self._get_ankle_position_at_frame(frame_data, hs1['frame'], 'right')
                
                # HS2에서 양발목 위치 가져오기
                left_ankle_hs2 = self._get_ankle_position_at_frame(frame_data, hs2['frame'], 'left')
                right_ankle_hs2 = self._get_ankle_position_at_frame(frame_data, hs2['frame'], 'right')
                
                if left_ankle_hs1 and right_ankle_hs1 and left_ankle_hs2 and right_ankle_hs2:
                    # 정규화 좌표를 픽셀 좌표로 변환
                    left_px_hs1 = (left_ankle_hs1.x * self.frame_width, left_ankle_hs1.y * self.frame_height)
                    right_px_hs1 = (right_ankle_hs1.x * self.frame_width, right_ankle_hs1.y * self.frame_height)
                    left_px_hs2 = (left_ankle_hs2.x * self.frame_width, left_ankle_hs2.y * self.frame_height)
                    right_px_hs2 = (right_ankle_hs2.x * self.frame_width, right_ankle_hs2.y * self.frame_height)
                    
                    print(f"   📍 HS1 왼발목: ({left_px_hs1[0]:.1f}, {left_px_hs1[1]:.1f})")
                    print(f"   📍 HS1 오른발목: ({right_px_hs1[0]:.1f}, {right_px_hs1[1]:.1f})")
                    
                    # HS1에서 양발목 간 거리 계산
                    hs1_distance_px = np.sqrt((left_px_hs1[0] - right_px_hs1[0])**2 + (left_px_hs1[1] - right_px_hs1[1])**2)
                    hs1_distance_m = hs1_distance_px * self.pixel_to_meter_ratio
                    print(f"   📏 HS1 양발목 간 거리: {hs1_distance_px:.1f} px → {hs1_distance_m:.3f} m")
                    
                    print(f"   📍 HS2 왼발목: ({left_px_hs2[0]:.1f}, {left_px_hs2[1]:.1f})")
                    print(f"   📍 HS2 오른발목: ({right_px_hs2[0]:.1f}, {right_px_hs2[1]:.1f})")
                    
                    # HS2에서 양발목 간 거리 계산
                    hs2_distance_px = np.sqrt((left_px_hs2[0] - right_px_hs2[0])**2 + (left_px_hs2[1] - right_px_hs2[1])**2)
                    hs2_distance_m = hs2_distance_px * self.pixel_to_meter_ratio
                    print(f"   📏 HS2 양발목 간 거리: {hs2_distance_px:.1f} px → {hs2_distance_m:.3f} m")
                    
                    # Stride Length = HS1 거리 + HS2 거리
                    stride_length = hs1_distance_m + hs2_distance_m
                    print(f"   📐 Stride Length: {hs1_distance_m:.3f}m + {hs2_distance_m:.3f}m = {stride_length:.3f} m")
                    
                    # Velocity 계산
                    velocity = stride_length / stride_time if stride_time > 0 else 0
                    print(f"   🚀 Velocity: {stride_length:.3f}m / {stride_time:.3f}s = {velocity:.3f} m/s")
                    
                    # 결과 저장
                    results['stride_times'].append(stride_time)
                    results['stride_lengths'].append(stride_length)
                    results['velocities'].append(velocity)
                    
                    results['details'].append({
                        'foot': 'right',
                        'start_frame': hs1['frame'],
                        'end_frame': hs2['frame'],
                        'start_time': hs1['time'],
                        'end_time': hs2['time'],
                        'stride_time': stride_time,
                        'stride_length': stride_length,
                        'velocity': velocity
                    })
                    print(f"   ✅ Stride #{stride_num} 계산 완료")
                else:
                    print(f"   ❌ Stride #{stride_num}: 발목 위치를 찾을 수 없음")
        
        # 5. 평균값 계산
        if results['stride_times']:
            results['mean_stride_time'] = np.mean(results['stride_times'])
            results['mean_stride_length'] = np.mean(results['stride_lengths'])
            results['mean_velocity'] = np.mean(results['velocities'])
            
            print(f"\n📊 전체 보행 주기 요약:")
            print(f"   📈 총 측정된 stride: {len(results['stride_times'])}개")
            print(f"   ⏱️ 평균 Stride Time: {results['mean_stride_time']:.3f}s")
            print(f"   📏 평균 Stride Length: {results['mean_stride_length']:.3f}m")
            print(f"   🚀 평균 Velocity: {results['mean_velocity']:.3f}m/s")
            
            # 개별 값들도 출력
            print(f"\n🔍 개별 측정값들:")
            print(f"   Stride Times: {[f'{t:.3f}' for t in results['stride_times']]}")
            print(f"   Stride Lengths: {[f'{l:.3f}' for l in results['stride_lengths']]}")
            print(f"   Velocities: {[f'{v:.3f}' for v in results['velocities']]}")
        else:
            print(f"\n⚠️ 계산된 stride가 없습니다.")
        
        return results
    
    def _get_ankle_position_at_frame(self, frame_data: List[FrameData], 
                                   frame: int, foot: str) -> Optional[JointPosition]:
        """특정 프레임에서의 발목 위치 반환"""
        for frame_data_item in frame_data:
            if frame_data_item.frame_number == frame:
                return frame_data_item.get_joint(f'{foot}_ankle')
        return None    
    def calculate_gait_parameters(self, joint_data_list: List[Dict], 
                                timestamps: List[float],
                                support_labels: List[Dict] = None,
                                use_phase_method: bool = True) -> Dict[str, Any]:
        """
        전체 보행 파라미터 계산 메인 함수
        
        Args:
            joint_data_list: MediaPipe 관절 데이터 리스트
            timestamps: 각 프레임의 타임스탬프
            support_labels: 지지 라벨 데이터
            use_phase_method: True면 새로운 phase 시퀀스 방법, False면 기존 HS 방법
            
        Returns:
            계산된 보행 파라미터
        """
        # 데이터 전처리
        frame_data = self.process_joint_data(joint_data_list, timestamps)
        
        if not frame_data:
            return {'error': 'No valid frame data'}
        
        # 보행 방향 감지
        self.detect_walking_direction(frame_data)
        
        # 픽셀-미터 비율 계산
        self.calculate_pixel_to_meter_ratio(frame_data)
        
        # 파라미터 계산 (방법 선택)
        if use_phase_method and support_labels:
            print("🔄 Phase 시퀀스 기반 계산 방법 사용")
            stride_results = self.calculate_stride_parameters_by_phases(frame_data, support_labels)
        else:
            print("🔄 기존 Heel Strike 기반 계산 방법 사용")
            stride_results = self.calculate_stride_parameters(frame_data, support_labels or [])
        
        # 결과 구성
        results = {
            'total_frames': len(frame_data),
            'calculation_method': 'phase_sequence' if use_phase_method else 'heel_strike',
            'parameters': {
                'stride_time': {
                    'values': stride_results['stride_times'],
                    'mean': stride_results['mean_stride_time'],
                    'count': len(stride_results['stride_times'])
                },
                'stride_length': {
                    'values': stride_results['stride_lengths'],
                    'mean': stride_results['mean_stride_length'],
                    'count': len(stride_results['stride_lengths'])
                },
                'velocity': {
                    'values': stride_results['velocities'],
                    'mean': stride_results['mean_velocity'],
                    'count': len(stride_results['velocities'])
                }
            },
            'details': stride_results['details']
        }
        
        return results

    def calculate_stride_parameters_by_phases(self, frame_data: List[FrameData], 
                                            support_labels: List[Dict]) -> Dict[str, Any]:
        """
        Phase 시퀀스 기반 stride 계산
        
        Right Stride: double_stance → right_stance → double_stance → left_stance
        Left Stride: double_stance → left_stance → double_stance → right_stance
        
        Args:
            frame_data: 관절 위치 데이터
            support_labels: 지지 라벨 데이터
            
        Returns:
            계산된 보행 파라미터
        """
        results = {
            'stride_times': [],
            'stride_lengths': [],
            'velocities': [],
            'mean_stride_time': 0.0,
            'mean_stride_length': 0.0,
            'mean_velocity': 0.0,
            'details': []
        }
        
        if not support_labels or len(support_labels) < 4:
            print("⚠️ Phase 기반 계산을 위한 충분한 라벨 데이터가 없습니다.")
            return results
        
        print(f"\n🔄 Phase 시퀀스 기반 stride 계산 시작...")
        print(f"📊 총 {len(support_labels)}개의 phase 라벨")
        
        # Phase sequence 매핑
        phase_mapping = {
            'double_support': 'double_stance',
            'single_support_left': 'left_stance', 
            'left_support': 'left_stance',
            'single_support_right': 'right_stance',
            'right_support': 'right_stance'
        }
        
        # 라벨 데이터 정규화
        normalized_labels = []
        for label in support_labels:
            phase = label.get('phase', '')
            normalized_phase = phase_mapping.get(phase, phase)
            if normalized_phase in ['double_stance', 'left_stance', 'right_stance', 'non_gait']:
                normalized_labels.append({
                    'phase': normalized_phase,
                    'start_frame': label.get('start_frame', 0),
                    'end_frame': label.get('end_frame', 0)
                })
        
        print(f"🔍 정규화된 라벨: {len(normalized_labels)}개")
        
        # Right Stride 계산
        right_strides = self._find_stride_cycles(normalized_labels, 
                                               ['double_stance', 'right_stance', 'double_stance', 'left_stance'],
                                               'right')
        
        # Left Stride 계산  
        left_strides = self._find_stride_cycles(normalized_labels,
                                              ['double_stance', 'left_stance', 'double_stance', 'right_stance'], 
                                              'left')
        
        # 각 stride에 대해 길이와 시간 계산
        all_strides = right_strides + left_strides
        
        for stride_info in all_strides:
            stride_result = self._calculate_stride_from_sequence(frame_data, stride_info)
            if stride_result:
                results['stride_times'].append(stride_result['stride_time'])
                results['stride_lengths'].append(stride_result['stride_length'])
                results['velocities'].append(stride_result['velocity'])
                results['details'].append(stride_result)
        
        # 평균값 계산
        if results['stride_times']:
            results['mean_stride_time'] = np.mean(results['stride_times'])
            results['mean_stride_length'] = np.mean(results['stride_lengths'])
            results['mean_velocity'] = np.mean(results['velocities'])
            
            print(f"\n📊 Phase 기반 stride 계산 결과:")
            print(f"   📈 총 측정된 stride: {len(results['stride_times'])}개")
            print(f"   ⏱️ 평균 Stride Time: {results['mean_stride_time']:.3f}s")
            print(f"   📏 평균 Stride Length: {results['mean_stride_length']:.3f}m")
            print(f"   🚀 평균 Velocity: {results['mean_velocity']:.3f}m/s")
        
        return results
    
    def _find_stride_cycles(self, labels: List[Dict], sequence: List[str], stride_type: str) -> List[Dict]:
        """
        특정 phase 시퀀스로 구성된 stride cycle들을 찾기
        
        Args:
            labels: 정규화된 라벨 데이터
            sequence: 찾을 phase 시퀀스 (예: ['double_stance', 'right_stance', 'double_stance', 'left_stance'])
            stride_type: 'left' 또는 'right'
            
        Returns:
            발견된 stride cycle 정보 리스트
        """
        stride_cycles = []
        i = 0
        
        print(f"\n🔍 {stride_type.upper()} stride 시퀀스 탐색: {' → '.join(sequence)}")
        
        while i <= len(labels) - len(sequence):
            # 현재 위치에서 시퀀스가 매치되는지 확인
            match = True
            sequence_labels = []
            
            for j, expected_phase in enumerate(sequence):
                if i + j >= len(labels):
                    match = False
                    break
                    
                current_label = labels[i + j]
                if current_label['phase'] != expected_phase:
                    match = False
                    break
                    
                sequence_labels.append(current_label)
            
            if match:
                # 매치된 시퀀스 정보 저장
                start_frame = sequence_labels[0]['start_frame']
                end_frame = sequence_labels[-1]['end_frame']
                
                stride_info = {
                    'type': stride_type,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'sequence': sequence_labels,
                    'heel_strike_frame': None,  # heel strike 시점 계산
                    'toe_off_frame': None       # toe off 시점 계산
                }
                
                # Heel Strike 시점 계산
                if stride_type == 'right':
                    # right stride: left_stance 시작 = 오른발 heel strike
                    for label in sequence_labels:
                        if label['phase'] == 'left_stance':
                            stride_info['heel_strike_frame'] = label['start_frame']
                            break
                elif stride_type == 'left':
                    # left stride: right_stance 시작 = 왼발 heel strike  
                    for label in sequence_labels:
                        if label['phase'] == 'right_stance':
                            stride_info['heel_strike_frame'] = label['start_frame']
                            break
                
                stride_cycles.append(stride_info)
                print(f"   ✅ {stride_type} stride 발견: frame {start_frame}-{end_frame}, HS@{stride_info['heel_strike_frame']}")
                
                # 다음 탐색은 현재 시퀀스의 두 번째 요소부터 시작 (겹치는 부분 허용)
                i += 1
            else:
                i += 1
        
        print(f"   📊 총 {len(stride_cycles)}개의 {stride_type} stride 발견")
        return stride_cycles
    
    def _calculate_stride_from_sequence(self, frame_data: List[FrameData], stride_info: Dict) -> Optional[Dict]:
        """
        Phase 시퀀스로부터 stride 길이, 시간, 속도 계산
        
        Args:
            frame_data: 관절 데이터
            stride_info: stride 시퀀스 정보
            
        Returns:
            계산된 stride 정보
        """
        stride_type = stride_info['type']
        sequence_labels = stride_info['sequence']
        
        print(f"\n📏 {stride_type.upper()} stride 계산 중...")
        
        # 시간 계산
        start_time = sequence_labels[0]['start_frame'] / self.fps
        end_time = sequence_labels[-1]['end_frame'] / self.fps
        stride_time = end_time - start_time
        
        print(f"   ⏱️ 시간: {start_time:.3f}s ~ {end_time:.3f}s = {stride_time:.3f}s")
        
        # 거리 계산을 위한 두 지점 찾기
        distance1_frame = None
        distance2_frame = None
        
        if stride_type == 'right':
            # Right stride: left_stance 시작 + right_stance 종료에서 양발목 거리
            for label in sequence_labels:
                if label['phase'] == 'left_stance' and distance1_frame is None:
                    distance1_frame = label['start_frame']
                elif label['phase'] == 'right_stance':
                    distance2_frame = label['end_frame']
        elif stride_type == 'left':
            # Left stride: right_stance 시작 + left_stance 종료에서 양발목 거리
            for label in sequence_labels:
                if label['phase'] == 'right_stance' and distance1_frame is None:
                    distance1_frame = label['start_frame']
                elif label['phase'] == 'left_stance':
                    distance2_frame = label['end_frame']
        
        if distance1_frame is None or distance2_frame is None:
            print(f"   ❌ 거리 계산 프레임을 찾을 수 없음")
            return None
        
        print(f"   📍 거리 계산 프레임: {distance1_frame}, {distance2_frame}")
        
        # 각 프레임에서 양발목 거리 계산
        distance1 = self._calculate_ankle_distance_at_frame(frame_data, distance1_frame)
        distance2 = self._calculate_ankle_distance_at_frame(frame_data, distance2_frame)
        
        if distance1 is None or distance2 is None:
            print(f"   ❌ 발목 거리 계산 실패")
            return None
        
        # Stride Length = distance1 + distance2
        stride_length = distance1 + distance2
        
        # Velocity 계산
        velocity = stride_length / stride_time if stride_time > 0 else 0
        
        print(f"   📐 거리1: {distance1:.3f}m, 거리2: {distance2:.3f}m")
        print(f"   📏 Stride Length: {stride_length:.3f}m")
        print(f"   🚀 Velocity: {velocity:.3f}m/s")
        
        return {
            'foot': stride_type,
            'start_frame': sequence_labels[0]['start_frame'],
            'end_frame': sequence_labels[-1]['end_frame'],
            'start_time': start_time,
            'end_time': end_time,
            'stride_time': stride_time,
            'stride_length': stride_length,
            'velocity': velocity,
            'distance1_frame': distance1_frame,
            'distance2_frame': distance2_frame,
            'distance1': distance1,
            'distance2': distance2,
            'sequence': [label['phase'] for label in sequence_labels]
        }
    
    def _calculate_ankle_distance_at_frame(self, frame_data: List[FrameData], frame_num: int) -> Optional[float]:
        """특정 프레임에서 양발목 간 거리 계산 (미터 단위)"""
        # 해당 프레임 찾기
        target_frame = None
        for frame in frame_data:
            if frame.frame_number == frame_num:
                target_frame = frame
                break
        
        if not target_frame:
            return None
        
        # 양발목 위치 가져오기
        left_ankle = target_frame.get_joint('left_ankle')
        right_ankle = target_frame.get_joint('right_ankle')
        
        if not left_ankle or not right_ankle:
            return None
        
        # 정규화 좌표를 픽셀 좌표로 변환
        left_px = (left_ankle.x * self.frame_width, left_ankle.y * self.frame_height)
        right_px = (right_ankle.x * self.frame_width, right_ankle.y * self.frame_height)
        
        # 픽셀 거리 계산
        distance_px = np.sqrt((left_px[0] - right_px[0])**2 + (left_px[1] - right_px[1])**2)
        
        # 미터 단위로 변환
        distance_m = distance_px * self.pixel_to_meter_ratio
        
        return distance_m


# 편의 함수들
def create_gait_engine(fps: float = 30.0) -> GaitCalculationEngine:
    """보행 계산 엔진 생성"""
    return GaitCalculationEngine(fps)


def calculate_gait_parameters(joint_data: List[Dict], 
                            timestamps: List[float],
                            fps: float = 30.0,
                            support_labels: List[Dict] = None,
                            use_phase_method: bool = True) -> Dict[str, Any]:
    """
    원샷 보행 파라미터 계산 함수
    
    Args:
        joint_data: MediaPipe 관절 데이터
        timestamps: 프레임 타임스탬프
        fps: 프레임 레이트
        support_labels: 지지 라벨
        use_phase_method: True면 새로운 phase 시퀀스 방법, False면 기존 HS 방법
        
    Returns:
        계산된 보행 파라미터 (stride_time, stride_length, velocity)
    """
    engine = create_gait_engine(fps)
    return engine.calculate_gait_parameters(joint_data, timestamps, support_labels, use_phase_method)


def test_phase_based_calculation():
    """
    Phase 기반 stride 계산 테스트 함수
    """
    print("🧪 Phase 기반 stride 계산 테스트 시작...")
    
    # 샘플 support labels (4단계 완전한 stride cycle)
    sample_labels = [
        {'phase': 'double_support', 'start_frame': 0, 'end_frame': 10},
        {'phase': 'single_support_right', 'start_frame': 11, 'end_frame': 30},
        {'phase': 'double_support', 'start_frame': 31, 'end_frame': 40},
        {'phase': 'single_support_left', 'start_frame': 41, 'end_frame': 60},
        {'phase': 'double_support', 'start_frame': 61, 'end_frame': 70},
        {'phase': 'single_support_right', 'start_frame': 71, 'end_frame': 90},
        {'phase': 'double_support', 'start_frame': 91, 'end_frame': 100},
        {'phase': 'single_support_left', 'start_frame': 101, 'end_frame': 120}
    ]
    
    # 샘플 관절 데이터 (간단한 발목 위치)
    sample_joint_data = []
    sample_timestamps = []
    
    for i in range(121):  # 0~120 프레임
        # 간단한 보행 시뮬레이션: 발목이 좌우로 움직임
        left_x = 0.4 + 0.1 * np.sin(i * 0.1)  # 왼발목 X 좌표
        right_x = 0.6 - 0.1 * np.sin(i * 0.1)  # 오른발목 X 좌표
        
        joint_data = {
            'left_ankle': {'x': left_x, 'y': 0.8, 'z': 0.5},
            'right_ankle': {'x': right_x, 'y': 0.8, 'z': 0.5},
            'left_knee': {'x': left_x, 'y': 0.6, 'z': 0.5},
            'right_knee': {'x': right_x, 'y': 0.6, 'z': 0.5}
        }
        
        sample_joint_data.append(joint_data)
        sample_timestamps.append(i / 30.0)  # 30fps 가정
    
    # 테스트 실행
    try:
        results = calculate_gait_parameters(
            joint_data=sample_joint_data,
            timestamps=sample_timestamps, 
            fps=30.0,
            support_labels=sample_labels,
            use_phase_method=True
        )
        
        print("✅ 테스트 성공!")
        print(f"📊 계산 방법: {results['calculation_method']}")
        print(f"📈 측정된 stride 수: {results['parameters']['stride_time']['count']}")
        
        if results['parameters']['stride_time']['count'] > 0:
            print(f"⏱️ 평균 Stride Time: {results['parameters']['stride_time']['mean']:.3f}s")
            print(f"📏 평균 Stride Length: {results['parameters']['stride_length']['mean']:.3f}m")
            print(f"🚀 평균 Velocity: {results['parameters']['velocity']['mean']:.3f}m/s")
        
        return results
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 테스트 실행
    test_phase_based_calculation()