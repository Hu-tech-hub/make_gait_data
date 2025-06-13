#!/usr/bin/env python3
"""
Stride Inference Pipeline
라벨링된 IMU 데이터에서 stride cycle을 추출하고 학습된 TCN 모델로 보폭 예측

Input:
- support_labels.csv: 지지 라벨 데이터 
- walking_data.csv: IMU 센서 데이터
- trained model: models_2/best_fold_5.keras

Process:
1. 라벨 데이터에서 stride cycle 추출
2. IMU 센서 데이터에서 해당 구간 추출
3. 보조 특성 계산 (stride_time, height, foot_id)
4. 정규화 적용
5. 모델 예측
6. 결과 출력

Author: Assistant
Date: 2025-01-12
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import re
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TCN 커스텀 레이어들을 등록하기 위해 import (데코레이터 방식)
try:
    import tcn_model  # 데코레이터로 자동 등록됨
    logger.info("✅ TCN 모델 클래스들이 등록되었습니다.")
except ImportError as e:
    logger.warning(f"tcn_model을 import할 수 없습니다: {e}")
    raise ImportError("TCN 모델을 사용하려면 tcn_model.py가 필요합니다.")


class StrideInferencePipeline:
    """Stride 길이 예측 파이프라인"""
    
    def __init__(self, model_path="models/best_fold_5.keras", 
                 metadata_dir="metadata"):
        """
        초기화
        
        Args:
            model_path: 학습된 모델 경로
            metadata_dir: 정규화 통계 등 메타데이터 디렉토리
        """
        # 절대 경로로 변환
        base_dir = Path(__file__).parent
        self.model_path = base_dir / model_path if not Path(model_path).is_absolute() else Path(model_path)
        self.metadata_dir = base_dir / metadata_dir if not Path(metadata_dir).is_absolute() else Path(metadata_dir)
        self.model = None
        self.normalization_stats = None
        self.auxiliary_stats = None
        self.fps = 30.0  # 기본 FPS
        
        # Subject별 키 정보 (cm)
        self.subject_heights = {
            'SA01': 175,
            'SA02': 170, 
            'SA03': 180,
            'SA04': 160,
            'SA05': 160
        }
        
        # Foot mapping
        self.foot_mapping = {'left': 0, 'right': 1}
        
    def load_model_and_stats(self):
        """모델과 정규화 통계 로드"""
        try:
            # 모델 로드 (데코레이터로 등록된 클래스들은 자동으로 인식됨)
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ 모델 로드 성공: {self.model_path}")
            logger.info("   (데코레이터로 등록된 커스텀 레이어 사용)")
            
            # 정규화 통계 로드
            norm_file = self.metadata_dir / "global_norm_enhanced.npz"
            if norm_file.exists():
                norm_data = np.load(norm_file)
                
                if 'sequence_mean' in norm_data.files:
                    self.normalization_stats = {
                        'mean': norm_data['sequence_mean'],
                        'std': norm_data['sequence_std']
                    }
                    self.auxiliary_stats = {
                        'height_mean': norm_data['height_mean'],
                        'height_std': norm_data['height_std'],
                        'stride_time_mean': norm_data['stride_time_mean'],
                        'stride_time_std': norm_data['stride_time_std']
                    }
                else:
                    self.normalization_stats = {
                        'mean': norm_data['mean'],
                        'std': norm_data['std']
                    }
                    self.auxiliary_stats = {
                        'height_mean': norm_data.get('aux_height_mean', 170.0),
                        'height_std': norm_data.get('aux_height_std', 10.0),
                        'stride_time_mean': norm_data.get('aux_stride_time_mean', 1.0),
                        'stride_time_std': norm_data.get('aux_stride_time_std', 0.3)
                    }
                logger.info("✅ 정규화 통계 로드 성공")
            else:
                # 기본 global_norm.npz 시도
                norm_file = self.metadata_dir / "global_norm.npz"
                if norm_file.exists():
                    norm_data = np.load(norm_file)
                    
                    self.normalization_stats = {
                        'mean': norm_data['sequence_mean'],
                        'std': norm_data['sequence_std']
                    }
                    self.auxiliary_stats = {
                        'height_mean': 170.0,
                        'height_std': 10.0,
                        'stride_time_mean': 1.0,
                        'stride_time_std': 0.3
                    }
                    logger.info("✅ 기본 정규화 통계 로드 성공")
                else:
                    raise FileNotFoundError(f"정규화 통계 파일을 찾을 수 없습니다: {self.metadata_dir}")
            
        except Exception as e:
            logger.error(f"❌ 모델/통계 로드 실패: {e}")
            raise
    
    def load_support_labels(self, labels_file: str) -> List[Dict]:
        """지지 라벨 CSV 로드"""
        try:
            df = pd.read_csv(labels_file)
            labels = []
            
            for _, row in df.iterrows():
                label = {
                    'start_frame': int(row['start_frame']),
                    'end_frame': int(row['end_frame']),
                    'phase': row['phase'].strip()
                }
                labels.append(label)
            
            logger.info(f"✅ 지지 라벨 로드: {len(labels)}개 라벨")
            return labels
            
        except Exception as e:
            logger.error(f"❌ 지지 라벨 로드 실패: {e}")
            raise
    
    def load_walking_data(self, walking_file: str) -> pd.DataFrame:
        """IMU 센서 데이터 로드"""
        try:
            df = pd.read_csv(walking_file)
            
            # 필요한 컬럼 확인
            required_cols = ['frame_number', 'sync_timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"누락된 컬럼: {missing_cols}")
            
            logger.info(f"✅ IMU 데이터 로드: {len(df)}개 프레임")
            return df
            
        except Exception as e:
            logger.error(f"❌ IMU 데이터 로드 실패: {e}")
            raise
    
    def find_stride_cycles(self, labels: List[Dict], stride_type: str) -> List[Dict]:
        """
        특정 stride type의 cycle들을 찾기
        gait_calculation_engine.py의 _find_stride_cycles 참고
        """
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
        for label in labels:
            phase = label.get('phase', '')
            normalized_phase = phase_mapping.get(phase, phase)
            if normalized_phase in ['double_stance', 'left_stance', 'right_stance', 'non_gait']:
                normalized_labels.append({
                    'phase': normalized_phase,
                    'start_frame': label.get('start_frame', 0),
                    'end_frame': label.get('end_frame', 0)
                })
        
        # Stride sequence 정의
        if stride_type == 'right':
            sequence = ['double_stance', 'right_stance', 'double_stance', 'left_stance']
        elif stride_type == 'left':
            sequence = ['double_stance', 'left_stance', 'double_stance', 'right_stance']
        else:
            raise ValueError(f"Invalid stride_type: {stride_type}")
        
        stride_cycles = []
        i = 0
        
        while i <= len(normalized_labels) - len(sequence):
            # 현재 위치에서 시퀀스가 매치되는지 확인
            match = True
            sequence_labels = []
            
            for j, expected_phase in enumerate(sequence):
                if i + j >= len(normalized_labels):
                    match = False
                    break
                    
                current_label = normalized_labels[i + j]
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
                    'sequence': sequence_labels
                }
                
                stride_cycles.append(stride_info)
                i += 1
            else:
                i += 1
        
        logger.info(f"✅ {stride_type} stride 발견: {len(stride_cycles)}개")
        return stride_cycles
    
    def extract_cycle_sequence(self, walking_df: pd.DataFrame, 
                             start_frame: int, end_frame: int) -> List[List[float]]:
        """stride cycle 구간의 센서 데이터 추출"""
        try:
            # 프레임 범위 필터링
            cycle_data = walking_df[
                (walking_df['frame_number'] >= start_frame) & 
                (walking_df['frame_number'] <= end_frame)
            ].copy()
            
            if cycle_data.empty:
                logger.warning(f"프레임 범위 {start_frame}-{end_frame}에 데이터 없음")
                return []
            
            # 센서 데이터 추출
            sequence = []
            for idx, row in cycle_data.iterrows():
                try:
                    frame_data = [
                        float(row['accel_x']),
                        float(row['accel_y']),
                        float(row['accel_z']),
                        float(row['gyro_x']),
                        float(row['gyro_y']),
                        float(row['gyro_z'])
                    ]
                    
                    # 유효성 검사
                    if len(frame_data) != 6 or any(not np.isfinite(val) for val in frame_data):
                        continue
                        
                    sequence.append(frame_data)
                    
                except Exception as e:
                    logger.error(f"프레임 {row['frame_number']} 처리 오류: {e}")
                    continue
            
            if len(sequence) == 0:
                logger.error(f"유효한 프레임 없음: {start_frame}-{end_frame}")
                return []
            
            return sequence
            
        except Exception as e:
            logger.error(f"Cycle sequence 추출 실패: {e}")
            return []
    
    def calculate_stride_time(self, start_frame: int, end_frame: int) -> float:
        """Stride time 계산"""
        return (end_frame - start_frame) / self.fps
    
    def extract_subject_id(self, filename: str) -> str:
        """파일명에서 subject ID 추출 (S01 -> SA01)"""
        pattern = r'(S\d+)'
        match = re.search(pattern, filename)
        if match:
            subject_num = match.group(1)
            return 'SA' + subject_num[1:]  # S01 -> SA01
        return 'SA01'  # 기본값
    
    def get_subject_height(self, subject_id: str) -> int:
        """Subject의 키 반환"""
        return self.subject_heights.get(subject_id, 170)  # 기본값 170cm
    
    def normalize_sequence(self, sequence: List[List[float]]) -> np.ndarray:
        """시퀀스 정규화"""
        seq_array = np.array(sequence)
        
        if seq_array.shape != (len(sequence), 6):
            logger.warning(f"시퀀스 형태 불일치: {seq_array.shape}")
        
        # 센서 데이터 정규화
        try:
            normalized = (seq_array - self.normalization_stats['mean']) / self.normalization_stats['std']
        except Exception as e:
            logger.error(f"정규화 실패: {e}")
            raise
        
        return normalized.astype(np.float32)
    
    def normalize_auxiliary_features(self, height: float, stride_time: float, foot_id: int) -> np.ndarray:
        """보조 특성 정규화"""
        # Height 정규화
        norm_height = (height - self.auxiliary_stats['height_mean']) / self.auxiliary_stats['height_std']
        
        # Stride time 정규화
        norm_stride_time = (stride_time - self.auxiliary_stats['stride_time_mean']) / self.auxiliary_stats['stride_time_std']
        
        # Foot ID는 one-hot encoding된 상태로 입력 (0 or 1)
        foot_binary = float(foot_id)
        
        return np.array([norm_height, norm_stride_time, foot_binary], dtype=np.float32)
    
    def prepare_model_input(self, cycles_data: List[Dict]) -> Tuple[tf.Tensor, tf.Tensor]:
        """모델 입력 형태로 데이터 준비"""
        sequences = []
        auxiliary_features = []
        
        for cycle in cycles_data:
            # 시퀀스 정규화
            normalized_seq = self.normalize_sequence(cycle['sequence'])
            sequences.append(normalized_seq)
            
            # 보조 특성 정규화
            aux_features = self.normalize_auxiliary_features(
                cycle['height'], 
                cycle['stride_time'],
                cycle['foot_id']
            )
            auxiliary_features.append(aux_features)
        
        # 모델이 기대하는 고정 길이 사용 (학습 시와 동일)
        MODEL_EXPECTED_LENGTH = 67
        
        # 패딩된 시퀀스 생성
        padded_sequences = []
        for seq in sequences:
            if len(seq) < MODEL_EXPECTED_LENGTH:
                # 0으로 패딩
                padding_needed = MODEL_EXPECTED_LENGTH - len(seq)
                padding = np.zeros((padding_needed, 6), dtype=np.float32)
                seq_array = np.array(seq, dtype=np.float32)
                padded_seq = np.concatenate([seq_array, padding], axis=0)
            else:
                padded_seq = np.array(seq, dtype=np.float32)[:MODEL_EXPECTED_LENGTH]
                
            padded_sequences.append(padded_seq)
        
        # 텐서 생성
        padded_sequences_array = np.array(padded_sequences, dtype=np.float32)
        sequence_tensor = tf.constant(padded_sequences_array, dtype=tf.float32)
        auxiliary_tensor = tf.constant(auxiliary_features, dtype=tf.float32)
        
        return sequence_tensor, auxiliary_tensor
    
    def predict_stride_lengths(self, sequences: tf.Tensor, 
                             auxiliary: tf.Tensor) -> np.ndarray:
        """모델을 사용하여 stride length 예측"""
        try:
            predictions = self.model([sequences, auxiliary])
            return predictions.numpy().flatten()
            
        except Exception as e:
            logger.error(f"❌ 모델 예측 실패: {e}")
            raise
    
    def process_single_session(self, labels_file: str, walking_file: str) -> Dict:
        """단일 세션 처리"""
        logger.info(f"세션 처리 시작: {Path(labels_file).stem}")
        
        try:
            # 1. 데이터 로드
            support_labels = self.load_support_labels(labels_file)
            walking_df = self.load_walking_data(walking_file)
            
            # 2. Subject 정보 추출
            subject_id = self.extract_subject_id(Path(labels_file).stem)
            height = self.get_subject_height(subject_id)
            
            logger.info(f"Subject: {subject_id}, Height: {height}cm")
            
            # 3. Stride cycles 추출
            left_cycles = self.find_stride_cycles(support_labels, 'left')
            right_cycles = self.find_stride_cycles(support_labels, 'right')
            
            all_cycles_data = []
            
            # 4. Left cycles 처리
            for i, cycle in enumerate(left_cycles):
                sequence = self.extract_cycle_sequence(
                    walking_df, cycle['start_frame'], cycle['end_frame']
                )
                
                if len(sequence) < 15:  # 최소 길이 체크
                    logger.warning(f"Left cycle {i+1}: 시퀀스 길이 부족 ({len(sequence)})")
                    continue
                
                stride_time = self.calculate_stride_time(
                    cycle['start_frame'], cycle['end_frame']
                )
                
                cycle_data = {
                    'cycle_number': len(all_cycles_data) + 1,
                    'foot': 'left',
                    'foot_id': self.foot_mapping['left'],
                    'start_frame': cycle['start_frame'],
                    'end_frame': cycle['end_frame'],
                    'sequence': sequence,
                    'height': height,
                    'stride_time': stride_time,
                    'sequence_length': len(sequence)
                }
                all_cycles_data.append(cycle_data)
            
            # 5. Right cycles 처리
            for i, cycle in enumerate(right_cycles):
                sequence = self.extract_cycle_sequence(
                    walking_df, cycle['start_frame'], cycle['end_frame']
                )
                
                if len(sequence) < 15:  # 최소 길이 체크
                    logger.warning(f"Right cycle {i+1}: 시퀀스 길이 부족 ({len(sequence)})")
                    continue
                
                stride_time = self.calculate_stride_time(
                    cycle['start_frame'], cycle['end_frame']
                )
                
                cycle_data = {
                    'cycle_number': len(all_cycles_data) + 1,
                    'foot': 'right',
                    'foot_id': self.foot_mapping['right'],
                    'start_frame': cycle['start_frame'],
                    'end_frame': cycle['end_frame'],
                    'sequence': sequence,
                    'height': height,
                    'stride_time': stride_time,
                    'sequence_length': len(sequence)
                }
                all_cycles_data.append(cycle_data)
            
            if not all_cycles_data:
                logger.warning("⚠️ 유효한 stride cycle이 없습니다.")
                return {
                    'subject_id': subject_id,
                    'total_cycles': 0,
                    'predictions': [],
                    'error': 'No valid cycles found'
                }
            
            # 6. 모델 입력 준비 및 예측
            sequences, auxiliary = self.prepare_model_input(all_cycles_data)
            predictions = self.predict_stride_lengths(sequences, auxiliary)
            
            # 7. 결과 정리
            results = []
            for i, (cycle_data, pred_length) in enumerate(zip(all_cycles_data, predictions)):
                result = {
                    'cycle_number': cycle_data['cycle_number'],
                    'foot': cycle_data['foot'],
                    'start_frame': cycle_data['start_frame'],
                    'end_frame': cycle_data['end_frame'],
                    'sequence_length': cycle_data['sequence_length'],
                    'stride_time': cycle_data['stride_time'],
                    'predicted_stride_length': float(pred_length),
                    'predicted_velocity': float(pred_length / cycle_data['stride_time']) if cycle_data['stride_time'] > 0 else 0.0
                }
                results.append(result)
            
            # 8. 통계 계산
            stride_lengths = [r['predicted_stride_length'] for r in results]
            velocities = [r['predicted_velocity'] for r in results]
            
            summary = {
                'subject_id': subject_id,
                'height': height,
                'total_cycles': len(results),
                'left_cycles': len([r for r in results if r['foot'] == 'left']),
                'right_cycles': len([r for r in results if r['foot'] == 'right']),
                'mean_stride_length': np.mean(stride_lengths),
                'std_stride_length': np.std(stride_lengths),
                'mean_velocity': np.mean(velocities),
                'std_velocity': np.std(velocities),
                'predictions': results
            }
            
            logger.info(f"예측 완료: {len(results)}개 cycle (평균 stride: {summary['mean_stride_length']:.3f}m)")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 세션 처리 실패: {e}")
            return {
                'subject_id': subject_id if 'subject_id' in locals() else 'Unknown',
                'total_cycles': 0,
                'predictions': [],
                'error': str(e)
            }
    
    def save_results(self, results: Dict, output_file: str):
        """결과를 JSON 파일로 저장"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 결과 저장: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ 결과 저장 실패: {e}")
    
    def run_inference(self, labels_file: str, walking_file: str, 
                     output_file: Optional[str] = None) -> Dict:
        """전체 추론 파이프라인 실행"""
        logger.info(f"🚀 Stride Inference Pipeline 시작")
        
        try:
            # 1. 모델과 통계 로드
            if self.model is None:
                self.load_model_and_stats()
            
            # 2. 세션 처리
            results = self.process_single_session(labels_file, walking_file)
            
            # 3. 결과 저장 (옵션)
            if output_file:
                self.save_results(results, output_file)
            
            logger.info(f"🎉 추론 완료!")
            return results
            
        except Exception as e:
            logger.error(f"❌ 추론 파이프라인 실패: {e}")
            raise


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stride Inference Pipeline')
    parser.add_argument('--labels', required=True, help='Support labels CSV 파일')
    parser.add_argument('--walking', required=True, help='Walking data CSV 파일')
    parser.add_argument('--model', default='models_2/best_fold_5.keras', help='모델 파일 경로')
    parser.add_argument('--output', help='결과 JSON 파일 (옵션)')
    parser.add_argument('--metadata_dir', default='metadata', help='메타데이터 디렉토리')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    pipeline = StrideInferencePipeline(args.model, args.metadata_dir)
    results = pipeline.run_inference(args.labels, args.walking, args.output)
    
    # 간단한 결과 출력
    if 'error' not in results:
        print(f"\n{'='*50}")
        print(f"STRIDE INFERENCE RESULTS")
        print(f"{'='*50}")
        print(f"Subject: {results['subject_id']} (Height: {results['height']}cm)")
        print(f"Total Cycles: {results['total_cycles']}")
        print(f"Mean Stride Length: {results['mean_stride_length']:.3f} ± {results['std_stride_length']:.3f} m")
        print(f"Mean Velocity: {results['mean_velocity']:.3f} ± {results['std_velocity']:.3f} m/s")
        print(f"{'='*50}")


if __name__ == "__main__":
    # 예시 실행
    if len(sys.argv) == 1:
        # 기본 예시 데이터로 테스트
        base_dir = Path(__file__).parent
        labels_file = str(base_dir / "support_label_data/SA01/S01T01R01_support_labels.csv")
        walking_file = str(base_dir / "walking_data/SA01/S01T01R01.csv")
        
        pipeline = StrideInferencePipeline()
        results = pipeline.run_inference(labels_file, walking_file, "inference_results.json")
    else:
        main()