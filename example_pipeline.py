#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example_pipeline.py - 통합 보행 분석 시스템 예제 파이프라인

이 스크립트는 전체 보행 분석 파이프라인을 단계별로 실행하는 예제입니다.

사용법:
    python example_pipeline.py --video_path path/to/video.mp4 --imu_path path/to/imu_data.csv
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

# 프로젝트 모듈 임포트
from gait_class import GaitAnalyzer, GaitEvent
from gait_metrics_calculator import GaitMetricsCalculator, GaitCycle
from time_series_model import GaitMetricsPredictor, IMUFeatureExtractor
from data_processing_utils import GaitDatasetBuilder, ModelEvaluator, ExperimentLogger


class GaitAnalysisPipeline:
    """통합 보행 분석 파이프라인 클래스"""
    
    def __init__(self, config: Dict):
        """
        초기화
        
        Args:
            config (Dict): 파이프라인 설정
        """
        self.config = config
        self.results = {}
        
        # 컴포넌트 초기화
        self.gait_analyzer = None
        self.metrics_calculator = GaitMetricsCalculator()
        self.model_predictor = None
        self.dataset_builder = GaitDatasetBuilder(
            window_size=config.get('window_size', 90),
            overlap=config.get('overlap', 0.5)
        )
        self.evaluator = ModelEvaluator()
        self.logger = ExperimentLogger(config.get('log_dir', './experiments'))
        
    def run_full_pipeline(self, video_path: str, imu_path: str, 
                         output_dir: str = './results') -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            video_path (str): 비디오 파일 경로
            imu_path (str): IMU 데이터 파일 경로
            output_dir (str): 결과 출력 디렉토리
            
        Returns:
            Dict: 분석 결과
        """
        print("=" * 60)
        print("통합 보행 분석 파이프라인 시작")
        print("=" * 60)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: 데이터 로드 및 검증
            print("\nStep 1: 데이터 로드 및 검증")
            imu_data = self._load_and_validate_data(video_path, imu_path)
            
            # Step 2: 보행 이벤트 검출
            print("\nStep 2: 보행 이벤트 검출")
            gait_events = self._detect_gait_events(video_path, imu_path)
            
            # Step 3: 보행 지표 계산 (라벨 데이터 기반)
            print("\nStep 3: 보행 지표 계산")
            # 라벨 데이터 로드 (예시 - 실제로는 데이터에 따라 조정)
            support_labels = self._load_support_labels(video_path)
            gait_cycles = self._calculate_gait_metrics_with_labels(video_path, support_labels, output_dir)
            
            # Step 4: 학습 데이터셋 생성
            print("\nStep 4: 학습 데이터셋 생성")
            X, y = self._create_training_dataset(imu_data, gait_cycles, output_dir)
            
            # Step 5: 모델 학습
            print("\nStep 5: 모델 학습")
            model_metrics = self._train_model(X, y, output_dir)
            
            # Step 6: 결과 정리 및 저장
            print("\nStep 6: 결과 정리 및 저장")
            final_results = self._finalize_results(output_dir, model_metrics)
            
            print("\n" + "=" * 60)
            print("파이프라인 완료!")
            print(f"결과 저장 위치: {output_dir}")
            print("=" * 60)
            
            return final_results
            
        except Exception as e:
            print(f"\n오류 발생: {e}")
            raise
    
    def _load_and_validate_data(self, video_path: str, imu_path: str) -> pd.DataFrame:
        """데이터 로드 및 검증"""
        # 파일 존재 여부 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        if not os.path.exists(imu_path):
            raise FileNotFoundError(f"IMU 파일을 찾을 수 없습니다: {imu_path}")
        
        # IMU 데이터 로드
        imu_data = pd.read_csv(imu_path)
        print(f"✓ IMU 데이터 로드 완료: {len(imu_data)} 샘플")
        
        # 비디오 정보 확인
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        print(f"✓ 비디오 정보: {frame_count} 프레임, {fps} FPS, {duration:.2f}초")
        
        # 동기화 검증
        imu_duration = imu_data['sync_timestamp'].max()
        time_diff = abs(duration - imu_duration)
        
        if time_diff > 1.0:  # 1초 이상 차이
            print(f"⚠ 경고: 시간 동기화 오차가 큽니다 ({time_diff:.2f}초)")
        else:
            print(f"✓ 동기화 검증 완료: 시간 차이 {time_diff:.2f}초")
        
        return imu_data
    
    def _detect_gait_events(self, video_path: str, imu_path: str) -> List[GaitEvent]:
        """보행 이벤트 검출"""
        self.gait_analyzer = GaitAnalyzer(video_path, imu_path)
        
        print("  - 보행 방향 감지 중...")
        direction = self.gait_analyzer.detect_walking_direction()
        print(f"  ✓ 보행 방향: {direction}")
        
        print("  - 관절 좌표 추출 중...")
        landmarks_data = self.gait_analyzer.extract_pose_landmarks()
        print(f"  ✓ 관절 좌표 추출 완료: {len(landmarks_data)} 프레임")
        
        print("  - 보행 이벤트 검출 중...")
        gait_events = self.gait_analyzer.detect_gait_events()
        print(f"  ✓ 보행 이벤트 검출 완료: {len(gait_events)} 이벤트")
        
        # 이벤트 통계
        hs_events = [e for e in gait_events if e.event_type == "HS"]
        to_events = [e for e in gait_events if e.event_type == "TO"]
        
        print(f"    - Heel Strike: {len(hs_events)} 개")
        print(f"    - Toe Off: {len(to_events)} 개")
        
        return gait_events
    
    def _calculate_gait_metrics_with_labels(self, video_path: str, support_labels: List[Dict], 
                                          output_dir: str) -> List[GaitCycle]:
        """라벨 데이터와 MediaPipe를 사용한 보행 지표 계산"""
        print("  - MediaPipe 관절 추정 중...")
        
        # 픽셀-미터 비율 설정 (실제 환경에 맞게 조정 필요)
        self.metrics_calculator.pixel_to_meter_ratio = self.config.get('pixel_to_meter_ratio', 0.001)
        
        # 1단계: 관절 좌표 추출
        joint_coords = self.metrics_calculator.extract_joint_coordinates(video_path)
        print(f"  ✓ 관절 좌표 추출 완료: {len(joint_coords)} 프레임")
        
        # 2단계: 라벨 기반 보행 지표 계산
        print("  - 라벨 기반 보행 지표 계산 중...")
        gait_cycles = self.metrics_calculator.calculate_gait_metrics_from_labels(
            video_path, joint_coords, support_labels
        )
        
        print(f"  ✓ 보행 지표 계산 완료: {len(gait_cycles)} 주기")
        
        # 지표 통계 출력
        if gait_cycles:
            data = [cycle.to_dict() for cycle in gait_cycles]
            df = pd.DataFrame(data)
            
            print(f"    - 평균 보폭: {df['stride_length'].mean():.3f} ± {df['stride_length'].std():.3f} m")
            print(f"    - 평균 속도: {df['velocity'].mean():.3f} ± {df['velocity'].std():.3f} m/s")
            print(f"    - 평균 보행률: {df['cadence'].mean():.1f} ± {df['cadence'].std():.1f} steps/min")
            
            # 결과 저장
            metrics_path = os.path.join(output_dir, "gait_metrics.csv")
            df.to_csv(metrics_path, index=False)
            print(f"  ✓ 보행 지표 저장: {metrics_path}")
        
        return gait_cycles
    
    def _load_support_labels(self, video_path: str) -> List[Dict]:
        """
        지지 단계 라벨 데이터 로드 (예시 함수)
        실제 구현에서는 데이터 경로에 맞게 조정 필요
        """
        # 예시: 비디오 경로에서 라벨 파일 경로 추정
        # 실제로는 GUI에서와 같은 매핑 로직 필요
        try:
            import os
            video_dir = os.path.dirname(video_path)
            
            # support_label_data 폴더에서 해당 라벨 찾기
            # 실제 구현에서는 더 정교한 매핑 필요
            support_labels = []
            
            # 임시로 빈 리스트 반환 (실제로는 CSV 파일 로드)
            print("  ⚠ 라벨 데이터 로드 로직 구현 필요")
            return support_labels
            
        except Exception as e:
            print(f"  ⚠ 라벨 데이터 로드 실패: {e}")
            return []
    
    def _create_training_dataset(self, imu_data: pd.DataFrame, gait_cycles: List[GaitCycle], 
                                output_dir: str) -> tuple:
        """학습 데이터셋 생성"""
        print("  - IMU 데이터와 보행 지표 정렬 중...")
        
        # 데이터 정렬
        aligned_data = self.dataset_builder.align_imu_with_gait_cycles(imu_data, gait_cycles)
        print(f"  ✓ 정렬된 데이터: {len(aligned_data)} 주기")
        
        # 특징 추출 및 데이터셋 생성
        print("  - 특징 추출 및 데이터셋 생성 중...")
        X, y = self.dataset_builder.create_training_dataset(aligned_data)
        print(f"  ✓ 데이터셋 생성 완료: {X.shape[0]} 샘플, {X.shape[1]} 특징")
        
        # 데이터셋 정보 저장
        dataset_info = {
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'n_targets': int(y.shape[1]),
            'window_size': self.dataset_builder.window_size,
            'overlap': self.dataset_builder.overlap,
            'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
            'target_names': [
                'stride_length', 'velocity', 'cycle_time', 'cadence',
                'hip_rom', 'knee_rom', 'ankle_rom', 'stance_ratio'
            ]
        }
        
        dataset_info_path = os.path.join(output_dir, "dataset_info.json")
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # 데이터셋 저장
        np.save(os.path.join(output_dir, "features.npy"), X)
        np.save(os.path.join(output_dir, "targets.npy"), y)
        print(f"  ✓ 데이터셋 저장 완료")
        
        return X, y
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, output_dir: str) -> Dict:
        """모델 학습"""
        model_type = self.config.get('model_type', 'lstm').lower()
        print(f"  - {model_type.upper()} 모델 학습 중...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = self.dataset_builder.split_dataset(
            X, y, test_size=self.config.get('test_size', 0.2)
        )
        
        # 정규화
        X_train_scaled, X_test_scaled, feature_scaler = self.dataset_builder.normalize_features(
            X_train, X_test
        )
        y_train_scaled, y_test_scaled, target_scaler = self.dataset_builder.normalize_targets(
            y_train, y_test
        )
        
        print(f"  ✓ 데이터 분할: 학습 {X_train.shape[0]}, 테스트 {X_test.shape[0]}")
        
        # 모델 생성 및 학습
        self.model_predictor = GaitMetricsPredictor(
            model_type=model_type,
            window_size=self.config.get('window_size', 90)
        )
        
        # 실제 모델 학습
        print(f"    - 모델 타입: {model_type.upper()}")
        print(f"    - 입력 형태: {X_train_scaled.shape}")
        print(f"    - 출력 형태: {y_train_scaled.shape}")
        
        history = self.model_predictor.train_model(
            X_train_scaled, y_train_scaled,
            X_test_scaled, y_test_scaled,
            epochs=self.config.get('epochs', 50),
            batch_size=self.config.get('batch_size', 32),
            verbose=1
        )
        
        # 모델 평가
        metrics = self.model_predictor.evaluate_model(X_test_scaled, y_test_scaled)
        
        # 스케일러를 원래 범위로 복원하여 실제 성능 확인
        y_pred_scaled = self.model_predictor.predict(X_test_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_test_original = target_scaler.inverse_transform(y_test_scaled)
        
        # 원래 스케일에서의 성능 계산
        original_metrics = self.evaluator.calculate_metrics(y_test_original, y_pred)
        metrics['original_scale'] = original_metrics
        
        print(f"  ✓ 모델 학습 완료")
        print(f"    - 전체 MAE: {metrics['overall']['mae']:.4f}")
        print(f"    - 전체 RMSE: {metrics['overall']['rmse']:.4f}")
        print(f"    - 전체 R²: {metrics['overall']['r2']:.4f}")
        
        # 모델 저장
        model_path = os.path.join(output_dir, f"{model_type}_model.h5")
        scaler_path = os.path.join(output_dir, f"{model_type}_scalers.pkl")
        
        self.model_predictor.save_model(model_path, scaler_path)
        
        # 스케일러도 저장
        import pickle
        scalers = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        print(f"  ✓ 모델 저장: {model_path}")
        print(f"  ✓ 스케일러 저장: {scaler_path}")
        
        return metrics
    
    def _finalize_results(self, output_dir: str, model_metrics: Dict) -> Dict:
        """결과 정리 및 저장"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'model_metrics': model_metrics,
            'output_directory': output_dir,
            'files': {
                'gait_metrics': 'gait_metrics.csv',
                'dataset_info': 'dataset_info.json',
                'features': 'features.npy',
                'targets': 'targets.npy',
                'model': f"{self.config.get('model_type', 'lstm')}_model.h5"
            }
        }
        
        # 전체 결과 저장
        results_path = os.path.join(output_dir, "analysis_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 실험 로그 저장
        experiment_name = f"gait_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.log_experiment(experiment_name, self.config, model_metrics)
        
        print(f"  ✓ 분석 결과 저장: {results_path}")
        print(f"  ✓ 실험 로그 저장: ./experiments/{experiment_name}.json")
        
        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="통합 보행 분석 시스템 예제 파이프라인")
    
    parser.add_argument('--video_path', required=True, help="비디오 파일 경로")
    parser.add_argument('--imu_path', required=True, help="IMU 데이터 파일 경로")
    parser.add_argument('--output_dir', default='./pipeline_results', help="결과 출력 디렉토리")
    parser.add_argument('--config', help="설정 파일 경로 (JSON)")
    
    args = parser.parse_args()
    
    # 기본 설정
    default_config = {
        'model_type': 'lstm',
        'window_size': 90,
        'overlap': 0.5,
        'test_size': 0.2,
        'pixel_to_meter_ratio': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'log_dir': './experiments'
    }
    
    # 설정 파일 로드 (있는 경우)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # 기본 설정과 병합
        default_config.update(config)
    
    config = default_config
    
    print(f"사용 설정:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 파이프라인 실행
    pipeline = GaitAnalysisPipeline(config)
    
    try:
        results = pipeline.run_full_pipeline(
            args.video_path, 
            args.imu_path, 
            args.output_dir
        )
        
        print(f"\n✅ 분석 완료!")
        print(f"결과 확인: {args.output_dir}/analysis_results.json")
        
    except Exception as e:
        print(f"\n❌ 분석 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()