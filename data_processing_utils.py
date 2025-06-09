# ======================================
# Data Processing Utilities Module
# ======================================
"""
data_processing_utils.py - 데이터 처리 유틸리티

이 모듈은 다음 기능을 제공합니다:
1. IMU 데이터와 보행 지표 매핑
2. 학습 데이터셋 생성
3. 데이터 전처리 및 정규화
4. 교차 검증 및 성능 평가
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from gait_metrics_calculator import GaitCycle
from time_series_model import IMUFeatureExtractor
import json
import os


class GaitDatasetBuilder:
    """보행 데이터셋 생성 클래스"""
    
    def __init__(self, window_size: int = 90, overlap: float = 0.5):
        """
        초기화
        
        Args:
            window_size (int): IMU 윈도우 크기
            overlap (float): 윈도우 겹침 비율
        """
        self.window_size = window_size
        self.overlap = overlap
        self.feature_extractor = IMUFeatureExtractor(window_size, overlap)
        
    def align_imu_with_gait_cycles(self, imu_data: pd.DataFrame, 
                                  gait_cycles: List[GaitCycle]) -> List[Dict]:
        """
        IMU 데이터와 보행 주기 정렬
        
        Args:
            imu_data (pd.DataFrame): IMU 데이터
            gait_cycles (List[GaitCycle]): 보행 주기 리스트
            
        Returns:
            List[Dict]: 정렬된 데이터 리스트
        """
        aligned_data = []
        
        for cycle in gait_cycles:
            # 해당 주기의 IMU 데이터 추출
            start_time = cycle.start_frame / 30.0  # 30 FPS 가정
            end_time = cycle.end_frame / 30.0
            
            cycle_imu = imu_data[
                (imu_data['sync_timestamp'] >= start_time) & 
                (imu_data['sync_timestamp'] <= end_time)
            ].copy()
            
            if len(cycle_imu) < self.window_size:
                continue  # 데이터가 부족한 주기는 건너뛰기
            
            # IMU 컬럼명 표준화
            if 'accel_x' not in cycle_imu.columns:
                # 기존 컬럼명을 표준 컬럼명으로 변경
                column_mapping = {
                    'accel_x': 'accel_x', 'accel_y': 'accel_y', 'accel_z': 'accel_z',
                    'gyro_x': 'gyro_x', 'gyro_y': 'gyro_y', 'gyro_z': 'gyro_z'
                }
                # 실제 컬럼명에 맞게 매핑 수정 필요
                for old_col, new_col in column_mapping.items():
                    if old_col in cycle_imu.columns:
                        cycle_imu[new_col] = cycle_imu[old_col]
            
            aligned_data.append({
                'imu_data': cycle_imu,
                'gait_cycle': cycle,
                'start_time': start_time,
                'end_time': end_time
            })
        
        return aligned_data
    
    def create_training_dataset(self, aligned_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습 데이터셋 생성
        
        Args:
            aligned_data (List[Dict]): 정렬된 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (특징 벡터, 타겟 값)
        """
        features_list = []
        targets_list = []
        
        for data_item in aligned_data:
            imu_data = data_item['imu_data']
            gait_cycle = data_item['gait_cycle']
            
            # IMU 특징 추출
            try:
                windows, features = self.feature_extractor.create_sliding_windows(imu_data)
                
                if len(features) > 0:
                    # 각 윈도우에 대해 해당 주기의 보행 지표를 타겟으로 사용
                    for feature_vector in features:
                        features_list.append(feature_vector)
                        
                        # 타겟 값 (보행 지표)
                        target = np.array([
                            gait_cycle.stride_length,
                            gait_cycle.velocity,
                            gait_cycle.cycle_time,
                            gait_cycle.cadence,
                            gait_cycle.hip_rom,
                            gait_cycle.knee_rom,
                            gait_cycle.ankle_rom,
                            gait_cycle.stance_ratio
                        ])
                        targets_list.append(target)
                        
            except Exception as e:
                print(f"특징 추출 오류: {e}")
                continue
        
        if not features_list:
            raise ValueError("특징 벡터가 생성되지 않았습니다.")
        
        X = np.array(features_list)
        y = np.array(targets_list)
        
        return X, y
    
    def split_dataset(self, X: np.ndarray, y: np.ndarray, 
                     test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        데이터셋 분할
        
        Args:
            X (np.ndarray): 특징 벡터
            y (np.ndarray): 타겟 값
            test_size (float): 테스트 셋 비율
            random_state (int): 랜덤 시드
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def normalize_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """
        특징 정규화
        
        Args:
            X_train (np.ndarray): 학습 특징
            X_test (np.ndarray): 테스트 특징
            
        Returns:
            Tuple: (정규화된 X_train, 정규화된 X_test, scaler)
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def normalize_targets(self, y_train: np.ndarray, y_test: np.ndarray) -> Tuple:
        """
        타겟 정규화
        
        Args:
            y_train (np.ndarray): 학습 타겟
            y_test (np.ndarray): 테스트 타겟
            
        Returns:
            Tuple: (정규화된 y_train, 정규화된 y_test, scaler)
        """
        scaler = StandardScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_test_scaled = scaler.transform(y_test)
        
        return y_train_scaled, y_test_scaled, scaler


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self):
        """초기화"""
        self.metric_names = [
            'stride_length', 'velocity', 'cycle_time', 'cadence',
            'hip_rom', 'knee_rom', 'ankle_rom', 'stance_ratio'
        ]
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        회귀 성능 지표 계산
        
        Args:
            y_true (np.ndarray): 실제값
            y_pred (np.ndarray): 예측값
            
        Returns:
            Dict: 성능 지표 딕셔너리
        """
        metrics = {}
        
        # 전체 성능
        metrics['overall'] = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        # 각 지표별 성능
        for i, metric_name in enumerate(self.metric_names):
            if i < y_true.shape[1]:
                y_true_metric = y_true[:, i]
                y_pred_metric = y_pred[:, i]
                
                metrics[metric_name] = {
                    'mae': mean_absolute_error(y_true_metric, y_pred_metric),
                    'rmse': np.sqrt(mean_squared_error(y_true_metric, y_pred_metric)),
                    'r2': r2_score(y_true_metric, y_pred_metric)
                }
        
        return metrics
    
    def cross_validate(self, model, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5) -> Dict:
        """
        교차 검증
        
        Args:
            model: 모델 객체
            X (np.ndarray): 특징 벡터
            y (np.ndarray): 타겟 값
            cv_folds (int): 교차 검증 폴드 수
            
        Returns:
            Dict: 교차 검증 결과
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'mae': [],
            'rmse': [],
            'r2': []
        }
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 모델 학습 (실제 구현에서는 모델 타입에 따라 다름)
            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_val)
            
            # 임시로 더미 예측값 사용
            y_pred = np.random.random(y_val.shape)
            
            # 성능 계산
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            cv_scores['mae'].append(mae)
            cv_scores['rmse'].append(rmse)
            cv_scores['r2'].append(r2)
        
        # 평균 및 표준편차 계산
        result = {}
        for metric in cv_scores:
            scores = cv_scores[metric]
            result[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return result
    
    def plot_prediction_comparison(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  save_path: Optional[str] = None):
        """
        예측값과 실제값 비교 플롯
        
        Args:
            y_true (np.ndarray): 실제값
            y_pred (np.ndarray): 예측값
            save_path (Optional[str]): 저장 경로
        """
        n_metrics = min(len(self.metric_names), y_true.shape[1])
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(n_metrics):
            ax = axes[i]
            
            # 산점도
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            
            # 완벽한 예측선 (y=x)
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('실제값')
            ax.set_ylabel('예측값')
            ax.set_title(f'{self.metric_names[i]}')
            
            # R² 표시
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      save_path: Optional[str] = None):
        """
        잔차 플롯
        
        Args:
            y_true (np.ndarray): 실제값
            y_pred (np.ndarray): 예측값
            save_path (Optional[str]): 저장 경로
        """
        residuals = y_true - y_pred
        n_metrics = min(len(self.metric_names), y_true.shape[1])
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(n_metrics):
            ax = axes[i]
            
            # 잔차 vs 예측값
            ax.scatter(y_pred[:, i], residuals[:, i], alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
            
            ax.set_xlabel('예측값')
            ax.set_ylabel('잔차')
            ax.set_title(f'{self.metric_names[i]} 잔차')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


class ExperimentLogger:
    """실험 로그 관리 클래스"""
    
    def __init__(self, log_dir: str = "./experiments"):
        """
        초기화
        
        Args:
            log_dir (str): 로그 디렉토리
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_experiment(self, experiment_name: str, config: Dict, 
                      metrics: Dict, model_path: str = None):
        """
        실험 결과 로그
        
        Args:
            experiment_name (str): 실험 이름
            config (Dict): 실험 설정
            metrics (Dict): 성능 지표
            model_path (str): 모델 저장 경로
        """
        from datetime import datetime
        
        log_data = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'model_path': model_path
        }
        
        log_file = os.path.join(self.log_dir, f"{experiment_name}.json")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"실험 로그 저장: {log_file}")
    
    def load_experiment(self, experiment_name: str) -> Dict:
        """
        실험 로그 로드
        
        Args:
            experiment_name (str): 실험 이름
            
        Returns:
            Dict: 실험 데이터
        """
        log_file = os.path.join(self.log_dir, f"{experiment_name}.json")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[str]:
        """
        실험 목록 반환
        
        Returns:
            List[str]: 실험 이름 리스트
        """
        experiments = []
        for file in os.listdir(self.log_dir):
            if file.endswith('.json'):
                experiments.append(file[:-5])  # .json 제거
        
        return experiments