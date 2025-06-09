# =====================================
# Time Series Regression Model Module
# =====================================
"""
time_series_model.py - IMU 데이터 기반 보행 지표 예측 모델

이 모듈은 다음 기능을 제공합니다:
1. IMU 시계열 데이터 전처리 및 특징 추출
2. 딥러닝 회귀 모델 (LSTM, TCN, 1D CNN) 구현
3. 모델 학습 및 검증
4. 실시간 보행 지표 예측
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import List, Tuple, Dict, Optional, Union
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class IMUFeatureExtractor:
    """IMU 시계열 데이터 특징 추출 클래스"""
    
    def __init__(self, window_size: int = 90, overlap: float = 0.5):
        """
        초기화
        
        Args:
            window_size (int): 슬라이딩 윈도우 크기 (프레임 수)
            overlap (float): 윈도우 간 겹침 비율 (0.0~1.0)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        
        # 스케일러 초기화
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_time_domain_features(self, signal: np.ndarray) -> np.ndarray:
        """
        시간 도메인 특징 추출
        
        Args:
            signal (np.ndarray): IMU 신호 (윈도우 크기 x 6축)
            
        Returns:
            np.ndarray: 시간 도메인 특징 벡터
        """
        features = []
        
        for axis in range(signal.shape[1]):  # 각 축(AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
            axis_signal = signal[:, axis]
            
            # 기본 통계 특징
            features.extend([
                np.mean(axis_signal),      # 평균
                np.std(axis_signal),       # 표준편차
                np.var(axis_signal),       # 분산
                np.min(axis_signal),       # 최솟값
                np.max(axis_signal),       # 최댓값
                np.median(axis_signal),    # 중앙값
                np.percentile(axis_signal, 25),  # 25% 분위수
                np.percentile(axis_signal, 75),  # 75% 분위수
            ])
            
            # 형태 특징
            features.extend([
                np.sqrt(np.mean(axis_signal**2)),  # RMS
                np.mean(np.abs(axis_signal)),      # 평균 절댓값
                np.sum(np.abs(np.diff(axis_signal))),  # 총 변화량
            ])
            
        return np.array(features)
    
    def extract_frequency_domain_features(self, signal: np.ndarray, fs: float = 30.0) -> np.ndarray:
        """
        주파수 도메인 특징 추출
        
        Args:
            signal (np.ndarray): IMU 신호
            fs (float): 샘플링 주파수 (Hz)
            
        Returns:
            np.ndarray: 주파수 도메인 특징 벡터
        """
        features = []
        
        for axis in range(signal.shape[1]):
            axis_signal = signal[:, axis]
            
            # FFT 계산
            fft = np.fft.fft(axis_signal)
            freqs = np.fft.fftfreq(len(axis_signal), 1/fs)
            magnitude = np.abs(fft)
            
            # 양의 주파수 성분만 사용
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            if len(positive_magnitude) > 0:
                # 스펙트럼 특징
                features.extend([
                    np.mean(positive_magnitude),         # 평균 스펙트럼 크기
                    np.std(positive_magnitude),          # 스펙트럼 표준편차
                    positive_freqs[np.argmax(positive_magnitude)],  # 주파수 피크
                    np.sum(positive_magnitude),          # 총 스펙트럼 에너지
                ])
                
                # 주파수 대역별 에너지
                low_band = (positive_freqs <= 2.0)     # 0-2 Hz
                mid_band = (positive_freqs > 2.0) & (positive_freqs <= 8.0)  # 2-8 Hz
                high_band = (positive_freqs > 8.0)     # 8+ Hz
                
                features.extend([
                    np.sum(positive_magnitude[low_band]),   # 저주파 에너지
                    np.sum(positive_magnitude[mid_band]),   # 중주파 에너지
                    np.sum(positive_magnitude[high_band]),  # 고주파 에너지
                ])
            else:
                features.extend([0] * 7)  # 기본값으로 채움
                
        return np.array(features)
    
    def create_sliding_windows(self, imu_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        슬라이딩 윈도우로 IMU 데이터 분할
        
        Args:
            imu_data (pd.DataFrame): IMU 데이터
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (원시 윈도우, 특징 벡터)
        """
        # IMU 컬럼 추출
        imu_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        imu_values = imu_data[imu_columns].values
        
        windows = []
        features = []
        
        for start_idx in range(0, len(imu_values) - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window = imu_values[start_idx:end_idx]
            
            # 원시 윈도우 저장
            windows.append(window)
            
            # 특징 추출
            time_features = self.extract_time_domain_features(window)
            freq_features = self.extract_frequency_domain_features(window)
            combined_features = np.concatenate([time_features, freq_features])
            
            features.append(combined_features)
        
        return np.array(windows), np.array(features)
    
    def fit_scaler(self, features: np.ndarray):
        """
        스케일러 학습
        
        Args:
            features (np.ndarray): 특징 벡터 배열
        """
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """
        특징 정규화
        
        Args:
            features (np.ndarray): 원시 특징 벡터
            
        Returns:
            np.ndarray: 정규화된 특징 벡터
        """
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다. fit_scaler()를 먼저 호출하세요.")
        
        return self.scaler.transform(features)

class GaitMetricsPredictor:
    """보행 지표 예측 모델 클래스"""
    
    def __init__(self, model_type: str = "lstm", window_size: int = 90):
        """
        초기화
        
        Args:
            model_type (str): 모델 타입 ("lstm", "tcn", "cnn1d")
            window_size (int): 입력 시퀀스 길이
        """
        self.model_type = model_type
        self.window_size = window_size
        self.model = None
        self.feature_extractor = IMUFeatureExtractor(window_size)
        
        # 출력 지표 정의
        self.output_metrics = [
            'stride_length', 'velocity', 'cycle_time', 'cadence',
            'hip_rom', 'knee_rom', 'ankle_rom', 'stance_ratio'
        ]
        
        # 출력 스케일러
        self.output_scaler = StandardScaler()
        self.output_scaler_fitted = False
    
    def create_lstm_model(self, input_shape: Tuple, output_dim: int) -> keras.Model:
        """
        LSTM 모델 생성
        
        Args:
            input_shape (Tuple): 입력 형태
            output_dim (int): 출력 차원
            
        Returns:
            keras.Model: LSTM 모델
        """
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_tcn_model(self, input_shape: Tuple, output_dim: int) -> keras.Model:
        """
        Temporal Convolutional Network (TCN) 모델 생성
        
        Args:
            input_shape (Tuple): 입력 형태
            output_dim (int): 출력 차원
            
        Returns:
            keras.Model: TCN 모델
        """
        inputs = layers.Input(shape=input_shape)
        
        # TCN 블록들
        x = inputs
        for i, filters in enumerate([64, 128, 64]):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=3,
                dilation_rate=2**i,
                padding='causal',
                activation='relu'
            )(x)
            x = layers.Dropout(0.2)(x)
        
        # Global pooling and dense layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(output_dim, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def create_cnn1d_model(self, input_shape: Tuple, output_dim: int) -> keras.Model:
        """
        1D CNN 모델 생성
        
        Args:
            input_shape (Tuple): 입력 형태
            output_dim (int): 출력 차원
            
        Returns:
            keras.Model: 1D CNN 모델
        """
        model = keras.Sequential([
            layers.Conv1D(64, 5, activation='relu', input_shape=input_shape),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu'),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 100, batch_size: int = 32, 
                   verbose: int = 1) -> Dict:
        """
        모델 학습
        
        Args:
            X_train (np.ndarray): 학습 특징 데이터
            y_train (np.ndarray): 학습 타겟 데이터
            X_val (np.ndarray): 검증 특징 데이터
            y_val (np.ndarray): 검증 타겟 데이터
            epochs (int): 에포크 수
            batch_size (int): 배치 크기
            verbose (int): 출력 레벨
            
        Returns:
            Dict: 학습 히스토리
        """
        # 특징 벡터를 사용하는 경우 입력 형태 조정
        if len(X_train.shape) == 2:
            # 특징 벡터를 시퀀스로 변환 (batch_size, 1, features)
            input_shape = (1, X_train.shape[1])
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            if X_val is not None:
                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
        else:
            # 이미 시퀀스 형태인 경우
            input_shape = (X_train.shape[1], X_train.shape[2])
        
        output_dim = y_train.shape[1]
        
        # 모델 생성
        if self.model_type.lower() == "lstm":
            self.model = self.create_lstm_model(input_shape, output_dim)
        elif self.model_type.lower() == "tcn":
            self.model = self.create_tcn_model(input_shape, output_dim)
        elif self.model_type.lower() == "cnn1d":
            self.model = self.create_cnn1d_model(input_shape, output_dim)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")
        
        # 콜백 설정
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # 검증 데이터 준비
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # 모델 학습
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X (np.ndarray): 입력 데이터
            
        Returns:
            np.ndarray: 예측 결과
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. train_model()을 먼저 호출하세요.")
        
        # 입력 형태 조정
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """
        모델 저장
        
        Args:
            model_path (str): 모델 저장 경로
            scaler_path (str): 스케일러 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(model_path)
        
        if scaler_path and self.feature_extractor.is_fitted:
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_extractor.scaler, f)
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        모델 로드
        
        Args:
            model_path (str): 모델 파일 경로
            scaler_path (str): 스케일러 파일 경로
        """
        self.model = keras.models.load_model(model_path)
        
        if scaler_path:
            import pickle
            with open(scaler_path, 'rb') as f:
                self.feature_extractor.scaler = pickle.load(f)
                self.feature_extractor.is_fitted = True
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        모델 평가
        
        Args:
            X_test (np.ndarray): 테스트 특징 데이터
            y_test (np.ndarray): 테스트 타겟 데이터
            
        Returns:
            Dict: 평가 결과
        """
        if self.model is None:
            raise ValueError("평가할 모델이 없습니다.")
        
        # 예측 수행
        y_pred = self.predict(X_test)
        
        # 성능 지표 계산
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        # 전체 성능
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'overall': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            }
        }
        
        # 각 지표별 성능
        for i, metric_name in enumerate(self.output_metrics):
            if i < y_test.shape[1]:
                mae_metric = mean_absolute_error(y_test[:, i], y_pred[:, i])
                mse_metric = mean_squared_error(y_test[:, i], y_pred[:, i])
                rmse_metric = np.sqrt(mse_metric)
                r2_metric = r2_score(y_test[:, i], y_pred[:, i])
                
                results[metric_name] = {
                    'mae': mae_metric,
                    'mse': mse_metric,
                    'rmse': rmse_metric,
                    'r2': r2_metric
                }
        
        return results