#!/usr/bin/env python3
"""
TCN (Temporal Convolutional Network) 기반 보폭 예측 모델

아키텍처:
- 두 갈래 입력: (67×6) IMU 시퀀스 + (3,) 보조 벡터
- Masking(mask_value=0.0) 레이어로 패딩 무시
- Dilated Conv1D (dilation 1,2,4,8) × 4스택
- 각 Conv: 64필터, 커널3, causal 패딩
- LayerNorm + ReLU + Residual 연결 + Dropout(0.1)
- GlobalAveragePooling1D → 128차원 임베딩
- 보조벡터와 concat → 131차원
- Dense(64) + ReLU → Dense(1) 최종 출력
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Optional, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@keras.utils.register_keras_serializable()
class DilatedConvBlock(layers.Layer):
    """
    Dilated Convolution 블록
    Conv1D → LayerNorm → ReLU → Dropout
    """
    
    def __init__(self, filters: int, kernel_size: int = 3, dilation_rate: int = 1, 
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Conv1D with causal padding
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=None,
            name=f'conv1d_dilation_{dilation_rate}'
        )
        
        # Layer Normalization
        self.layer_norm = layers.LayerNormalization(name=f'layer_norm_dilation_{dilation_rate}')
        
        # ReLU activation
        self.relu = layers.ReLU(name=f'relu_dilation_{dilation_rate}')
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate, name=f'dropout_dilation_{dilation_rate}')
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x, training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation_rate': self.dilation_rate,
            'dropout_rate': self.dropout_rate
        })
        return config

@keras.utils.register_keras_serializable()
class TCNStack(layers.Layer):
    """
    TCN 스택: Dilated Conv (1,2,4,8) + Residual 연결
    """
    
    def __init__(self, filters: int = 64, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        
        # Dilated Conv 블록들 (dilation 1, 2, 4, 8)
        self.conv_blocks = [
            DilatedConvBlock(filters, dilation_rate=1, dropout_rate=dropout_rate),
            DilatedConvBlock(filters, dilation_rate=2, dropout_rate=dropout_rate),
            DilatedConvBlock(filters, dilation_rate=4, dropout_rate=dropout_rate),
            DilatedConvBlock(filters, dilation_rate=8, dropout_rate=dropout_rate)
        ]
        
        # Residual 연결을 위한 1x1 Conv (채널 수 맞추기)
        self.residual_conv = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same',
            activation=None,
            name='residual_conv'
        )
        
        # 최종 Dropout
        self.final_dropout = layers.Dropout(dropout_rate, name='stack_dropout')
    
    def call(self, inputs, training=None):
        # 입력을 residual 연결용으로 저장
        residual = self.residual_conv(inputs)
        
        # Dilated Conv 블록들을 순차적으로 통과
        x = inputs
        for conv_block in self.conv_blocks:
            x = conv_block(x, training=training)
        
        # Residual 연결
        x = x + residual
        
        # 최종 Dropout
        x = self.final_dropout(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'dropout_rate': self.dropout_rate
        })
        return config

class StrideTCNModel:
    """
    TCN 기반 보폭 예측 모델
    """
    
    def __init__(self, 
                 sequence_length: int = 67,
                 sequence_features: int = 6,
                 auxiliary_features: int = 3,
                 tcn_filters: int = 64,
                 tcn_stacks: int = 4,
                 dropout_rate: float = 0.1,
                 dense_units: int = 64,
                 **kwargs):
        """
        TCN 모델 초기화
        
        Args:
            sequence_length: 시퀀스 길이 (기본값: 67)
            sequence_features: 시퀀스 특성 수 (기본값: 6, IMU 6축)
            auxiliary_features: 보조 특성 수 (기본값: 3, stride_time/height/foot_id)
            tcn_filters: TCN 필터 수 (기본값: 64)
            tcn_stacks: TCN 스택 수 (기본값: 4)
            dropout_rate: 드롭아웃 비율 (기본값: 0.1)
            dense_units: Dense 레이어 유닛 수 (기본값: 64)
        """
        self.sequence_length = sequence_length
        self.sequence_features = sequence_features
        self.auxiliary_features = auxiliary_features
        self.tcn_filters = tcn_filters
        self.tcn_stacks = tcn_stacks
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        
        self.model = None
        
    def build_model(self) -> keras.Model:
        """모델 구축"""
        
        # 입력 레이어들
        sequence_input = layers.Input(
            shape=(self.sequence_length, self.sequence_features),
            name='sequence_input'
        )
        
        auxiliary_input = layers.Input(
            shape=(self.auxiliary_features,),
            name='auxiliary_input'
        )
        
        # 시퀀스 처리 브랜치
        # Masking 레이어 (0-패딩 무시)
        x = layers.Masking(mask_value=0.0, name='masking')(sequence_input)
        
        # TCN 스택들
        for i in range(self.tcn_stacks):
            x = TCNStack(
                filters=self.tcn_filters,
                dropout_rate=self.dropout_rate,
                name=f'tcn_stack_{i+1}'
            )(x)
        
        # Global Average Pooling (마스킹 정보 유지)
        sequence_embedding = layers.GlobalAveragePooling1D(
            name='global_avg_pooling'
        )(x)
        
        # 특성 융합
        # 시퀀스 임베딩(128차원)과 보조 특성(3차원) 연결 → 131차원
        fused_features = layers.Concatenate(
            name='feature_fusion'
        )([sequence_embedding, auxiliary_input])
        
        # 최종 예측 레이어들
        x = layers.Dense(
            self.dense_units,
            activation='relu',
            name='dense_hidden'
        )(fused_features)
        
        x = layers.Dropout(
            self.dropout_rate,
            name='final_dropout'
        )(x)
        
        # 출력 레이어 (보폭 예측)
        output = layers.Dense(
            1,
            activation='linear',
            name='stride_length_output'
        )(x)
        
        # 모델 생성
        model = keras.Model(
            inputs=[sequence_input, auxiliary_input],
            outputs=output,
            name='StrideTCNModel'
        )
        
        self.model = model
        return model
    
    def compile_model(self, 
                     learning_rate: float = 1e-3,
                     loss: str = 'mae',
                     metrics: Optional[list] = None):
        """모델 컴파일"""
        
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        if metrics is None:
            metrics = ['mae', 'mse']
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with lr={learning_rate}, loss={loss}")
        
    def get_model(self) -> keras.Model:
        """컴파일된 모델 반환"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model
    
    def summary(self):
        """모델 요약 출력"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        return self.model.summary()
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model

def create_tcn_model(sequence_length: int = 67,
                     sequence_features: int = 6,
                     auxiliary_features: int = 3,
                     tcn_filters: int = 64,
                     tcn_stacks: int = 4,
                     dropout_rate: float = 0.1,
                     dense_units: int = 64,
                     learning_rate: float = 1e-3,
                     compile_model: bool = True) -> keras.Model:
    """
    TCN 모델 생성 헬퍼 함수
    
    Args:
        sequence_length: 시퀀스 길이
        sequence_features: 시퀀스 특성 수
        auxiliary_features: 보조 특성 수
        tcn_filters: TCN 필터 수
        tcn_stacks: TCN 스택 수
        dropout_rate: 드롭아웃 비율
        dense_units: Dense 레이어 유닛 수
        learning_rate: 학습률
        compile_model: 모델 컴파일 여부
        
    Returns:
        keras.Model: 생성된 TCN 모델
    """
    
    tcn_model = StrideTCNModel(
        sequence_length=sequence_length,
        sequence_features=sequence_features,
        auxiliary_features=auxiliary_features,
        tcn_filters=tcn_filters,
        tcn_stacks=tcn_stacks,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )
    
    model = tcn_model.build_model()
    
    if compile_model:
        tcn_model.compile_model(learning_rate=learning_rate)
    
    return model

def get_model_callbacks(patience_early: int = 10,
                       patience_lr: int = 5,
                       lr_factor: float = 0.5,
                       min_lr: float = 1e-6,
                       monitor: str = 'val_loss',
                       save_best_path: Optional[str] = None) -> list:
    """
    학습용 콜백 함수들 생성
    
    Args:
        patience_early: EarlyStopping patience
        patience_lr: ReduceLROnPlateau patience
        lr_factor: 학습률 감소 비율
        min_lr: 최소 학습률
        monitor: 모니터링할 메트릭
        save_best_path: 최적 모델 저장 경로
        
    Returns:
        list: 콜백 함수들
    """
    
    callbacks = []
    
    # Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience_early,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce Learning Rate on Plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=lr_factor,
        patience=patience_lr,
        min_lr=min_lr,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # Model Checkpoint (선택적)
    if save_best_path:
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=save_best_path,
            monitor=monitor,
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
    
    return callbacks

# 테스트 및 예시
if __name__ == "__main__":
    print("=== TCN 모델 테스트 ===")
    
    # 모델 생성
    model = create_tcn_model(
        tcn_filters=64,
        tcn_stacks=4,
        dropout_rate=0.1,
        learning_rate=1e-3
    )
    
    print("✅ 모델 생성 완료")
    print(f"📊 모델 요약:")
    model.summary()
    
    # 더미 데이터로 테스트
    print("\n=== 더미 데이터 테스트 ===")
    batch_size = 4
    
    # 더미 입력 생성
    dummy_sequences = np.random.randn(batch_size, 67, 6).astype(np.float32)
    dummy_auxiliary = np.random.randn(batch_size, 3).astype(np.float32)
    
    # 예측 테스트
    predictions = model.predict([dummy_sequences, dummy_auxiliary], verbose=0)
    
    print(f"✅ 예측 테스트 완료")
    print(f"   입력 시퀀스 shape: {dummy_sequences.shape}")
    print(f"   입력 보조 특성 shape: {dummy_auxiliary.shape}")
    print(f"   출력 예측 shape: {predictions.shape}")
    print(f"   예측값 범위: {predictions.min():.4f} ~ {predictions.max():.4f}")
    
    # 콜백 함수 테스트
    print("\n=== 콜백 함수 테스트 ===")
    callbacks = get_model_callbacks(
        patience_early=10,
        patience_lr=5,
        save_best_path="test_best_model.keras"
    )
    
    print(f"✅ 콜백 함수 생성 완료: {len(callbacks)}개")
    for i, callback in enumerate(callbacks):
        print(f"   {i+1}. {type(callback).__name__}")
    
    print("\n🎉 모든 테스트 완료!") 