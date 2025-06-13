#!/usr/bin/env python3
"""
RaggedTensor 기반 Stride Analysis 데이터 제너레이터
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

class RaggedStrideDataGenerator:
    """RaggedTensor 기반 stride analysis 데이터 제너레이터"""
    
    def __init__(self, pkl_dir="stride_train_data_pkl", metadata_dir="metadata"):
        self.pkl_dir = Path(pkl_dir)
        self.metadata_dir = Path(metadata_dir)
        self.cv_splits_file = self.metadata_dir / "cv_splits.json"
        self.cv_splits = None
        self.normalization_stats = None
        self.auxiliary_stats = None
        
        self.load_cv_splits()
        self.load_normalization_stats()
        self.calculate_auxiliary_stats()
    
    def load_cv_splits(self):
        """교차검증 split 정보 로드"""
        try:
            with open(self.cv_splits_file, 'r', encoding='utf-8') as f:
                self.cv_splits = json.load(f)
            logger.info(f"Loaded {len(self.cv_splits)} CV splits")
        except FileNotFoundError:
            logger.error(f"CV splits file not found: {self.cv_splits_file}")
            self.cv_splits = None
    
    def load_normalization_stats(self):
        """전역 정규화 통계 로드"""
        global_norm_file = self.metadata_dir / 'global_norm.npz'
        try:
            stats = np.load(global_norm_file)
            self.normalization_stats = {
                'mean': stats['mean'],
                'std': stats['std'],
                'n_samples': stats['n_samples']
            }
            logger.info(f"Loaded global normalization statistics from {global_norm_file}")
        except FileNotFoundError:
            logger.warning(f"Global normalization file not found: {global_norm_file}")
            self.normalization_stats = None
    
    def calculate_auxiliary_stats(self):
        """보조 특징(stride_time, height)의 전역 정규화 통계 계산"""
        if not self.pkl_dir.exists():
            logger.warning("PKL directory not found")
            return
        
        all_stride_times = []
        all_heights = []
        
        for pkl_file in self.pkl_dir.glob("*.pkl"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                for cycle in data['cycles']:
                    stride_time = cycle.get('stride_time', 0.0)
                    height = cycle.get('height', 0.0)
                    
                    if stride_time > 0:
                        all_stride_times.append(stride_time)
                    if height > 0:
                        all_heights.append(height)
                        
            except Exception as e:
                logger.warning(f"Error reading {pkl_file}: {e}")
                continue
        
        if all_stride_times and all_heights:
            self.auxiliary_stats = {
                'stride_time_mean': np.mean(all_stride_times),
                'stride_time_std': np.std(all_stride_times),
                'height_mean': np.mean(all_heights),
                'height_std': np.std(all_heights),
                'n_samples': len(all_stride_times)
            }
            logger.info(f"Calculated auxiliary stats from {len(all_stride_times)} samples")
        else:
            logger.warning("No valid auxiliary features found")
            self.auxiliary_stats = None
    
    def load_pkl_file(self, pkl_file: str) -> Optional[Dict]:
        """개별 PKL 파일 로드"""
        pkl_path = self.pkl_dir / pkl_file
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            logger.error(f"Error loading {pkl_file}: {str(e)}")
            return None
    
    def extract_data_for_generator(self, pkl_files: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Dict]]:
        """PKL 파일들에서 RaggedTensor용 데이터 추출"""
        all_sequences = []
        all_auxiliary = []
        all_labels = []
        all_metadata = []
        
        for pkl_file in pkl_files:
            data = self.load_pkl_file(pkl_file)
            if data is None:
                continue
            
            subject = data['subject']
            task = data['task']
            rep = data['rep']
            
            for cycle_idx, cycle in enumerate(data['cycles']):
                # 시퀀스 데이터 (가변 길이)
                sequence = np.array(cycle['sequence'], dtype=np.float32)
                all_sequences.append(sequence)
                
                # 보조 특징 벡터 [stride_time, height, foot_id]
                stride_time = cycle.get('stride_time', 0.0)
                height = cycle.get('height', 0.0)
                foot = cycle.get('foot', 'unknown')
                
                # foot_id 변환: left=0, right=1, unknown=-1
                if foot.lower() == 'left':
                    foot_id = 0.0
                elif foot.lower() == 'right':
                    foot_id = 1.0
                else:
                    foot_id = -1.0
                
                auxiliary = np.array([stride_time, height, foot_id], dtype=np.float32)
                all_auxiliary.append(auxiliary)
                
                # 라벨 (stride_length)
                stride_length = cycle.get('stride_length', 0.0)
                all_labels.append(stride_length)
                
                # 메타데이터
                metadata = {
                    'subject': subject,
                    'task': task,
                    'rep': rep,
                    'cycle_idx': cycle_idx,
                    'pkl_file': pkl_file,
                    'sequence_length': len(sequence),
                    'stride_time': stride_time,
                    'height': height,
                    'foot': foot
                }
                all_metadata.append(metadata)
        
        logger.info(f"Extracted {len(all_sequences)} variable-length sequences")
        return all_sequences, all_auxiliary, all_labels, all_metadata
    
    def normalize_sequences(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """시퀀스 정규화 (z-score)"""
        if self.normalization_stats is None:
            logger.warning("No normalization stats available")
            return sequences
        
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        
        normalized_sequences = []
        for seq in sequences:
            normalized_seq = (seq - mean) / std
            normalized_sequences.append(normalized_seq.astype(np.float32))
        
        logger.info("Applied z-score normalization to sequences")
        return normalized_sequences
    
    def normalize_auxiliary_features(self, auxiliary_features: List[np.ndarray]) -> List[np.ndarray]:
        """보조 특징 정규화 (stride_time, height만 z-score, foot_id는 그대로)"""
        if self.auxiliary_stats is None:
            logger.warning("No auxiliary stats available")
            return auxiliary_features
        
        stride_time_mean = self.auxiliary_stats['stride_time_mean']
        stride_time_std = self.auxiliary_stats['stride_time_std']
        height_mean = self.auxiliary_stats['height_mean']
        height_std = self.auxiliary_stats['height_std']
        
        normalized_auxiliary = []
        for aux in auxiliary_features:
            stride_time, height, foot_id = aux
            
            # stride_time과 height만 정규화, foot_id는 그대로
            norm_stride_time = (stride_time - stride_time_mean) / stride_time_std if stride_time_std > 0 else 0.0
            norm_height = (height - height_mean) / height_std if height_std > 0 else 0.0
            
            normalized_aux = np.array([norm_stride_time, norm_height, foot_id], dtype=np.float32)
            normalized_auxiliary.append(normalized_aux)
        
        logger.info("Applied z-score normalization to auxiliary features")
        return normalized_auxiliary
    
    def create_tf_dataset(self, sequences: List[np.ndarray], auxiliary_features: List[np.ndarray], 
                         labels: List[float], batch_size: int = 64, shuffle: bool = True,
                         buffer_size: Optional[int] = None) -> tf.data.Dataset:
        """패딩된 Dense Tensor 기반 TensorFlow Dataset 생성"""
        
        if buffer_size is None:
            buffer_size = len(sequences)
        
        # 고정 길이 설정 (p99 + margin)
        MAX_LEN = 67
        
        # 시퀀스 길이 수집 및 패딩
        sequence_lengths = [len(seq) for seq in sequences]
        logger.info(f"Sequence length range: {min(sequence_lengths)} ~ {max(sequence_lengths)}")
        
        # 시퀀스를 MAX_LEN으로 패딩/자르기
        padded_sequences = []
        for seq in sequences:
            if len(seq) < MAX_LEN:
                # 0으로 패딩 (마스킹 값)
                padding = np.zeros((MAX_LEN - len(seq), seq.shape[1]), dtype=np.float32)
                padded_seq = np.concatenate([seq, padding], axis=0)
            else:
                # 너무 긴 경우 자르기 (outlier)
                padded_seq = seq[:MAX_LEN]
            padded_sequences.append(padded_seq)
        
        # NumPy 배열로 변환
        padded_sequences_array = np.array(padded_sequences, dtype=np.float32)
        auxiliary_array = np.array(auxiliary_features, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.float32)
        
        logger.info(f"Dataset shapes: sequences={padded_sequences_array.shape}, auxiliary={auxiliary_array.shape}, labels={labels_array.shape}")
        logger.info(f"Padded to fixed length: {MAX_LEN}")
        
        # TensorFlow Dataset 생성 (Dense Tensor)
        dataset = tf.data.Dataset.from_tensor_slices({
            'sequences': padded_sequences_array,
            'auxiliary': auxiliary_array,
            'labels': labels_array
        })
        
        # 입력과 라벨 분리
        dataset = dataset.map(
            lambda x: ((x['sequences'], x['auxiliary']), x['labels']),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        logger.info(f"Created dataset with padded sequences (max_length={MAX_LEN})")
        return dataset
    
    def get_fold_datasets(self, fold_idx: int, batch_size: int = 64, 
                         normalize: bool = True) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
        """특정 fold의 train/validation TensorFlow Dataset 반환"""
        if self.cv_splits is None:
            raise ValueError("CV splits not loaded")
        
        if fold_idx >= len(self.cv_splits):
            raise ValueError(f"Invalid fold index: {fold_idx}")
        
        fold_info = self.cv_splits[fold_idx]
        
        # Train 데이터 추출
        train_sequences, train_auxiliary, train_labels, train_metadata = self.extract_data_for_generator(
            fold_info['train_files']
        )
        
        # Validation 데이터 추출
        val_sequences, val_auxiliary, val_labels, val_metadata = self.extract_data_for_generator(
            fold_info['val_files']
        )
        
        # 정규화
        if normalize:
            train_sequences = self.normalize_sequences(train_sequences)
            val_sequences = self.normalize_sequences(val_sequences)
            train_auxiliary = self.normalize_auxiliary_features(train_auxiliary)
            val_auxiliary = self.normalize_auxiliary_features(val_auxiliary)
        
        # TensorFlow Dataset 생성
        train_dataset = self.create_tf_dataset(
            train_sequences, train_auxiliary, train_labels, 
            batch_size=batch_size, shuffle=True, buffer_size=len(train_sequences)
        )
        
        val_dataset = self.create_tf_dataset(
            val_sequences, val_auxiliary, val_labels,
            batch_size=batch_size, shuffle=False
        )
        
        # fold 정보 업데이트
        fold_info_extended = {
            **fold_info,
            'n_train_cycles': len(train_sequences),
            'n_val_cycles': len(val_sequences),
            'train_metadata': train_metadata,
            'val_metadata': val_metadata
        }
        
        logger.info(f"Created datasets for fold {fold_idx+1}: "
                   f"train={len(train_sequences)} cycles, val={len(val_sequences)} cycles")
        
        return train_dataset, val_dataset, fold_info_extended 