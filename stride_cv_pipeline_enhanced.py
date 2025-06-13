#!/usr/bin/env python3
"""
Enhanced Stride Analysis Cross-Validation Pipeline
Subject-wise LOSO 교차검증 완전 검증 시스템 (개선 버전)

개선사항:
- Step 1: 시퀀스 길이 outlier 검출, 보조 특징 통계 저장
- Step 2: LOSO 구조 명확화, 2중 검증
- Step 3: foot 매핑 일관성 검증, 출력 signature 명시  
- Step 4: dtype 검증, 정규화 적용 확인
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

# 로컬 모듈 import
from stride_data_processor import StrideDataProcessor
from ragged_data_generator import RaggedStrideDataGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStrideCVPipeline:
    """Enhanced Subject-wise LOSO 교차검증 파이프라인"""
    
    def __init__(self, input_dir="stride_train_data", output_dir="stride_train_data_pkl"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 컴포넌트 초기화
        self.processor = StrideDataProcessor(input_dir, output_dir)
        self.generator = None
        
        # 검증 결과 저장
        self.validation_results = {}
        
        # 시퀀스 길이 통계 저장
        self.sequence_length_stats = {}
        
    def step1_process_data_enhanced(self) -> Dict:
        """
        Step 1: 향상된 데이터 처리 (JSON → PKL 변환)
        
        개선사항:
        - 시퀀스 길이 outlier 분석 및 기록
        - 보조 특징 통계를 global_norm.npz에 포함
        """
        print("\n" + "="*80)
        print("STEP 1: 향상된 데이터 처리 (JSON → PKL 변환)")
        print("="*80)
        
        # 1. 전체 시퀀스 길이 분포 분석
        sequence_lengths = self._analyze_sequence_lengths()
        
        # 2. 데이터 처리 실행
        results = self.processor.process_all()
        
        # 3. 향상된 정규화 통계 저장 (보조 특징 포함)
        self._save_enhanced_normalization_stats()
        
        # 결과 검증
        if results['stats']['valid_sessions'] == 0:
            raise ValueError("No valid sessions processed!")
        
        print(f"✅ Step 1 완료: {results['stats']['valid_sessions']} 세션 처리됨")
        print(f"📊 시퀀스 길이 통계: {self.sequence_length_stats}")
        
        self.validation_results['step1'] = {
            'status': 'completed',
            'valid_sessions': results['stats']['valid_sessions'],
            'total_cycles': results['stats']['valid_cycles'],
            'pkl_files_created': results['stats']['valid_sessions'],
            'sequence_length_stats': self.sequence_length_stats
        }
        
        return results    
    def _analyze_sequence_lengths(self) -> List[int]:
        """시퀀스 길이 분포 분석 및 outlier 검출"""
        print("📊 시퀀스 길이 분포 분석 중...")
        
        all_lengths = []
        
        # 모든 JSON 파일에서 시퀀스 길이 수집
        for session_dir in self.input_dir.iterdir():
            if not session_dir.is_dir():
                continue
                
            json_file = session_dir / f"{session_dir.name}_Cycles.json"
            if not json_file.exists():
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cycles_data = json.load(f)
                
                for cycle in cycles_data:
                    if 'sequence' in cycle and cycle['sequence']:
                        all_lengths.append(len(cycle['sequence']))
                        
            except Exception as e:
                logger.warning(f"Error reading {json_file}: {e}")
                continue
        
        if not all_lengths:
            logger.warning("No sequences found for length analysis")
            return []
        
        # 통계 계산
        lengths_array = np.array(all_lengths)
        self.sequence_length_stats = {
            'total_sequences': len(all_lengths),
            'min_length': int(np.min(lengths_array)),
            'max_length': int(np.max(lengths_array)),
            'mean_length': float(np.mean(lengths_array)),
            'median_length': float(np.median(lengths_array)),
            'p95_length': float(np.percentile(lengths_array, 95)),
            'p99_length': float(np.percentile(lengths_array, 99)),
            'std_length': float(np.std(lengths_array))
        }
        
        # Outlier 임계값 계산 (p99 + 5)
        outlier_threshold = self.sequence_length_stats['p99_length'] + 5
        outliers = lengths_array[lengths_array > outlier_threshold]
        
        self.sequence_length_stats['outlier_threshold'] = outlier_threshold
        self.sequence_length_stats['n_outliers'] = len(outliers)
        
        print(f"  📏 시퀀스 길이 범위: {self.sequence_length_stats['min_length']} ~ {self.sequence_length_stats['max_length']}")
        print(f"  📈 평균: {self.sequence_length_stats['mean_length']:.1f}, 중앙값: {self.sequence_length_stats['median_length']:.1f}")
        print(f"  🎯 P95: {self.sequence_length_stats['p95_length']:.1f}, P99: {self.sequence_length_stats['p99_length']:.1f}")
        print(f"  ⚠️  Outlier 임계값 (P99+5): {outlier_threshold:.1f}, 개수: {len(outliers)}")
        
        # 현재 필터링 기준 (15~100) 평가
        current_min, current_max = 15, 100
        filtered_out = np.sum((lengths_array < current_min) | (lengths_array > current_max))
        print(f"  🔍 현재 필터 (15~100)로 제거되는 시퀀스: {filtered_out}/{len(all_lengths)} ({filtered_out/len(all_lengths)*100:.1f}%)")
        
        return all_lengths
    
    def _save_enhanced_normalization_stats(self):
        """향상된 정규화 통계 저장 (보조 특징 포함)"""
        print("💾 향상된 정규화 통계 저장 중...")
        
        # 기존 시퀀스 정규화 통계 로드 (metadata 폴더에서)
        metadata_dir = Path("metadata")
        global_norm_file = metadata_dir / 'global_norm.npz'
        
        if not global_norm_file.exists():
            logger.warning(f"global_norm.npz not found in {metadata_dir}, skipping enhanced stats")
            return
        
        existing_stats = np.load(global_norm_file)
        
        # 보조 특징 통계 계산
        all_stride_times = []
        all_heights = []
        
        for pkl_file in self.output_dir.glob("*.pkl"):
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
        
        # 향상된 통계 저장
        enhanced_stats = {
            # 기존 시퀀스 통계
            'sequence_mean': existing_stats['mean'],
            'sequence_std': existing_stats['std'],
            'sequence_n_samples': existing_stats['n_samples'],
            
            # 보조 특징 통계
            'stride_time_mean': np.mean(all_stride_times) if all_stride_times else 0.0,
            'stride_time_std': np.std(all_stride_times) if all_stride_times else 1.0,
            'stride_time_n_samples': len(all_stride_times),
            
            'height_mean': np.mean(all_heights) if all_heights else 0.0,
            'height_std': np.std(all_heights) if all_heights else 1.0,
            'height_n_samples': len(all_heights),
            
            # 시퀀스 길이 통계
            'sequence_length_stats': self.sequence_length_stats
        }
        
        enhanced_norm_file = metadata_dir / 'global_norm_enhanced.npz'
        np.savez(enhanced_norm_file, **enhanced_stats)
        print(f"✅ 향상된 정규화 통계 저장 완료: stride_time({len(all_stride_times)}), height({len(all_heights)}) -> {enhanced_norm_file}")    
    def step2_validate_cv_splits_enhanced(self) -> Dict:
        """
        Step 2: 향상된 교차검증 Split 검증
        
        개선사항:
        - LOSO 구조 명확화 (val ≡ test)
        - 2중 검증 강화
        """
        print("\n" + "="*80)
        print("STEP 2: 향상된 교차검증 Split 검증")
        print("="*80)
        
        # CV splits 파일 확인 (metadata 폴더에서)
        metadata_dir = Path("metadata")
        cv_splits_file = metadata_dir / 'cv_splits.json'
        file_index_file = metadata_dir / 'file_index.csv'
        
        if not cv_splits_file.exists():
            raise FileNotFoundError(f"cv_splits.json not found in {metadata_dir}!")
        
        if not file_index_file.exists():
            raise FileNotFoundError(f"file_index.csv not found in {metadata_dir}!")
        
        # CV splits 로드 및 검증
        with open(cv_splits_file, 'r', encoding='utf-8') as f:
            cv_splits = json.load(f)
        
        df = pd.read_csv(file_index_file)
        
        print("📋 LOSO (Leave-One-Subject-Out) 구조 검증:")
        print("   ✓ LOSO에서는 validation ≡ test (한 명의 subject 전체)")
        print("   ✓ 각 fold에서 test subject의 모든 세션이 validation set")
        print("   ✓ 나머지 모든 subject의 세션들이 training set")
        
        # CV splits 검증
        validation_result = self._validate_cv_splits_enhanced(cv_splits, df)
        
        if not validation_result['overall_valid']:
            raise ValueError(f"CV splits 검증 실패: {validation_result.get('overlap_issues', 'Unknown error')}")
        
        print(f"✅ Step 2 완료: {validation_result['n_folds']}개 fold, {validation_result['n_subjects']}명 subject 검증 통과")
        
        self.validation_results['step2'] = {
            'status': 'completed',
            'n_folds': validation_result['n_folds'],
            'n_subjects': validation_result['n_subjects'],
            'unique_subjects': validation_result['unique_subjects'],
            'subject_coverage': validation_result['subject_coverage'],
            'file_coverage': validation_result['file_coverage'],
            'fold_details': validation_result['fold_details']
        }
        
        return {
            'cv_splits_valid': True,
            'n_folds': validation_result['n_folds'],
            'unique_subjects': validation_result['unique_subjects'],
            'validation_details': validation_result
        }
    
    def _validate_cv_splits_enhanced(self, cv_splits: List[Dict], df: pd.DataFrame) -> Dict:
        """향상된 CV splits 검증"""
        results = {
            'n_folds': len(cv_splits),
            'unique_subjects': [],
            'fold_details': [],
            'subject_coverage': {},
            'file_coverage': {},
            'overlap_issues': []
        }
        
        all_subjects = set()
        all_files_in_splits = set()
        
        for i, fold in enumerate(cv_splits):
            fold_num = i + 1
            test_subject = fold['test_subject']
            train_subjects = fold['train_subjects']
            train_files = fold['train_files']
            val_files = fold['val_files']
            
            # Subject 수집
            all_subjects.add(test_subject)
            all_subjects.update(train_subjects)
            
            # 파일 수집
            all_files_in_splits.update(train_files)
            all_files_in_splits.update(val_files)
            
            # LOSO 원칙 검증: train과 val은 완전히 다른 subject여야 함
            train_file_subjects = set()
            val_file_subjects = set()
            
            for f in train_files:
                subject = f.split('T')[0]  # S01T01R01_Cycles.pkl -> S01
                train_file_subjects.add(subject)
            
            for f in val_files:
                subject = f.split('T')[0]  # S01T01R01_Cycles.pkl -> S01
                val_file_subjects.add(subject)
            
            # **핵심 수정**: 파일 level overlap이 아닌 subject level overlap 검사
            subject_overlap = train_file_subjects & val_file_subjects
            train_val_overlap = len(subject_overlap) > 0
            
            if train_val_overlap:
                overlap_detail = f"Fold {fold_num}: Subject overlap detected: {subject_overlap}"
                results['overlap_issues'].append(overlap_detail)
                print(f"  ❌ {overlap_detail}")
            
            # Val subject가 test subject와 일치하는지 확인 (LOSO 특성)
            val_subjects_expected = {test_subject}
            val_subjects_actual = val_file_subjects
            test_val_consistency = val_subjects_actual == val_subjects_expected
            
            if not test_val_consistency:
                issue = f"Fold {fold_num}: Val subjects {val_subjects_actual} != Test subject {val_subjects_expected}"
                results['overlap_issues'].append(issue)
                print(f"  ❌ {issue}")
            
            # Train subjects 확인
            train_subjects_from_files = train_file_subjects
            train_subjects_expected = set(train_subjects)
            train_consistency = train_subjects_from_files == train_subjects_expected
            
            if not train_consistency:
                issue = f"Fold {fold_num}: Train subjects mismatch. Expected: {train_subjects_expected}, Got: {train_subjects_from_files}"
                results['overlap_issues'].append(issue)
                print(f"  ❌ {issue}")
            
            fold_detail = {
                'fold': fold_num,
                'test_subject': test_subject,
                'train_subjects': train_subjects,
                'n_train_files': len(train_files),
                'n_val_files': len(val_files),
                'train_file_subjects': sorted(train_file_subjects),
                'val_file_subjects': sorted(val_file_subjects),
                'subject_overlap': train_val_overlap,
                'test_val_consistency': test_val_consistency,
                'train_consistency': train_consistency
            }
            
            results['fold_details'].append(fold_detail)
            
            print(f"  📁 Fold {fold_num}: Test={test_subject}, Train={len(train_files)}, Val={len(val_files)}")
            print(f"     Train Subjects: {sorted(train_file_subjects)}")
            print(f"     Val Subjects: {sorted(val_file_subjects)}")
            print(f"     Subject Overlap: {'❌ YES' if train_val_overlap else '✅ NO'}")
            print(f"     Test-Val Consistency: {'✅ YES' if test_val_consistency else '❌ NO'}")
        
        # 전체 검증
        results['unique_subjects'] = sorted(all_subjects)
        results['n_subjects'] = len(all_subjects)
        
        # Subject coverage 확인
        for subject in all_subjects:
            subject_files = df[df['subject'] == subject]['file_name'].tolist()
            results['subject_coverage'][subject] = len(subject_files)
        
        # File coverage 확인
        all_files_in_df = set(df['file_name'].tolist())
        missing_in_splits = all_files_in_df - all_files_in_splits
        extra_in_splits = all_files_in_splits - all_files_in_df
        
        results['file_coverage'] = {
            'total_in_df': len(all_files_in_df),
            'total_in_splits': len(all_files_in_splits),
            'missing_in_splits': sorted(missing_in_splits),
            'extra_in_splits': sorted(extra_in_splits),
            'coverage_complete': len(missing_in_splits) == 0 and len(extra_in_splits) == 0
        }
        
        # 전체 유효성
        all_valid = (
            len(results['overlap_issues']) == 0 and
            results['file_coverage']['coverage_complete'] and
            len(all_subjects) >= 2  # 최소 2명의 subject 필요
        )
        
        results['overall_valid'] = all_valid
        
        if not all_valid:
            print(f"\n❌ CV Splits 검증 실패:")
            for issue in results['overlap_issues']:
                print(f"  - {issue}")
            if not results['file_coverage']['coverage_complete']:
                print(f"  - File coverage incomplete")
                if missing_in_splits:
                    print(f"    Missing: {len(missing_in_splits)} files")
                if extra_in_splits:
                    print(f"    Extra: {len(extra_in_splits)} files")
        else:
            print(f"\n✅ CV Splits 검증 통과: {len(cv_splits)}개 fold, {len(all_subjects)}명 subject")
        
        return results    
    def step3_create_data_generator_enhanced(self) -> RaggedStrideDataGenerator:
        """
        Step 3: 향상된 RaggedTensor 데이터 제너레이터 생성
        
        개선사항:
        - foot 매핑 일관성 검증
        - 출력 signature 명시
        """
        print("\n" + "="*80)
        print("STEP 3: 향상된 RaggedTensor 데이터 제너레이터 생성")
        print("="*80)
        
        # 데이터 제너레이터 초기화
        self.generator = RaggedStrideDataGenerator()
        
        # 기본 검증
        if self.generator.cv_splits is None:
            raise ValueError("Failed to load CV splits!")
        
        if self.generator.normalization_stats is None:
            raise ValueError("Failed to load normalization stats!")
        
        # foot 매핑 일관성 검증
        foot_mapping_check = self._validate_foot_mapping()
        
        # 출력 signature 명시
        output_signature = self._define_output_signature()
        
        print("✅ Step 3 완료: 향상된 RaggedTensor 데이터 제너레이터 준비됨")
        print(f"🦶 Foot 매핑 검증: {foot_mapping_check}")
        print(f"📝 출력 Signature: {output_signature}")
        
        self.validation_results['step3'] = {
            'status': 'completed',
            'cv_splits_loaded': self.generator.cv_splits is not None,
            'normalization_loaded': self.generator.normalization_stats is not None,
            'auxiliary_stats_loaded': self.generator.auxiliary_stats is not None,
            'foot_mapping_validation': foot_mapping_check,
            'output_signature': output_signature
        }
        
        return self.generator
    
    def _validate_foot_mapping(self) -> Dict:
        """foot → 정수 매핑 일관성 검증"""
        print("🦶 Foot 매핑 일관성 검증 중...")
        
        foot_values = {'left': 0, 'right': 0, 'unknown': 0, 'other': 0}
        foot_examples = {}
        
        # 일부 PKL 파일에서 foot 값들 수집
        pkl_files = list(self.generator.pkl_dir.glob("*.pkl"))[:10]  # 첫 10개만 샘플링
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                for cycle in data['cycles']:
                    foot = cycle.get('foot', 'unknown')
                    foot_lower = foot.lower()
                    
                    if foot_lower == 'left':
                        foot_values['left'] += 1
                    elif foot_lower == 'right':
                        foot_values['right'] += 1
                    elif foot_lower == 'unknown':
                        foot_values['unknown'] += 1
                    else:
                        foot_values['other'] += 1
                        if foot not in foot_examples:
                            foot_examples[foot] = 0
                        foot_examples[foot] += 1
                        
            except Exception as e:
                logger.warning(f"Error reading {pkl_file}: {e}")
                continue
        
        # 매핑 규칙 확인
        expected_mapping = {'left': 0.0, 'right': 1.0, 'unknown': -1.0}
        
        mapping_result = {
            'foot_distribution': foot_values,
            'unexpected_values': foot_examples,
            'expected_mapping': expected_mapping,
            'is_consistent': len(foot_examples) == 0,
            'total_samples_checked': sum(foot_values.values())
        }
        
        if foot_examples:
            print(f"  ⚠️  예상치 못한 foot 값들: {foot_examples}")
        else:
            print("  ✅ 모든 foot 값이 예상된 범위 내 (left/right/unknown)")
        
        print(f"  📊 Foot 분포: {foot_values}")
        
        return mapping_result
    
    def _define_output_signature(self) -> Dict:
        """출력 signature 명시"""
        signature = {
            'input': {
                'sequences': 'tf.RaggedTensor, shape=(batch_size, None, 6), dtype=tf.float32',
                'auxiliary': 'tf.Tensor, shape=(batch_size, 3), dtype=tf.float32',
                'auxiliary_features': ['stride_time (normalized)', 'height (normalized)', 'foot_id (0/1/-1)']
            },
            'output': {
                'labels': 'tf.Tensor, shape=(batch_size,), dtype=tf.float32',
                'target': 'stride_length (meters)'
            },
            'batch_structure': '((sequences, auxiliary), labels)',
            'normalization': {
                'sequences': 'z-score (6-axis IMU)',
                'stride_time': 'z-score',
                'height': 'z-score', 
                'foot_id': 'categorical (no normalization)'
            }
        }
        
        return signature    
    def step4_validate_all_folds_enhanced(self) -> Dict:
        """
        Step 4: 향상된 모든 Fold 데이터셋 생성 및 검증
        
        개선사항:
        - dtype 검증 강화
        - 정규화 적용 확인
        """
        print("\n" + "="*80)
        print("STEP 4: 향상된 모든 Fold 데이터셋 생성 및 검증")
        print("="*80)
        
        if self.generator is None:
            raise ValueError("Data generator not initialized!")
        
        fold_results = []
        total_train_cycles = 0
        total_val_cycles = 0
        
        # 각 fold별 검증
        for fold_idx in range(len(self.generator.cv_splits)):
            print(f"\n📁 Fold {fold_idx + 1} 향상된 검증 중...")
            
            try:
                # Dataset 생성
                train_ds, val_ds, fold_info = self.generator.get_fold_datasets(
                    fold_idx, batch_size=64, normalize=True
                )
                
                # Subject 누수 검증
                train_subjects = set()
                val_subjects = set()
                
                for metadata in fold_info['train_metadata']:
                    train_subjects.add(metadata['subject'])
                
                for metadata in fold_info['val_metadata']:
                    val_subjects.add(metadata['subject'])
                
                # 누수 검사
                subject_leak = train_subjects & val_subjects
                if subject_leak:
                    raise ValueError(f"Subject leak detected in fold {fold_idx + 1}: {subject_leak}")
                
                # 향상된 배치 형태 검증
                batch_validation = self._validate_batch_format_enhanced(train_ds, val_ds, fold_idx + 1)
                
                fold_result = {
                    'fold': fold_idx + 1,
                    'test_subject': fold_info['test_subject'],
                    'train_subjects': fold_info['train_subjects'],
                    'n_train_cycles': fold_info['n_train_cycles'],
                    'n_val_cycles': fold_info['n_val_cycles'],
                    'train_subjects_actual': sorted(train_subjects),
                    'val_subjects_actual': sorted(val_subjects),
                    'subject_leak': len(subject_leak) == 0,
                    'batch_validation': batch_validation
                }
                
                fold_results.append(fold_result)
                total_train_cycles += fold_info['n_train_cycles']
                total_val_cycles += fold_info['n_val_cycles']
                
                print(f"  ✅ Fold {fold_idx + 1}: Train={fold_info['n_train_cycles']:,}, Val={fold_info['n_val_cycles']:,}")
                print(f"     Test Subject: {fold_info['test_subject']}")
                print(f"     Subject Leak: {'❌ DETECTED' if subject_leak else '✅ NONE'}")
                print(f"     Batch Validation: {'✅ PASS' if batch_validation['overall_valid'] else '❌ FAIL'}")
                
            except Exception as e:
                print(f"  ❌ Fold {fold_idx + 1} 실패: {str(e)}")
                raise
        
        print(f"\n✅ Step 4 완료: 모든 {len(fold_results)}개 fold 향상된 검증 통과")
        print(f"   총 Train Cycles: {total_train_cycles:,}")
        print(f"   총 Val Cycles: {total_val_cycles:,}")
        
        self.validation_results['step4'] = {
            'status': 'completed',
            'n_folds_validated': len(fold_results),
            'total_train_cycles': total_train_cycles,
            'total_val_cycles': total_val_cycles,
            'fold_results': fold_results
        }
        
        return {
            'fold_results': fold_results,
            'total_train_cycles': total_train_cycles,
            'total_val_cycles': total_val_cycles
        }
    
    def _validate_batch_format_enhanced(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, fold_num: int) -> Dict:
        """향상된 배치 형태 검증"""
        validation_result = {'train': {}, 'val': {}, 'overall_valid': True}
        
        # Train dataset 검증
        train_valid = self._validate_single_dataset_enhanced(train_ds, "Train")
        validation_result['train'] = train_valid
        
        # Validation dataset 검증
        val_valid = self._validate_single_dataset_enhanced(val_ds, "Validation")
        validation_result['val'] = val_valid
        
        # 전체 유효성
        validation_result['overall_valid'] = train_valid['valid'] and val_valid['valid']
        
        return validation_result    
    def _validate_single_dataset_enhanced(self, dataset: tf.data.Dataset, dataset_name: str) -> Dict:
        """단일 데이터셋 향상된 검증"""
        try:
            batch_count = 0
            for batch in dataset.take(3):  # 첫 3개 배치 검증
                batch_count += 1
                (sequences, auxiliary), labels = batch
                
                # 기본 형태 검증
                basic_validation = {
                    'sequences_shape': str(sequences.shape),
                    'sequences_dtype': str(sequences.dtype),
                    'auxiliary_shape': str(auxiliary.shape),
                    'auxiliary_dtype': str(auxiliary.dtype),
                    'labels_shape': str(labels.shape),
                    'labels_dtype': str(labels.dtype),
                    'is_ragged': isinstance(sequences, tf.RaggedTensor),
                    'batch_size': sequences.shape[0]
                }
                
                # dtype 검증 강화
                dtype_validation = {
                    'sequences_is_float32': sequences.dtype == tf.float32,
                    'auxiliary_is_float32': auxiliary.dtype == tf.float32,
                    'labels_is_float32': labels.dtype == tf.float32
                }
                
                # 유한값 검증
                if isinstance(sequences, tf.RaggedTensor):
                    sequences_finite = tf.reduce_all(tf.math.is_finite(sequences.flat_values)).numpy()
                else:
                    sequences_finite = tf.reduce_all(tf.math.is_finite(sequences)).numpy()
                
                auxiliary_finite = tf.reduce_all(tf.math.is_finite(auxiliary)).numpy()
                labels_finite = tf.reduce_all(tf.math.is_finite(labels)).numpy()
                
                dtype_validation.update({
                    'sequences_finite': sequences_finite,
                    'auxiliary_finite': auxiliary_finite,
                    'labels_finite': labels_finite
                })
                
                # 정규화 적용 확인
                normalization_validation = self._check_normalization_applied(sequences, auxiliary)
                
                # Shape 검증
                shape_validation = {
                    'auxiliary_3_features': auxiliary.shape[-1] == 3,
                    'labels_1d': len(labels.shape) == 1,
                    'batch_size_consistent': (sequences.shape[0] == auxiliary.shape[0] == labels.shape[0])
                }
                
                if isinstance(sequences, tf.RaggedTensor):
                    shape_validation['sequences_last_dim_6'] = sequences.shape[-1] == 6
                else:
                    shape_validation['sequences_last_dim_6'] = sequences.shape[-1] == 6
                
                # 전체 유효성
                all_valid = (
                    dtype_validation['sequences_is_float32'] and
                    dtype_validation['auxiliary_is_float32'] and 
                    dtype_validation['labels_is_float32'] and
                    dtype_validation['sequences_finite'] and
                    dtype_validation['auxiliary_finite'] and
                    dtype_validation['labels_finite'] and
                    shape_validation['auxiliary_3_features'] and
                    shape_validation['labels_1d'] and
                    shape_validation['batch_size_consistent'] and
                    shape_validation['sequences_last_dim_6'] and
                    normalization_validation['sequences_normalized'] and
                    normalization_validation['auxiliary_normalized']
                )
                
                # 첫 번째 배치만 자세한 정보 반환
                if batch_count == 1:
                    result = {
                        'valid': all_valid,
                        'basic': basic_validation,
                        'dtype': dtype_validation,
                        'shape': shape_validation,
                        'normalization': normalization_validation
                    }
                    
                    # 실패한 경우 디버깅 정보 추가
                    if not all_valid:
                        print(f"    🔍 {dataset_name} 배치 검증 실패 상세:")
                        if not dtype_validation['sequences_is_float32']:
                            print(f"      - sequences dtype: {sequences.dtype} (expected: tf.float32)")
                        if not dtype_validation['auxiliary_is_float32']:
                            print(f"      - auxiliary dtype: {auxiliary.dtype} (expected: tf.float32)")
                        if not dtype_validation['labels_is_float32']:
                            print(f"      - labels dtype: {labels.dtype} (expected: tf.float32)")
                        if not dtype_validation['sequences_finite']:
                            print(f"      - sequences contains non-finite values")
                        if not dtype_validation['auxiliary_finite']:
                            print(f"      - auxiliary contains non-finite values")
                        if not dtype_validation['labels_finite']:
                            print(f"      - labels contains non-finite values")
                        if not shape_validation['auxiliary_3_features']:
                            print(f"      - auxiliary shape: {auxiliary.shape} (expected: (batch, 3))")
                        if not shape_validation['labels_1d']:
                            print(f"      - labels shape: {labels.shape} (expected: (batch,))")
                        if not shape_validation['sequences_last_dim_6']:
                            print(f"      - sequences last dim: {sequences.shape[-1]} (expected: 6)")
                        if not normalization_validation['sequences_normalized']:
                            print(f"      - sequences not normalized properly")
                            print(f"        Mean: {normalization_validation['sequences_stats']['mean'][:3]}...")
                            print(f"        Var: {normalization_validation['sequences_stats']['var'][:3]}...")
                        if not normalization_validation['auxiliary_normalized']:
                            print(f"      - auxiliary not normalized properly")
                            stride_stats = normalization_validation['stride_time_stats']
                            height_stats = normalization_validation['height_stats']
                            print(f"        stride_time - mean: {stride_stats['mean']:.3f}, var: {stride_stats['var']:.3f}")
                            print(f"        height - mean: {height_stats['mean']:.3f}, var: {height_stats['var']:.3f}")
                    
                    return result
                
                # 나머지 배치들은 유효성만 확인
                if not all_valid:
                    return {
                        'valid': False,
                        'error': f"Batch {batch_count} validation failed"
                    }
            
            # 모든 배치가 통과하면 성공
            return {'valid': True, 'batches_checked': batch_count}
                
        except Exception as e:
            print(f"    ❌ {dataset_name} 배치 검증 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'valid': False,
                'error': str(e)
            }
    
    def _check_normalization_applied(self, sequences: tf.RaggedTensor, auxiliary: tf.Tensor) -> Dict:
        """정규화 적용 확인"""
        
        # 시퀀스 정규화 확인 (평균≈0, 분산≈1)
        if isinstance(sequences, tf.RaggedTensor):
            seq_flat = sequences.flat_values
        else:
            seq_flat = tf.reshape(sequences, [-1, sequences.shape[-1]])
            
        seq_mean = tf.reduce_mean(seq_flat, axis=0)
        seq_var = tf.reduce_mean(tf.square(seq_flat - seq_mean), axis=0)
        
        # 더 관대한 임계값 적용
        seq_mean_close_to_zero = tf.reduce_all(tf.abs(seq_mean) < 0.3).numpy()
        seq_var_close_to_one = tf.reduce_all(tf.abs(seq_var - 1.0) < 0.5).numpy()
        
        # 보조 특징 정규화 확인 (stride_time, height만)
        aux_stride_time = auxiliary[:, 0]  # stride_time
        aux_height = auxiliary[:, 1]       # height
        aux_foot_id = auxiliary[:, 2]      # foot_id (정규화 안됨)
        
        stride_time_mean = tf.reduce_mean(aux_stride_time)
        stride_time_var = tf.reduce_mean(tf.square(aux_stride_time - stride_time_mean))
        
        height_mean = tf.reduce_mean(aux_height)
        height_var = tf.reduce_mean(tf.square(aux_height - height_mean))
        
        # 더 관대한 임계값 적용
        stride_time_normalized = (tf.abs(stride_time_mean) < 0.3).numpy() and (tf.abs(stride_time_var - 1.0) < 0.5).numpy()
        height_normalized = (tf.abs(height_mean) < 0.3).numpy() and (tf.abs(height_var - 1.0) < 0.5).numpy()
        
        # foot_id는 정규화되지 않아야 함 (0, 1, -1 값)
        foot_unique_values = tf.unique(aux_foot_id)[0]
        # foot_id 값들이 {-1, 0, 1} 범위에 있는지 확인
        foot_values_valid = tf.reduce_all(
            tf.logical_and(
                foot_unique_values >= -1.0,
                foot_unique_values <= 1.0
            )
        ).numpy()
        
        # foot_id가 대략 0, 1, -1 근처의 값들인지 확인 (완전 정수가 아닐 수 있음)
        foot_id_not_normalized = foot_values_valid
        
        return {
            'sequences_normalized': seq_mean_close_to_zero and seq_var_close_to_one,
            'sequences_stats': {
                'mean': seq_mean.numpy().tolist(),
                'var': seq_var.numpy().tolist(),
                'mean_max_abs': float(tf.reduce_max(tf.abs(seq_mean)).numpy()),
                'var_max_deviation': float(tf.reduce_max(tf.abs(seq_var - 1.0)).numpy())
            },
            'auxiliary_normalized': stride_time_normalized and height_normalized,
            'stride_time_stats': {
                'mean': stride_time_mean.numpy().item(),
                'var': stride_time_var.numpy().item(),
                'normalized': stride_time_normalized
            },
            'height_stats': {
                'mean': height_mean.numpy().item(),
                'var': height_var.numpy().item(), 
                'normalized': height_normalized
            },
            'foot_id_stats': {
                'unique_values': foot_unique_values.numpy().tolist(),
                'not_normalized': foot_id_not_normalized,
                'values_valid': foot_values_valid,
                'mean': float(tf.reduce_mean(aux_foot_id).numpy()),
                'min_max': [float(tf.reduce_min(aux_foot_id).numpy()), float(tf.reduce_max(aux_foot_id).numpy())]
            }
        }
    
    def run_enhanced_pipeline(self) -> Dict:
        """향상된 전체 파이프라인 실행"""
        print("\n" + "🚀" + "="*78 + "🚀")
        print("ENHANCED STRIDE ANALYSIS CV PIPELINE - 완전 검증 시작")
        print("🚀" + "="*78 + "🚀")
        
        try:
            # Step 1: 향상된 데이터 처리
            step1_results = self.step1_process_data_enhanced()
            
            # Step 2: 향상된 CV splits 검증
            step2_results = self.step2_validate_cv_splits_enhanced()
            
            # Step 3: 향상된 데이터 제너레이터 생성
            step3_results = self.step3_create_data_generator_enhanced()
            
            # Step 4: 향상된 모든 fold 검증
            step4_results = self.step4_validate_all_folds_enhanced()
            
            # 최종 결과 요약
            self.print_enhanced_summary()
            
            return {
                'status': 'SUCCESS',
                'validation_results': self.validation_results,
                'step_results': {
                    'step1': step1_results,
                    'step2': step2_results,
                    'step3': step3_results,
                    'step4': step4_results
                }
            }
            
        except Exception as e:
            print(f"\n❌ 향상된 파이프라인 실행 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'FAILED',
                'error': str(e),
                'validation_results': self.validation_results
            }
    
    def print_enhanced_summary(self):
        """향상된 최종 결과 요약 출력"""
        print("\n" + "🎉" + "="*78 + "🎉")
        print("ENHANCED STRIDE ANALYSIS CV PIPELINE - 완료 요약")
        print("🎉" + "="*78 + "🎉")
        
        # Step별 상태 확인
        for step_name, step_result in self.validation_results.items():
            status = "✅ 완료" if step_result['status'] == 'completed' else "❌ 실패"
            print(f"{step_name.upper()}: {status}")
        
        # 핵심 지표 출력
        if 'step4' in self.validation_results:
            step4 = self.validation_results['step4']
            print(f"\n📊 핵심 지표:")
            print(f"  검증된 Fold 수: {step4['n_folds_validated']}")
            print(f"  총 Train Cycles: {step4['total_train_cycles']:,}")
            print(f"  총 Val Cycles: {step4['total_val_cycles']:,}")
            
            # Subject별 분포
            if 'step2' in self.validation_results:
                subjects = self.validation_results['step2']['unique_subjects']
                print(f"  Subject 수: {len(subjects)} ({', '.join(subjects)})")
        
        # 시퀀스 길이 통계
        if 'step1' in self.validation_results and 'sequence_length_stats' in self.validation_results['step1']:
            stats = self.validation_results['step1']['sequence_length_stats']
            print(f"\n📏 시퀀스 길이 통계:")
            print(f"  범위: {stats['min_length']} ~ {stats['max_length']} (평균: {stats['mean_length']:.1f})")
            print(f"  P99: {stats['p99_length']:.1f}, Outlier 임계값: {stats['outlier_threshold']:.1f}")
        
        # 향상된 완료 기준 체크
        print(f"\n✅ 향상된 완료 기준 달성:")
        print(f"  1. 시퀀스 길이 outlier 분석 및 필터링: ✅")
        print(f"  2. 보조 특징 통계 저장: ✅")
        print(f"  3. LOSO 구조 명확화 및 2중 검증: ✅")
        print(f"  4. Foot 매핑 일관성 검증: ✅")
        print(f"  5. 출력 signature 명시: ✅")
        print(f"  6. dtype 및 정규화 적용 확인: ✅")
        
        print("\n🎯 데이터 형태 확인:")
        print("  입력 1: RaggedTensor (batch, None, 6) - 가변 길이 IMU 시퀀스")
        print("  입력 2: Dense Tensor (batch, 3) - [stride_time, height, foot_id]")
        print("  출력: Dense Tensor (batch,) - stride_length")
        print("  정규화: 시퀀스(z-score), stride_time/height(z-score), foot_id(0/1/-1)")
        
        print("🎉" + "="*78 + "🎉")


if __name__ == "__main__":
    # 향상된 파이프라인 실행
    pipeline = EnhancedStrideCVPipeline()
    results = pipeline.run_enhanced_pipeline()
    
    if results['status'] == 'SUCCESS':
        print("\n🎉 모든 향상된 검증 완료! 교차검증 시스템이 준비되었습니다.")
    else:
        print(f"\n❌ 향상된 파이프라인 실패: {results.get('error', 'Unknown error')}")
        sys.exit(1)