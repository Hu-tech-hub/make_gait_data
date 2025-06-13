#!/usr/bin/env python3
"""
Stride Training Data Processor
stride_train_data 폴더의 JSON 파일들을 처리하여 학습용 PKL 데이터로 변환
Subject-wise LOSO 5-Fold 교차검증 지원
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
import logging
from sklearn.model_selection import LeaveOneGroupOut

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrideDataProcessor:
    def __init__(self, input_dir="stride_train_data", output_dir="stride_train_data_pkl", metadata_dir="metadata"):
        """
        초기화
        
        Args:
            input_dir: JSON 파일들이 있는 입력 디렉토리
            output_dir: PKL 파일들을 저장할 출력 디렉토리
            metadata_dir: 메타데이터 파일들을 저장할 디렉토리 (인덱스, 정규화, CV splits 등)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = Path(metadata_dir)
        self.min_sequence_length = 15
        self.max_sequence_length = 100
        
        # 통계 정보 저장
        self.processed_stats = {
            'total_sessions': 0,
            'valid_sessions': 0,
            'total_cycles': 0,
            'valid_cycles': 0,
            'filtered_cycles': 0,
            'empty_sessions': 0
        }
        
        # 전역 정규화를 위한 데이터 수집
        self.all_sequences = []
        
        # 파일 인덱스 데이터
        self.file_index = []

    def parse_session_name(self, session_name):
        """
        세션 이름에서 subject, task, rep 추출
        예: S01T01R01 -> subject="S01", task="T01", rep="R01"
        Subject-wise LOSO를 위해 subject는 "S01" 형태로 유지
        """
        pattern = r'(S\d+)(T\d+)(R\d+)'
        match = re.match(pattern, session_name)
        if match:
            return match.groups()  # ("S01", "T01", "R01")
        else:
            logger.warning(f"Invalid session name format: {session_name}")
            return None, None, None

    def is_valid_sequence(self, sequence):
        """
        시퀀스가 유효한지 검사
        - 길이가 15~100 사이인지
        - NaN, Inf 값이 없는지
        - 빈 배열이 아닌지
        """
        if not sequence or len(sequence) == 0:
            return False
        
        if len(sequence) < self.min_sequence_length or len(sequence) > self.max_sequence_length:
            return False
        
        # NumPy 배열로 변환하여 NaN, Inf 체크
        try:
            seq_array = np.array(sequence)
            if np.any(np.isnan(seq_array)) or np.any(np.isinf(seq_array)):
                return False
        except (ValueError, TypeError):
            return False
        
        return True

    def filter_cycles(self, cycles):
        """
        사이클 리스트에서 유효하지 않은 사이클들을 필터링
        """
        valid_cycles = []
        filtered_count = 0
        
        for cycle in cycles:
            if 'sequence' not in cycle:
                filtered_count += 1
                continue
                
            if self.is_valid_sequence(cycle['sequence']):
                valid_cycles.append(cycle)
            else:
                filtered_count += 1
        
        return valid_cycles, filtered_count

    def process_session(self, session_path):
        """
        개별 세션 폴더 처리
        """
        session_name = session_path.name
        subject, task, rep = self.parse_session_name(session_name)
        
        if not all([subject, task, rep]):
            logger.warning(f"Skipping invalid session: {session_name}")
            return None
        
        # JSON 파일 찾기
        json_file = session_path / f"{session_name}_Cycles.json"
        if not json_file.exists():
            logger.warning(f"JSON file not found: {json_file}")
            return None
        
        try:
            # JSON 로드
            with open(json_file, 'r', encoding='utf-8') as f:
                cycles_data = json.load(f)
            
            self.processed_stats['total_cycles'] += len(cycles_data)
            
            # 사이클 필터링
            valid_cycles, filtered_count = self.filter_cycles(cycles_data)
            self.processed_stats['filtered_cycles'] += filtered_count
            
            if len(valid_cycles) == 0:
                logger.info(f"Empty session after filtering: {session_name}")
                self.processed_stats['empty_sessions'] += 1
                return None
            
            self.processed_stats['valid_cycles'] += len(valid_cycles)
            
            # 전역 정규화를 위해 시퀀스 수집
            for cycle in valid_cycles:
                self.all_sequences.extend(cycle['sequence'])
            
            # PKL 데이터 구조 생성 (Subject-wise LOSO를 위한 구조)
            pkl_data = {
                'subject': subject,    # "S01" 형태로 저장
                'task': task,         # "T01" 형태로 저장
                'rep': rep,           # "R01" 형태로 저장
                'cycles': valid_cycles
            }
            
            # PKL 파일 저장
            pkl_file = self.output_dir / f"{session_name}_Cycles.pkl"
            with open(pkl_file, 'wb') as f:
                pickle.dump(pkl_data, f)
            
            # 파일 인덱스에 추가
            self.file_index.append({
                'file_name': f"{session_name}_Cycles.pkl",
                'subject': subject,
                'task': task,
                'rep': rep,
                'n_cycles': len(valid_cycles)
            })
            
            logger.info(f"Processed {session_name}: {len(valid_cycles)} valid cycles")
            return pkl_data
            
        except Exception as e:
            logger.error(f"Error processing {session_name}: {str(e)}")
            return None

    def calculate_global_normalization(self):
        """
        전역 정규화 통계 계산 (6축 각각의 평균, 표준편차)
        """
        if not self.all_sequences:
            logger.warning("No sequences collected for normalization")
            return None
        
        # 모든 시퀀스를 NumPy 배열로 변환
        all_data = np.array(self.all_sequences)  # shape: (N, 6)
        
        # 6축 각각의 평균과 표준편차 계산
        global_mean = np.mean(all_data, axis=0)  # shape: (6,)
        global_std = np.std(all_data, axis=0)    # shape: (6,)
        
        # 0으로 나누기 방지
        global_std = np.where(global_std == 0, 1.0, global_std)
        
        normalization_stats = {
            'mean': global_mean,
            'std': global_std,
            'n_samples': len(all_data)
        }
        
        # metadata 폴더에 global_norm.npz로 저장
        norm_file = self.metadata_dir / 'global_norm.npz'
        np.savez(norm_file, **normalization_stats)
        
        logger.info(f"Global normalization calculated from {len(all_data)} samples")
        logger.info(f"Mean: {global_mean}")
        logger.info(f"Std: {global_std}")
        logger.info(f"Normalization stats saved to: {norm_file}")
        
        return normalization_stats

    def save_file_index(self):
        """
        파일 인덱스를 CSV로 저장
        """
        if not self.file_index:
            logger.warning("No file index data to save")
            return
        
        df = pd.DataFrame(self.file_index)
        index_file = self.metadata_dir / 'file_index.csv'
        df.to_csv(index_file, index=False)
        logger.info(f"File index saved: {len(df)} entries to {index_file}")

    def generate_cross_validation_splits(self):
        """
        Subject-wise LOSO 5-Fold 교차검증 split 생성
        각 subject를 한 번씩 validation set으로 사용
        """
        if not self.file_index:
            logger.warning("No file index available for cross-validation")
            return None
        
        df = pd.DataFrame(self.file_index)
        
        # 고유한 subject 목록 추출
        unique_subjects = sorted(df['subject'].unique())
        logger.info(f"Found {len(unique_subjects)} unique subjects: {unique_subjects}")
        
        # 5명이 아닌 경우 경고
        if len(unique_subjects) != 5:
            logger.warning(f"Expected 5 subjects for 5-Fold CV, but found {len(unique_subjects)}")
        
        # 각 fold별 train/validation split 생성
        cv_splits = []
        for i, test_subject in enumerate(unique_subjects):
            # 현재 subject는 validation, 나머지는 training
            train_mask = df['subject'] != test_subject
            val_mask = df['subject'] == test_subject
            
            train_files = df[train_mask]['file_name'].tolist()
            val_files = df[val_mask]['file_name'].tolist()
            
            fold_info = {
                'fold': i + 1,
                'test_subject': test_subject,
                'train_subjects': [s for s in unique_subjects if s != test_subject],
                'train_files': train_files,
                'val_files': val_files,
                'n_train': len(train_files),
                'n_val': len(val_files)
            }
            
            cv_splits.append(fold_info)
            
            logger.info(f"Fold {i+1}: Test subject {test_subject} "
                       f"({len(val_files)} sessions), "
                       f"Train subjects {fold_info['train_subjects']} "
                       f"({len(train_files)} sessions)")
        
        # CV splits를 JSON으로 저장
        cv_file = self.metadata_dir / 'cv_splits.json'
        with open(cv_file, 'w', encoding='utf-8') as f:
            json.dump(cv_splits, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Cross-validation splits saved to {cv_file}")
        return cv_splits

    def analyze_data_distribution(self):
        """
        데이터 분포 분석 (subject별, task별)
        """
        if not self.file_index:
            logger.warning("No file index available for analysis")
            return None
        
        df = pd.DataFrame(self.file_index)
        
        print("\n" + "="*60)
        print("DATA DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Subject별 분포
        subject_stats = df.groupby('subject').agg({
            'file_name': 'count',
            'n_cycles': ['sum', 'mean']
        }).round(2)
        subject_stats.columns = ['n_sessions', 'total_cycles', 'avg_cycles_per_session']
        print("\n📊 Subject별 분포:")
        print(subject_stats)
        
        # Task별 분포
        task_stats = df.groupby('task').agg({
            'file_name': 'count',
            'n_cycles': ['sum', 'mean']
        }).round(2)
        task_stats.columns = ['n_sessions', 'total_cycles', 'avg_cycles_per_session']
        print("\n📊 Task별 분포:")
        print(task_stats)
        
        # Subject-Task 조합별 분포
        print("\n📊 Subject-Task 조합별 세션 수:")
        pivot_table = df.pivot_table(
            values='file_name', 
            index='subject', 
            columns='task', 
            aggfunc='count',
            fill_value=0
        )
        print(pivot_table)
        
        # 전체 통계
        total_sessions = len(df)
        total_cycles = df['n_cycles'].sum()
        avg_cycles = df['n_cycles'].mean()
        
        print(f"\n📈 전체 통계:")
        print(f"- 총 세션 수: {total_sessions:,}")
        print(f"- 총 사이클 수: {total_cycles:,}")
        print(f"- 세션당 평균 사이클: {avg_cycles:.1f}")
        print("="*60)
        
        return {
            'subject_stats': subject_stats,
            'task_stats': task_stats,
            'pivot_table': pivot_table,
            'total_stats': {
                'total_sessions': total_sessions,
                'total_cycles': total_cycles,
                'avg_cycles': avg_cycles
            }
        }

    def process_all(self):
        """
        전체 처리 파이프라인 실행
        """
        logger.info("Starting stride data processing pipeline...")
        
        # 출력 디렉토리들 생성
        self.output_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output directories created:")
        logger.info(f"  PKL files: {self.output_dir}")
        logger.info(f"  Metadata: {self.metadata_dir}")
        
        # 모든 세션 폴더 찾기
        session_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        session_dirs.sort()  # 정렬
        
        self.processed_stats['total_sessions'] = len(session_dirs)
        logger.info(f"Found {len(session_dirs)} session directories")
        
        # 각 세션 처리
        valid_sessions = 0
        for session_path in tqdm(session_dirs, desc="Processing sessions"):
            result = self.process_session(session_path)
            if result is not None:
                valid_sessions += 1
        
        self.processed_stats['valid_sessions'] = valid_sessions
        
        # 전역 정규화 통계 계산
        normalization_stats = self.calculate_global_normalization()
        
        # 파일 인덱스 저장
        self.save_file_index()
        
        # 데이터 분포 분석
        distribution_stats = self.analyze_data_distribution()
        
        # 교차검증 splits 생성
        cv_splits = self.generate_cross_validation_splits()
        
        # 처리 결과 출력
        self.print_summary()
        
        return {
            'stats': self.processed_stats,
            'normalization': normalization_stats,
            'file_index_size': len(self.file_index),
            'distribution': distribution_stats,
            'cv_splits': cv_splits
        }

    def print_summary(self):
        """
        처리 결과 요약 출력
        """
        stats = self.processed_stats
        
        print("\n" + "="*60)
        print("STRIDE DATA PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Sessions Found    : {stats['total_sessions']:,}")
        print(f"Valid Sessions Processed: {stats['valid_sessions']:,}")
        print(f"Empty Sessions (filtered): {stats['empty_sessions']:,}")
        print(f"")
        print(f"Total Cycles Found      : {stats['total_cycles']:,}")
        print(f"Valid Cycles Kept       : {stats['valid_cycles']:,}")
        print(f"Filtered Cycles Removed : {stats['filtered_cycles']:,}")
        print(f"")
        print(f"PKL Files Created       : {stats['valid_sessions']:,}")
        print(f"File Index Entries      : {len(self.file_index):,}")
        print(f"Global Normalization    : {'✓' if self.all_sequences else '✗'}")
        print(f"Cross-Validation Splits : {'✓' if self.file_index else '✗'}")
        print("="*60)


def main():
    """
    메인 실행 함수
    """
    processor = StrideDataProcessor()
    results = processor.process_all()
    
    print(f"\nProcessing completed successfully!")
    print(f"📁 PKL files: 'stride_train_data_pkl/' folder")
    print(f"📊 Metadata files: 'metadata/' folder")
    print(f"  - file_index.csv: 파일 인덱스 및 메타데이터")
    print(f"  - global_norm.npz: 전역 정규화 통계")
    print(f"  - cv_splits.json: 교차검증 split 정보")


if __name__ == "__main__":
    main() 