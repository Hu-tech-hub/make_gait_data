#!/usr/bin/env python3
"""
TCN 기반 보폭 예측 모델 교차검증 트레이너

학습 흐름:
1. metadata/file_index.csv와 metadata/cv_splits.json 로드
2. 무결성 검사 (PKL 경로, 중복, 피험자 누수)
3. RaggedStrideDataGenerator로 fold별 데이터셋 생성
4. TCN 모델 빌드 및 학습
5. EarlyStopping/ReduceLROnPlateau 콜백 적용
6. 최적 모델 저장 (models/best_fold{k}.keras)
7. 교차검증 결과 리포트 생성 (models/cv_report.txt)
8. 선택적 전체 재훈련 (models/final_model.keras)
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tensorflow as tf

# 로컬 모듈 import
from tcn_model import create_tcn_model, get_model_callbacks
from ragged_data_generator import RaggedStrideDataGenerator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('tcn_training.log')
    ]
)
logger = logging.getLogger(__name__)

class TCNTrainer:
    """TCN 모델 교차검증 트레이너"""
    
    def __init__(self, 
                 metadata_dir: str = "metadata",
                 pkl_dir: str = "stride_train_data_pkl",
                 models_dir: str = "models",
                 logs_dir: str = "logs",
                 strict_mode: bool = True):
        """
        트레이너 초기화
        
        Args:
            metadata_dir: 메타데이터 디렉토리
            pkl_dir: PKL 파일 디렉토리
            models_dir: 모델 저장 디렉토리
            logs_dir: 로그 저장 디렉토리
            strict_mode: 엄격 모드 (무결성 검사 실패시 중단)
        """
        self.metadata_dir = Path(metadata_dir)
        self.pkl_dir = Path(pkl_dir)
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.strict_mode = strict_mode
        
        # 디렉토리 생성
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # 데이터 제너레이터
        self.generator = None
        
        # 결과 저장
        self.cv_results = []
        self.training_history = {}
        
    def load_and_validate_metadata(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """메타데이터 로드 및 검증"""
        
        logger.info("=== 메타데이터 로드 및 검증 ===")
        
        # 파일 존재 확인
        file_index_path = self.metadata_dir / 'file_index.csv'
        cv_splits_path = self.metadata_dir / 'cv_splits.json'
        
        if not file_index_path.exists():
            raise FileNotFoundError(f"file_index.csv not found: {file_index_path}")
        
        if not cv_splits_path.exists():
            raise FileNotFoundError(f"cv_splits.json not found: {cv_splits_path}")
        
        # 파일 로드
        logger.info(f"📁 Loading file index: {file_index_path}")
        df = pd.read_csv(file_index_path)
        
        logger.info(f"📁 Loading CV splits: {cv_splits_path}")
        with open(cv_splits_path, 'r', encoding='utf-8') as f:
            cv_splits = json.load(f)
        
        logger.info(f"✅ Loaded {len(df)} files, {len(cv_splits)} folds")
        
        # 무결성 검사
        self._validate_data_integrity(df, cv_splits)
        
        return df, cv_splits
    
    def _validate_data_integrity(self, df: pd.DataFrame, cv_splits: List[Dict]):
        """데이터 무결성 검사"""
        
        logger.info("🔍 데이터 무결성 검사 중...")
        
        issues = []
        
        # 1. PKL 파일 존재 확인
        missing_files = []
        for _, row in df.iterrows():
            pkl_path = self.pkl_dir / row['file_name']
            if not pkl_path.exists():
                missing_files.append(row['file_name'])
        
        if missing_files:
            issues.append(f"Missing PKL files: {len(missing_files)} files")
            if len(missing_files) <= 5:
                issues.append(f"  Examples: {missing_files}")
        
        # 2. CV splits 파일 중복 확인
        all_files_in_splits = set()
        duplicate_files = []
        
        for fold in cv_splits:
            train_files = set(fold['train_files'])
            val_files = set(fold['val_files'])
            
            # 같은 fold 내 중복
            overlap = train_files & val_files
            if overlap:
                issues.append(f"Fold {fold['fold']}: train-val overlap: {len(overlap)} files")
            
            # 전체 중복
            for f in train_files | val_files:
                if f in all_files_in_splits:
                    duplicate_files.append(f)
                all_files_in_splits.add(f)
        
        if duplicate_files:
            issues.append(f"Duplicate files across folds: {len(set(duplicate_files))} files")
        
        # 3. 피험자 누수 확인
        for fold in cv_splits:
            test_subject = fold['test_subject']
            train_subjects = set(fold['train_subjects'])
            
            if test_subject in train_subjects:
                issues.append(f"Fold {fold['fold']}: subject leak: {test_subject} in both train and test")
        
        # 4. 파일 커버리지 확인
        df_files = set(df['file_name'].tolist())
        split_files = all_files_in_splits
        
        missing_in_splits = df_files - split_files
        extra_in_splits = split_files - df_files
        
        if missing_in_splits:
            issues.append(f"Files in index but not in splits: {len(missing_in_splits)}")
        
        if extra_in_splits:
            issues.append(f"Files in splits but not in index: {len(extra_in_splits)}")
        
        # 결과 처리
        if issues:
            logger.warning(f"❌ 무결성 검사에서 {len(issues)}개 이슈 발견:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            if self.strict_mode:
                raise ValueError(f"Strict mode: Data integrity check failed with {len(issues)} issues")
            else:
                logger.warning("⚠️ 비엄격 모드: 이슈가 있지만 계속 진행")
        else:
            logger.info("✅ 무결성 검사 통과")
    
    def initialize_data_generator(self):
        """데이터 제너레이터 초기화"""
        
        logger.info("📊 데이터 제너레이터 초기화 중...")
        
        self.generator = RaggedStrideDataGenerator(
            pkl_dir=str(self.pkl_dir),
            metadata_dir=str(self.metadata_dir)
        )
        
        if self.generator.cv_splits is None:
            raise ValueError("Failed to load CV splits in data generator")
        
        if self.generator.normalization_stats is None:
            raise ValueError("Failed to load normalization stats in data generator")
        
        logger.info(f"✅ 데이터 제너레이터 초기화 완료")
        logger.info(f"   CV folds: {len(self.generator.cv_splits)}")
        logger.info(f"   Normalization stats: ✓")
        logger.info(f"   Auxiliary stats: {'✓' if self.generator.auxiliary_stats else '✗'}")
    
    def train_single_fold(self, 
                         fold_idx: int,
                         model_config: Dict,
                         training_config: Dict) -> Dict:
        """단일 fold 학습"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1} 학습 시작")
        logger.info(f"{'='*60}")
        
        fold_start_time = datetime.now()
        
        # 데이터셋 생성
        logger.info("📊 데이터셋 생성 중...")
        train_ds, val_ds, fold_info = self.generator.get_fold_datasets(
            fold_idx=fold_idx,
            batch_size=training_config['batch_size'],
            normalize=True
        )
        
        logger.info(f"✅ 데이터셋 생성 완료")
        logger.info(f"   Test subject: {fold_info['test_subject']}")
        logger.info(f"   Train cycles: {fold_info['n_train_cycles']:,}")
        logger.info(f"   Val cycles: {fold_info['n_val_cycles']:,}")
        
        # 모델 생성
        logger.info("🧠 모델 생성 중...")
        model = create_tcn_model(**model_config)
        
        logger.info(f"✅ 모델 생성 완료")
        logger.info(f"   총 파라미터: {model.count_params():,}")
        
        # 콜백 함수 준비
        model_save_path = self.models_dir / f"best_fold_{fold_idx + 1}.keras"
        callbacks = get_model_callbacks(
            patience_early=training_config['patience_early'],
            patience_lr=training_config['patience_lr'],
            lr_factor=training_config['lr_factor'],
            min_lr=training_config['min_lr'],
            monitor='val_loss',
            save_best_path=str(model_save_path)
        )
        
        # 학습 실행
        logger.info(f"🏃 학습 시작 (최대 {training_config['epochs']} epochs)...")
        
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 결과 분석
        fold_duration = (datetime.now() - fold_start_time).total_seconds() / 60
        
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = np.min(history.history['val_loss'])
        best_val_mae = history.history['val_mae'][best_epoch - 1]
        
        final_train_loss = history.history['loss'][-1]
        final_train_mae = history.history['mae'][-1]
        
        fold_result = {
            'fold': fold_idx + 1,
            'test_subject': fold_info['test_subject'],
            'train_subjects': fold_info['train_subjects'],
            'n_train_cycles': fold_info['n_train_cycles'],
            'n_val_cycles': fold_info['n_val_cycles'],
            'best_epoch': best_epoch,
            'total_epochs': len(history.history['loss']),
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'final_train_loss': final_train_loss,
            'final_train_mae': final_train_mae,
            'duration_minutes': fold_duration,
            'model_path': str(model_save_path)
        }
        
        logger.info(f"✅ Fold {fold_idx + 1} 학습 완료")
        logger.info(f"   최적 epoch: {best_epoch}/{len(history.history['loss'])}")
        logger.info(f"   최적 val MAE: {best_val_mae:.4f}")
        logger.info(f"   최종 train MAE: {final_train_mae:.4f}")
        logger.info(f"   학습 시간: {fold_duration:.1f}분")
        logger.info(f"   모델 저장: {model_save_path}")
        
        # 히스토리 저장
        history_path = self.logs_dir / f"fold_{fold_idx + 1}_history.json"
        history_dict = {k: [float(v) for v in values] for k, values in history.history.items()}
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        return fold_result
    
    def run_cross_validation(self,
                           model_config: Optional[Dict] = None,
                           training_config: Optional[Dict] = None) -> Dict:
        """교차검증 실행"""
        
        logger.info("\n" + "🚀" + "="*58 + "🚀")
        logger.info("TCN 교차검증 학습 시작")
        logger.info("🚀" + "="*58 + "🚀")
        
        cv_start_time = datetime.now()
        
        # 기본 설정
        if model_config is None:
            model_config = {
                'tcn_filters': 64,
                'tcn_stacks': 4,
                'dropout_rate': 0.1,
                'dense_units': 64,
                'learning_rate': 1e-3
            }
        
        if training_config is None:
            training_config = {
                'epochs': 100,
                'batch_size': 32,
                'patience_early': 10,
                'patience_lr': 5,
                'lr_factor': 0.5,
                'min_lr': 1e-6
            }
        
        logger.info(f"📋 모델 설정: {model_config}")
        logger.info(f"📋 학습 설정: {training_config}")
        
        # 메타데이터 로드 및 검증
        df, cv_splits = self.load_and_validate_metadata()
        
        # 데이터 제너레이터 초기화
        self.initialize_data_generator()
        
        # 각 fold 학습
        self.cv_results = []
        n_folds = len(cv_splits)
        
        for fold_idx in range(n_folds):
            try:
                fold_result = self.train_single_fold(
                    fold_idx=fold_idx,
                    model_config=model_config,
                    training_config=training_config
                )
                self.cv_results.append(fold_result)
                
            except Exception as e:
                logger.error(f"❌ Fold {fold_idx + 1} 학습 실패: {str(e)}")
                if self.strict_mode:
                    raise
                else:
                    logger.warning(f"⚠️ 비엄격 모드: Fold {fold_idx + 1} 건너뛰고 계속 진행")
                    continue
        
        # 교차검증 결과 분석
        cv_duration = (datetime.now() - cv_start_time).total_seconds() / 60
        cv_summary = self._analyze_cv_results(cv_duration)
        
        # 결과 리포트 생성
        self._generate_cv_report(cv_summary, model_config, training_config)
        
        logger.info(f"\n🎉 교차검증 완료!")
        logger.info(f"   완료된 folds: {len(self.cv_results)}/{n_folds}")
        logger.info(f"   평균 val MAE: {cv_summary['mean_val_mae']:.4f} ± {cv_summary['std_val_mae']:.4f}")
        logger.info(f"   총 소요 시간: {cv_duration:.1f}분")
        
        return cv_summary
    
    def _analyze_cv_results(self, cv_duration: float) -> Dict:
        """교차검증 결과 분석"""
        
        if not self.cv_results:
            raise ValueError("No fold results to analyze")
        
        # 메트릭 수집
        val_maes = [r['best_val_mae'] for r in self.cv_results]
        val_losses = [r['best_val_loss'] for r in self.cv_results]
        train_maes = [r['final_train_mae'] for r in self.cv_results]
        
        # 통계 계산
        summary = {
            'n_completed_folds': len(self.cv_results),
            'mean_val_mae': np.mean(val_maes),
            'std_val_mae': np.std(val_maes),
            'min_val_mae': np.min(val_maes),
            'max_val_mae': np.max(val_maes),
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_train_mae': np.mean(train_maes),
            'std_train_mae': np.std(train_maes),
            'total_duration_minutes': cv_duration,
            'avg_duration_per_fold': cv_duration / len(self.cv_results),
            'fold_results': self.cv_results
        }
        
        return summary
    
    def _generate_cv_report(self, cv_summary: Dict, model_config: Dict, training_config: Dict):
        """교차검증 결과 리포트 생성"""
        
        report_path = self.models_dir / 'cv_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("TCN 교차검증 결과 리포트\n")
            f.write("="*80 + "\n")
            f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 모델 및 학습 설정
            f.write("📋 모델 설정:\n")
            for key, value in model_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("📋 학습 설정:\n")
            for key, value in training_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # 전체 결과 요약
            f.write("📊 전체 결과 요약:\n")
            f.write(f"  완료된 folds: {cv_summary['n_completed_folds']}\n")
            f.write(f"  평균 val MAE: {cv_summary['mean_val_mae']:.4f} ± {cv_summary['std_val_mae']:.4f}\n")
            f.write(f"  최소 val MAE: {cv_summary['min_val_mae']:.4f}\n")
            f.write(f"  최대 val MAE: {cv_summary['max_val_mae']:.4f}\n")
            f.write(f"  평균 train MAE: {cv_summary['mean_train_mae']:.4f} ± {cv_summary['std_train_mae']:.4f}\n")
            f.write(f"  총 소요 시간: {cv_summary['total_duration_minutes']:.1f}분\n")
            f.write(f"  fold당 평균 시간: {cv_summary['avg_duration_per_fold']:.1f}분\n\n")
            
            # Fold별 상세 결과
            f.write("📁 Fold별 상세 결과:\n")
            f.write("-" * 120 + "\n")
            f.write(f"{'Fold':<4} {'Test Subject':<12} {'Train':<6} {'Val':<6} {'Best Epoch':<10} {'Val MAE':<8} {'Train MAE':<9} {'Duration':<8} {'Model Path':<30}\n")
            f.write("-" * 120 + "\n")
            
            for result in cv_summary['fold_results']:
                f.write(f"{result['fold']:<4} "
                       f"{result['test_subject']:<12} "
                       f"{result['n_train_cycles']:<6} "
                       f"{result['n_val_cycles']:<6} "
                       f"{result['best_epoch']:<10} "
                       f"{result['best_val_mae']:<8.4f} "
                       f"{result['final_train_mae']:<9.4f} "
                       f"{result['duration_minutes']:<8.1f} "
                       f"{Path(result['model_path']).name:<30}\n")
            
            f.write("-" * 120 + "\n")
        
        logger.info(f"📄 교차검증 리포트 저장: {report_path}")
    
    def retrain_final_model(self,
                           model_config: Optional[Dict] = None,
                           training_config: Optional[Dict] = None) -> str:
        """전체 데이터로 최종 모델 재훈련"""
        
        logger.info("\n" + "🔄" + "="*58 + "🔄")
        logger.info("전체 데이터로 최종 모델 재훈련")
        logger.info("🔄" + "="*58 + "🔄")
        
        if model_config is None:
            model_config = {
                'tcn_filters': 64,
                'tcn_stacks': 4,
                'dropout_rate': 0.1,
                'dense_units': 64,
                'learning_rate': 1e-3
            }
        
        if training_config is None:
            training_config = {
                'epochs': 50,  # 재훈련은 더 적은 epoch
                'batch_size': 32,
                'patience_early': 10,
                'patience_lr': 5,
                'lr_factor': 0.5,
                'min_lr': 1e-6
            }
        
        # 전체 데이터셋 생성 (train + val 합침)
        logger.info("📊 전체 데이터셋 생성 중...")
        
        # 모든 fold의 train + val 데이터를 합쳐서 사용
        all_train_data = []
        all_val_data = []
        
        for fold_idx in range(len(self.generator.cv_splits)):
            train_ds, val_ds, _ = self.generator.get_fold_datasets(
                fold_idx=fold_idx,
                batch_size=training_config['batch_size'],
                normalize=True
            )
            all_train_data.append(train_ds)
            all_val_data.append(val_ds)
        
        # 데이터셋 합치기 (첫 번째 fold를 전체 데이터로 사용)
        full_train_ds, _, fold_info = self.generator.get_fold_datasets(
            fold_idx=0,
            batch_size=training_config['batch_size'],
            normalize=True
        )
        
        total_cycles = sum(len(self.generator.cv_splits[i]['train_files']) + 
                          len(self.generator.cv_splits[i]['val_files']) 
                          for i in range(len(self.generator.cv_splits)))
        
        logger.info(f"✅ 전체 데이터셋 준비 완료")
        logger.info(f"   총 cycles: 약 {total_cycles:,}")
        
        # 모델 생성
        logger.info("🧠 최종 모델 생성 중...")
        final_model = create_tcn_model(**model_config)
        
        # 콜백 함수 (validation 없이)
        final_model_path = self.models_dir / "final_model.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(final_model_path),
                save_best_only=False,
                verbose=1
            )
        ]
        
        # 재훈련 실행
        logger.info(f"🏃 최종 모델 재훈련 시작...")
        
        history = final_model.fit(
            full_train_ds,
            epochs=training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"✅ 최종 모델 재훈련 완료")
        logger.info(f"   최종 loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"   최종 MAE: {history.history['mae'][-1]:.4f}")
        logger.info(f"   모델 저장: {final_model_path}")
        
        return str(final_model_path)

def parse_arguments():
    """명령행 인자 파싱"""
    
    parser = argparse.ArgumentParser(description='TCN 교차검증 트레이너')
    
    # 디렉토리 설정
    parser.add_argument('--metadata_dir', default='metadata', help='메타데이터 디렉토리')
    parser.add_argument('--pkl_dir', default='stride_train_data_pkl', help='PKL 파일 디렉토리')
    parser.add_argument('--models_dir', default='models', help='모델 저장 디렉토리')
    parser.add_argument('--logs_dir', default='logs', help='로그 저장 디렉토리')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--tcn_filters', type=int, default=64, help='TCN 필터 수')
    parser.add_argument('--tcn_stacks', type=int, default=4, help='TCN 스택 수')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='드롭아웃 비율')
    parser.add_argument('--dense_units', type=int, default=64, help='Dense 레이어 유닛 수')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='학습률')
    
    # 학습 설정
    parser.add_argument('--epochs', type=int, default=100, help='최대 epoch 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--patience_early', type=int, default=10, help='EarlyStopping patience')
    parser.add_argument('--patience_lr', type=int, default=5, help='ReduceLROnPlateau patience')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='학습률 감소 비율')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='최소 학습률')
    
    # 기타 옵션
    parser.add_argument('--strict_mode', action='store_true', help='엄격 모드 (기본값: False)')
    parser.add_argument('--full_retrain', action='store_true', help='전체 재훈련 수행')
    parser.add_argument('--export_tflite', action='store_true', help='TensorFlow Lite 변환')
    
    return parser.parse_args()

def main():
    """메인 실행 함수"""
    
    args = parse_arguments()
    
    # 트레이너 초기화
    trainer = TCNTrainer(
        metadata_dir=args.metadata_dir,
        pkl_dir=args.pkl_dir,
        models_dir=args.models_dir,
        logs_dir=args.logs_dir,
        strict_mode=args.strict_mode
    )
    
    # 모델 설정
    model_config = {
        'tcn_filters': args.tcn_filters,
        'tcn_stacks': args.tcn_stacks,
        'dropout_rate': args.dropout_rate,
        'dense_units': args.dense_units,
        'learning_rate': args.learning_rate
    }
    
    # 학습 설정
    training_config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patience_early': args.patience_early,
        'patience_lr': args.patience_lr,
        'lr_factor': args.lr_factor,
        'min_lr': args.min_lr
    }
    
    try:
        # 교차검증 실행
        cv_summary = trainer.run_cross_validation(
            model_config=model_config,
            training_config=training_config
        )
        
        # 전체 재훈련 (선택적)
        if args.full_retrain:
            final_model_path = trainer.retrain_final_model(
                model_config=model_config,
                training_config=training_config
            )
            logger.info(f"🎯 최종 모델 저장: {final_model_path}")
        
        # TensorFlow Lite 변환 (선택적)
        if args.export_tflite:
            logger.info("🔄 TensorFlow Lite 변환 중...")
            # TODO: TFLite 변환 구현
            logger.info("⚠️ TensorFlow Lite 변환은 아직 구현되지 않았습니다")
        
        logger.info("\n🎉 모든 작업 완료!")
        
    except Exception as e:
        logger.error(f"❌ 실행 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()