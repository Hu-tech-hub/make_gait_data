#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMU 데이터와 보행 분석 결과 융합 시스템

make_data.py에서 수집한 데이터와 gait_analyzer.py의 분석 결과를 결합하여
통합된 멀티모달 보행 분석 데이터를 생성합니다.

주요 기능:
1. experiment_data 디렉토리에서 세션 데이터 자동 스캔
2. 각 세션의 비디오에 대해 보행 분석 수행
3. IMU 데이터와 보행 이벤트를 시간/프레임 기준으로 동기화
4. 통합 분석 결과 CSV 생성

사용법:
    python data_fusion.py [--input-dir experiment_data] [--output-dir fusion_results]
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# 로컬 모듈 import
from gait_class import GaitAnalyzer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fusion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataFusionProcessor:
    """IMU와 보행 분석 데이터 융합 처리기"""
    
    def __init__(self, input_dir: str = "experiment_data", output_dir: str = "fusion_results"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 결과 저장용
        self.fusion_results = []
        
    def scan_sessions(self) -> List[Path]:
        """experiment_data 디렉토리에서 세션 폴더들을 스캔"""
        if not self.input_dir.exists():
            logger.error(f"입력 디렉토리가 존재하지 않습니다: {self.input_dir}")
            return []
        
        session_dirs = []
        for item in self.input_dir.iterdir():
            if item.is_dir() and item.name.startswith('session_'):
                # 필수 파일들이 존재하는지 확인
                video_file = item / "video.mp4"
                imu_file = item / "imu_data.csv"
                metadata_file = item / "metadata.json"
                
                if all([video_file.exists(), imu_file.exists(), metadata_file.exists()]):
                    session_dirs.append(item)
                    logger.info(f"유효한 세션 발견: {item.name}")
                else:
                    missing_files = []
                    if not video_file.exists(): missing_files.append("video.mp4")
                    if not imu_file.exists(): missing_files.append("imu_data.csv")
                    if not metadata_file.exists(): missing_files.append("metadata.json")
                    logger.warning(f"세션 {item.name}에서 누락된 파일: {', '.join(missing_files)}")
        
        session_dirs.sort()  # 시간순 정렬
        logger.info(f"총 {len(session_dirs)}개의 유효한 세션을 발견했습니다.")
        return session_dirs
    
    def load_session_data(self, session_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """세션 데이터 로드 (IMU + 메타데이터)"""
        try:
            # IMU 데이터 로드
            imu_file = session_dir / "imu_data.csv"
            imu_data = pd.read_csv(imu_file)
            logger.info(f"IMU 데이터 로드: {len(imu_data)} 샘플")
            
            # 메타데이터 로드
            metadata_file = session_dir / "metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return imu_data, metadata
            
        except Exception as e:
            logger.error(f"세션 데이터 로드 실패 ({session_dir.name}): {e}")
            return None, None
    
    def analyze_gait(self, session_dir: Path) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """비디오에 대해 보행 분석 수행"""
        try:
            video_file = session_dir / "video.mp4"
            
            # 임시 출력 디렉토리 생성
            temp_output = self.output_dir / f"{session_dir.name}_gait_temp"
            temp_output.mkdir(exist_ok=True)
            
            # GaitAnalyzer로 분석 수행 (고속모드)
            logger.info(f"보행 분석 시작: {session_dir.name}")
            analyzer = GaitAnalyzer(str(video_file), str(temp_output), enable_fast_mode=True)
            
            # 4단계 분석 수행
            frame_mapping = analyzer.step1_prepare_video_data()
            joint_data = analyzer.step2_extract_joint_signals()
            events_data = analyzer.step3_detect_gait_events()
            analyzer.step4_visualize_and_export()
            
            logger.info(f"보행 분석 완료: 관절 데이터 {joint_data.shape}, 이벤트 {len(events_data)}")
            
            return joint_data, events_data
            
        except Exception as e:
            logger.error(f"보행 분석 실패 ({session_dir.name}): {e}")
            return None, None
    
    def synchronize_data(self, imu_data: pd.DataFrame, gait_data: pd.DataFrame, 
                        events_data: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
        """IMU 데이터와 보행 분석 결과를 동기화"""
        logger.info("데이터 동기화 시작...")
        
        # 기본 프레임 정보로 시작 (gait_data 기준)
        synchronized_data = gait_data.copy()
        
        # IMU 데이터 매핑 (frame_number 기준)
        imu_dict = {}
        for _, row in imu_data.iterrows():
            frame_num = int(row['frame_number'])
            imu_dict[frame_num] = {
                'imu_accel_x': row['accel_x'],
                'imu_accel_y': row['accel_y'], 
                'imu_accel_z': row['accel_z'],
                'imu_gyro_x': row['gyro_x'],
                'imu_gyro_y': row['gyro_y'],
                'imu_gyro_z': row['gyro_z'],
                'imu_sync_timestamp': row['sync_timestamp']
            }
        
        # IMU 데이터를 gait_data에 추가
        imu_columns = ['imu_accel_x', 'imu_accel_y', 'imu_accel_z', 
                      'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z', 'imu_sync_timestamp']
        
        for col in imu_columns:
            synchronized_data[col] = np.nan
        
        for idx, row in synchronized_data.iterrows():
            frame_idx = int(row['frame_idx'])
            if frame_idx in imu_dict:
                for col in imu_columns:
                    synchronized_data.loc[idx, col] = imu_dict[frame_idx][col]
        
        # 보행 이벤트 정보 추가
        synchronized_data['gait_event'] = ''
        synchronized_data['gait_event_details'] = ''
        
        for _, event in events_data.iterrows():
            event_frame = int(event['frame_idx'])
            event_type = event['event_type']
            
            # 해당 프레임에 이벤트 정보 추가
            mask = synchronized_data['frame_idx'] == event_frame
            if mask.any():
                current_events = synchronized_data.loc[mask, 'gait_event'].iloc[0]
                if current_events:
                    synchronized_data.loc[mask, 'gait_event'] = current_events + ',' + event_type
                else:
                    synchronized_data.loc[mask, 'gait_event'] = event_type
                
                # 상세 정보 추가
                details = f"{event_type}(ankle_x:{event['ankle_x']:.3f},ankle_y:{event['ankle_y']:.3f})"
                current_details = synchronized_data.loc[mask, 'gait_event_details'].iloc[0]
                if current_details:
                    synchronized_data.loc[mask, 'gait_event_details'] = current_details + ';' + details
                else:
                    synchronized_data.loc[mask, 'gait_event_details'] = details
        
        # 메타데이터 정보 추가
        synchronized_data['session_id'] = metadata['session_id']
        synchronized_data['session_duration'] = metadata['duration']
        synchronized_data['video_fps'] = metadata['video_fps']
        synchronized_data['imu_hz'] = metadata['imu_hz']
        
        # IMU 데이터가 있는 행만 필터링 (옵션)
        valid_imu_mask = synchronized_data['imu_sync_timestamp'].notna()
        valid_data_count = valid_imu_mask.sum()
        total_data_count = len(synchronized_data)
        
        logger.info(f"동기화 완료: 전체 {total_data_count} 프레임 중 {valid_data_count} 프레임에 IMU 데이터 매핑")
        
        return synchronized_data
    
    def calculate_multimodal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """멀티모달 특성 계산"""
        logger.info("멀티모달 특성 계산 중...")
        
        # IMU 기반 특성
        if 'imu_accel_x' in data.columns:
            # 가속도 크기 (벡터 크기)
            data['imu_accel_magnitude'] = np.sqrt(
                data['imu_accel_x']**2 + data['imu_accel_y']**2 + data['imu_accel_z']**2
            )
            
            # 자이로 크기
            data['imu_gyro_magnitude'] = np.sqrt(
                data['imu_gyro_x']**2 + data['imu_gyro_y']**2 + data['imu_gyro_z']**2
            )
            
            # 보행 방향 추정 (주 움직임 축)
            data['imu_walking_axis'] = np.argmax(
                np.abs([data['imu_accel_x'], data['imu_accel_y'], data['imu_accel_z']]), axis=0
            )
        
        # 비전 기반 특성과 IMU 상관관계
        if all(col in data.columns for col in ['left_ankle_x', 'imu_accel_x']):
            # 발목 움직임과 IMU 가속도 상관관계 (윈도우별)
            window_size = 30  # 1초 윈도우 (30fps)
            correlations = []
            
            for i in range(len(data)):
                start_idx = max(0, i - window_size//2)
                end_idx = min(len(data), i + window_size//2)
                
                if end_idx - start_idx > 10:  # 최소 데이터 포인트
                    window_data = data.iloc[start_idx:end_idx]
                    
                    # 발목 x 움직임과 IMU x 가속도 상관관계
                    ankle_x = window_data['left_ankle_x'].dropna()
                    imu_x = window_data['imu_accel_x'].dropna()
                    
                    if len(ankle_x) > 5 and len(imu_x) > 5:
                        # 길이를 맞춰서 상관관계 계산
                        min_len = min(len(ankle_x), len(imu_x))
                        if min_len > 5:
                            corr = np.corrcoef(ankle_x.iloc[:min_len], imu_x.iloc[:min_len])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(corr)
                            else:
                                correlations.append(0.0)
                        else:
                            correlations.append(0.0)
                    else:
                        correlations.append(0.0)
                else:
                    correlations.append(0.0)
            
            data['vision_imu_correlation'] = correlations
        
        # 이벤트 기반 특성
        data['is_gait_event'] = data['gait_event'] != ''
        data['hs_event_count'] = data['gait_event'].str.contains('HS', na=False).astype(int)
        data['to_event_count'] = data['gait_event'].str.contains('TO', na=False).astype(int)
        
        logger.info("멀티모달 특성 계산 완료")
        return data
    
    def process_session(self, session_dir: Path) -> Optional[pd.DataFrame]:
        """단일 세션 전체 처리"""
        logger.info(f"\n{'='*60}")
        logger.info(f"세션 처리 시작: {session_dir.name}")
        logger.info(f"{'='*60}")
        
        # 1. 세션 데이터 로드
        imu_data, metadata = self.load_session_data(session_dir)
        if imu_data is None or metadata is None:
            logger.error(f"세션 데이터 로드 실패: {session_dir.name}")
            return None
        
        # 2. 보행 분석 수행
        gait_results = self.analyze_gait(session_dir)
        if gait_results is None:
            logger.error(f"보행 분석 실패: {session_dir.name}")
            return None
        
        joint_data, events_data = gait_results
        
        # 3. 데이터 동기화
        synchronized_data = self.synchronize_data(imu_data, joint_data, events_data, metadata)
        
        # 4. 멀티모달 특성 계산
        enriched_data = self.calculate_multimodal_features(synchronized_data)
        
        # 5. 세션별 결과 저장
        session_output_file = self.output_dir / f"{session_dir.name}_fusion_result.csv"
        enriched_data.to_csv(session_output_file, index=False)
        logger.info(f"세션 결과 저장: {session_output_file}")
        
        # 6. 요약 통계 생성
        self.generate_session_summary(enriched_data, session_dir.name, metadata)
        
        return enriched_data
    
    def generate_session_summary(self, data: pd.DataFrame, session_name: str, metadata: Dict):
        """세션별 요약 통계 생성"""
        summary = {
            'session_name': session_name,
            'session_id': metadata['session_id'],
            'duration_seconds': metadata['duration'],
            'total_frames': len(data),
            'valid_imu_frames': data['imu_sync_timestamp'].notna().sum(),
            'total_gait_events': data['is_gait_event'].sum(),
            'hs_events': data['hs_event_count'].sum(),
            'to_events': data['to_event_count'].sum(),
            'avg_ankle_distance': data['ankle_distance'].mean() if 'ankle_distance' in data.columns else None,
            'avg_imu_accel_magnitude': data['imu_accel_magnitude'].mean() if 'imu_accel_magnitude' in data.columns else None,
            'avg_vision_imu_correlation': data['vision_imu_correlation'].mean() if 'vision_imu_correlation' in data.columns else None,
            'data_completeness': (data['imu_sync_timestamp'].notna().sum() / len(data) * 100) if len(data) > 0 else 0
        }
        
        self.fusion_results.append(summary)
        logger.info(f"세션 요약: {summary['total_gait_events']}개 이벤트, "
                   f"{summary['data_completeness']:.1f}% 데이터 완성도")
    
    def process_all_sessions(self):
        """모든 세션 처리"""
        logger.info(f"\n🚀 데이터 융합 처리 시작")
        logger.info(f"입력 디렉토리: {self.input_dir}")
        logger.info(f"출력 디렉토리: {self.output_dir}")
        
        session_dirs = self.scan_sessions()
        if not session_dirs:
            logger.error("처리할 세션이 없습니다.")
            return
        
        total_sessions = len(session_dirs)
        successful_sessions = 0
        
        all_fusion_data = []
        
        for i, session_dir in enumerate(session_dirs, 1):
            logger.info(f"\n📊 진행률: {i}/{total_sessions} ({i/total_sessions*100:.1f}%)")
            
            try:
                fusion_data = self.process_session(session_dir)
                if fusion_data is not None:
                    all_fusion_data.append(fusion_data)
                    successful_sessions += 1
                    logger.info(f"✅ {session_dir.name} 처리 완료")
                else:
                    logger.error(f"❌ {session_dir.name} 처리 실패")
            except Exception as e:
                logger.error(f"❌ {session_dir.name} 처리 중 오류: {e}")
        
        # 전체 통합 결과 생성
        if all_fusion_data:
            self.generate_integrated_results(all_fusion_data)
        
        # 최종 요약
        logger.info(f"\n📈 데이터 융합 완료")
        logger.info(f"총 세션: {total_sessions}")
        logger.info(f"성공한 세션: {successful_sessions}")
        logger.info(f"실패한 세션: {total_sessions - successful_sessions}")
        logger.info(f"성공률: {successful_sessions/total_sessions*100:.1f}%")
    
    def generate_integrated_results(self, all_data: List[pd.DataFrame]):
        """통합 결과 생성"""
        logger.info("통합 결과 생성 중...")
        
        # 1. 전체 데이터 통합
        integrated_data = pd.concat(all_data, ignore_index=True)
        integrated_file = self.output_dir / "integrated_fusion_results.csv"
        integrated_data.to_csv(integrated_file, index=False)
        logger.info(f"통합 데이터 저장: {integrated_file} ({len(integrated_data)} 행)")
        
        # 2. 세션별 요약 저장
        if self.fusion_results:
            summary_df = pd.DataFrame(self.fusion_results)
            summary_file = self.output_dir / "session_summaries.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"세션 요약 저장: {summary_file}")
        
        # 3. 전체 통계 생성
        overall_stats = {
            'total_sessions': len(self.fusion_results),
            'total_frames': len(integrated_data),
            'total_gait_events': integrated_data['is_gait_event'].sum(),
            'total_hs_events': integrated_data['hs_event_count'].sum(),
            'total_to_events': integrated_data['to_event_count'].sum(),
            'avg_data_completeness': np.mean([r['data_completeness'] for r in self.fusion_results]),
            'avg_session_duration': np.mean([r['duration_seconds'] for r in self.fusion_results]),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        stats_file = self.output_dir / "overall_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        logger.info(f"전체 통계 저장: {stats_file}")
        logger.info(f"📊 전체 통계: {overall_stats['total_sessions']}개 세션, "
                   f"{overall_stats['total_gait_events']}개 이벤트, "
                   f"{overall_stats['avg_data_completeness']:.1f}% 평균 완성도")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='IMU-보행분석 데이터 융합 시스템')
    parser.add_argument('--input-dir', type=str, default='experiment_data',
                       help='입력 데이터 디렉토리 (기본값: experiment_data)')
    parser.add_argument('--output-dir', type=str, default='fusion_results',
                       help='출력 결과 디렉토리 (기본값: fusion_results)')
    parser.add_argument('--session', type=str, default=None,
                       help='특정 세션만 처리 (예: session_20250604_210219)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🔗 IMU-보행분석 데이터 융합 시스템")
    logger.info("=" * 80)
    
    # 데이터 융합 프로세서 생성
    processor = DataFusionProcessor(args.input_dir, args.output_dir)
    
    if args.session:
        # 특정 세션만 처리
        session_path = Path(args.input_dir) / args.session
        if session_path.exists():
            logger.info(f"특정 세션 처리: {args.session}")
            processor.process_session(session_path)
        else:
            logger.error(f"세션을 찾을 수 없습니다: {session_path}")
    else:
        # 모든 세션 처리
        processor.process_all_sessions()


if __name__ == "__main__":
    main() 