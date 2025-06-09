#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_system.py - 통합 보행 분석 시스템 테스트 스크립트

이 스크립트는 시스템의 주요 기능을 테스트합니다.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

def test_imports():
    """모듈 임포트 테스트"""
    print("1. 모듈 임포트 테스트...")
    
    try:
        from gait_class import GaitAnalyzer, GaitEvent
        print("  ✓ gait_class 모듈")
        
        from gait_metrics_calculator import GaitMetricsCalculator, GaitCycle
        print("  ✓ gait_metrics_calculator 모듈")
        
        from time_series_model import GaitMetricsPredictor, IMUFeatureExtractor
        print("  ✓ time_series_model 모듈")
        
        from data_processing_utils import GaitDatasetBuilder, ModelEvaluator
        print("  ✓ data_processing_utils 모듈")
        
        from integrated_gait_system_gui import IntegratedGaitSystemGUI
        print("  ✓ integrated_gait_system_gui 모듈")
        
        print("  ✅ 모든 모듈 임포트 성공!")
        return True
        
    except ImportError as e:
        print(f"  ❌ 모듈 임포트 실패: {e}")
        return False

def test_dependencies():
    """의존성 라이브러리 테스트"""
    print("\n2. 의존성 라이브러리 테스트...")
    
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('opencv-python', 'cv2'),
        ('mediapipe', 'mp'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('PyQt5', 'PyQt5'),
        ('pyqtgraph', 'pg'),
    ]
    
    success_count = 0
    for package_name, import_name in required_packages:
        try:
            if import_name == 'np':
                import numpy as np
            elif import_name == 'pd':
                import pandas as pd
            elif import_name == 'cv2':
                import cv2
            elif import_name == 'mp':
                import mediapipe as mp
            elif import_name == 'scipy':
                import scipy
            elif import_name == 'sklearn':
                import sklearn
            elif import_name == 'plt':
                import matplotlib.pyplot as plt
            elif import_name == 'sns':
                import seaborn as sns
            elif import_name == 'PyQt5':
                import PyQt5
            elif import_name == 'pg':
                import pyqtgraph as pg
            
            print(f"  ✓ {package_name}")
            success_count += 1
            
        except ImportError:
            print(f"  ❌ {package_name} - 설치 필요")
    
    if success_count == len(required_packages):
        print("  ✅ 모든 의존성 라이브러리 확인!")
        return True
    else:
        print(f"  ⚠ {len(required_packages) - success_count}개 라이브러리 누락")
        return False

def test_data_structure():
    """데이터 구조 테스트"""
    print("\n3. 데이터 구조 테스트...")
    
    # 실험 데이터 디렉토리 확인
    data_dirs = ['experiment_data', 'support_label_data']
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"  ✓ {data_dir} 디렉토리 존재")
            
            # 하위 디렉토리 확인
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"    - 하위 디렉토리: {len(subdirs)}개 ({', '.join(subdirs[:3])}{'...' if len(subdirs) > 3 else ''})")
            
        else:
            print(f"  ❌ {data_dir} 디렉토리 없음")
    
    # 샘플 데이터 확인
    sample_session = None
    if os.path.exists('experiment_data'):
        for root, dirs, files in os.walk('experiment_data'):
            if 'video.mp4' in files and 'imu_data.csv' in files:
                sample_session = root
                break
    
    if sample_session:
        print(f"  ✓ 샘플 세션 발견: {sample_session}")
        
        # 파일 크기 확인
        video_path = os.path.join(sample_session, 'video.mp4')
        imu_path = os.path.join(sample_session, 'imu_data.csv')
        
        video_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        imu_size = os.path.getsize(imu_path) / 1024  # KB
        
        print(f"    - 비디오 크기: {video_size:.1f} MB")
        print(f"    - IMU 데이터 크기: {imu_size:.1f} KB")
        
        return sample_session
    else:
        print("  ❌ 완전한 샘플 세션 없음")
        return None

def test_basic_functionality(sample_session):
    """기본 기능 테스트"""
    print("\n4. 기본 기능 테스트...")
    
    if not sample_session:
        print("  ⚠ 샘플 데이터가 없어 기본 기능 테스트를 건너뜁니다.")
        return False
    
    try:
        # IMU 데이터 로드 테스트
        imu_path = os.path.join(sample_session, 'imu_data.csv')
        imu_data = pd.read_csv(imu_path)
        print(f"  ✓ IMU 데이터 로드: {len(imu_data)} 샘플")
        
        # 비디오 정보 확인 테스트
        import cv2
        video_path = os.path.join(sample_session, 'video.mp4')
        cap = cv2.VideoCapture(video_path)
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  ✓ 비디오 정보: {frame_count} 프레임, {fps} FPS")
            cap.release()
        else:
            print("  ❌ 비디오 파일 열기 실패")
            return False
        
        # MediaPipe 초기화 테스트
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        print("  ✓ MediaPipe 초기화")
        pose.close()
        
        # 클래스 인스턴스 생성 테스트
        from gait_metrics_calculator import GaitMetricsCalculator
        calculator = GaitMetricsCalculator()
        print("  ✓ GaitMetricsCalculator 생성")
        
        from time_series_model import IMUFeatureExtractor
        extractor = IMUFeatureExtractor(window_size=90)
        print("  ✓ IMUFeatureExtractor 생성")
        
        print("  ✅ 기본 기능 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"  ❌ 기본 기능 테스트 실패: {e}")
        return False

def test_gui_components():
    """GUI 컴포넌트 테스트"""
    print("\n5. GUI 컴포넌트 테스트...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        import sys
        
        # QApplication이 이미 존재하는지 확인
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        from integrated_gait_system_gui import (
            DataSynchronizationWidget, 
            GaitMetricsWidget,
            ModelTrainingWidget,
            PredictionVisualizationWidget
        )
        
        # 위젯 생성 테스트
        sync_widget = DataSynchronizationWidget()
        print("  ✓ DataSynchronizationWidget 생성")
        
        metrics_widget = GaitMetricsWidget()
        print("  ✓ GaitMetricsWidget 생성")
        
        training_widget = ModelTrainingWidget()
        print("  ✓ ModelTrainingWidget 생성")
        
        prediction_widget = PredictionVisualizationWidget()
        print("  ✓ PredictionVisualizationWidget 생성")
        
        print("  ✅ GUI 컴포넌트 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"  ❌ GUI 컴포넌트 테스트 실패: {e}")
        return False

def create_test_report(results):
    """테스트 결과 리포트 생성"""
    print("\n" + "=" * 60)
    print("테스트 결과 리포트")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
    
    print(f"\n전체 결과: {passed_tests}/{total_tests} 통과")
    
    if passed_tests == total_tests:
        print("🎉 모든 테스트 통과! 시스템을 사용할 준비가 되었습니다.")
        return True
    else:
        print("⚠ 일부 테스트 실패. 문제를 해결하고 다시 테스트하세요.")
        return False

def main():
    """메인 테스트 함수"""
    print("통합 보행 분석 시스템 테스트 시작")
    print("=" * 60)
    
    # 테스트 실행
    results = {}
    
    # 1. 모듈 임포트 테스트
    results['모듈 임포트'] = test_imports()
    
    # 2. 의존성 라이브러리 테스트
    results['의존성 라이브러리'] = test_dependencies()
    
    # 3. 데이터 구조 테스트
    sample_session = test_data_structure()
    results['데이터 구조'] = sample_session is not None
    
    # 4. 기본 기능 테스트
    results['기본 기능'] = test_basic_functionality(sample_session)
    
    # 5. GUI 컴포넌트 테스트
    results['GUI 컴포넌트'] = test_gui_components()
    
    # 테스트 리포트 생성
    success = create_test_report(results)
    
    if success:
        print("\n다음 단계:")
        print("1. GUI 실행: python integrated_gait_system_gui.py")
        print("2. 예제 파이프라인 실행: python example_pipeline.py --help")
        print("3. 사용 가이드 확인: system_guide.md")
    else:
        print("\n문제 해결:")
        print("1. requirements.txt의 의존성 설치: pip install -r requirements.txt")
        print("2. 데이터 디렉토리 구조 확인")
        print("3. 오류 메시지 확인 및 수정")

if __name__ == "__main__":
    main()