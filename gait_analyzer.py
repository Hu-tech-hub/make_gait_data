"""
보행 분석 메인 실행 스크립트
- MediaPipe 기반 보행 분석 파이프라인 실행
- 비디오에서 관절 데이터 추출 후 보행 이벤트(HS/TO) 검출
- 4단계 분석 프로세스를 순차적으로 실행
"""

# 표준 라이브러리 import
import logging
import os
import sys
from datetime import datetime
import argparse
import time

# 로컬 모듈 import - GaitAnalyzer 클래스 가져오기
from gait_class import GaitAnalyzer

# 로깅 설정 - 분석 과정 추적을 위한 로그 구성
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gait_analysis.log'),  # 파일 로그
        logging.StreamHandler(sys.stdout)          # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

def get_next_output_directory(base_dir: str = "./gait_analysis_output") -> str:
    """
    번호가 매겨진 출력 디렉토리 생성
    
    Args:
        base_dir: 기본 출력 디렉토리 경로
        
    Returns:
        str: 사용 가능한 번호가 매겨진 디렉토리 경로 (예: ./gait_analysis_output/output(1))
    """
    # 기본 디렉토리가 없으면 생성
    os.makedirs(base_dir, exist_ok=True)
    
    # 번호가 매겨진 하위 디렉토리 찾기
    counter = 1
    while True:
        numbered_dir = os.path.join(base_dir, f"output({counter})")
        
        # 해당 번호의 디렉토리가 존재하지 않으면 생성하고 반환
        if not os.path.exists(numbered_dir):
            os.makedirs(numbered_dir, exist_ok=True)
            logger.info(f"새 출력 디렉토리 생성: {numbered_dir}")
            return numbered_dir
        
        # 존재하면 다음 번호로 증가
        counter += 1
        
        # 안전장치: 1000개 이상은 방지
        if counter > 1000:
            raise RuntimeError("출력 디렉토리 번호가 1000을 초과했습니다. 기존 폴더를 정리해주세요.")

def main():
    """
    메인 실행 함수
    
    사용법:
    python gait_analyzer.py [--normal-mode] [--video-path path/to/video.mp4]
    
    --normal-mode: 일반 연산 모드 (고정밀도, 기본값은 고속모드)
    --video-path: 분석할 비디오 파일 경로
    """
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='보행 분석 프로그램 (기본값: 고속모드)')
    parser.add_argument('--normal-mode', action='store_true', 
                       help='일반 연산 모드 활성화 (고정밀도, 기본값은 고속모드)')
    parser.add_argument('--video-path', type=str, 
                       default="experiment_data/normal_gait/session_20250604_210219/video.mp4",
                       help='분석할 비디오 파일 경로')
    
    args = parser.parse_args()
    
    # 비디오 파일 경로
    video_path = args.video_path
    
    # 고속 모드 설정 (기본값: True, --normal-mode 옵션으로 False)
    enable_fast_mode = not args.normal_mode
    
    # 출력 디렉토리 자동 생성
    base_output_dir = "gait_analysis_output"
    output_dir = get_next_output_directory(base_output_dir)
    
    logger.info("=" * 80)
    logger.info("보행 분석 시스템 시작")
    logger.info("=" * 80)
    logger.info(f"입력 비디오: {video_path}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"연산 모드: {'⚡ 고속 모드 (좌표 3자리, 각도 5자리)' if enable_fast_mode else '🔬 일반 모드 (고정밀도)'}")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 비디오 파일 존재 확인
    if not os.path.exists(video_path):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        return
    
    # GaitAnalyzer 인스턴스 생성 (고속 모드 설정)
    analyzer = GaitAnalyzer(video_path, output_dir, enable_fast_mode=enable_fast_mode)
    
    try:
        # 전체 실행 시간 측정
        total_start_time = time.time()
        
        # === Step 1: 비디오 데이터 준비 ===
        logger.info("=== Step 1: 비디오 데이터 준비 ===")
        step1_start = time.time()
        frame_mapping = analyzer.step1_prepare_video_data()
        step1_time = time.time() - step1_start
        logger.info(f"프레임 매핑 완료: {len(frame_mapping)} 프레임 (소요시간: {step1_time:.2f}초)")
        
        # === Step 2: 관절 시계열 신호 추출 ===
        logger.info("\n=== Step 2: 관절 시계열 신호 추출 ===")
        step2_start = time.time()
        joint_data = analyzer.step2_extract_joint_signals()
        step2_time = time.time() - step2_start
        logger.info(f"관절 데이터 추출 완료: {joint_data.shape} (소요시간: {step2_time:.2f}초)")
        
        # === Step 3: 보행 이벤트 검출 ===
        logger.info("\n=== Step 3: 보행 이벤트 검출 ===")
        step3_start = time.time()
        events = analyzer.step3_detect_gait_events()
        step3_time = time.time() - step3_start
        logger.info(f"검출된 이벤트 수: {len(events)} (소요시간: {step3_time:.2f}초)")
        
        # === Step 4: 시각화 및 데이터 구조화 ===
        logger.info("\n=== Step 4: 시각화 및 데이터 구조화 ===")
        step4_start = time.time()
        analyzer.step4_visualize_and_export()
        step4_time = time.time() - step4_start
        logger.info(f"시각화 및 내보내기 완료 (소요시간: {step4_time:.2f}초)")
        
        # 전체 실행 시간
        total_time = time.time() - total_start_time
        
        # === 성능 요약 ===
        logger.info("\n=== 성능 요약 ===")
        logger.info(f"Step 1 (데이터 준비): {step1_time:.2f}초")
        logger.info(f"Step 2 (관절 추출): {step2_time:.2f}초 {'(고속 모드)' if enable_fast_mode else '(일반 모드)'}")
        logger.info(f"Step 3 (이벤트 검출): {step3_time:.2f}초")
        logger.info(f"Step 4 (시각화): {step4_time:.2f}초")
        logger.info(f"전체 실행 시간: {total_time:.2f}초")
        logger.info(f"초당 프레임 처리: {len(frame_mapping) / total_time:.1f} FPS")
        logger.info(f"연산 모드: {'⚡ 고속 모드 (좌표 3자리, 각도 5자리)' if enable_fast_mode else '🔬 일반 모드 (고정밀도)'}")
        
        # === 분석 완료 메시지 ===
        logger.info("\n=== 보행 분석 완료 ===")
        logger.info(f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"모든 결과 파일이 저장되었습니다:")
        logger.info(f"📁 출력 디렉토리: {output_dir}")
        
        # 생성된 주요 파일 목록 출력
        output_files = [
            "frame_timestamp_mapping.csv",    # 프레임-타임스탬프 매핑
            "joint_time_series.csv",          # 관절 시계열 데이터
            "gait_events.csv",                # 검출된 보행 이벤트
            "gait_events_plot.png",           # 이벤트 검출 시각화
            "gait_analysis_overlay.mp4",      # 스켈레톤 오버레이 비디오
            "gait_analysis_complete.csv",     # 통합 분석 데이터
            "analysis_summary.json"           # 분석 요약 통계
        ]
        
        logger.info("\n📄 생성된 파일 목록:")
        success_count = 0
        for file in output_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                logger.info(f"  ✅ {file} ({file_size:,} bytes)")
                success_count += 1
            else:
                logger.warning(f"  ❌ {file} (생성되지 않음)")
        
        logger.info(f"\n📊 완료 요약: {success_count}/{len(output_files)} 파일 생성 성공")
        logger.info(f"🗂️  결과 폴더 경로: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        # 오류 발생 시 상세 정보 출력 및 디버깅 도움말 제공
        logger.error(f"분석 중 오류 발생: {str(e)}")
        logger.error("오류 해결 방법:")
        logger.error("1. 비디오 파일 경로가 올바른지 확인")
        logger.error("2. 필요한 Python 패키지가 모두 설치되었는지 확인")
        logger.error("3. 비디오 파일이 손상되지 않았는지 확인")
        logger.error("4. 충분한 디스크 공간이 있는지 확인")
        raise

def check_dependencies():
    """
    필수 라이브러리 설치 여부 확인
    분석 실행 전 의존성 체크
    """
    required_packages = [
        'cv2', 'mediapipe', 'numpy', 'pandas', 
        'scipy', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} 설치됨")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} 설치 필요")
    
    if missing_packages:
        logger.error("다음 패키지를 설치해주세요:")
        logger.error("pip install opencv-python mediapipe numpy pandas scipy matplotlib")
        return False
    
    return True

if __name__ == "__main__":
    """
    스크립트 직접 실행 시 진입점
    """
    print("=" * 60)
    print("보행 분석 시스템 v1.0")
    print("MediaPipe 기반 보행 이벤트 검출")
    print("=" * 60)
    
    # 의존성 확인
    logger.info("필수 라이브러리 확인 중...")
    if not check_dependencies():
        sys.exit(1)
    
    # 메인 분석 실행
    main()