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
    메인 실행 함수 - 4단계 보행 분석 파이프라인 실행
    
    분석 단계:
    1. 비디오 데이터 준비 및 프레임-타임스탬프 매핑 생성
    2. MediaPipe를 통한 관절 좌표 추출 및 시계열 신호 생성  
    3. 규칙 기반 알고리즘으로 보행 이벤트(HS/TO) 검출
    4. 결과 시각화 및 구조화된 데이터 내보내기
    """
    
    # === 파일 경로 및 출력 디렉토리 설정 ===
    # TODO: 실제 분석할 비디오 파일 경로로 변경 필요
    video_path = "walking_video_2.mp4"  # 입력 비디오 파일 경로
    
    # 비디오 파일 존재 여부 확인
    if not os.path.exists(video_path):
        logger.error(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        logger.info("video_path 변수를 실제 비디오 파일 경로로 수정해주세요.")
        return
    
    # 번호가 매겨진 출력 디렉토리 자동 생성
    output_dir = get_next_output_directory()  # 결과 파일 저장 디렉토리
    
    # === 보행 분석기 초기화 ===
    # GaitAnalyzer 객체 생성 - 모든 분석 기능을 포함하는 메인 클래스
    analyzer = GaitAnalyzer(video_path, output_dir)
    logger.info(f"보행 분석기 초기화 완료")
    logger.info(f"입력 비디오: {video_path}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"분석 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # === Step 1: 비디오 데이터 준비 ===
        # 비디오를 프레임 단위로 분해하고 각 프레임에 타임스탬프 할당
        # 결과: frame_timestamp_mapping.csv 파일 생성
        logger.info("=== Step 1: 비디오 데이터 준비 ===")
        frame_mapping = analyzer.step1_prepare_video_data()
        logger.info(f"프레임 매핑 완료: {len(frame_mapping)} 프레임")
        
        # === Step 2: 관절 시계열 신호 추출 ===
        # MediaPipe로 각 프레임에서 주요 관절 좌표 추출
        # Savitzky-Golay 필터로 노이즈 제거 및 신호 평활화
        # 관절 간 거리, 각도 등 파생 변수 계산
        # 결과: joint_time_series.csv 파일 생성
        logger.info("\n=== Step 2: 관절 시계열 신호 추출 ===")
        joint_data = analyzer.step2_extract_joint_signals()
        logger.info(f"관절 데이터 추출 완료: {joint_data.shape}")
        
        # === Step 3: 보행 이벤트 검출 ===
        # 발목 x좌표 시계열에서 피크 검출 알고리즘 적용 (논문 방법론)
        # HS(Heel Strike): 발목 x축 변위의 피크(최대값) - 발이 앞으로 최대한 나아간 시점
        # TO(Toe Off): 발목 x축 변위의 계곡(최소값) - 발이 뒤로 최대한 당겨진 시점
        # 결과: gait_events.csv 파일 및 시각화 플롯(무릎 관절 각도 포함) 생성
        logger.info("\n=== Step 3: 보행 이벤트 검출 ===")
        events = analyzer.step3_detect_gait_events()
        logger.info(f"검출된 이벤트 수: {len(events)}")
        
        # === Step 4: 시각화 및 데이터 구조화 ===
        # 원본 비디오에 스켈레톤과 이벤트 정보 오버레이
        # 모든 데이터를 통합하여 최종 분석 결과 생성
        # 보행 주기, 보폭 등 요약 통계 계산
        # 결과: 오버레이 비디오, 통합 CSV, 요약 JSON 파일 생성
        logger.info("\n=== Step 4: 시각화 및 데이터 구조화 ===")
        analyzer.step4_visualize_and_export()
        
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