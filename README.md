# 🚶‍♂️ 보행 분석 시스템 (Gait Analysis System)

MediaPipe 기반 보행 이벤트 검출 및 분석 도구

## 📋 프로젝트 개요

이 프로젝트는 비디오에서 사람의 보행을 분석하여 다음과 같은 정보를 추출합니다:

- **보행 이벤트 검출**: Heel Strike (HS), Toe Off (TO)
- **관절 좌표 추출**: MediaPipe Pose를 통한 실시간 관절 위치 추적
- **보행 주기 분석**: 보폭, 보행 속도, 리듬 분석
- **시각화**: 스켈레톤 오버레이 비디오 및 분석 그래프 생성

## 🏗️ 시스템 구조

```
📁 보행 분석 시스템
├── 📄 gait_class.py          # 핵심 분석 클래스 (권장명: gait_analyzer_core.py)
├── 📄 gait_analyzer.py       # 메인 실행 스크립트 (권장명: main_gait_analysis.py)
├── 📄 requirements.txt       # Python 의존성 목록
└── 📄 README.md              # 프로젝트 가이드 (현재 파일)
```

## 🔧 설치 및 환경 설정

### 1. Python 환경 요구사항
- Python 3.8 이상
- pip 패키지 관리자

### 2. 의존성 설치

```bash
# 저장소 클론 (또는 파일 다운로드)
git clone [repository-url]
cd vision_gait

# 가상환경 생성 (권장)
# 방법 1: 시스템 기본 Python 사용
python -m venv vision_gait_env

# 방법 2: 특정 Python 버전 명시 (권장)
# python3.8 -m venv vision_gait_env
# python3.9 -m venv vision_gait_env
# python3.10 -m venv vision_gait_env

# 가상환경 활성화
# Windows:
vision_gait_env\Scripts\activate
# macOS/Linux:
source vision_gait_env/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 핵심 의존성 확인

설치가 완료되면 다음 라이브러리들이 포함됩니다:

- **OpenCV**: 비디오 처리 및 이미지 조작
- **MediaPipe**: Google의 포즈 추정 라이브러리
- **NumPy**: 수치 연산
- **Pandas**: 데이터 조작 및 분석
- **SciPy**: 신호 처리 및 필터링
- **Matplotlib**: 시각화

## 🚀 사용 방법

### 1. 비디오 파일 준비
분석할 보행 비디오를 준비합니다. 지원 형식: MP4, AVI, MOV 등

### 2. 스크립트 실행

```python
# gait_analyzer.py 파일에서 비디오 경로 수정
video_path = "your_walking_video.mp4"  # 실제 파일 경로로 변경

# 스크립트 실행
python gait_analyzer.py
```

### 3. 단계별 분석 과정

프로그램은 4단계로 분석을 진행합니다:

#### 📊 Step 1: 비디오 데이터 준비
- 비디오를 프레임 단위로 분해
- 프레임-타임스탬프 매핑 테이블 생성
- 출력: `frame_timestamp_mapping.csv`

**📅 타임스탬프 정보:**
- `frame_idx`: 프레임 번호 (0부터 시작)
- `timestamp`: 비디오 내 재생 시간 (초)
- `timestamp_ms`: 비디오 내 재생 시간 (밀리초)

#### 🎯 Step 2: 관절 시계열 신호 추출
- MediaPipe로 각 프레임에서 관절 좌표 추출
- Savitzky-Golay 필터로 노이즈 제거
- 관절 간 거리 및 각도 계산
- 출력: `joint_time_series.csv`

#### 🔍 Step 3: 보행 이벤트 검출
- 발목 x좌표 시계열에서 피크 검출 알고리즘 적용 (논문 방법론)
- HS(Heel Strike): 발목 x축 변위의 피크(최대값) - 발이 앞으로 최대한 나아간 시점
- TO(Toe Off): 발목 x축 변위의 계곡(최소값) - 발이 뒤로 최대한 당겨진 시점
- 출력: `gait_events.csv`, `gait_events_plot.png`

#### 🎬 Step 4: 시각화 및 결과 통합
- 스켈레톤 오버레이 비디오 생성
- 모든 데이터 통합 및 요약 통계 계산
- 출력: `gait_analysis_overlay.mp4`, `gait_analysis_complete.csv`, `analysis_summary.json`

## 📁 출력 파일 설명

분석 완료 후 `./gait_analysis_output/output(1)/` 디렉토리에 다음 파일들이 생성됩니다:

**📂 자동 디렉토리 관리:**
- 첫 번째 분석: `./gait_analysis_output/output(1)/`
- 두 번째 분석: `./gait_analysis_output/output(2)/`
- 세 번째 분석: `./gait_analysis_output/output(3)/`
- ... (자동으로 다음 번호 할당)

**📄 생성되는 파일들:**

| 파일명 | 설명 |
|--------|------|
| `frame_timestamp_mapping.csv` | 프레임-타임스탬프 매핑 테이블 |
| `joint_time_series.csv` | 관절 좌표 시계열 데이터 |
| `gait_events.csv` | 검출된 보행 이벤트 목록 |
| `gait_events_plot.png` | 이벤트 검출 시각화 그래프 (x축 변위 + 무릎 관절 각도) |
| `gait_analysis_overlay.mp4` | 스켈레톤 오버레이 비디오 |
| `gait_analysis_complete.csv` | 통합 분석 데이터 |
| `analysis_summary.json` | 분석 요약 통계 |

## 📊 분석 결과 해석

### 보행 이벤트
- **HS (Heel Strike)**: 발뒤꿈치가 지면에 닿는 순간 (x축 변위의 피크)
- **TO (Toe Off)**: 발가락이 지면에서 떨어지는 순간 (x축 변위의 계곡)

### 방법론 배경
이 분석은 측면 촬영 시 발목의 전후(anterior-posterior) 방향 움직임을 기반으로 합니다:
- **x축**: 보행 진행 방향 (전후 방향)
- **피크 시점**: 발이 앞으로 최대한 나아간 후 지면에 닿는 HS
- **계곡 시점**: 발이 뒤로 최대한 당겨진 후 지면에서 떨어지는 TO

### 주요 측정값
- **보행 주기 (Stride Time)**: 한 발의 HS에서 다음 HS까지의 시간
- **보행률 (Cadence)**: 분당 걸음 수
- **관절 각도**: 무릎, 엉덩이 관절의 굴곡/신전 각도

## 🛠️ 커스터마이징

### 파라미터 조정

`gait_class.py`에서 다음 파라미터들을 조정할 수 있습니다:

```python
# 피크 검출 파라미터
prominence=0.02,  # 피크 prominence 임계값
distance=15       # 최소 피크 간격 (프레임)

# 필터링 파라미터
window_length = 11  # Savitzky-Golay 윈도우 크기
polyorder = 3       # 다항식 차수
```

### 관절 추가

`JOINT_INDICES` 딕셔너리에 새로운 관절을 추가할 수 있습니다:

```python
JOINT_INDICES = {
    'custom_joint': 33,  # MediaPipe 관절 인덱스
    # ... 기존 관절들
}
```

## 🔧 문제 해결

### 일반적인 오류들

1. **비디오 파일을 찾을 수 없음**
   - 파일 경로가 정확한지 확인
   - 지원되는 비디오 형식인지 확인

2. **MediaPipe 오류**
   - MediaPipe 버전 확인: `pip show mediapipe`
   - 비디오 해상도가 너무 높은 경우 리사이징 고려

3. **메모리 부족**
   - 긴 비디오의 경우 프레임 샘플링 적용
   - 배치 처리 구현 고려

## 📈 성능 최적화

- **GPU 가속**: CUDA 지원 OpenCV 설치
- **멀티프로세싱**: 프레임 처리 병렬화
- **메모리 관리**: 큰 비디오 파일의 경우 스트리밍 처리

## 📝 라이선스

이 프로젝트는 연구 및 교육 목적으로 제공됩니다.

## 🤝 기여

버그 리포트나 기능 제안은 이슈로 등록해 주세요.

## 📞 지원

질문이나 지원이 필요한 경우 다음을 확인하세요:
1. 이 README 파일
2. 코드 내 주석 및 docstring
3. 로그 파일 (`gait_analysis.log`) 