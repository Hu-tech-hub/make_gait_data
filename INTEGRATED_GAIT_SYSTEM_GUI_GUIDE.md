# Integrated Gait Analysis System GUI Guide

## 개요

Integrated Gait Analysis System GUI는 PyQt5 기반의 통합 보행 분석 시스템으로, 데이터 동기화, 보행 지표 계산, 영상 검증 기능을 하나의 인터페이스에서 제공합니다.

## 시스템 아키텍처

```
integrated_gait_system_gui.py (메인 애플리케이션)
├── main_window.py (메인 윈도우 컨테이너)
├── data_sync_widget.py (데이터 동기화 & 시각화)
├── gait_metrics_widget.py (보행 지표 계산)
├── video_validation_widget.py (영상 & 데이터 검증)
├── gait_param_class.py (공통 유틸리티 & 설정)
├── gait_calculation_engine.py (보행 계산 엔진)
└── gait_class.py (보행 분석 핵심 로직)
```

## 핵심 구성 요소

### 1. integrated_gait_system_gui.py
**역할**: 메인 애플리케이션 진입점
- QApplication 초기화 및 스타일 설정
- MainWindow 인스턴스 생성 및 실행
- 전역 스타일시트 적용

```python
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        # 통합된 UI 스타일 정의
        QMainWindow { background-color: #f8f9fa; }
        QPushButton { background-color: #007bff; color: white; }
        # ... 추가 스타일
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 2. main_window.py
**역할**: 메인 윈도우 컨테이너 및 탭 관리
- 3개 주요 기능을 탭으로 구성
- 위젯 간 데이터 공유 및 통신 관리
- 전체 시스템 상태 관리

#### 탭 구성
1. **데이터 동기화 & 시각화** (`DataSynchronizationWidget`)
2. **보행 지표 계산 & 분석** (`GaitMetricsWidget`)
3. **영상 & 데이터 검증** (`VideoValidationWidget`)

```python
class MainWindow(QMainWindow):
    def init_ui(self):
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        
        # 각 기능별 위젯 추가
        self.sync_widget = DataSynchronizationWidget()
        self.metrics_widget = GaitMetricsWidget()
        self.validation_widget = VideoValidationWidget()
        
        # 위젯 간 참조 설정
        self.sync_widget.main_window = self
```

### 3. gait_param_class.py
**역할**: 공통 설정 및 유틸리티 제공

#### GaitAnalysisConfig 클래스
```python
class GaitAnalysisConfig:
    # 보행 타입 매핑
    GAIT_TYPE_MAPPING = {
        'normal_gait': 'T01',
        'ataxic_gait': 'T02', 
        'pain_gait': 'T04',
        'hemiparetic_gait': 'T03',
        'parkinson_gait': 'T05'
    }
    
    # 라벨 색상 매핑
    LABEL_COLORS = {
        'single_support_left': (100, 255, 100, 80),
        'single_support_right': (100, 100, 255, 80),
        'double_support': (255, 100, 100, 80),
        'non_gait': (200, 200, 200, 60)
    }
```

#### GaitAnalysisUtils 클래스
- 파일명 생성 및 검증
- 비디오 정보 추출
- 세션 데이터 유효성 검사
- 동기화 품질 계산

## 주요 기능별 상세 설명

### 1. 데이터 동기화 & 시각화 (data_sync_widget.py)

#### 핵심 기능
- **스마트 세션 선택**: 피험자 → 보행 타입 → 세션 Run 순차 선택
- **자동 파일 매칭**: 비디오, IMU 데이터, 라벨 파일 자동 연결
- **실시간 시각화**: IMU 센서 데이터와 라벨 정보 동기화 그래프
- **동기화 품질 평가**: 비디오와 IMU 데이터 간 시간 동기화 상태 확인

#### 데이터 구조
```
experiment_data/
├── SA01/
│   ├── normal_gait/
│   │   └── session_001/
│   │       ├── video.mp4
│   │       ├── imu_data.csv
│   │       └── metadata.json
│   └── ataxic_gait/
└── SA02/
```

#### 주요 메소드
```python
class DataSynchronizationWidget:
    def scan_experiment_data(self):
        """experiment_data 폴더 스캔하여 세션 목록 구성"""
    
    def load_session_data(self):
        """선택된 세션의 모든 데이터 로드"""
    
    def create_sync_visualization(self):
        """IMU 데이터와 라벨 정보 동기화 그래프 생성"""
    
    def enable_gait_metrics_calculation(self):
        """2번 탭 보행 지표 계산 기능 활성화"""
```

#### 시각화 기능
- **메타데이터 탭**: 세션 정보, 파일 상태, 동기화 품질
- **동기화 테이블**: 프레임별 IMU 데이터와 라벨 매핑
- **시간축 그래프**: 실시간 센서 데이터 시각화 (PyQtGraph 사용)

### 2. 보행 지표 계산 & 분석 (gait_metrics_widget.py)

#### 핵심 기능
- **MediaPipe 기반 관절 추정**: 실시간 포즈 검출 및 관절 좌표 추출
- **보행 이벤트 검출**: Heel Strike(HS), Toe Off(TO) 자동 감지
- **보행 파라미터 계산**: Stride Time, Stride Length, Velocity 계산
- **결과 시각화**: 실시간 그래프 및 통계 분석

#### 계산 방법
1. **방향 감지**: 발목 Z축 좌표 분석으로 보행 방향 판별
2. **노이즈 제거**: Butterworth 필터 + 가우시안 스무딩
3. **이벤트 검출**: 발목 X축 좌표의 피크 검출
4. **파라미터 계산**: 이벤트 간 시간/거리 분석

```python
class GaitMetricsWidget:
    def calculate_gait_metrics(self):
        """전체 보행 지표 계산 파이프라인"""
        # 1. 관절 데이터 추출
        # 2. 보행 방향 감지
        # 3. 이벤트 검출
        # 4. 파라미터 계산
        # 5. 결과 시각화
```

### 3. 영상 & 데이터 검증 (video_validation_widget.py)

#### 세션 기반 워크플로우
```
세션 1: 라벨링 데이터 생성
├── MediaPipe 관절 추출
├── 지지 라벨 생성
└── 관절 데이터 저장

세션 2: 보행 파라미터 계산
├── 보행 계산 엔진 실행
├── Stride 분석
└── 통계 결과 생성

세션 3: 시각화 모드
├── 모든 오버레이 활성화
├── 실시간 재생 제어
└── 결과 검증
```

#### VideoPlayer 클래스
- **다중 오버레이**: 관절, 지지 라벨, 보폭 정보 동시 표시
- **실시간 재생**: 프레임별 이동 및 자동 재생
- **세션 통합**: 이전 세션 결과를 활용한 시각화

#### SessionManager 클래스
```python
class SessionManager:
    def complete_session1(self, support_labels, joint_data, timestamps):
        """세션 1 완료 및 데이터 저장"""
    
    def complete_session2(self, gait_parameters, stride_details):
        """세션 2 완료 및 데이터 저장"""
    
    def get_session_status(self):
        """현재 세션 진행 상태 반환"""
```

### 4. 보행 계산 엔진 (gait_calculation_engine.py)

#### 핵심 알고리즘
- **Heel Strike 검출**: 라벨 데이터 기반 HS 이벤트 추출
- **Stride 계산**: 양발목 간격 기반 보폭 측정
- **픽셀-미터 변환**: 사용자 키 정보를 활용한 스케일 보정

#### 계산 방식
```python
class GaitCalculationEngine:
    def calculate_stride_parameters_by_phases(self, frame_data, support_labels):
        """Phase 시퀀스 기반 stride 계산"""
        # Right Stride: double_stance → right_stance → double_stance → left_stance
        # Left Stride: double_stance → left_stance → double_stance → right_stance
        
    def _calculate_ankle_distance_at_frame(self, frame_data, frame_num):
        """특정 프레임에서 양발목 간 거리 계산"""
        # Stride Length = HS1 거리 + HS2 거리
```

### 5. 보행 분석 핵심 로직 (gait_class.py)

#### GaitAnalyzer 클래스
- **MediaPipe 통합**: 포즈 추정 및 관절 좌표 추출
- **신호 처리**: 노이즈 제거 및 필터링
- **이벤트 검출**: HS/TO 이벤트 자동 감지
- **보행 단계 분석**: 이중지지, 단일지지 구간 분류

#### 처리 파이프라인
```python
class GaitAnalyzer:
    def detect_walking_direction(self):
        """보행 방향 감지 (forward/backward)"""
    
    def apply_enhanced_noise_reduction(self, signal):
        """4단계 노이즈 제거 파이프라인"""
    
    def detect_gait_events(self):
        """HS/TO 이벤트 검출"""
    
    def analyze_gait_phases(self):
        """보행 단계 분석"""
```

## 데이터 흐름

### 1. 세션 선택 및 로드
```
사용자 선택 → 파일 스캔 → 유효성 검사 → 데이터 로드 → 위젯 간 공유
```

### 2. 보행 분석 파이프라인
```
비디오 입력 → MediaPipe 처리 → 관절 좌표 추출 → 신호 처리 → 이벤트 검출 → 파라미터 계산
```

### 3. 결과 시각화
```
계산 결과 → 그래프 생성 → 실시간 업데이트 → 사용자 인터랙션
```

## 사용자 워크플로우

### 1단계: 데이터 준비
1. **세션 선택**: 피험자 → 보행 타입 → Run 번호 선택
2. **데이터 확인**: 비디오, IMU, 라벨 파일 상태 확인
3. **로드 실행**: "세션 데이터 로드" 버튼 클릭

### 2단계: 보행 분석
1. **2번 탭 이동**: "보행 지표 계산 & 분석" 탭 선택
2. **분석 실행**: "보행 지표 계산" 버튼 클릭
3. **결과 확인**: 실시간 그래프 및 통계 확인

### 3단계: 결과 검증
1. **3번 탭 이동**: "영상 & 데이터 검증" 탭 선택
2. **세션 진행**: 세션 1 → 세션 2 → 세션 3 순차 실행
3. **시각화 확인**: 모든 오버레이로 결과 검증

## 기술적 특징

### UI/UX 설계
- **일관된 스타일**: 전역 스타일시트로 통일된 디자인
- **직관적 네비게이션**: 탭 기반 기능 분리
- **실시간 피드백**: 진행 상태 및 결과 즉시 표시

### 성능 최적화
- **지연 로딩**: 필요시에만 데이터 처리
- **샘플링**: 대용량 데이터의 효율적 시각화
- **메모리 관리**: 적절한 데이터 해제 및 재사용

### 확장성
- **모듈화 설계**: 각 기능별 독립적 위젯
- **플러그인 구조**: 새로운 분석 방법 쉽게 추가 가능
- **설정 관리**: 중앙화된 설정 및 매핑

## 오류 처리 및 검증

### 데이터 유효성 검사
- **파일 존재 확인**: 필수 파일들의 존재 여부 검증
- **형식 검증**: CSV, 비디오 파일 형식 확인
- **동기화 검증**: 시간 정보 일치성 확인

### 사용자 피드백
- **상태 표시**: 각 단계별 진행 상태 실시간 표시
- **오류 메시지**: 구체적이고 해결 방안 포함된 메시지
- **경고 시스템**: 잠재적 문제 사전 알림

## 설정 및 커스터마이징

### 보행 타입 추가
```python
# gait_param_class.py에서 수정
GAIT_TYPE_MAPPING = {
    'normal_gait': 'T01',
    'new_gait_type': 'T06',  # 새로운 타입 추가
}
```

### 라벨 색상 변경
```python
LABEL_COLORS = {
    'single_support_left': (100, 255, 100, 80),  # 색상 조정
    'custom_phase': (255, 255, 0, 80),           # 새로운 단계 추가
}
```

### 계산 파라미터 조정
```python
# gait_class.py에서 수정
prominence_threshold = 0.015  # 피크 검출 민감도
min_distance = 15            # 최소 피크 간격
```

## 출력 및 결과

### 저장되는 파일들
- `gait_events.csv`: 검출된 HS/TO 이벤트
- `gait_phases.csv`: 보행 단계 정보
- `analysis_summary.csv`: 분석 요약 통계
- `event_timeline.csv`: 프레임별 이벤트 타임라인

### 실시간 결과
- **그래프**: 센서 데이터, 이벤트, 파라미터 시각화
- **통계**: 평균, 표준편차, 개수 등
- **품질 지표**: 동기화 품질, 검출 신뢰도

## 시스템 요구사항

### 필수 라이브러리
```
PyQt5>=5.15.0
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.3.0
pyqtgraph>=0.12.0
scipy>=1.7.0
```

### 하드웨어 권장사항
- **CPU**: Intel i5 이상 또는 동급
- **RAM**: 8GB 이상
- **GPU**: MediaPipe 가속을 위한 GPU 권장
- **저장공간**: 분석 데이터에 따라 가변

## 문제 해결

### 일반적인 문제들

1. **MediaPipe 설치 오류**
   - 해결: `pip install mediapipe` 재실행
   - 대안: conda 환경에서 설치

2. **비디오 코덱 문제**
   - 해결: OpenCV 추가 코덱 설치
   - 대안: 비디오 형식 변환 (MP4 권장)

3. **메모리 부족**
   - 해결: 샘플링 비율 조정
   - 대안: 비디오 해상도 축소

4. **동기화 품질 불량**
   - 해결: 타임스탬프 데이터 확인
   - 대안: 수동 시간 오프셋 조정

## 향후 개발 계획

### 단기 목표
- **실시간 분석**: 웹캠 입력 실시간 처리
- **배치 처리**: 다중 세션 자동 분석
- **결과 내보내기**: PDF, Excel 형태 리포트

### 장기 목표
- **머신러닝 통합**: 이상 보행 패턴 자동 감지
- **클라우드 연동**: 원격 분석 및 협업 기능
- **모바일 앱**: 스마트폰 기반 간편 분석

## 참고 자료

- **MediaPipe 문서**: https://mediapipe.dev/
- **PyQt5 가이드**: https://doc.qt.io/qtforpython/
- **보행 분석 이론**: 관련 학술 논문 및 교재
- **OpenCV 튜토리얼**: https://opencv.org/

이 통합 시스템은 연구자와 임상의가 보행 데이터를 효율적으로 분석하고 시각화할 수 있도록 설계된 종합적인 도구입니다. 