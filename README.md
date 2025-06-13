# Vision Gait Analysis System

## 개요

Vision Gait Analysis System은 비디오와 IMU 센서 데이터를 활용한 종합적인 보행 분석 시스템입니다. MediaPipe 기반 포즈 추정, 머신러닝 기반 보폭 예측, 그리고 통합 GUI를 통해 연구자들이 보행 데이터를 효율적으로 분석할 수 있도록 지원합니다.

## 시스템 구조

```
vision_gait/
├── 📊 데이터 전처리 파이프라인
│   ├── batch_gait_analyzer.py          # Step 1: 보행 단계 라벨 생성
│   ├── batch_stride_analyzer.py        # Step 2: 보폭 분석
│   ├── stride_dataset_generator.py     # Step 3: JSON 데이터셋 생성
│   ├── stride_data_processor.py        # Step 4: PKL 변환 및 정규화
│   └── stride_cv_pipeline_enhanced.py  # Step 5: 전체 검증
│
├── 🤖 머신러닝 모델
│   ├── tcn_model.py                    # TCN 모델 정의
│   ├── tcn_trainer.py                  # 모델 학습 스크립트
│   ├── tcn_trainer_gui.py              # 학습 GUI
│   ├── ragged_data_generator.py        # RaggedTensor 데이터 제너레이터
│   └── stride_inference_pipeline.py    # 추론 파이프라인
│
├── 🖥️ GUI 애플리케이션
│   ├── integrated_gait_system_gui.py   # 통합 보행 분석 GUI
│   ├── main_window.py                  # 메인 윈도우 컨테이너
│   ├── gait_analyzer_gui.py            # 개별 보행 분석 GUI
│   └── stride_inference_gui.py         # 보폭 추론 GUI
│
├── 🔧 핵심 분석 엔진
│   ├── gait_class.py                   # 보행 분석 핵심 로직
│   ├── gait_calculation_engine.py      # Phase 기반 보행 지표 계산 엔진
│   ├── gait_param_class.py             # 공통 유틸리티 및 설정
│   └── make_data.py                    # 데이터 생성 도구
│
├── 🎛️ GUI 위젯 컴포넌트
│   ├── data_sync_widget.py             # 데이터 동기화 위젯
│   ├── gait_metrics_widget.py          # 보행 지표 계산 및 표시
│   └── video_validation_widget.py      # 영상 검증 위젯
│
├── 📁 데이터 폴더
│   ├── experiment_data/                # 원시 실험 데이터
│   ├── support_label_data/             # 보행 단계 라벨
│   ├── stride_analysis_results/        # 보폭 분석 결과
│   ├── walking_data/                   # IMU 센서 데이터
│   ├── stride_train_data/              # JSON 학습 데이터
│   ├── stride_train_data_pkl/          # PKL 학습 데이터
│   ├── models/                         # 학습된 모델
│   ├── models_2/                       # 추가 모델
│   ├── metadata/                       # 메타데이터
│   └── logs/                           # 로그 파일
│
└── 📚 문서
    ├── README.md                       # 이 파일
    ├── DATA_PREPROCESSING_PIPELINE_GUIDE.md  # 데이터 전처리 가이드
    ├── STRIDE_INFERENCE_PIPELINE_GUIDE.md    # 추론 파이프라인 가이드
    ├── INTEGRATED_GAIT_SYSTEM_GUI_GUIDE.md   # 통합 GUI 가이드
    ├── system_guide.md                 # 시스템 가이드
    ├── make_data_guide.md              # 데이터 생성 가이드
    └── requirements.txt                # 의존성 패키지
```

## 주요 기능

### 🎯 핵심 분석 기능
- **포즈 기반 보행 분석**: MediaPipe를 활용한 실시간 포즈 추정
- **보행 이벤트 검출**: Heel strike, toe off 자동 감지
- **보행 지표 계산**: Stride length, stride time, velocity 등
- **머신러닝 보폭 예측**: TCN 모델 기반 정확한 보폭 추정

### 📊 데이터 처리 파이프라인
- **5단계 자동화 파이프라인**: 원시 데이터 → 머신러닝 준비 완료
- **배치 처리**: 여러 세션 동시 분석
- **품질 관리**: 각 단계별 검증 및 오류 처리
- **교차검증**: Subject-wise LOSO 5-Fold 지원

### 🖥️ 사용자 인터페이스
- **통합 GUI**: 모든 기능을 하나의 인터페이스에서
- **실시간 시각화**: 센서 데이터 및 분석 결과 그래프
- **세션 기반 워크플로우**: 단계별 분석 진행
- **다중 오버레이**: 관절, 라벨, 보폭 정보 동시 표시

## 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv vision_gait_env
source vision_gait_env/bin/activate  # Windows: vision_gait_env\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 통합 GUI 실행
```bash
python integrated_gait_system_gui.py
```

### 3. 데이터 전처리 파이프라인 실행
```bash
# 전체 파이프라인 자동 실행
python stride_cv_pipeline_enhanced.py

# 또는 단계별 실행
python batch_gait_analyzer.py          # 보행 단계 라벨 생성
python batch_stride_analyzer.py        # 보폭 분석
python stride_dataset_generator.py     # JSON 데이터셋 생성
python stride_data_processor.py        # PKL 변환
```

### 4. 모델 학습
```bash
python tcn_trainer_gui.py              # GUI로 학습
# 또는
python tcn_trainer.py                  # 스크립트로 학습
```

### 5. 보폭 추론
```bash
python stride_inference_gui.py         # GUI로 추론
# 또는
python stride_inference_pipeline.py    # 스크립트로 추론
```

## 상세 가이드

### 📖 문서별 설명
- **[DATA_PREPROCESSING_PIPELINE_GUIDE.md](DATA_PREPROCESSING_PIPELINE_GUIDE.md)**: 5단계 데이터 전처리 파이프라인 상세 설명
- **[STRIDE_INFERENCE_PIPELINE_GUIDE.md](STRIDE_INFERENCE_PIPELINE_GUIDE.md)**: TCN 모델 기반 보폭 추론 시스템
- **[INTEGRATED_GAIT_SYSTEM_GUI_GUIDE.md](INTEGRATED_GAIT_SYSTEM_GUI_GUIDE.md)**: 통합 GUI 사용법 및 구조
- **[system_guide.md](system_guide.md)**: 전체 시스템 개요
- **[make_data_guide.md](make_data_guide.md)**: 데이터 생성 도구 사용법

### 🔧 모듈별 설명

#### 데이터 전처리 모듈
- **`batch_gait_analyzer.py`**: 실험 비디오에서 MediaPipe로 보행 단계 라벨 자동 생성
- **`batch_stride_analyzer.py`**: 보행 단계 라벨과 비디오를 결합하여 보폭 분석
- **`stride_dataset_generator.py`**: 보폭 분석 결과와 IMU 데이터를 JSON 형태로 결합
- **`stride_data_processor.py`**: JSON을 PKL로 변환하고 정규화, 교차검증 분할 생성
- **`stride_cv_pipeline_enhanced.py`**: 전체 파이프라인 검증 및 품질 관리

#### 머신러닝 모듈
- **`tcn_model.py`**: Temporal Convolutional Network 모델 정의
- **`tcn_trainer.py`**: 모델 학습 스크립트 (Subject-wise LOSO 교차검증)
- **`tcn_trainer_gui.py`**: 학습 과정 모니터링 GUI
- **`ragged_data_generator.py`**: 가변 길이 시퀀스 처리용 RaggedTensor 제너레이터
- **`stride_inference_pipeline.py`**: 학습된 모델로 새로운 데이터 추론

#### GUI 모듈
- **`integrated_gait_system_gui.py`**: 모든 기능을 통합한 메인 GUI
- **`gait_analyzer_gui.py`**: 개별 세션 보행 분석 GUI
- **`stride_inference_gui.py`**: 보폭 추론 전용 GUI
- **`tcn_trainer_gui.py`**: 학습 과정 모니터링 GUI
- **`batch_gait_analyzer.py`**: 실험 비디오에서 MediaPipe로 보행 단계 라벨 자동 생성 GUI
- **`batch_stride_analyzer.py`**: 보행 단계 라벨과 비디오를 결합하여 보폭 분석 GUI
- **`stride_dataset_generator.py`**: 보폭 분석 결과와 IMU 데이터를 JSON 형태로 결합 GUI

#### 핵심 엔진 모듈
- **`gait_class.py`**: MediaPipe 통합, 보행 이벤트 검출, 포즈 분석
- **`gait_calculation_engine.py`**: Phase 기반 보행 지표 계산 엔진
- **`gait_param_class.py`**: 공통 설정, 유틸리티 함수, 상수 정의

#### 위젯 모듈
- **`data_sync_widget.py`**: 스마트 세션 선택 및 실시간 데이터 시각화
- **`gait_metrics_widget.py`**: 보행 지표 계산 및 표시
- **`video_validation_widget.py`**: 3단계 세션 기반 검증 워크플로우

## 데이터 구조

### 입력 데이터 형식
```
experiment_data/
├── SA01/                               # 피험자 ID
│   ├── normal_gait/                    # 보행 타입
│   │   └── session_20250604_213127/    # 세션 폴더
│   │       ├── video.mp4               # 비디오 파일
│   │       ├── imu_data.csv            # IMU 센서 데이터
│   │       └── metadata.json           # 세션 메타데이터
│   ├── ataxic_gait/
│   └── pain_gait/
├── SA02/
└── SA03/
```

### 출력 데이터 형식
```
support_label_data/SA01/S01T01R01_support_labels.csv    # 보행 단계 라벨
stride_analysis_results/S01T01R01_stride_labels.csv     # 보폭 분석 결과
stride_train_data/S01T01R01/S01T01R01_Cycles.json       # JSON 학습 데이터
stride_train_data_pkl/S01T01R01_Cycles.pkl              # PKL 학습 데이터
models_2/tcn_model_fold1.keras                          # 학습된 모델
```

## 시스템 요구사항

### 소프트웨어
- **Python**: 3.8 이상
- **주요 라이브러리**:
  - OpenCV (4.5+)
  - MediaPipe (0.8+)
  - TensorFlow (2.8+)
  - PyQt5 (5.15+)
  - pandas, numpy, scikit-learn
  - matplotlib, seaborn

### 하드웨어
- **RAM**: 8GB 이상 권장
- **저장공간**: 10GB 이상 (중간 파일 포함)
- **GPU**: CUDA 지원 GPU (선택적, 학습 가속화)
- **웹캠**: 실시간 분석용 (선택적)

## 사용 사례

### 1. 연구자용 - 전체 파이프라인
```bash
# 1. 실험 데이터 수집 후 experiment_data/ 폴더에 배치
# 2. 통합 GUI로 전체 분석
python integrated_gait_system_gui.py

# 3. 또는 파이프라인 자동 실행
python stride_cv_pipeline_enhanced.py
```

### 2. 임상의용 - 개별 세션 분석
```bash
# 개별 세션 분석 GUI
python gait_analyzer_gui.py
```

### 3. 개발자용 - 모델 개발
```bash
# 데이터 전처리
python stride_data_processor.py

# 모델 학습
python tcn_trainer_gui.py

# 추론 테스트
python stride_inference_gui.py
```

## 문제 해결

### 일반적인 문제
1. **MediaPipe 설치 오류**: `pip install mediapipe --upgrade`
2. **GPU 메모리 부족**: 배치 크기 줄이기 또는 CPU 사용
3. **파일 경로 오류**: 절대 경로 사용 권장
4. **의존성 충돌**: 가상환경 사용 권장

### 로그 확인
- GUI 애플리케이션: 내장 로그 패널 확인
- 스크립트 실행: 터미널 출력 및 `logs/` 폴더 확인

## 기여 방법

### 개발 환경 설정
```bash
git clone <repository-url>
cd vision_gait
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt
```

### 코드 스타일
- PEP 8 준수
- 함수/클래스에 docstring 작성
- 타입 힌트 사용 권장

### 테스트
```bash
# 단위 테스트 실행
python -m pytest tests/

# 통합 테스트
python stride_cv_pipeline_enhanced.py
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 연락처

- **개발팀**: [이메일 주소]
- **이슈 리포트**: GitHub Issues
- **문서 개선**: Pull Request 환영

## 업데이트 로그

### v1.0.0 (2025-01-12)
- 초기 릴리스
- 5단계 데이터 전처리 파이프라인 구현
- TCN 기반 보폭 예측 모델
- 통합 GUI 시스템
- Subject-wise LOSO 교차검증 지원

---

**Vision Gait Analysis System**으로 정확하고 효율적인 보행 분석을 시작하세요! 🚀