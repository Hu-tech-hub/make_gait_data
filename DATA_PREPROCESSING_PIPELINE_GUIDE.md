# Data Preprocessing Pipeline Guide

## 개요

Vision Gait Analysis 시스템의 데이터 전처리 파이프라인은 실험 데이터에서 머신러닝 학습용 데이터셋까지의 전체 흐름을 자동화합니다. 이 가이드는 5개의 핵심 모듈을 통한 단계별 데이터 처리 과정을 설명합니다.

## 파이프라인 구조

```
Raw Experiment Data
├── experiment_data/SA01/normal_gait/session_xxx/
│   ├── video.mp4 (비디오 파일)
│   └── imu_data.csv (IMU 센서 데이터)
│
↓ [Step 1: batch_gait_analyzer.py]
│
Support Labels
├── support_label_data/SA01/
│   └── S01T01R01_support_labels.csv (보행 단계 라벨)
│
↓ [Step 2: batch_stride_analyzer.py]
│
Stride Analysis Results
├── stride_analysis_results/
│   └── S01T01R01_stride_labels.csv (보폭 분석 결과)
│
↓ [Step 3: stride_dataset_generator.py]
│
JSON Training Data
├── stride_train_data/S01T01R01/
│   └── S01T01R01_Cycles.json (사이클별 JSON 데이터)
│
↓ [Step 4: stride_data_processor.py]
│
PKL Training Data + Metadata
├── stride_train_data_pkl/
│   └── S01T01R01_Cycles.pkl (정규화된 학습 데이터)
└── metadata/
    ├── file_index.csv (파일 인덱스)
    ├── global_norm.npz (정규화 통계)
    └── cv_splits.json (교차검증 분할)
│
↓ [Step 5: stride_cv_pipeline_enhanced.py]
│
Validated ML Dataset
└── Ready for TCN Model Training
```

## 단계별 상세 설명

### Step 1: batch_gait_analyzer.py
**목적**: 실험 비디오에서 보행 단계 라벨 생성

**입력**:
- `experiment_data/SA01/gait_type/session_xxx/video.mp4`
- `experiment_data/SA01/gait_type/session_xxx/imu_data.csv` (선택적)

**처리 과정**:
1. MediaPipe를 사용한 포즈 추출
2. 보행 방향 자동 감지
3. 보행 이벤트 검출 (heel strike, toe off)
4. 보행 단계 분석 (stance, swing phase)

**출력**:
- `support_label_data/SA01/S01T01R01_support_labels.csv`

**주요 특징**:
- 배치 처리로 여러 세션 동시 분석
- 자동 파일명 매핑 (SA01/normal_gait → S01T01R01)
- GUI 기반 진행상황 모니터링

### Step 2: batch_stride_analyzer.py
**목적**: 보행 단계 라벨을 기반으로 보폭(stride) 분석

**입력**:
- `experiment_data/SA01/gait_type/session_xxx/video.mp4`
- `support_label_data/SA01/S01T01R01_support_labels.csv`

**처리 과정**:
1. 비디오에서 MediaPipe로 관절 좌표 추출
2. Support labels와 동기화
3. GaitCalculationEngine으로 보폭 계산
4. Phase 기반 stride cycle 분석

**출력**:
- `stride_analysis_results/S01T01R01_stride_labels.csv`

**주요 특징**:
- 자동 파일 매칭 (support_labels ↔ video)
- 피험자별 신장 정보 활용
- 한국어 컬럼명으로 결과 저장

### Step 3: stride_dataset_generator.py
**목적**: Stride 분석 결과와 IMU 데이터를 결합하여 JSON 데이터셋 생성

**입력**:
- `stride_analysis_results/S01T01R01_stride_labels.csv`
- `walking_data/SA01/S01T01R01.csv` (IMU 센서 데이터)

**처리 과정**:
1. Stride labels에서 cycle 정보 추출
2. Walking data에서 해당 프레임 구간의 IMU 시퀀스 추출
3. 각 cycle별로 sequence + metadata 결합
4. JSON 형태로 구조화

**출력**:
- `stride_train_data/S01T01R01/S01T01R01_Cycles.json`

**주요 특징**:
- 자동 파일 매칭 및 배치 처리
- 6축 IMU 데이터 (accel_x,y,z + gyro_x,y,z)
- Cycle별 메타데이터 (height, stride_time, foot 등)

### Step 4: stride_data_processor.py
**목적**: JSON 데이터를 머신러닝 학습용 PKL 형태로 변환 및 정규화

**입력**:
- `stride_train_data/S01T01R01/S01T01R01_Cycles.json`

**처리 과정**:
1. JSON 파일들을 스캔하여 유효성 검증
2. 시퀀스 길이 필터링 (15~100 프레임)
3. 전역 정규화 통계 계산 (z-score)
4. Subject-wise LOSO 교차검증 분할 생성
5. PKL 형태로 직렬화

**출력**:
- `stride_train_data_pkl/S01T01R01_Cycles.pkl`
- `metadata/file_index.csv`
- `metadata/global_norm.npz`
- `metadata/cv_splits.json`

**주요 특징**:
- Subject별 5-Fold LOSO 교차검증
- 전역 정규화 (시퀀스 + 보조 특징)
- 데이터 분포 분석 및 통계 생성

### Step 5: stride_cv_pipeline_enhanced.py
**목적**: 전체 파이프라인 검증 및 머신러닝 준비 완료 확인

**입력**:
- 모든 이전 단계의 출력물

**처리 과정**:
1. 데이터 처리 결과 검증
2. 교차검증 분할 무결성 확인
3. RaggedTensor 데이터 제너레이터 생성
4. 모든 fold의 데이터셋 형태 검증
5. 정규화 적용 확인

**출력**:
- 검증된 머신러닝 준비 완료 데이터셋
- 상세한 검증 리포트

**주요 특징**:
- 4단계 향상된 검증 프로세스
- Subject 누수 방지 확인
- RaggedTensor 형태 검증
- dtype 및 정규화 적용 확인

## 데이터 형태 변환

### 1. Raw Video → Support Labels
```
video.mp4 (30fps, 1920x1080)
↓ MediaPipe Pose Detection
pose_landmarks (33 joints × frames)
↓ Gait Event Detection
support_labels.csv:
  phase | start_frame | end_frame
  stance|     120     |    180
  swing |     180     |    240
```

### 2. Support Labels → Stride Analysis
```
support_labels.csv + video.mp4
↓ Phase-based Stride Calculation
stride_labels.csv:
  번호|피험자ID|키(cm)|발|시작프레임|종료프레임|Stride Time(s)|Stride Length(m)|Velocity(m/s)
   1 | SA01  | 175 |left|   120    |    240   |     1.2      |     1.4       |    1.17
```

### 3. Stride Labels → JSON Dataset
```
stride_labels.csv + walking_data.csv
↓ IMU Sequence Extraction
Cycles.json:
[
  {
    "sequence": [[ax,ay,az,gx,gy,gz], ...],  # 120 frames × 6 axes
    "height": 175,
    "stride_time": 1.2,
    "stride_length": 1.4,
    "foot": "left"
  }
]
```

### 4. JSON → PKL + Normalization
```
Cycles.json
↓ Validation + Normalization
Cycles.pkl:
{
  "subject": "S01",
  "task": "T01", 
  "rep": "R01",
  "cycles": [normalized_cycles]
}

global_norm.npz:
{
  "mean": [6-axis means],
  "std": [6-axis stds]
}
```

### 5. PKL → ML-Ready Dataset
```
Cycles.pkl + metadata
↓ RaggedTensor Generation
tf.data.Dataset:
  Input: (RaggedTensor(batch, None, 6), Tensor(batch, 3))
  Output: Tensor(batch,)
  
  Features: [IMU_sequences, auxiliary_features]
  Labels: [stride_lengths]
```

## 실행 순서

### 1. 순차 실행 (권장)
```bash
# Step 1: 보행 단계 라벨 생성
python batch_gait_analyzer.py

# Step 2: 보폭 분석
python batch_stride_analyzer.py

# Step 3: JSON 데이터셋 생성
python stride_dataset_generator.py

# Step 4: PKL 변환 및 정규화
python stride_data_processor.py

# Step 5: 전체 검증
python stride_cv_pipeline_enhanced.py
```

### 2. 통합 실행
```bash
# 전체 파이프라인 자동 실행
python stride_cv_pipeline_enhanced.py
# (내부적으로 필요한 단계들을 자동 호출)
```

## 품질 관리

### 데이터 검증 체크포인트
1. **Step 1**: 비디오 품질, 포즈 검출 성공률
2. **Step 2**: Support labels 매칭, 보폭 계산 정확도
3. **Step 3**: IMU-라벨 동기화, 시퀀스 완정성
4. **Step 4**: 정규화 통계, 교차검증 분할 무결성
5. **Step 5**: 최종 데이터셋 형태, Subject 누수 방지

### 오류 처리
- 각 단계별 상세한 로그 및 진행상황 표시
- 실패한 세션 건너뛰기 및 오류 리포트
- 중간 결과물 저장으로 재시작 가능

### 성능 최적화
- 배치 처리로 여러 세션 동시 처리
- 멀티스레딩 지원 (GUI 응답성 유지)
- 메모리 효율적인 대용량 데이터 처리

## 출력 데이터 구조

### 최종 학습 데이터 형태
```python
# 입력 데이터
sequences: tf.RaggedTensor  # shape=(batch, None, 6)
auxiliary: tf.Tensor        # shape=(batch, 3) [stride_time, height, foot_id]

# 출력 데이터  
labels: tf.Tensor          # shape=(batch,) [stride_length]

# 정규화
sequences: z-score normalized (6-axis IMU)
stride_time: z-score normalized
height: z-score normalized
foot_id: categorical (0=left, 1=right, -1=unknown)
```

### 교차검증 구조
```
5-Fold Subject-wise LOSO:
Fold 1: Train=[S02,S03,S04,S05], Test=[S01]
Fold 2: Train=[S01,S03,S04,S05], Test=[S02]
Fold 3: Train=[S01,S02,S04,S05], Test=[S03]
Fold 4: Train=[S01,S02,S03,S05], Test=[S04]
Fold 5: Train=[S01,S02,S03,S04], Test=[S05]
```

## 요구사항

### 시스템 요구사항
- Python 3.8+
- OpenCV, MediaPipe
- TensorFlow 2.x
- PyQt5 (GUI)
- pandas, numpy, scikit-learn

### 데이터 요구사항
- 비디오: MP4 형식, 30fps 권장
- IMU: CSV 형식, 6축 데이터 (accel_x,y,z + gyro_x,y,z)
- 폴더 구조: `experiment_data/SA01/gait_type/session_xxx/`

### 하드웨어 권장사항
- RAM: 8GB 이상
- 저장공간: 10GB 이상 (중간 파일 포함)
- GPU: TensorFlow 가속 지원 (선택적)

이 파이프라인을 통해 원시 실험 데이터에서 머신러닝 학습 준비가 완료된 고품질 데이터셋을 자동으로 생성할 수 있습니다. 