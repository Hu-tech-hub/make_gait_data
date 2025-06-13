# Stride Inference Pipeline Guide

## 개요

Stride Inference Pipeline은 라벨링된 IMU 센서 데이터에서 stride cycle을 추출하고, 학습된 TCN(Temporal Convolutional Network) 모델을 사용하여 보폭(stride length)을 예측하는 시스템입니다.

## 주요 기능

- **Stride Cycle 추출**: 지지 라벨 데이터를 기반으로 left/right stride cycle 자동 감지
- **IMU 데이터 처리**: 가속도계 및 자이로스코프 센서 데이터 전처리
- **TCN 모델 추론**: 학습된 모델을 사용한 실시간 보폭 예측
- **보조 특성 계산**: stride time, 신장, foot ID 등 추가 특성 활용
- **결과 분석**: 통계적 분석 및 JSON 형태 결과 출력

## 시스템 구조

```
Input Data
├── support_labels.csv (지지 라벨 데이터)
├── walking_data.csv (IMU 센서 데이터)
└── trained_model.keras (학습된 TCN 모델)

Processing Pipeline
├── Label Processing → Stride Cycle Detection
├── Sensor Data → Feature Extraction
├── Normalization → Model Input Preparation
└── TCN Model → Stride Length Prediction

Output
└── Results JSON (예측 결과 및 통계)
```

## 입력 데이터 형식

### 1. Support Labels CSV (`{filename}_support_labels.csv`)
```csv
start_frame,end_frame,phase
0,45,double_support
46,120,single_support_left
121,165,double_support
166,240,single_support_right
...
```

**필수 컬럼:**
- `start_frame`: 구간 시작 프레임
- `end_frame`: 구간 종료 프레임
- `phase`: 보행 단계 (double_support, single_support_left, single_support_right)

### 2. Walking Data CSV (`{filename}.csv`)
```csv
frame_number,sync_timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0,0.0333,-0.123,9.756,0.456,0.012,-0.034,0.067
1,0.0667,-0.145,9.734,0.478,0.023,-0.045,0.078
...
```

**필수 컬럼:**
- `frame_number`: 프레임 번호
- `sync_timestamp`: 동기화 타임스탬프
- `accel_x, accel_y, accel_z`: 3축 가속도 데이터
- `gyro_x, gyro_y, gyro_z`: 3축 각속도 데이터

## 핵심 클래스: StrideInferencePipeline

### 초기화 매개변수

```python
pipeline = StrideInferencePipeline(
    model_path="models_2/best_fold_5.keras",  # 학습된 모델 경로
    metadata_dir="metadata"                   # 정규화 통계 디렉토리
)
```

### 주요 메소드

#### 1. `load_model_and_stats()`
- TCN 모델 및 정규화 통계 로드
- 정규화 파라미터 (mean, std) 로드

#### 2. `find_stride_cycles(labels, stride_type)`
- 지지 라벨에서 특정 stride type의 cycle 탐지
- **Left Stride**: `double_stance → left_stance → double_stance → right_stance`
- **Right Stride**: `double_stance → right_stance → double_stance → left_stance`

#### 3. `extract_cycle_sequence(walking_df, start_frame, end_frame)`
- stride cycle 구간의 IMU 센서 데이터 추출
- 6차원 특성 벡터 (accel_x,y,z + gyro_x,y,z) 생성
- 데이터 유효성 검증 및 오류 처리

#### 4. `prepare_model_input(cycles_data)`
- 모델 입력 형태로 데이터 전처리
- **고정 길이 패딩**: 67 프레임으로 시퀀스 정규화
- **정규화**: Z-score normalization 적용
- **보조 특성**: height, stride_time, foot_id 정규화

#### 5. `predict_stride_lengths(sequences, auxiliary)`
- TCN 모델을 통한 보폭 예측
- 다중 입력 (시퀀스 + 보조 특성) 처리
- 배치 예측 지원

## 처리 과정

### 1. 데이터 로드 및 검증
```python
# 지지 라벨 로드
support_labels = pipeline.load_support_labels(labels_file)

# IMU 센서 데이터 로드
walking_df = pipeline.load_walking_data(walking_file)
```

### 2. Subject 정보 추출
```python
# 파일명에서 subject ID 추출 (S01 → SA01)
subject_id = pipeline.extract_subject_id(filename)

# Subject별 신장 정보 매핑
height = pipeline.get_subject_height(subject_id)
```

**지원 Subject 정보:**
- SA01: 175cm
- SA02: 170cm  
- SA03: 180cm
- SA04: 160cm
- SA05: 160cm

### 3. Stride Cycle 탐지
```python
left_cycles = pipeline.find_stride_cycles(support_labels, 'left')
right_cycles = pipeline.find_stride_cycles(support_labels, 'right')
```

### 4. 특성 추출 및 정규화
```python
for cycle in all_cycles:
    # IMU 시퀀스 추출
    sequence = extract_cycle_sequence(walking_df, start_frame, end_frame)
    
    # 보조 특성 계산
    stride_time = (end_frame - start_frame) / fps
    foot_id = foot_mapping[foot_type]  # left=0, right=1
    
    # 정규화 적용
    normalized_seq = normalize_sequence(sequence)
    normalized_aux = normalize_auxiliary_features(height, stride_time, foot_id)
```

### 5. 모델 추론
```python
# 고정 길이 패딩 (67 프레임)
padded_sequences = apply_padding(sequences, target_length=67)

# TCN 모델 예측
predictions = model.predict([padded_sequences, auxiliary_features])
```

## 사용법

### 1. 명령행 인터페이스

```bash
# 기본 사용법
python stride_inference_pipeline.py \
    --labels support_label_data/SA01/S01T01R01_support_labels.csv \
    --walking walking_data/SA01/S01T01R01.csv \
    --output results.json

# 고급 옵션
python stride_inference_pipeline.py \
    --labels path/to/labels.csv \
    --walking path/to/walking.csv \
    --model models_2/custom_model.keras \
    --metadata_dir custom_metadata \
    --output detailed_results.json
```

### 2. Python API 사용

```python
from stride_inference_pipeline import StrideInferencePipeline

# 파이프라인 초기화
pipeline = StrideInferencePipeline(
    model_path="models_2/best_fold_5.keras",
    metadata_dir="metadata"
)

# 추론 실행
results = pipeline.run_inference(
    labels_file="support_label_data/SA01/S01T01R01_support_labels.csv",
    walking_file="walking_data/SA01/S01T01R01.csv",
    output_file="results.json"
)

# 결과 확인
print(f"Total Cycles: {results['total_cycles']}")
print(f"Mean Stride Length: {results['mean_stride_length']:.3f}m")
```

## 출력 결과 형식

### JSON 결과 구조
```json
{
  "subject_id": "SA01",
  "height": 175,
  "total_cycles": 12,
  "left_cycles": 6,
  "right_cycles": 6,
  "mean_stride_length": 1.286,
  "std_stride_length": 0.087,
  "mean_velocity": 1.523,
  "std_velocity": 0.124,
  "predictions": [
    {
      "cycle_number": 1,
      "foot": "left",
      "start_frame": 0,
      "end_frame": 67,
      "sequence_length": 68,
      "stride_time": 2.267,
      "predicted_stride_length": 1.234,
      "predicted_velocity": 0.544
    },
    ...
  ]
}
```

### 결과 해석

**전체 통계:**
- `total_cycles`: 감지된 총 stride cycle 수
- `mean_stride_length`: 평균 보폭 (미터)
- `std_stride_length`: 보폭 표준편차
- `mean_velocity`: 평균 보행 속도 (m/s)

**개별 Cycle 정보:**
- `cycle_number`: Cycle 순서
- `foot`: 발 구분 (left/right)
- `start_frame`, `end_frame`: 프레임 범위
- `sequence_length`: 실제 데이터 길이
- `stride_time`: Stride 시간 (초)
- `predicted_stride_length`: 예측된 보폭 (미터)
- `predicted_velocity`: 계산된 속도 (보폭/시간)

## 모델 요구사항

### TCN 모델 구조
```python
# 필요한 입력 형태
sequence_input: (batch_size, 67, 6)  # 패딩된 IMU 시퀀스
auxiliary_input: (batch_size, 3)     # [height, stride_time, foot_id]

# 출력 형태
predictions: (batch_size, 1)         # 예측된 stride length
```

### 정규화 통계 파일
```
metadata/
├── global_norm_enhanced.npz  # 선호되는 통계 파일
└── global_norm.npz          # 대체 통계 파일
```

**필수 통계 정보:**
- `sequence_mean, sequence_std`: IMU 데이터 정규화
- `height_mean, height_std`: 신장 정규화
- `stride_time_mean, stride_time_std`: Stride time 정규화

## 오류 처리 및 검증

### 데이터 품질 검사
- **최소 시퀀스 길이**: 15 프레임 이상
- **센서 데이터 유효성**: NaN, Inf 값 제거
- **프레임 연속성**: 시작-종료 프레임 일치성 확인

### 일반적인 오류 및 해결방법

1. **"Broadcasting error"**
   - 원인: 모델 입력 형태 불일치
   - 해결: 고정 길이 패딩 (67 프레임) 적용

2. **"정규화 통계 파일 없음"**
   - 원인: metadata 디렉토리 또는 통계 파일 누락
   - 해결: 정확한 경로 확인 및 파일 존재 여부 검증

3. **"유효한 stride cycle 없음"**
   - 원인: 라벨 데이터의 phase 순서 불일치
   - 해결: 라벨 데이터 형식 및 phase naming 확인

## 성능 최적화

### 배치 처리
- 다중 cycle 동시 처리로 GPU 활용도 극대화
- 메모리 효율적인 패딩 및 정규화

### 데이터 전처리
- 벡터화 연산을 통한 속도 향상
- 불필요한 데이터 복사 최소화

## 확장 가능성

### 새로운 Subject 추가
```python
# subject_heights 딕셔너리에 추가
self.subject_heights = {
    'SA01': 175,
    'SA02': 170,
    # 새로운 subject 추가
    'SA06': 180,
}
```

### 다른 센서 모달리티 지원
- 현재: 6-DOF IMU (가속도계 + 자이로스코프)
- 확장 가능: 자력계, 압력 센서 등

### 실시간 처리
- 스트리밍 데이터 처리를 위한 버퍼링 구조
- 온라인 학습 및 적응형 정규화

## 관련 파일

- `tcn_model.py`: TCN 모델 구조 정의
- `gait_calculation_engine.py`: Stride cycle 탐지 로직
- `ragged_data_generator.py`: 데이터 패딩 참조 구현
- `stride_inference_gui.py`: GUI 인터페이스

## 참고 문헌

- TCN (Temporal Convolutional Networks) 아키텍처
- IMU 기반 보행 분석 방법론
- Gait cycle 단계 분류 시스템 