# 통합 보행 분석 시스템 사용 가이드

## 시스템 개요

이 시스템은 IMU 센서 데이터와 영상 데이터를 기반으로 보행 분석 및 낙상 위험 예측을 수행하는 통합 시스템입니다.

### 주요 기능

1. **데이터 시각화 및 정제**: 센서 데이터, 보행 영상, 이벤트 정보를 시간 축 기준으로 동기화하여 시각적으로 확인
2. **자동 보행 지표 계산**: MediaPipe 기반 관절 추정을 통한 보폭, 속도, 주기, 보행률, ROM 계산
3. **시계열 회귀 모델 학습**: IMU 데이터만으로 보행 지표를 예측하는 딥러닝 모델 학습
4. **실시간 예측 및 검증**: 학습된 모델을 사용한 보행 지표 추론 및 시각적 검증

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 준비

다음과 같은 구조로 데이터를 준비하세요:

```
experiment_data/
├── SA01/
│   ├── normal_gait/
│   │   ├── session_YYYYMMDD_HHMMSS/
│   │   │   ├── video.mp4
│   │   │   ├── imu_data.csv
│   │   │   └── metadata.json
│   │   └── ...
│   └── ...
└── ...
```

#### IMU 데이터 형식 (imu_data.csv)

```csv
frame_number,sync_timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0,0.000,0.086,9.701,1.834,-1.95420,0.05344,-1.72519
1,0.033,0.077,9.852,1.801,-1.50745,2.10028,-1.74367
...
```

#### 메타데이터 형식 (metadata.json)

```json
{
  "session_id": 16,
  "sync_start_time_seoul": "2025-06-04 21:02:19",
  "sync_end_time_seoul": "2025-06-04 21:02:31",
  "duration": 11.221202611923218,
  "video_fps": 30,
  "video_frames": 337,
  "imu_hz": 30,
  "imu_samples": 336,
  "sync_method": "frame_sync_interpolation",
  "timestamp": "2025-06-04T21:02:32.020654"
}
```

## 시스템 사용법

### 1. GUI 실행

```bash
python integrated_gait_system_gui.py
```

### 2. 단계별 분석 프로세스

#### Step 1: 데이터 동기화
1. "1. 데이터 동기화" 탭으로 이동
2. 비디오 파일 선택 (*.mp4)
3. IMU 데이터 파일 선택 (*.csv)
4. 선택사항: 라벨 파일 선택 (*.json 또는 *.csv)
5. "데이터 동기화 검증" 버튼 클릭

#### Step 2: 이벤트 검출
1. "2. 이벤트 검출" 탭으로 이동
2. 보행 이벤트(HS/TO) 자동 검출 실행
3. 검출된 이벤트를 시각적으로 확인 및 수정
4. 보행 상태 라벨(이중지지, 단일지지, non-gait) 검토

#### Step 3: 보행 지표 계산
1. "3. 보행 지표 계산" 탭으로 이동
2. 픽셀-미터 변환 비율 설정
3. "보행 지표 계산 시작" 버튼 클릭
4. 계산된 지표를 테이블에서 확인
5. 이상치 제거 및 결과 저장

#### Step 4: 모델 학습
1. "4. 모델 학습" 탭으로 이동
2. 모델 타입 선택 (LSTM, TCN, 1D CNN)
3. 하이퍼파라미터 설정 (윈도우 크기, 학습/검증 비율)
4. "모델 학습 시작" 버튼 클릭
5. 학습 완료 후 모델 저장

#### Step 5: 예측 및 검증
1. "5. 예측 및 검증" 탭으로 이동
2. 학습된 모델로 새로운 데이터 예측
3. 예측값과 실제값 비교 시각화
4. 성능 지표 확인 (MAE, RMSE, R²)

### 3. 배치 처리

기존 배치 분석 스크립트 사용:

```bash
python batch_gait_analyzer.py
```

## 출력 파일 형식

### 보행 지표 CSV

```csv
start_frame,end_frame,foot,stride_length,velocity,cycle_time,cadence,hip_rom,knee_rom,ankle_rom,stance_ratio
45,89,left,0.523,0.892,1.467,40.9,23.4,45.2,15.8,62.3
```

### 예측 결과 JSON

```json
{
  "model_type": "LSTM",
  "window_size": 90,
  "predictions": [
    {
      "timestamp": "2025-01-01T10:00:00",
      "predicted_metrics": {
        "stride_length": 0.521,
        "velocity": 0.885,
        "cadence": 41.2,
        "knee_rom": 44.8
      },
      "confidence": 0.89
    }
  ]
}
```

## 성능 최적화 팁

### 1. 데이터 전처리
- IMU 데이터의 노이즈 제거를 위해 적절한 필터링 적용
- 관절 추정 품질이 낮은 프레임 제외
- 이상치 검출 및 제거

### 2. 모델 학습
- 적절한 윈도우 크기 선택 (보행 주기의 1-2배)
- 교차 검증을 통한 모델 성능 평가
- 하이퍼파라미터 튜닝

### 3. 실시간 처리
- GPU 가속 활용 (CUDA 지원)
- 모델 경량화 (quantization, pruning)
- 배치 추론을 통한 처리 속도 향상

## 트러블슈팅

### 자주 발생하는 문제

1. **MediaPipe 관절 추정 실패**
   - 조명 조건 개선
   - 카메라 각도 조정
   - 배경 단순화

2. **IMU-영상 동기화 오류**
   - 시간 스탬프 확인
   - 샘플링 레이트 일치 여부 확인
   - 동기화 신호 사용

3. **모델 성능 저하**
   - 학습 데이터 품질 검토
   - 특징 엔지니어링 개선
   - 모델 아키텍처 조정

### 로그 확인

시스템 로그는 다음 위치에 저장됩니다:
- 분석 로그: `./logs/analysis.log`
- 모델 학습 로그: `./experiments/[experiment_name].json`
- 오류 로그: `./logs/error.log`

## 확장 기능

### 1. 실시간 분석
```python
from real_time_analyzer import RealTimeGaitAnalyzer

analyzer = RealTimeGaitAnalyzer(model_path="trained_model.h5")
analyzer.start_real_time_analysis()
```

### 2. 사용자 정의 지표
```python
from custom_metrics import CustomGaitMetrics

custom_metrics = CustomGaitMetrics()
custom_metrics.add_metric("step_asymmetry", calculate_step_asymmetry)
```

### 3. 다중 모달 분석
```python
from multimodal_analyzer import MultiModalAnalyzer

analyzer = MultiModalAnalyzer()
analyzer.add_modality("pressure", pressure_data)
analyzer.add_modality("emg", emg_data)
```

## 참고 자료

- MediaPipe Documentation: https://mediapipe.dev/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- 보행 분석 이론: [관련 논문 및 자료]

## 지원 및 문의

기술적 문의나 버그 리포트는 다음으로 연락하세요:
- 이메일: [contact@email.com]
- GitHub Issues: [repository_url]