# 통합 보행 분석 시스템 (Integrated Gait Analysis System)

IMU 센서 데이터와 영상 데이터를 기반으로 보행 분석 및 낙상 위험 예측을 수행하는 통합 시스템입니다.

## 🎯 주요 기능

### 1. 시각적 데이터 확인 및 수정
- 센서 데이터, 보행 영상, 이벤트 정보의 시간 축 기준 동기화
- 보행 구간 선택 및 HS/TO 시점 검토/수정
- 영상 프레임 위 관절 추정 결과 시각적 확인

### 2. 자동 보행 지표 계산
- MediaPipe 기반 관절 추정 수행
- 보폭, 속도, 보행 주기, 보행률, ROM 등 주요 지표 자동 산출
- IMU 시계열과 정렬된 학습용 정답(label) 생성

### 3. 시계열 회귀 모델 학습
- LSTM, TCN, 1D CNN 등 다양한 모델 아키텍처 지원
- IMU 데이터만으로 보행 지표 예측하는 회귀 모델 학습
- 교차 검증 및 성능 평가

### 4. 실시간 예측 및 검증
- 학습된 모델을 통한 새로운 IMU 데이터 보행 지표 추론
- 예측값과 실제값 비교 시각화
- 오차 분석 및 모델 성능 평가

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    통합 GUI 시스템                           │
├─────────────────────────────────────────────────────────────┤
│ 1. 데이터 동기화 │ 2. 이벤트 검출 │ 3. 지표 계산 │ 4. 모델 학습 │ 5. 예측 검증 │
└─────────────────────────────────────────────────────────────┘
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│  데이터 처리     │   보행 지표      │   시계열 모델    │   성능 평가      │
│  및 시각화      │   계산          │   학습/예측     │   및 검증       │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
```

## 📋 시스템 요구사항

- **Python**: 3.8+
- **주요 라이브러리**:
  - OpenCV 4.8+
  - MediaPipe 0.10+
  - TensorFlow 2.13+
  - PyQt5 5.15+
  - scikit-learn 1.3+
  - NumPy, Pandas, SciPy

## 🚀 설치 및 설정

### 1. 저장소 클론
```bash
git clone [repository-url]
cd vision_gait
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 시스템 테스트
```bash
python test_system.py
```

## 📊 데이터 형식

### IMU 데이터 (CSV)
```csv
frame_number,sync_timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0,0.000,0.086,9.701,1.834,-1.95420,0.05344,-1.72519
1,0.033,0.077,9.852,1.801,-1.50745,2.10028,-1.74367
```

### 메타데이터 (JSON)
```json
{
  "session_id": 16,
  "video_fps": 30,
  "video_frames": 337,
  "imu_hz": 30,
  "imu_samples": 336,
  "duration": 11.22
}
```

### 보행 지표 출력 (CSV)
```csv
start_frame,end_frame,foot,stride_length,velocity,cycle_time,cadence,hip_rom,knee_rom,ankle_rom,stance_ratio
45,89,left,0.523,0.892,1.467,40.9,23.4,45.2,15.8,62.3
```

## 🖥️ 사용법

### 1. 통합 GUI 실행
```bash
python integrated_gait_system_gui.py
```

### 2. 예제 파이프라인 실행
```bash
python example_pipeline.py --video_path path/to/video.mp4 --imu_path path/to/imu_data.csv
```

### 3. 기존 GUI (이벤트 검출 전용)
```bash
python main.py
```

### 4. 배치 처리
```bash
python batch_gait_analyzer.py
```

## 📁 프로젝트 구조

```
vision_gait/
├── integrated_gait_system_gui.py    # 통합 GUI 시스템
├── gait_metrics_calculator.py       # 보행 지표 계산 모듈
├── time_series_model.py             # 시계열 회귀 모델
├── data_processing_utils.py         # 데이터 처리 유틸리티
├── gait_class.py                    # 기존 보행 분석 클래스
├── gait_analyzer_gui.py             # 기존 GUI (이벤트 검출)
├── example_pipeline.py              # 전체 파이프라인 예제
├── test_system.py                   # 시스템 테스트 스크립트
├── system_guide.md                  # 상세 사용 가이드
├── requirements.txt                 # 의존성 목록
├── experiment_data/                 # 실험 데이터
├── support_label_data/              # 지원 라벨 데이터
└── README.md                        # 이 파일
```

## 🔄 분석 워크플로우

### Step 1: 데이터 준비
1. 동기화된 IMU 데이터와 보행 영상 준비
2. 메타데이터 파일 확인
3. 데이터 품질 검증

### Step 2: 이벤트 검출
1. MediaPipe 기반 관절 추정
2. HS/TO 이벤트 자동 검출
3. 시각적 검토 및 수정

### Step 3: 보행 지표 계산
1. 보행 주기별 공간적 지표 산출
2. 관절 가동 범위(ROM) 계산
3. 품질 검증 및 이상치 제거

### Step 4: 모델 학습
1. IMU 특징 추출 및 데이터셋 생성
2. 시계열 회귀 모델 학습
3. 교차 검증 및 성능 평가

### Step 5: 예측 및 검증
1. 새로운 IMU 데이터로 보행 지표 예측
2. 실제값과 예측값 비교
3. 모델 성능 분석

## 📈 출력 결과

### 보행 지표
- **보폭 (Stride Length)**: 동일 발의 연속 HS 간 거리
- **속도 (Velocity)**: 보폭/보행주기
- **보행률 (Cadence)**: 분당 걸음 수
- **관절 ROM**: 엉덩이, 무릎, 발목 가동 범위
- **입각기 비율**: 전체 주기 대비 입각기 시간

### 모델 성능 지표
- **MAE (Mean Absolute Error)**: 평균 절대 오차
- **RMSE (Root Mean Square Error)**: 평균 제곱근 오차
- **R² (Coefficient of Determination)**: 결정 계수

## 🔧 고급 설정

### 모델 하이퍼파라미터
```python
config = {
    'model_type': 'lstm',        # 'lstm', 'tcn', 'cnn1d'
    'window_size': 90,           # IMU 윈도우 크기 (프레임)
    'overlap': 0.5,              # 윈도우 겹침 비율
    'test_size': 0.2,            # 테스트 셋 비율
    'pixel_to_meter_ratio': 0.001 # 픽셀-미터 변환 비율
}
```

### 실시간 처리 설정
```python
# GPU 가속 활용
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 🧪 실험 및 검증

### 교차 검증
```bash
python example_pipeline.py --config cross_validation_config.json
```

### 성능 벤치마크
```bash
python benchmark_models.py --models lstm,tcn,cnn1d
```

## 📚 참고 자료

- **MediaPipe**: https://mediapipe.dev/
- **TensorFlow**: https://www.tensorflow.org/
- **보행 분석 이론**: 관련 논문 및 연구 자료
- **시스템 가이드**: `system_guide.md` 참조

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- MediaPipe 팀의 포즈 추정 기술
- TensorFlow/Keras 커뮤니티
- PyQt5 GUI 프레임워크
- OpenCV 컴퓨터 비전 도구들
- 보행 분석 연구 커뮤니티

---

**개발팀**: 보행 분석 연구팀  
**연락처**: [contact@email.com]  
**버전**: 1.0.0  
**최종 업데이트**: 2025년 1월