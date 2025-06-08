# Vision Gait Analysis System

> MediaPipe 기반 보행 분석 시스템 - 비디오와 IMU 데이터를 이용한 보행 패턴 분석 및 HS/TO 이벤트 검출

## 📋 프로젝트 개요

본 시스템은 MediaPipe Pose 모델을 활용하여 비디오에서 보행 패턴을 분석하고, 다양한 보행 유형(정상, 마비성, 파킨슨, 운동실조성, 통증성 보행)에 대한 **Heel Strike (HS)**와 **Toe Off (TO)** 이벤트를 자동으로 검출하는 통합 분석 도구입니다.

### 주요 특징
- 🎯 **MediaPipe 기반 관절 추정**: 실시간 포즈 인식을 통한 발목 좌표 추출
- 📊 **보행 이벤트 검출**: HS/TO 이벤트 자동 감지 및 시각화
- 🔄 **배치 처리 지원**: 다수 세션 일괄 분석 기능
- 📈 **실시간 시각화**: PyQt5 기반 GUI로 분석 결과 실시간 확인
- 🌊 **신호 처리**: 고급 필터링 및 노이즈 제거 알고리즘 적용
- 📱 **IMU 데이터 연동**: 센서 데이터와 비디오 동기화 분석

## 🛠️ 시스템 아키텍처

```
vision_gait/
├── main.py                     # 메인 진입점
├── gait_analyzer_gui.py        # GUI 기반 단일 분석 도구
├── batch_gait_analyzer.py      # 배치 처리 GUI
├── gait_class.py              # 핵심 보행 분석 알고리즘
├── make_data.py               # 데이터 수집 시스템
├── experiment_data/           # 실험 데이터 저장소
│   ├── SA01/                 # 피험자별 폴더
│   │   ├── normal_gait/      # 정상 보행 세션들
│   │   ├── ataxic_gait/      # 운동실조성 보행
│   │   ├── hemiparetic_gait/ # 편마비성 보행
│   │   ├── pain_gait/        # 통증성 보행
│   │   └── parkinson_gait/   # 파킨슨성 보행
│   └── SA02/, SA03/...       # 추가 피험자들
├── support_label_data/        # 분석 결과 라벨 데이터
└── gait_analysis_output/      # 분석 결과 출력
```

## 🎯 지원하는 보행 유형

| 보행 유형 | 코드 | 설명 |
|-----------|------|------|
| Normal Gait | T01 | 정상 보행 패턴 |
| Ataxic Gait | T02 | 운동실조성 보행 (균형 장애) |
| Hemiparetic Gait | T03 | 편마비성 보행 (뇌졸중 후유증) |
| Pain Gait | T04 | 통증성 보행 (족부/하지 통증) |
| Parkinson Gait | T05 | 파킨슨성 보행 (경직성 보행) |

## 🚀 설치 및 환경 설정

### 필수 요구사항
- Python 3.8 이상
- OpenCV 호환 웹캠 또는 비디오 파일
- (선택사항) IMU 센서 (MPU6050 등)

### 설치 방법

1. **저장소 클론**
```bash
git clone https://github.com/your-username/vision_gait.git
cd vision_gait
```

2. **가상환경 생성 (권장)**
```bash
python -m venv vision_gait_env
# Windows
vision_gait_env\Scripts\activate
# Mac/Linux
source vision_gait_env/bin/activate
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

## 🖥️ 사용 방법

### 1. GUI 기반 단일 분석

```bash
python main.py
```

**주요 기능:**
- 비디오 파일 로드 및 프레임별 탐색
- 실시간 발목 좌표 시각화
- HS/TO 이벤트 마커 표시
- 보행 단계별 색상 구분
- 분석 결과 저장 (CSV, JSON)

### 2. 배치 처리 분석

```bash
python batch_gait_analyzer.py
```

**주요 기능:**
- 다수 세션 선택 및 일괄 처리
- 진행률 실시간 모니터링  
- 결과 파일 자동 명명 규칙 적용
- 처리 로그 및 오류 리포트

### 3. 데이터 수집 시스템

```bash
python make_data.py
```

**주요 기능:**
- 실시간 비디오 녹화
- IMU 센서 데이터 TCP 수신
- 정밀한 타임스탬프 동기화
- 세션별 자동 저장

## 📊 분석 결과 형식

### 1. 보행 이벤트 (gait_events.json)
```json
{
  "events": [
    {
      "frame": 45,
      "event": "HS",
      "foot": "left",
      "timestamp": 1.5
    }
  ],
  "analysis_info": {
    "walking_direction": "forward",
    "total_events": 24,
    "video_duration": 10.0
  }
}
```

### 2. 보행 단계 (support_labels.csv)
```csv
frame,timestamp,left_support,right_support,phase
0,0.000,1,0,left_single_support
15,0.500,0,1,right_single_support
30,1.000,1,1,double_support
```

### 3. 발목 좌표 데이터 (ankle_coordinates.csv)
```csv
frame,timestamp,left_ankle_x,right_ankle_x,left_ankle_filtered,right_ankle_filtered
0,0.000,0.45,0.55,0.451,0.549
1,0.033,0.46,0.54,0.452,0.548
```

## 🔧 알고리즘 상세

### 1. 보행 방향 감지
- 초기 15프레임에서 발목 Z축 좌표 분석
- 전진/후진 보행 자동 판별
- 좌표계 보정 및 정규화

### 2. 신호 처리 파이프라인
```python
# 노이즈 제거 다단계 처리
signal_filtered = apply_enhanced_noise_reduction(raw_signal)
# 1. 중앙값 필터 (스파이크 제거)
# 2. 가우시안 필터 (고주파 노이즈)  
# 3. 버터워스 저역통과 필터
```

### 3. 이벤트 검출 알고리즘
- **HS 검출**: 발목 X좌표 극값(peak) 기반
- **TO 검출**: 발목 X좌표 극값(valley) 기반
- 적응적 임계값 및 최소 간격 제약 적용

### 4. 보행 단계 분류
- Single Support: 한 발만 지지
- Double Support: 양발 동시 지지  
- Swing Phase: 유각기 (발이 공중에 있는 상태)

## 📈 성능 지표

| 측정 항목 | 성능 |
|-----------|------|
| 프레임 처리 속도 | ~30-60 FPS |
| HS/TO 검출 정확도 | ~85-95% |
| 메모리 사용량 | ~500MB (1080p 비디오) |
| 지원 비디오 형식 | MP4, AVI, MOV |

## 🔬 연구 및 임상 활용

### 적용 분야
- **임상 보행 분석**: 재활의학과, 정형외과
- **스포츠 과학**: 보행/달리기 폼 분석
- **연구**: 신경학적 질환 보행 패턴 연구
- **웨어러블 기술**: IMU 기반 보행 모니터링

### 출력 데이터 활용
- 보행 속도 계산
- 보행 주기 분석
- 좌우 비대칭성 평가
- 보행 안정성 지표 산출

## 🔧 고급 설정

### IMU 센서 연동 설정
```python
# 라즈베리파이에서 실행
# get_data.py 수정 필요
SERVER_IP = 'YOUR_PC_IP'  # PC IP 주소
SERVER_PORT = 5000        # 포트 번호
TARGET_HZ = 30           # 샘플링 주파수
```

### 분석 파라미터 조정
```python
# gait_class.py에서 설정 가능
NOISE_REDUCTION_STRENGTH = 0.7    # 노이즈 제거 강도
MIN_EVENT_INTERVAL = 0.3          # 최소 이벤트 간격 (초)
DIRECTION_DETECTION_FRAMES = 15   # 방향감지용 프레임 수
```

## 🐛 문제 해결

### 일반적인 문제

**Q: MediaPipe 설치 오류**
```bash
pip install --upgrade pip
pip install mediapipe --no-cache-dir
```

**Q: PyQt5 GUI가 표시되지 않음**
```bash
# Windows
pip install PyQt5 --upgrade
# Mac
brew install pyqt5
```

**Q: 비디오 코덱 문제**
```bash
pip install opencv-python-headless
# 또는
pip install opencv-contrib-python
```

### 성능 최적화

**메모리 사용량 감소:**
- 비디오 해상도 조정 (720p 권장)
- 배치 크기 조정
- 불필요한 데이터 저장 비활성화

**처리 속도 향상:**
- GPU 가속 활용 (CUDA 설치)
- MediaPipe 모델 복잡도 조정
- 멀티프로세싱 활용

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🤝 기여하기

1. Fork 프로젝트
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📞 지원 및 문의

- **이슈 리포트**: [GitHub Issues](https://github.com/your-username/vision_gait/issues)
- **기술 문의**: your-email@example.com
- **문서**: [Wiki](https://github.com/your-username/vision_gait/wiki)

## 📚 참고 자료

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [PyQt5 Documentation](https://doc.qt.io/qtforpython/)
- [Gait Analysis Fundamentals](https://www.physio-pedia.com/Gait_Analysis)

---

<div align="center">
  <strong>Vision Gait Analysis System</strong><br>
  MediaPipe 기반 지능형 보행 분석 도구
</div> 