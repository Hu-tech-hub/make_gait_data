# PyQt 기반 실험 데이터 수집 시스템 가이드라인

## 📋 시스템 개요

본 시스템은 비디오와 IMU 센서 데이터를 동기화하여 실시간으로 수집하고 저장하는 PyQt 기반 통합 실험 제어 시스템입니다.

### 주요 기능
- 실시간 비디오 프리뷰 및 녹화
- IMU 센서 데이터 TCP 수신 및 기록
- 정밀한 타임스탬프 기반 데이터 동기화
- 직관적인 GUI 인터페이스
- 세션별 자동 데이터 저장

## 🛠️ 시스템 구성요소

### 하드웨어 요구사항
- **라즈베리파이**: IMU 센서 장착 (I2C 연결)
- **IMU 센서**: MPU6050 또는 호환 센서
- **비디오 장비**: 웹캠/USB 카메라
- **PC**: PyQt 실행 환경

### 소프트웨어 요구사항
```bash
# Python 3.7 이상
pip install PyQt5
pip install opencv-python
pip install numpy

# 라즈베리파이 (IMU 송신용)
pip install smbus2
pip install bitstring
```

## 📝 섹션별 상세 설명

### <section1> 장비 및 소프트웨어 사전 준비

#### 1.1 라즈베리파이 설정
```bash
# I2C 활성화
sudo raspi-config
# Interface Options > I2C > Enable

# 필요 패키지 설치
sudo apt-get update
sudo apt-get install python3-smbus i2c-tools
pip3 install smbus2 bitstring
```

#### 1.2 IMU 센서 연결 확인
```bash
# I2C 장치 확인 (0x68이 보여야 함)
sudo i2cdetect -y 1
```

#### 1.3 개선된 get_data.py (30Hz 동기화 버전)
```python
#!/usr/bin/env python3
import smbus2
import time
import socket
import json
import threading
from bitstring import Bits

# 설정
BUS = smbus2.SMBus(1)
IMU_ADDR = 0x68
SERVER_IP = 'YOUR_PC_IP'  # PC IP 주소로 변경
SERVER_PORT = 5000
TARGET_HZ = 30  # 비디오와 동일한 Hz

# MPU6050 레지스터
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# 센서 초기화
BUS.write_byte_data(IMU_ADDR, PWR_MGMT_1, 0)

def read_sensor_data():
    """센서 데이터 읽기"""
    # 가속도 읽기 (6바이트)
    accel_data = BUS.read_i2c_block_data(IMU_ADDR, ACCEL_XOUT_H, 6)
    ax = (accel_data[0] << 8) | accel_data[1]
    ay = (accel_data[2] << 8) | accel_data[3]
    az = (accel_data[4] << 8) | accel_data[5]
    
    # 자이로 읽기 (6바이트)
    gyro_data = BUS.read_i2c_block_data(IMU_ADDR, GYRO_XOUT_H, 6)
    gx = (gyro_data[0] << 8) | gyro_data[1]
    gy = (gyro_data[2] << 8) | gyro_data[3]
    gz = (gyro_data[4] << 8) | gyro_data[5]
    
    # 2의 보수 변환
    def twos_comp(val):
        if val > 32767:
            val -= 65536
        return val
    
    # 단위 변환
    ax = twos_comp(ax) / 16384.0 * 9.80665  # m/s^2
    ay = twos_comp(ay) / 16384.0 * 9.80665
    az = twos_comp(az) / 16384.0 * 9.80665
    
    gx = twos_comp(gx) / 131.0  # deg/s
    gy = twos_comp(gy) / 131.0
    gz = twos_comp(gz) / 131.0
    
    return {
        'accel': {'x': ax, 'y': -ay, 'z': az},
        'gyro': {'x': gx, 'y': -gy, 'z': gz}
    }

def main():
    # TCP 연결
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_IP, SERVER_PORT))
    print(f"서버 연결됨: {SERVER_IP}:{SERVER_PORT}")
    
    start_time = time.time()
    sample_count = 0
    
    try:
        while True:
            # 센서 데이터 읽기
            sensor_data = read_sensor_data()
            
            # 타임스탬프 추가
            elapsed = time.time() - start_time
            sensor_data['timestamp'] = elapsed
            
            # JSON 전송
            json_data = json.dumps(sensor_data) + '\n'
            client.sendall(json_data.encode('utf-8'))
            
            sample_count += 1
            
            # 30Hz 유지
            next_sample = start_time + (sample_count / TARGET_HZ)
            sleep_time = next_sample - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        client.close()
        BUS.close()

if __name__ == '__main__':
    main()
```

### <section2> PyQt GUI 실행 및 사용자 준비

#### 2.1 GUI 구성요소

| 구성요소 | 설명 | 위치 |
|---------|------|------|
| 비디오 프리뷰 | 실시간 카메라 영상 표시 | 좌측 패널 |
| IMU 데이터 모니터 | 실시간 센서값 표시 | 우측 상단 |
| 실험 제어 | 시작/종료 버튼, 타이머 | 우측 중앙 |
| 시스템 로그 | 상태 메시지 표시 | 우측 하단 |

#### 2.2 실행 방법
```bash
# PC에서 실행
python experiment_control_system.py

# 라즈베리파이에서 실행 (PC 실행 후)
python get_data.py
```

### <section3> 실험 세션 시작 (동기화 커맨드)

#### 3.1 동기화 프로세스
```python
# 시작 버튼 클릭 시 내부 동작
sync_start_time = time.time()  # 동기화 기준 시간
video_capture.start_recording(sync_start_time)
imu_receiver.start_recording(sync_start_time)
```

#### 3.2 데이터 동기화 메커니즘
- **비디오**: 각 프레임에 `timestamp = current_time - sync_start_time` 기록
- **IMU**: 각 샘플에 `sync_timestamp = current_time - sync_start_time` 기록
- 모든 타임스탬프는 sync_start_time 기준 상대 시간

#### 3.3 GUI 상태 변화
- 시작 버튼: 비활성화
- 종료 버튼: 활성화
- 타이머: 실시간 경과 시간 표시
- 상태 표시: "녹화중..."

### <section4> 실험 종료 및 동기화

#### 4.1 종료 프로세스
```python
# 종료 버튼 클릭 시
sync_end_time = time.time()
duration = sync_end_time - sync_start_time

# 유효 구간 데이터만 추출
valid_video_frames = filter(lambda f: 0 <= f.timestamp <= duration, frames)
valid_imu_samples = filter(lambda s: 0 <= s.sync_timestamp <= duration, samples)
```

#### 4.2 자동 처리 작업
1. 비디오/IMU 버퍼 수집 중단
2. 유효 구간 데이터 클리핑
3. 저장 디렉토리 자동 생성
4. 파일 저장 및 메타데이터 생성

### <section5> 동기화된 데이터 저장

#### 5.1 저장 구조
```
experiment_data/
└── session_20250604_143022/
    ├── video.mp4          # 동기화된 비디오
    ├── imu_data.csv       # 동기화된 IMU 데이터
    └── metadata.json      # 세션 메타데이터
```

#### 5.2 IMU CSV 형식
```csv
sync_timestamp,timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
0.000,1234567890.123,0.123,-0.456,9.789,1.234,-5.678,0.012
0.033,1234567890.156,0.124,-0.457,9.790,1.235,-5.679,0.013
...
```

#### 5.3 메타데이터 JSON 구조
```json
{
  "session_id": 1,
  "sync_start_time": 1234567890.123,
  "sync_end_time": 1234567920.456,
  "duration": 30.333,
  "video_fps": 30,
  "video_frames": 910,
  "imu_samples": 910,
  "timestamp": "2025-06-04T14:30:22"
}
```

## 💻 사용 예시

### 기본 워크플로우
1. **시스템 준비**
   - PC에서 GUI 실행
   - 카메라 연결 확인 (프리뷰 표시)
   - 라즈베리파이에서 IMU 송신 시작
   - IMU 연결 상태 확인

2. **실험 시작**
   - F5 또는 '시작' 버튼 클릭
   - 실험 수행
   - 타이머로 진행 상황 확인

3. **실험 종료**
   - F6 또는 '종료' 버튼 클릭
   - 자동 저장 완료 대기
   - 로그에서 저장 결과 확인

### 입출력 예시

#### 입력
```
[14:30:00] 시스템 초기화 완료
[14:30:02] 비디오: 카메라 연결됨
[14:30:02] 비디오: 640x480 @ 30.0fps
[14:30:05] IMU: IMU 연결됨: 192.168.0.100
[14:30:10] 녹화 시작 - 세션 #1
[14:30:10] 저장 경로: experiment_data/session_20250604_143010
```

#### 출력
```
[14:30:40] 녹화 종료 - 세션 #1
[14:30:40] 총 시간: 30.00초
[14:30:40] 비디오 프레임: 900
[14:30:40] IMU 샘플: 900
[14:30:40] IMU 데이터 저장: 900 샘플
[14:30:40] 데이터 저장 완료!
```

## 🔧 문제 해결

### 일반적인 문제

1. **IMU 연결 실패**
   - 라즈베리파이 IP 주소 확인
   - 방화벽 설정 확인
   - 포트 5000 사용 가능 여부 확인

2. **비디오 프레임 드롭**
   - 카메라 USB 연결 확인
   - CPU 사용률 확인
   - 낮은 해상도로 테스트

3. **동기화 오차**
   - NTP 시간 동기화 확인
   - 네트워크 지연 최소화 (유선 연결 권장)

### 성능 최적화

1. **버퍼 크기 조정**
   ```python
   # 메모리가 부족한 경우
   MAX_BUFFER_SIZE = 1000  # 프레임/샘플
   ```

2. **샘플링 레이트 조정**
   ```python
   VIDEO_FPS = 15  # 저사양 시스템
   TARGET_HZ = 15  # IMU도 동일하게
   ```

## 📊 데이터 분석 팁

### 동기화 검증
```python
import pandas as pd
import cv2

# CSV 로드
imu_df = pd.read_csv('imu_data.csv')

# 비디오 정보
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 동기화 확인
print(f"IMU 시작: {imu_df['sync_timestamp'].min()}")
print(f"IMU 종료: {imu_df['sync_timestamp'].max()}")
print(f"비디오 길이: {frame_count/fps}초")
```

### 시각화 예제
```python
import matplotlib.pyplot as plt

# 가속도 데이터 플롯
plt.figure(figsize=(10, 6))
plt.plot(imu_df['sync_timestamp'], imu_df['accel_x'], label='X')
plt.plot(imu_df['sync_timestamp'], imu_df['accel_y'], label='Y')
plt.plot(imu_df['sync_timestamp'], imu_df['accel_z'], label='Z')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.show()
```

## 🚀 확장 가능성

- 다중 카메라 지원
- 추가 센서 통합 (GPS, 심박수 등)
- 실시간 데이터 분석
- 클라우드 저장 연동
- 이벤트 마킹 기능
- 자동 품질 검증

## 📄 라이선스

본 시스템은 연구 및 교육 목적으로 자유롭게 사용 가능합니다.