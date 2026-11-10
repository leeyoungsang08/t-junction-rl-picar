# PiCar Gazebo 검증 (SUMO → Gazebo → 실차)

SUMO에서 학습·선택한 교차로 제어(오래된 요청 우선 + 최대 대기 20 s, 또는 DQN)를 **실차와 같은 방식으로 달리는 차량** 위에서 돌려 보는 단계다.
SUMO는 차량을 "차로 위의 점"으로 움직이지만, Gazebo에서는 차량이 전면 카메라로 오른쪽 차선을 따라가고, 하부 IR로 정지선에서 서고, 교차로 안에서는 선 없이 돈다.
SUMO가 못 보여 주는 문제를 실차 전에 찾는 것이 목적이다. 차선 추종, 정지선 정지, 교차로 안 회전, 모터 편차, 앞차 감지가 여기에 해당한다.

2026-11-10 갱신: 코스를 사진 기반 추정 치수와 실제 선 배치(도로마다 3줄)로 바꿨다([PICAR_COURSE_ESTIMATE_KO.md](PICAR_COURSE_ESTIMATE_KO.md)).
차량 주행도 실차 방식(카메라 차선 추종 + IR 정지선)으로 새로 작성했고, 비보호 우회전 규칙을 추가했으며, DQN을 다시 학습했다(`runs/picar_v9_photo_rt`).

## 1. 구성

```
course.yaml (SUMO와 같은 코스) ──> world_gen.py ──> 바닥 텍스처 PNG + SDF 월드 + maneuvers.yaml + course_meta.yaml
                                                         │
Gazebo Fortress: 차량 N대 (DiffDrive, 전면 카메라 + 아래 보는 카메라 = 4채널 IR 대신)
   │ ros_gz_bridge
   ├─ ir_camera  (아래 카메라 → line_values 4채널)       ← 실차: IR 센서 노드 (정지선 감지)
   ├─ motor_model (지연·데드밴드·좌우 모터 편차)           ← 실차: motor_diff_node.py (GPIO PWM)
   ├─ ros2_camera_drive_node = CameraBrain (실차와 같은 코드) ← 실차: 그대로 사용
   │     전면 카메라 차선 추종 · IR 정지선 · 주행거리 기반 APPROACH · 교차로 안 회전 · 앞차 정지
   ├─ course_monitor (가상 출구 빔·초음파, 충돌/이탈 기록 metrics.json)  ← 실차: exit_beam.py, HC-SR04
   └─ picar_sim.central_server (UDP v2, 두 T자, 예약·허가, 비보호 우회전)  ← 실차: 같은 서버
```

실차와 공유하는 파일은 `picar_lab/car_agent/{lane_camera,camera_brain,agent,drive,calibration,ros2_camera_drive_node}.py`, `picar_sim/{central_server,right_turn}.py`다.
Gazebo 전용 파일은 `picar_gazebo/`에 있다(월드 생성, 카메라 → IR, 모터 모델, 모니터, 런치, 2D 렌더러).

### 카메라 차선 추종 (`lane_camera.py`)

UROP2 실차 방식을 그대로 따른다. 회색조 → 이진화 → 버드아이뷰 → 선 검출 → 중심 편차 → 조향 순서다.
- 버드아이뷰: 카메라 장착값(높이·숙인 각도·화각)에서 계산한 변환표를 쓴다. 실차에서는 실측 장착값이나 기존 4점 변환으로 바꾼다.
- 정지선처럼 한 줄 전체가 어두운 가로 테이프는 버드아이뷰에서 지우고, 남은 덩어리마다 2차 곡선을 맞춘다.
- 좌우 판정은 선의 가장 가까운 보이는 부분의 위치로 하고, 직전 프레임의 추적을 우선한다.
- 오른쪽 선을 우선으로 따라간다(왼쪽 선만 보이면 왼쪽 선 사용). 선을 법선 방향으로 반 차로 옮긴 곡선 위, 차에서 0.3 m(급한 곡선 0.21 m) 떨어진 점을 순수 추종(pure pursuit)으로 따라간다.
- **카메라 장착 가정**: 차 중심 앞 0.06 m, 높이 0.15 m, 35° 숙임, 수평 화각 62°(Pi 카메라 v2). 더 앞에 낮게 달면(0.10 m 앞, 0.13 m 높이) 안쪽 차로 급한 우회전에서 오른쪽 선이 시야 밖으로 나가 차로 이탈이 늘었다. 실차 장착값을 재서 `lane_camera.CAMERA`에 넣는다.

### 교차로

- 정지선: 하부 IR 4채널 중 3개 이상이 검으면 정지선이다. 차로 선은 양옆 10 cm에 있어 IR에 들어오지 않는다.
- 접근(APPROACH) 요청: 실물 코스에 접근 테이프가 없으므로, 교차로를 나온 뒤 주행 거리(`approach_after_exit_m`)로 보낸다. 정지선에 먼저 도착하면 그 자리에서 보낸다.
- 교차로 안: 섬 모서리가 연결로 쪽으로 휘어 있어 카메라를 따라가면 직진 차가 연결로로 빨려 든다. 그래서 주행 거리 기반 회전(`JunctionManeuver`)을 쓰고, 나온 뒤 카메라가 차선을 다시 잡으면 끝난다. 못 잡으면 탐색 후 FAULT다.
- 제자리 회전 중에는 초음파를 무시한다. 옆 차로에 서 있는 차를 장애물로 보고 교차로 안에서 멈추면 서로 기다리는 교착이 생기기 때문이다(2D 점검에서 발견).

### 비보호 우회전 (`picar_sim/right_turn.py`, `signal.unprotected_right`)

정지선에 멈춘(ARRIVE) 우회전 차는 세 조건을 모두 만족할 때 서버의 허가로 갈 수 있다. 다른 방향 차가 교차로 안에 없거나 예약하지 않았고, 녹색 쪽에서 다가오는 차가 없고, 교차로가 녹색이면서 정상 상태(황색·전환·결함 아님)여야 한다.
이 차가 지나는 동안 다른 방향 차는 기다린다. PiCar는 스스로 양보하지 못하므로 허가는 서버가 낸다. SUMO 환경과 실차 서버가 같은 함수를 쓴다.
이렇게 엄격한 이유가 있다. 차로가 0.21 m인데 0.24 × 0.17 m 차가 교차로 안에서 제자리 회전을 하면 모서리가 옆 차로로 약 4 cm 넘어간다. 실제로 2D 점검에서 J2의 두 우회전(W→N과 N→E)이 회전 지점 사이 0.21 m 거리에서 충돌했다.
그 결과 효과는 작다. SUMO 규칙 제어기 기준 6대 평균 지연이 3.08 → 3.03 s로 줄었고, 4대는 1.47 → 1.66 s로 오히려 늘었다.

## 2. 설치 (Ubuntu 22.04, 한 번)

```bash
git clone https://github.com/leeyoungsang08/t-junction-rl-picar.git ~/t-junction-rl-picar
cd ~/t-junction-rl-picar && git checkout picar-real2sim
bash picar_gazebo/scripts/setup_ubuntu2204.sh      # sudo 비밀번호 필요, 20~40분
```

이미 설치했다면 `git pull` 후 `cd ~/picar_ws && colcon build --symlink-install --packages-select picar_gazebo`.

## 3. 실행

```bash
ros2 launch picar_gazebo calibrate.launch.py                      # 차량 보정 (약 25 s) → ~/picar_gz_runs/car_calibration.yaml
ros2 launch picar_gazebo course.launch.py cars:=3                 # 기본 제어(오래된 요청 우선 + 최대 대기 20 s)
ros2 launch picar_gazebo course.launch.py cars:=3 \
    policy:=$PICAR_REPO/runs/picar_v9_photo_rt/seed42/best_policy.npz mode:=active   # DQN이 교차로 J 제어
ros2 launch picar_gazebo course.launch.py cars:=3 beam_fault_start:=60 beam_fault_stop:=80   # 결함 주입
```

창 없이: `gui:=false`. 결과: `~/picar_gz_runs/<시각>/metrics.json`, 서버 로그 `central.jsonl`.
상태 보기: `ros2 topic echo /car0/drive_state` (`mode:zone:edge`). 카메라 영상: `ros2 run rqt_image_view rqt_image_view /model/car0/front_cam`.

## 4. 합격 기준 (차량 3대, 10분, 시드 3개)

| 항목 | 기준 |
|---|---|
| 충돌 (차체 겹침) | 0 |
| 허가 없는 진입 (`legal=False`) | 0 |
| FAULT (차선 못 찾음 등) | 0 |
| 차로 이탈 (교차로 밖, 차로 중심에서 4.5 cm 초과 = 차체가 선 하나를 완전히 덮음) | 0 |
| 정지 위치 | 범퍼가 정지선 바깥 끝 기준 ±5 cm, 충돌구간(박스) 안으로 들어가지 않음 |
| 빔 결함 중 신규 진입 | 0 |
| 교착 (모든 차가 30 s 이상 정지) | 0 |

차로 이탈 기준은 차로가 0.21 m로 좁아져서 이전 8 cm에서 바꿨다(이전 기준으로는 차체가 선을 넘어 옆 차로에 6 cm 들어가도 통과였다).

한 번에 확인 (보정 → 시드 1·2·3 × 10분 → 빔 결함 60~80 s → 판정, 약 40분):

```bash
bash picar_gazebo/scripts/run_acceptance.sh                         # 기본 제어
CONTROLLERS="rule dqn" bash picar_gazebo/scripts/run_acceptance.sh  # 같은 시드로 규칙과 DQN 비교 (약 75분)
MINUTES=2 SEEDS="1" bash picar_gazebo/scripts/run_acceptance.sh     # 짧은 확인
# → ~/picar_gz_runs/acceptance_<시각>/summary.md (항목별 OK/FAIL + 교차로 J 성능표), 종료 코드 0 = 합격
```

교착은 `pose_logger.py`가 남긴 `poses.csv`에서 판정한다(모든 차가 1초에 1 cm 미만 이동한 상태가 30 s 이상).
빔 결함 중 신규 진입은 `central.jsonl`에서 판정한다(J 감독기가 fault인 동안 새로 나간 J 허가 수).

## 5. 2D 사전 점검 (ROS 없이, 이 저장소에서 바로)

Gazebo 전에 같은 `CameraBrain`을 2D 운동 모델로 돌린다. 전면 카메라 영상은 Gazebo가 그리는 바닥 텍스처를 같은 카메라 모델로 원근 투영해서 만들고, IR은 텍스처 픽셀에서 읽는다. 차량은 중앙 서버와 실제 UDP로 통신한다.
모터 모델은 명령 지연 0.05 s, 데드밴드 0.03 m/s, 오른쪽 바퀴 0.97배, 가속 제한이다. 차량 코드는 이 값을 모르고 보정으로만 안다.

```bash
PYTHONPATH=.:picar_gazebo python3 -m picar_gazebo.kinematic_check --cars 3 --seconds 600 --seed 1
PYTHONPATH=.:picar_gazebo python3 -m picar_gazebo.kinematic_check --cars 3 --seconds 600 --seed 1 \
    --policy runs/picar_v9_photo_rt/seed42/best_policy.npz --mode active
```

결과 (추정 치수, 차량 3대 × 600 s, 시드 1·2·3):

| 제어 | 시드 | 교차로 통과 | 충돌 | FAULT | 허가 없는 진입 | 차로 이탈 | 차로 중심 오차 p99 / 최대 [cm] | 정지 위치 오차 최대 [cm] | 비보호 우회전 허가 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 오래된 요청 우선 | 1 | 96 | 0 | 0 | 0 | 3 | 3.2 / 6.0 | 2.6 | 7 |
| 오래된 요청 우선 | 2 | 91 | 0 | 0 | 0 | 3 | 3.2 / 5.5 | 2.1 | 4 |
| 오래된 요청 우선 | 3 | 94 | 0 | 0 | 0 | 3 | 3.2 / 5.8 | 2.7 | 6 |
| DQN seed42 (active) | 1 | 95 | 0 | 0 | 0 | 4 | 3.4 / 5.8 | 1.6 | 7 |
| DQN seed42 (active) | 2 | 90 | 0 | 0 | 0 | 3 | 3.2 / 5.8 | 2.2 | 4 |
| DQN seed42 (active) | 3 | 94 | 0 | 0 | 0 | 1 | 3.2 / 4.8 | 2.7 | 8 |

원자료: `runs/gazebo_precheck/kinematic_camera_3cars_600s.json`

- 충돌, FAULT, 허가 없는 진입은 모든 조건에서 0이다. 통과 수는 규칙과 DQN이 비슷하다(90~96회). 정지 위치 오차는 최대 2.7 cm다.
- 차로 이탈은 run마다 1~4번이다. 모두 가장 급한 안쪽 차로 우회전 모서리(차로 중심 반경 0.245 m)에서 났다. 주행 시간의 99%는 차로 중심에서 3.4 cm 이내이고, 4.5 cm를 넘는 시간은 0.03~0.23%다(최대 4.8~6.0 cm). **이 항목은 합격 기준(0)을 아직 못 맞춘다.** 실차에서는 카메라 화각이 넓거나 차선을 더 일찍 보면 줄어든다. 실측 코스 반경이 더 크면 사라질 수 있다.
- 카메라 검출 단일 프레임 점검(모든 차로, 차로 중심 ±4 cm, 234장면): 목표점 오차 2.5 cm 이내가 234/234였다.

이 단계에서 찾아 고친 것:
1. 순환 경로의 마지막 → 첫 구간 U턴 → 10분 주행에서 차량 노드가 `KeyError`로 죽음(이전 커밋).
2. Gazebo Fortress PosePublisher 설정 → 차 위치가 안 나와 보정·측정이 멈춤(이전 커밋).
3. 곡선에서 차선이 기울면 차 기준 옆 방향 반 차로는 `lw/2 ÷ cos`가 됨 → 법선 방향으로 옮긴 중심선을 씀.
4. 급한 곡선에서 짧게 보인 선을 직선으로 늘리면 곡률을 1/3로 잡음 → 보이는 범위 안에서만 목표점을 잡음.
5. 짧은 앞보기 거리로 바꿀 때 목표점 부호가 뒤집힘 → 부호가 같을 때만 바꿈.
6. 제자리 회전 중 초음파 원뿔이 옆 차로 차를 쓸고 지나가며 교차로 안에서 정지 → 교착 → 회전 중 초음파 무시.
7. 비보호 우회전 두 대가 교차로 안에서 제자리 회전하다 충돌 → 다른 방향 이동과는 모두 상충으로 처리.

## 6. 실차로 옮길 때 (같은 코드, 바뀌는 것만)

| Gazebo | 실차 |
|---|---|
| `calibrate.launch.py` (위치 자동 측정) | `python3 -m picar_lab.car_agent.calib_drive_node --ros-args -p test:=straight` 등 4가지 시험 후 줄자·각도기로 측정 → `python3 -m picar_lab.car_agent.calibration --straight-m ... --drift-deg ... --roll-slow-m ... --roll-fast-m ... --spin-deg ...` |
| 전면 카메라 (`/model/carN/front_cam`) | Pi 카메라 → `image` 토픽 (해상도는 자동 반영). 장착값은 `camera_x/camera_z/camera_pitch_deg/camera_hfov_deg` 파라미터 |
| `ir_camera` | IR 센서 노드 (`line_values` 4채널, 검은 테이프 값 `dark_value`) |
| `motor_model` + DiffDrive | `motor_diff_node.py` (좌우 PWM; 기존 Yahboom motor_node는 주행 중 조향 불가) |
| `course_monitor` 가상 빔·초음파 | `exit_beam.py`, HC-SR04 (`range` 토픽) |
| `maneuvers.yaml` (world_gen) | 같은 파일. 실측 `course.yaml`로 다시 생성 |

실차 첫 단계 순서: 바퀴 들고 시험 → 1대 저속 → 서버 shadow → active. 차량별 `car_calibration.yaml`을 따로 둔다.

## 7. Gazebo에서 통과해도 실차에서 다를 수 있는 것

- 코스 치수: 사진 기반 추정값(±5%). 차로 폭 0.21 m가 맞는지가 가장 중요하다.
- 카메라: 장착값, 렌즈 왜곡, 조명·반사, 노출. Gazebo 영상은 깨끗하다.
- 바닥 마찰·바퀴 미끄러짐·배터리 전압에 따른 속도 변화. 보정은 측정한 그날 배터리 기준이라, 배터리가 바뀌면 다시 한다.
- 교차로 안 회전: 실차가 제자리 회전 대신 호를 그리며 돈다면 교차로 계획(`maneuvers.yaml`)과 우회전 상충 규칙을 다시 본다.
- 초음파의 실제 빔 폭·반사(곡선 추돌 위험은 실차에서 더 클 수 있음 → 처음엔 차 간격을 넓게).
- Wi-Fi 지연·끊김(서버는 TTL·하트비트로 처리하지만 실제 무선 상태는 실측 필요).
