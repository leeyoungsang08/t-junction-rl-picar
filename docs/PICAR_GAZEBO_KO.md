# PiCar Gazebo 검증 (SUMO → Gazebo → 실차)

SUMO에서 고른 교차로 제어(오래된 요청 우선 + 최대 대기 20 s, 또는 DQN)를 **실차와 같은 차량 코드**로 돌려 보는 단계다.
SUMO는 차량을 "차로 위의 점"으로 움직이지만, Gazebo에서는 차량이 바닥 테이프를 보고 스스로 달리고 교차로 안에서는 테이프 없이 돈다.
그래서 SUMO가 못 보여 주는 문제(라인 추종, 표식 인식, 교차로 안 회전, 모터 편차, 앞차 감지)를 실차 전에 찾는 것이 목적이다.

## 1. 구성

```
course.yaml (SUMO와 같은 코스) ──> world_gen.py ──> 바닥 텍스처 PNG + SDF 월드 + maneuvers.yaml + course_meta.yaml
                                                         │
Gazebo Fortress: 차량 N대 (DiffDrive, 아래 보는 카메라 = 4채널 IR 대신)
   │ ros_gz_bridge
   ├─ ir_camera  (영상 → line_values 4채널)          ← 실차: IR 센서 노드
   ├─ motor_model (지연·데드밴드·좌우 모터 편차)       ← 실차: motor_diff_node.py (GPIO PWM)
   ├─ ros2_drive_node = CarBrain (실차와 같은 코드)    ← 실차: 그대로 사용
   │     라인 추종 · 표식 인식 · 교차로 에이전트 · 교차로 안 회전 · 앞차 정지
   ├─ course_monitor (가상 출구 빔·초음파, 충돌/이탈 기록 metrics.json)  ← 실차: exit_beam.py, HC-SR04
   └─ picar_sim.central_server (UDP v2, 두 T자, 예약·허가)               ← 실차: 같은 서버
```

실차와 공유하는 파일: `picar_lab/car_agent/{agent,brain,drive,calibration,ros2_drive_node}.py`, `picar_sim/central_server.py`.
Gazebo 전용: `picar_gazebo/` (월드 생성, 카메라→IR, 모터 모델, 모니터, 런치).

## 2. 설치 (Ubuntu 22.04, 한 번)

```bash
git clone https://github.com/leeyoungsang08/t-junction-rl-picar.git ~/t-junction-rl-picar
cd ~/t-junction-rl-picar && git checkout picar-real2sim
bash picar_gazebo/scripts/setup_ubuntu2204.sh      # sudo 비밀번호 필요, 20~40분
```

스크립트가 하는 일: ROS 2 Humble(공식 `ros2-apt-source` 방식) + `ros-humble-ros-gz`(Gazebo Fortress) + colcon 설치 →
`~/picar_ws`에 `picar_gazebo` 빌드 → `~/.bashrc`에 `source`와 `PICAR_REPO` 추가 → ROS 없는 테스트와 2D 사전 점검 실행.
SUMO·PyTorch는 Gazebo 단계에 필요 없다.

## 3. 실행

새 터미널에서:

```bash
# (1) 차량 보정: 직진·정지·제자리 회전을 자동으로 하고 Gazebo 위치로 측정 (약 25 s)
ros2 launch picar_gazebo calibrate.launch.py
#     → ~/picar_gz_runs/car_calibration.yaml

# (2) 코스: 3대, 기본 제어(오래된 요청 우선 + 최대 대기 20 s)
ros2 launch picar_gazebo course.launch.py cars:=3

# (3) DQN이 실제로 교차로 J를 제어
ros2 launch picar_gazebo course.launch.py cars:=4 \
    policy:=$PICAR_REPO/runs/picar_v7_direct_guard20/seed43/best_policy.npz mode:=active

# (4) 결함 주입: 60~80 s 동안 J 출구 빔 끊김 → 신규 진입 0이어야 함
ros2 launch picar_gazebo course.launch.py cars:=3 beam_fault_start:=60 beam_fault_stop:=80

# (5) 모터를 더 나쁘게: 보정을 같은 값으로 다시 한 뒤 실행
ros2 launch picar_gazebo calibrate.launch.py motor_latency_s:=0.08 gain_right:=0.92
ros2 launch picar_gazebo course.launch.py cars:=3 motor_latency_s:=0.08 gain_right:=0.92
```

창 없이: `gui:=false`. 결과: `~/picar_gz_runs/<시각>/metrics.json`, 서버 로그 `central.jsonl`.
상태 보기: `ros2 topic echo /car0/drive_state` (`mode:zone:edge`).

## 4. 합격 기준 (차량 3대, 10분, 시드 3개)

| 항목 | 기준 |
|---|---|
| 충돌 (차체 겹침) | 0 |
| 허가 없는 진입 (`legal=False`) | 0 |
| 교차로 후 라인 못 찾음 (`FAULT`) | 0 |
| 차로 이탈 (중심선에서 8 cm 초과, 교차로 밖) | 0 |
| 정지 위치 | 범퍼가 정지선 바깥 끝 기준 ±5 cm, 충돌구간(박스) 안으로 들어가지 않음 |
| 빔 결함 중 신규 진입 | 0 |
| 교착 (모든 차가 30 s 이상 정지) | 0 |

한 번에 확인 (보정 → 시드 1·2·3 × 10분 → 빔 결함 60~80 s → 판정, 약 40분):

```bash
bash picar_gazebo/scripts/run_acceptance.sh          # ROS가 없으면 setup_ubuntu2204.sh부터 실행
# → ~/picar_gz_runs/acceptance_<시각>/summary.md (항목별 OK/FAIL), 종료 코드 0 = 합격
```

교착은 `pose_logger.py`가 남긴 `poses.csv`에서(모든 차가 1초에 1 cm 미만 이동한 상태가 30 s 이상),
빔 결함 중 신규 진입은 `central.jsonl`에서(J 감독기가 fault인 동안 새로 나간 J 허가 수) 판정한다.

통과하면 실차로 넘어간다. 실패하면 아래 2D 사전 점검에서 같은 조건을 재현해 고친다.

## 5. 2D 사전 점검 (ROS 없이, 이 저장소에서 바로)

Gazebo 전에 같은 `CarBrain`을 2D 운동 모델로 돌린다. 바닥 텍스처 PNG 픽셀을 IR 센서로 읽고, 중앙 서버와 실제 UDP로 통신한다.
모터 모델: 명령 지연 0.05 s, 데드밴드 0.03 m/s, 오른쪽 바퀴 0.97배, 가속 제한. 차량 코드는 이 값을 모르고 보정으로만 안다.

```bash
PYTHONPATH=.:picar_gazebo python3 -m picar_gazebo.kinematic_check --cars 6 --seconds 180 --video
```

결과 (차량 6대 × 180 s × 시드 5개, `runs/gazebo_precheck/kinematic_matrix_6cars.json`):

| 조건 | 교차로 통과 | 충돌 | FAULT | 차로 이탈 | 허가 없는 진입 |
|---|---:|---:|---:|---:|---:|
| 오래된 요청 우선 (보정함) | 213 | 0 | 0 | 0 | 0 |
| DQN v7 active (보정함) | 202 | 0 | 0 | 0 | 0 |
| **보정 안 함** | 25 | 0 | 7 | 12 | 0 |
| IR 잡음 2% (채널·프레임마다 뒤집힘) | 142 | 0 | 1 | 2 | 0 |
| 더 나쁜 모터 (지연 0.08 s, 오른쪽 0.92배, 감속 0.7) + 그에 맞춰 보정 | 205 | 1 | 0 | 0 | 0 |

읽는 법:
- 교차로 제어 쪽(허가 없는 진입)은 모든 조건에서 0. 문제가 생기는 곳은 **차량 주행**이다.
- **보정이 결정적**이다. 교차로 안에는 테이프가 없어 회전이 개루프(dead reckoning)인데, 좌우 모터 편차 3%만으로 0.8 m 직진 후 8° 틀어져 차선을 놓친다.
- DQN과 규칙의 통과 수 차이는 시드 5개 범위의 잡음 수준이다. 이 단계는 성능 비교가 아니라 "같은 정책이 실제 주행 위에서 안전하게 도는가"를 본다.
- 남은 충돌 1건은 곡선 구간에서 앞차가 초음파 원뿔(±15°) 밖에 있어 생긴 추돌이다. 실차에서도 같은 한계가 있다(아래 7장).

이 점검에서 찾아 고친 것:
1. 라인 테이프 18 mm는 센서 간격 15 mm와 거의 같아 두 센서가 동시에 보는 구간이 ±1.5 mm뿐 → 지그재그·라인 이탈. **25 mm 이상 권장**, PD 게인 kp 2.5 → 1.0.
2. 교차로 안 회전이 모터 편차·지연 때문에 빗나감 → 차량별 보정 4개 값(`trim_per_m`, `turn_scale`, `latency_s`, `accel_mps2`)과 정지 거리 예측 주행기록계(Odometer) 추가.
3. 정지선에서 멈췄다 출발하는 경우와 멈추지 않고 통과하는 경우 회전 지점이 4~5 cm 달라짐 → 정지선 감지 지점부터 굴러간 거리까지 포함해 측정.
4. 앞차 때문에 정지선 테이프 위에서 멈추면 정지선을 못 읽고 박스로 들어감 → 표식 인식을 멈춘 동안에도 추정 속도로 계속.
5. 교차로를 나온 직후 연결로의 APPROACH 표식 모서리를 차선으로 오인 → 3~4채널이 검으면 가로 테이프로 판단, 라인은 2프레임 연속 확인, 놓치면 ±60° 탐색.
6. IR 잡음 한 채널이 차선 위에서 "가로 테이프"를 만들어 가짜 정지선 → 4채널 모두 검은 프레임이 2개 이상일 때만 표식, APPROACH 표식 뒤 0.3 m 이내 정지선은 무시.
7. 곡선에서 앞차를 늦게 봄 → 앞차 정지 거리 0.15 → 0.20 m, 0.40 m부터 감속.

## 6. 실차로 옮길 때 (같은 코드, 바뀌는 것만)

| Gazebo | 실차 |
|---|---|
| `calibrate.launch.py` (위치 자동 측정) | `python3 -m picar_lab.car_agent.calib_drive_node --ros-args -p test:=straight` 등 4가지 시험 후 줄자·각도기로 측정 → `python3 -m picar_lab.car_agent.calibration --straight-m ... --drift-deg ... --roll-slow-m ... --roll-fast-m ... --spin-deg ...` |
| `ir_camera` | IR 센서 노드 (`line_values` 4채널, 검은 테이프 값 `dark_value`) |
| `motor_model` + DiffDrive | `motor_diff_node.py` (좌우 PWM; 기존 Yahboom motor_node는 주행 중 조향 불가) |
| `course_monitor` 가상 빔·초음파 | `exit_beam.py`, HC-SR04 (`range` 토픽) |
| 바닥 텍스처 | 실제 테이프: 차선 25 mm 이상, 정지선 50 mm, APPROACH 15 mm, 가로 테이프 길이 30 cm |

실차 첫 단계: 바퀴 들고 → 1대 저속 → 서버 shadow → active. 차량별 `car_calibration.yaml`을 따로 둔다.

## 7. Gazebo에서 통과해도 실차에서 다를 수 있는 것

- 바닥 마찰·바퀴 미끄러짐·배터리 전압에 따른 속도 변화 (보정은 측정한 그날 배터리 기준 → 배터리가 바뀌면 다시)
- IR 센서의 조명·바닥 반사 (Gazebo 카메라는 깨끗함; `ir_flip_prob`로 일부만 흉내)
- 초음파의 실제 빔 폭·반사 (곡선 추돌 위험은 실차에서 더 클 수 있음 → 처음엔 차 간격을 넓게)
- Wi-Fi 지연·끊김 (서버는 TTL·하트비트로 처리하지만 실제 무선 상태는 실측 필요)
- 코스 치수: 아직 사용자 설명 기반 가정값. 실측 후 `course.yaml` → SUMO와 Gazebo 둘 다 다시 생성

이 저장소에서는 ROS/Gazebo 패키지 저장소에 접근할 수 없어 Gazebo 실행 자체는 아직 확인하지 못했다.
ROS 없는 부분(월드 생성, 차량 코드, 보정, 2D 점검, 테스트 33개: `pytest picar_gazebo/test picar_lab/car_agent/tests picar_sim/tests`)은 확인했다.
