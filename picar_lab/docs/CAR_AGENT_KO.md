# PiCar 차량 쪽 교차로 에이전트와 중앙 서버 (프로토콜 v2)

Real-to-Sim 제어기(`picar_sim/`)를 실차에 붙이기 위한 코드다. **실차에서는 아직 실행하지 않았다.**
대신 SUMO 차량이 이 코드(차량 에이전트)를 그대로 돌리고, 실제 UDP 소켓으로 중앙 서버와 통신하는 소프트웨어 통합 시험(SIL)을 통과했다.

```text
라인 센서 ──line_values──► [ros2_gate_node] ◄──cmd_vel_raw── line_follower (cmd_vel 이름을 바꿔 연결)
   (표식 인식: 접근 표식/정지선)   │ CarAgent: 상태 보고, 요청 갱신, PERMIT 검증, 진입 게이트
                                   ├──cmd_vel──► motor_controller   (기본 STOP, 명령 watchdog)
                                   └◄══UDP v2══► picar_sim.central_server (J: 오래된 요청 우선/DQN, J2: 대기열 규칙)
출구 빔 센서 ──exit_beam.py──UDP──► 중앙 서버 (출구 통과 + heartbeat)
```

| 파일 | 역할 | 의존성 |
|---|---|---|
| `car_agent/protocol.py` | v2 메시지 정의·검증 | 표준 라이브러리 |
| `car_agent/agent.py` | `CarAgent`(상태 머신·허가·게이트), `MarkerDetector`(접근 표식/정지선 구분) | 표준 라이브러리 |
| `car_agent/exit_beam.py` | 출구 빔 감시(`ExitBeamMonitor`) + Pi 실행부 | 실행부만 RPi.GPIO |
| `car_agent/ros2_gate_node.py` | ROS2 노드: 라인 추종 명령을 게이트해서 모터로 전달 | rclpy (Pi) |
| `../picar_sim/central_server.py` | 중앙 PC 서버: 두 T자, 예약, 출구 빔 매칭 | numpy, PyYAML |
| `../picar_sim/sil.py` | SUMO ↔ 차량 에이전트 ↔ UDP ↔ 중앙 서버 통합 시험 | SUMO |

## 프로토콜 v2 (기존 `CENTRAL_UDP_KO.md` 계약을 따름)

| 방향 | 메시지 | 핵심 필드 |
|---|---|---|
| 차량 → 중앙 | `car_status` | car, boot, seq(매번 증가), state(REGISTER/APPROACH/ARRIVE/IN/CRUISE), junction, lane, **next**(다음 구간), edge |
| 중앙 → 차량 | `status_ack` | session, ack(seq) |
| 중앙 → 차량 | `permit` | session, car, boot, **req**(요청 seq), grant, junction, ttl_ms |
| 차량 → 중앙 | `permit_ack` | grant, result(applied/duplicate/rejected_*) |
| 빔 → 중앙 | `exit_event`, `beam_heartbeat` | junction, edge (차량 ID 없음) |

- **요청 갱신**: 접근·정지선 대기 중에는 0.2 s마다 새 seq로 요청을 다시 보낸다. 이것이 링크 heartbeat 역할도 한다.
- **TTL**: 차량이 그 요청을 **보낸 시각**부터 자기 단조시계로 계산한다. 같은 허가를 다시 받아도 연장되지 않는다. 중앙은 새 요청마다 새 grant를 보낸다.
- **진입 시점(commit)**: 정지선 앞에서 더 이상 멈출 수 없는 지점을 허가가 유효할 때 지나면, 이후 허가가 끝나도 통과를 마친다.
- **예약(RESERVED)**: 허가를 내면 그 차량과 다음 구간 한 칸을 잠근다. 예약된 차량은 IN 보고 전이라도 박스 안에 있는 것으로 센다. ACK 누락이나 TTL 경과만으로는 풀지 않는다. 풀리는 경우는 둘뿐이다: 출구 빔이 그 차량의 통과를 확인했을 때, 또는 허가가 끝난 뒤 더 새로운 요청이 왔을 때(= 통과하지 않았다는 증거).
- **순서**: 한 차로에서는 추월할 수 없으므로 요청이 먼저 온 순서로만 허가한다. 앞차가 막히면 뒤차를 건너뛰지 않는다. (SIL 첫 실행에서 뒤차가 자리를 먼저 예약해 교착이 났고, 이 규칙으로 고쳤다.)
- **출구 빔 매칭**: 빔은 차량을 구분하지 못한다. 그 교차로 안에 있고 다음 구간이 같은 차량 중 가장 먼저 들어간 차량으로 처리한다. 맞는 차량이 없으면 점유를 믿을 수 없으므로 신규 허가를 멈춘다.
- **결함**: 빔 heartbeat가 1 s 끊기거나 차량 보고가 0.5 s 끊기면 안전 제어기가 신규 허가를 멈춘다. 허가 TTL(0.5 s)은 전적색 최소(1 s)보다 짧아야 하며, 서버가 시작할 때 확인한다.

## 실행 (실차가 준비되면)

```bash
# 중앙 PC
python -m picar_sim.central_server --host 0.0.0.0 --port 5007                     # 오래된 요청 우선
python -m picar_sim.central_server --policy runs/picar_v7_direct_guard20/seed43/best_policy.npz --mode shadow
# 각 PiCar (route: 이 차가 도는 구간 순서, start_index: 출발 구간)
ros2 run line_follower line_follower_node --ros-args -r cmd_vel:=cmd_vel_raw
python3 -m picar_lab.car_agent.ros2_gate_node --ros-args -p car:=W1 -p central_host:=192.168.50.10 \
    -p route:="W_in,E_out,N_in,W_out" -p start_index:=0
# 출구 빔 (교차로마다 하나)
python3 -m picar_lab.car_agent.exit_beam --junction J  --pins W_out=36,E_out=38,N_out=40 --host 192.168.50.10
python3 -m picar_lab.car_agent.exit_beam --junction J2 --pins W_in=36,E_in=38,N_in=40 --host 192.168.50.10
```

**실차 전에 정해야 할 값**: 접근 표식·정지선 테이프 폭(`approach_mark_max_m`, `stop_line_min_m`), 실제 주행 속도, 빔 센서 핀·논리 레벨, 각 차량의 경로. 첫 실차 시험은 모터 전원을 끈 상태에서 상태 보고와 허가만 확인한다.

## 소프트웨어 통합 시험 결과 (2026-11-10)

```bash
python -m picar_sim.sil --cars 6 --seconds 240
python -m picar_sim.sil --cars 5 --seconds 180 --beam-fault 60 90
```

| 시나리오 | 통과 | 허가 없이 통과 | 충돌 | 상충 동시 점유 | 결함 중 진입 | 교착 |
|---|---:|---:|---:|---:|---:|---:|
| 6대 균등, 240 s | 59 | 0 | 0 | 0 | — | 없음 |
| 6대 연결로 편중, 240 s | 39 | 0 | 0 | 0 | — | 없음 |
| 5대, J 출구 빔 30 s 끊김 | 45 | 0 | 0 | 0 | 0 | 없음 |
| 5대, 차량 무선 10 s 끊김 | 45 | 0 | 0 | 0 | — | 없음 |
| 5대, DQN active | 54 | 0 | 0 | 0 | — | 없음 |

결함 중 진입은 결함 판정(1 s) + 이미 나간 허가 TTL(0.5 s)이 지난 뒤부터 센다.
실제 무선 지연·손실, 표식 인식 오류, 차량 정지 오차는 이 시험에 들어 있지 않다.
