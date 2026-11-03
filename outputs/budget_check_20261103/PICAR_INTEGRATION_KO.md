# UROP2 후속 검증 | PiCar 중앙 통신과 연동

## 확인한 기존 구현

노션 UROP2 정리에는 중앙 서버가 차량 ID·차선·이벤트를 UDP로 받아 GO/STOP을 보내고, PiCar는 로컬 차선 추종과 센서 정지를 수행한다고 기록돼 있다. 2025년 전시 데모는 사전 정의 상태 시퀀스를 사용했다. 이번 연동의 핵심은 **실차 관측으로 상태를 구성하고 서버가 학습 모델을 호출하는 폐루프**를 별도로 검증하는 것이다. [노션 UROP2 §9–16](https://app.notion.com/p/6149c197abc783a2868f81c943696784)

노션의 ROS2 첨부 두 개를 받아 실행 없이 소스를 검사했다. 두 파일 모두 `line_tracker → line_values → line_follower → cmd_vel → motor_controller` 흐름이다. `line_values`는 GPIO 센서 4개를 0.1초 간격으로 읽은 `Int32MultiArray`이고, `cmd_vel`은 `geometry_msgs/msg/Twist`다. 첨부 구현은 IR 라인트래킹이며, 카메라·OpenCV·PID 및 UDP 전시 설명은 UROP2 문서의 별도 범위다. [ROS2 첨부 ZIP 두 개](https://app.notion.com/p/39a9c197abc7805eb149d473a1338426) · [기존 트래킹 메모](https://app.notion.com/p/39a9c197abc78071bed3cd62a70b8cf1)

| 소스 | 모터 방식 | 실차 시작 전 확인할 부분 |
|---|---|---|
| `picar__ws (2).zip` | Yahboom I²C 보드. 드라이버 코드 기준 버스 1, 주소 0x16 | `motor_node`가 `YB_Pcb_Car` 모듈을 import하지만 파일은 `YB_Pcb_car.py`로 되어 있다. Linux에서 이름을 일치시킨다. `line_follower_node:main`이 등록돼 있으므로 누락된 `main()`을 구현하고 launch 설치 여부를 확인한다. |
| `picar__ws.zip` | BOARD 번호 기준 GPIO/PWM | launch 파일이 0바이트다. 실제 보드·배선이 이 방식과 일치하는지 먼저 확인한다. |
| 공통 | `Twist` 콜백으로 모터 명령 적용 | 두 `motor_node`에서 명령 무수신 시 주기적으로 모터를 멈추는 watchdog을 확인하지 못했다. 명령 신선도 검사와 0 출력은 모터 노드 자체에도 넣는다. |

현재 보드 모델·OS·IP와 현장 실행본은 확인되지 않았다. 위 결함은 첨부 스냅샷의 정적 분석이다. 하드웨어 구동 전 Pi에서 실제 사용 중인 코드·배선과 대조한다. ROS2 설치 메모는 Ubuntu 22.04/Humble을 전제로 한다. 기존 이미지를 먼저 확인한다. [노션 ROS2 메모](https://app.notion.com/p/39a9c197abc7804ba2c5ce7f54696ed3)

## 중앙 서버와 PiCar의 역할

```text
[PiCar 센서/ROS2] -- UDP 상태·확인 --> [중앙 PC]
  line_values/카메라/초음파              차량별 상태 원장
  ARRIVE / IN / EXIT                      W/E/S 상태 집계 → 정책 추론
  로컬 상태 주기 보고                    신호 FSM → 진입 허가/정지
       <-- seq에 대응하는 짧은 GO/STOP ----┘
  line_follower → cmd_vel_raw → 로컬 게이트 → cmd_vel → motor_node
                                       ↑ 장애물·라인 유실·E-STOP·타임아웃
```

- 중앙 PC에서 학습된 `Policy`를 CPU로 1회 로드하고, 완전한 센서 관측으로 5초마다 추론한다. 모델 출력은 `KEEP/SWITCH`이며 현재 구현의 W→E→S 순환 안에서 유지 시간을 정한다. 서버의 신호 상태 머신이 최소·최대 녹색과 황색·전적색 및 실제 점유 확인을 담당한다. 10–45초·3초·1초는 SUMO 실험 설정이며 축소 트랙의 통과·정지시간을 별도로 재서 결정한다.
- 각 PiCar는 기존 라인트래커를 계속 구동한다. 라인 추종 출력은 `/car1/cmd_vel_raw`, 게이트 출력만 `/car1/cmd_vel`로 연결하고 다른 차량은 별도 네임스페이스를 둔다. UDP 수신을 GPIO/PWM 직접 호출에 연결하지 않는다. 장애물, 라인 유실, 허가 만료 또는 센서 이상 시 로컬에서 모든 축 0의 Twist와 모터 정지로 처리한다. 센서 콜백이 멎어도 독립 타이머가 동작해야 한다.
- 정지선 도착 `ARRIVE`, 진입 확정 `IN`, 출구 통과 `EXIT`를 구분한다. ID별 상태 전이는 중복 제거·디바운싱을 적용하고, 첫 단계에서는 한 차량에만 진입을 허가한다. 허가를 보낸 시점에 슬롯을 예약한다. 허가 만료나 차량 무응답으로 슬롯이 비었다고 추정하지 않는다. 출구 센서 또는 상부 관측이 통과를 확인할 때 해제한다. 점유를 잃으면 신규 진입 허가를 닫고 운영자가 상태를 확인한다.
- 시험 메시지는 `{v,car,boot,seq,lane,zone,obstacle,line_ok}` 상태 스냅샷과 `{v,car,boot,ack,command,ttl_ms}` 응답으로 구성했다. 실차에서는 등록된 차량별 인증, 서버 세션, 증가하는 시퀀스, 요청 시각을 기준으로 한 로컬 단조시계 만료, 적용 결과 ACK를 추가한다. 중복과 순서 역전으로 기존 허가를 갱신하지 않는다. `GO`는 짧은 진입 허가다. 정지선 전까지는 제한 속도의 로컬 접근 주행을 허용하고, 정지선 횡단에만 신선한 진입 허가를 요구한다. 진입 후 연결을 잃어 차가 멈추면 새 진입을 막고 운영자가 직접 통제해 교차로를 비운다. [IETF UDP 지침 §3.3, §5.1](https://www.rfc-editor.org/rfc/rfc8085.html)

## 관측 스키마의 실차 매핑

| V2 입력 | 실차 확보 방법 |
|---|---|
| 접근 방향별 전체 차량·정지 큐 | 상부 카메라의 W/E/S 구역 추적이나 차량별 접근 구역·정지 상태 보고. 정지선 센서 하나로 뒤쪽 큐 길이를 추정하기 어렵다. |
| 방향별 평균 대기·최장 대기 | 차량별 연속 정차 시간과 누적 정차 시간을 따로 유지한다. 최장 대기에는 진입 전 대기도 반영한다. |
| 진입 대기·최근 30초 요청 | 상류의 요청 구역 또는 투입 대기 이벤트를 계수한다. 출발 전 대기가 존재하는 경우를 구별한다. |
| 내부 차량 수 | 정지선 통과와 출구 통과 이벤트를 짝지어 집계하고 상부 카메라 또는 독립 출구 센서로 교차 확인한다. |
| 현시, 녹색 경과, 전환 중 여부, 전환 잔여 | 중앙 신호 상태 머신이 제공한다. 관측 순서는 W/E/S로 고정한다. 실제 LANE_A/B/C 매핑은 트랙을 보고 등록한다. |

필수 관측이 비면 DQN을 shadow 모드로 유지한다. 값이 비었다는 사실과 실제로 차량이 0대라는 사실을 구분한다. 센서를 추가하기 어려우면 확보 가능한 관측으로 SUMO의 상태를 다시 정의하고 새 정책을 학습·평가한다. V2의 의미와 단위를 유지한 실차 측정값을 확보한 뒤에만 기존 정책의 실차 후보 여부를 판단한다.

## 연결 순서와 통과 조건

| 단계 | 작업 | 다음 단계로 넘어가는 기준 |
|---|---|---|
| 1. 같은 LAN | 중앙 PC는 실험용 공유기에 연결하고 고정 DHCP 예약을 사용한다. Pi마다 고유 hostname과 CAR ID를 부여한다. PC `ipconfig`, Pi `hostname -I`, `ssh <계정>@<Pi주소>`로 식별한다. UDP 예시 포트 5005를 실험 LAN 인터페이스에만 바인딩한다. | Pi 1대에서 상태→서버→ACK가 왕복하고 왕복시간·유실률을 기록한다. ping 성공과 UDP 성공은 별도로 확인한다. |
| 2. ROS2 단독 | 먼저 바퀴를 들어 올리거나 모터 전원을 분리한 상태에서 센서와 토픽만 확인한다. 현재 이미지에 맞는 모터 드라이버를 선택하고 위 소스 결함을 보완한다. | `ros2 topic list -t`, `ros2 topic info /car1/cmd_vel -v`로 타입·발행자 확인. 게이트가 유일한 모터 명령 발행자다. |
| 3. 고정/규칙 + 1대 | UDP 메시지를 ROS2 게이트에 연결한다. 저속으로 정지선 접근, STOP, GO, EXIT를 순서대로 시험한다. 모터 노드 watchdog과 물리 E-STOP을 점검한다. | AP 끊김, 서버·ROS 노드 종료, 초음파 차단, 라인 유실 때 실제 바퀴가 정지한다. 정지선 거리·속도·최악 지연을 측정해 허가 만료와 감속 지점을 설정한다. |
| 4. 상충 방향 2대, 이후 다차량 | 두 방향 동시 ARRIVE, 패킷 손실·중복·역순, EXIT 누락, 주행 중 연결 단절, 서버 재시작을 주입한다. 중앙 장부와 실제 점유를 대조한다. | 상충 방향 동시 허가 0건, 점유 미확인 상태 신규 허가 0건. 오류 후 재출발은 새 세션과 신선한 상태를 확인한 뒤 허용한다. |
| 5. DQN shadow → 제한 실험 | 같은 실차 관측으로 규칙과 DQN을 함께 계산하고 DQN은 로그만 남긴다. 관측이 확보된 이후 제한 속도·감독 아래 동일 차량 투입 순서로 두 정책을 반복 비교한다. | 실차에서 방향별 대기·통과율·최장 대기·긴급 정지·통신 손실이 기록되고 사전 합의 기준을 충족한다. |

LAN 접근은 [Raspberry Pi 공식 원격 접속 문서](https://www.raspberrypi.com/documentation/computers/remote-access.html), ROS 토픽 조회는 [ROS2 Humble 공식 튜토리얼 소스](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.rst)를 참조했다.

## 이번에 실행한 범위

Windows에서 로컬 UDP 소켓으로 가상 PiCar 두 대와 중앙 서버를 연결했다. 녹색 방향 GO, 상충 방향 STOP, 중복 응답 무시, 장애물·라인 유실 STOP, 전환·점유·미확인 관측 STOP, 응답 손실 시 만료, 지연 응답 거부, 신선한 요청 복구 등 **11개 검사**를 통과했다. 별도로 실제 SUMO GUI의 원시 관측을 CPU 모델에 재입력하여 행동 마스크와 신호 결정 **360개가 모두 일치**했다. 가상 차량의 고정 현시에서 UDP를 검증하고, SUMO 로그에서 정책 추론을 검증한 두 실험이다. 250ms는 로컬 시험값이다. 이번 실행 범위는 localhost의 모의 출력까지이며 Raspberry Pi, ROS2 실기기, GPIO·모터와 축소 트랙은 현장 검증 단계다.

프로젝트 루트의 PowerShell에서 재현한다.

```powershell
$env:PYTHONPATH = (Get-Location).Path
$run = 'outputs/budget_check_20261103'
& ../.venv/Scripts/python.exe "$run/picar_loopback.py" `
  --model outputs/verify_20261103/training/seed43/checkpoints/best_model.pt `
  --decisions outputs/verify_20261103/gui_verified_seed43/decisions.csv `
  --output "$run/picar_dry_run"
```

## 실험실 첫 연결 명령

`picar_lan_probe.py`는 표준 라이브러리만 사용하는 UDP 왕복 측정기다. 이 PC에서 두 프로세스로 20/20 응답을 확인했고, 수신 서버를 닫았을 때 3/3 무응답을 확인했다. 다음은 실제 LAN에서 먼저 실행할 명령이다. 주소는 예시이며 중앙 PC의 실험용 LAN 주소로 바꾼다.

```powershell
# 중앙 PC, 프로젝트 루트
python outputs/budget_check_20261103/picar_lan_probe.py listen --host 192.168.50.10 --port 5005
```

```bash
# PiCar, picar_lan_probe.py 파일을 복사한 디렉터리
python3 picar_lan_probe.py probe --host 192.168.50.10 --port 5005 --count 100
```

`matched`는 250ms 안에 요청과 일치한 응답 수, `lost_or_late`는 만료 수, `p95_rtt_ms`는 성공 응답의 왕복 지연이다. 중앙 PC에서 수신이 안 되면 장치의 동일 서브넷, 공유기의 클라이언트 격리, 실험 LAN 인터페이스의 UDP 수신 규칙을 점검한다. 다음으로 Pi의 실제 보드·OS와 ROS2 토픽을 확인한다.

