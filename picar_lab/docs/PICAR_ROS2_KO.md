# PiCar와 ROS2 설정

차량마다 [하드웨어 기록표](../config/hardware_inventory.template.yaml)를 복사한다. Pi의 `cat /etc/os-release`, `uname -m`, `printenv ROS_DISTRO`로 실제 환경을 확인한다. 노션 설치 메모는 Ubuntu 22.04/Humble을 전제로 한다. [노션 ROS2 메모](https://app.notion.com/p/39a9c197abc7804ba2c5ce7f54696ed3)

## 노션에 있던 두 버전

| 스냅샷 | 모터 방식 | 현장 적용 전 확인 |
|---|---|---|
| I²C | Yahboom 보드와 I²C 드라이버 | 파일명과 import의 대소문자가 다르다. Linux에서 이름을 일치시킨다. 소스의 주소·버스를 실제 보드와 대조한다. |
| GPIO/PWM | BOARD 핀 번호와 PWM | launch 파일이 비어 있다. 사용 중인 이미지의 실행 파일과 배선을 확인한다. |
| 공통 | line_values → line_follower → cmd_vel → motor_node | 등록된 라인팔로워 진입점의 main()이 누락돼 있다. 설치된 entry point를 확인한다. 두 모터 소스에는 명령 무수신 watchdog이 없다. |

`../ros2_reference/`는 읽기용 소스다. 실제 Pi 모델에 맞춰 한 가지 드라이버를 선택하고 배선과 설정을 확인한다. [노션 ROS2 원본](https://app.notion.com/p/39a9c197abc7805eb149d473a1338426)

## 토픽 배선

| 토픽 예시 | 발행 → 구독 | 데이터 |
|---|---|---|
| /car1/line_values | 센서 → 라인 추종기 | 센서 배열 |
| /car1/cmd_vel_raw | 라인 추종기 → 로컬 게이트 | 원하는 Twist |
| /car1/cmd_vel | 로컬 게이트만 발행 → 모터 노드 | 허용된 Twist 또는 0 Twist |
| /car1/lab_status | 차량 상태 노드 → UDP 어댑터 | 정지선, 진입, 출구, 센서 상태 |

라인팔로워만 `cmd_vel:=cmd_vel_raw`로 remap한다. 모터 노드 입력은 `cmd_vel`에 둔다. 차량마다 고유 namespace를 사용한다. 아래 명령은 라인팔로워의 entry point를 보완해 설치한 뒤 모터 전원이 분리된 상태에서 사용한다.

```bash
ros2 run line_follower line_follower_node --ros-args -r __ns:=/car1 -r cmd_vel:=cmd_vel_raw
ros2 topic list -t
ros2 topic info /car1/cmd_vel -v
ros2 topic echo /car1/cmd_vel_raw
```

모터 입력에는 게이트만 발행하는지 확인한다. 다른 차량의 토픽이 구독에 연결되지 않아야 한다. [ROS2 Humble 토픽 조회](https://github.com/ros2/ros2_documentation/blob/humble/source/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Topics/Understanding-ROS2-Topics.rst)

## 현장 구동 순서

1. **모터 전원 분리.** 센서 원시값, 라인 추종 출력, 중앙 ACK를 따로 확인한다. 정지선 표식과 주행선을 구별하고 디바운싱을 점검한다.
2. **바퀴를 들어 올린 시험.** 제한 출력과 독립 E-STOP을 사용한다. 로컬 게이트는 장애물·라인 유실·허가 만료·센서 이상에서 0 Twist를 보낸다. 모터 노드에도 독립 타이머를 둬 명령이 멎으면 Car_Stop 또는 GPIO/PWM 0을 적용한다. 프로세스 종료와 멈춤 상황에서 실제 차단을 확인한다.
3. **바닥에서 한 대.** 목적지를 지정하고 정지선에 멈춘 뒤 새 허가에서만 출발하게 한다. 분기에서는 사전 경로나 별도 표식을 사용한다. 진입 중 연결을 잃으면 로컬 정지하고 중앙은 새 진입을 차단한다.
4. **상충 방향 두 대 이상.** AP·서버·센서 노드 단절, 중복·지연, 출구 확인 누락을 각각 주입한다. 차량 정지와 신규 허가 차단을 별개로 확인한다.

이 배포본의 LAN 코드는 상태 확인까지 실행된다. 실차용 인증·허가→ROS2 게이트·센서 연동은 위 기준으로 구현하고 보드에서 검증한다.

