# T자 교차로 DQN · PiCar 연동

**PiCar Real-to-Sim (2026-11-09):** [진행 상황](PICAR_REAL2SIM_STATUS_KO.md) · 실물 코스를 본뜬 미터 단위 SUMO 맵, 실차와 같은 관측·안전 제어기, 3시드 학습·평가 → [계획·결과](docs/PICAR_REAL2SIM_PLAN_KO.md) · [`picar_sim/`](picar_sim/) · [Colab](notebooks/08_picar_real2sim_colab.ipynb)

**후속 연구 자료(2026-11-03~05):** [SUMO 다중 교차로·재학습·센서 및 대기 평가 기록](research/sumo_20261106/README.md). 후속 실험의 직접 방향 선택 모델은 아래 기존 KEEP/SWITCH 모델과 입력·행동 공간이 다르며 자동으로 교체하지 않았습니다.

SUMO의 T자 교차로에서 신호를 **유지하거나 다음 방향으로 전환**하는 DQN을 학습했습니다. 같은 교통 수요에서 고정 신호·규칙 기반 제어와 비교하고, 저장된 정책의 결정을 중앙 PC에서 UDP로 가상 PiCar에 전달하는 흐름을 검증했습니다.

**[한국어 시작 가이드](START_HERE_KO.md)** · [중앙 UDP](picar_lab/docs/CENTRAL_UDP_KO.md) · [PiCar·ROS2](picar_lab/docs/PICAR_ROS2_KO.md) · [트랙 도면과 현장 검증](picar_lab/docs/TRACK_VALIDATION_KO.md)

## 목적별로 시작하기

| 하려는 일 | 시작 위치 |
| --- | --- |
| Windows CPU에서 설치하고 모델 불러오기 | [시작 가이드](START_HERE_KO.md#pc-빠른-확인) |
| SUMO GUI로 차량과 신호 보기 | [GUI 실행 명령](docs/SUMO_RUN_KO.md#학습-결과-시험) · [실제 실행 영상](outputs/verify_20261103/gui_verified_seed43/playback_full.mp4) |
| 학습·평가를 재현하거나 체크포인트에서 재개하기 | [SUMO 실행·재개 가이드](docs/SUMO_RUN_KO.md) · [평가 프로토콜](docs/evaluation_protocol.md) |
| 중앙 PC와 Pi 사이의 UDP 상태·ACK 시험하기 | [PC·Pi 실행 명령](picar_lab/docs/CENTRAL_UDP_KO.md#pcpi-모터-없는-시험) |
| 파이카의 ROS2 토픽과 트랙 구성하기 | [ROS2 설정](picar_lab/docs/PICAR_ROS2_KO.md) · [6×4m 배치도](picar_lab/docs/track-layout.svg) · [현장 체크리스트](picar_lab/records/field_checks.csv) |

## 검증 결과

- **초기 재현**은 시드 42·43·44를 각 80회 새로 학습하고, 같은 수요에서 168회 비교했습니다. [초기 검증 보고서](VERIFICATION_KO.md)
- **후속 실험**은 각 시드의 80회 체크포인트에서 240회까지 이어 학습했습니다. 새 교통 시드 8개에서 320회 평가했고, 과포화·수요 전환의 사전 기준 충족 수가 **3/6에서 6/6**으로 늘었습니다. 240회 이내 검증 최우수 모델은 시드별로 220·140·180회차였습니다. [학습 예산 비교 보고서](outputs/budget_check_20261103/TRAINING_BUDGET_KO.md) · [평가 CSV](outputs/budget_check_20261103/evaluation/seed_level_results.csv)
- **통합 검증**에서 pytest 26개를 통과했습니다. CPU 모델 추론과 1,800초 SUMO 재실행, 로컬 UDP 상태·ACK 및 장애 시험, 저장된 모델 결정에서 가상 차량 진입 허가까지 확인했습니다. [정책→UDP 재현 기록](outputs/budget_check_20261103/picar_dry_run/policy_to_udp.json)

[80회·240회 성능 그래프](outputs/budget_check_20261103/evaluation/budget_comparison.png)

## 중앙 PC와 PiCar의 역할

```text
차량 수·대기·점유 관측 → 중앙 DQN의 KEEP/SWITCH → 신호 상태 머신
                                                    ↓ UDP 진입 허가
PiCar의 차선 추종 → 로컬 허가·안전 게이트 → ROS2 모터 명령
```

DQN은 W→E→S 순서에서 녹색을 더 유지할지 판단합니다. 중앙 신호 제어기가 녹색·황색·전적색과 교차로 점유를 확인하고, PiCar의 조향과 정지는 차량에서 처리합니다. 실습 코드는 모터 없는 통신까지 실행됩니다. 실제 PiCar 센서·모터·트랙은 [현장 검증 절차](picar_lab/docs/TRACK_VALIDATION_KO.md)에 따라 확인해야 합니다. [25개 상태와 보상 정의](docs/state_action_reward.md)

## 저장소 구조

| 경로 | 내용 |
| --- | --- |
| [`src/`](src/) · [`configs/`](configs/) · [`maps/`](maps/) | DQN, SUMO 환경, 신호 로직, 설정과 T자 맵 |
| [`scripts/`](scripts/) · [`tests/`](tests/) · [`notebooks/`](notebooks/) | 학습·평가·시각화, 테스트, Colab 노트북 |
| [`picar_lab/`](picar_lab/) | UDP 상태 실습, ROS2 참조 소스, 트랙 및 현장 문서 |
| [`outputs/verify_20261103/`](outputs/verify_20261103/) | 새 80회 모델, 초기 비교, SUMO GUI 실행 |
| [`outputs/budget_check_20261103/`](outputs/budget_check_20261103/) | 240회 체크포인트, 320회 결과, PiCar 연동 검증 |
| [`outputs/reference/`](outputs/reference/) · [`outputs/revision/`](outputs/revision/) | 입력 ZIP에 포함됐던 이전 모델과 실험 자료 |
