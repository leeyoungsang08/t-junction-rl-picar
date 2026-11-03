# 구성과 데이터 흐름

`generate_map` → `traffic_generator` → `TJunctionEnv` → `DQNAgent` → validation 선택 →
`evaluate` → `visualization` / `export_results` 순서입니다.

`TJunctionEnv`는 TraCI 연결 하나를 소유하고 매초 차량 subscription 결과를 수집합니다.
`StateAdapter`는 25개 특성을 정규화하고 `SignalController`는 녹색·황색·전적색을 독립 관리합니다.
`step()`은 DQN 기준 최대 5초 동안 실제 SUMO를 진행해 그 구간의 reward를 합산합니다.
종료 시간은 명세 horizon으로 고정합니다. 수요가 잠깐 비었다고 조기 종료하지 않습니다.

고정식은 정확한 20초 녹색을 위해 1초마다 스케줄을 확인합니다. 규칙식과 DQN은 5초 간격입니다.
최대 녹색은 모든 제어기에서 초 단위로 강제합니다. 전환 중 행동 0은 clearance를 계속 진행한다는 의미입니다.

신경망과 inference는 SUMO를 import하지 않습니다. StateAdapter 입력은 센서 상태의 Python dictionary입니다.
모델에는 정규화 및 신호 조건도 포함됩니다. 외부 제어와 연결할 때 SignalController를 통과해야 합니다.

로그: 초 단위 `timeseries.csv`, 판단 단위 `decisions.csv`, SUMO `tripinfo.xml`, 실행 `metrics.json`.
평가 수요 XML과 SHA-256은 모든 제어기에서 동일합니다. 다른 output 경로로 독립 실행하세요.
선택적 libsumo와 병렬 SUMO는 현재 검증·구현 범위에서 제외되어 명시적으로 거부합니다.

`decisions.csv`의 time은 해당 행동 구간의 종료 시각입니다. state 및 Q값은 행동 직전 값이며,
그 행에 합친 queue/phase 지표는 구간 종료 시점 값입니다. 실제 초별 신호는 timeseries.csv가 기준입니다.
