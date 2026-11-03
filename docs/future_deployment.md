# 정책 추론과 센서 연결

`python -m scripts.inference_demo --model outputs/revision/seed42/checkpoints/best_model.pt`로 SUMO 연결 없이 CPU 추론을 확인할 수 있습니다.

모델 옆 `observation_schema.json`, `normalization.json`, `action_schema.json`에 입력 순서와 단위, 출력 행동을 정의합니다.
실차 어댑터는 방향별 차량·정지 큐·대기·진입 요청과 신호 제어기의 시간 상태를 제공해야 합니다.
W/E/S 순서와 관측 주기를 유지해야 하며, 미측정 특성을 임의로 0으로 채우면 정책 입력의 의미가 달라집니다.
모델의 행동은 신호 제어기의 안전 규칙을 통과시킨 뒤 실행합니다. 실차 센서, 통신 및 제동은 별도 검증 대상입니다.
