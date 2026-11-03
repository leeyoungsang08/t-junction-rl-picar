# T자 교차로 DQN·PiCar 통합 실험 패키지

중앙 PC는 교차로 상태를 집계하고 신호를 판단한다. PiCar는 자기 차선을 따라가며, 진입 허가와 로컬 안전 조건이 유효할 때 정지선을 넘는다. 현재 DQN의 행동은 W→E→S 순환의 KEEP 또는 SWITCH다.

## 시작 순서

1. PC에서 아래 명령으로 모델 추론과 모터 없는 UDP 시험을 실행한다.
2. 실험실에 트랙을 표시하고 Pi의 실제 보드·OS·모터 방식을 확인한다. [트랙과 검증](picar_lab/docs/TRACK_VALIDATION_KO.md) · [PiCar·ROS2 설정](picar_lab/docs/PICAR_ROS2_KO.md)
3. 모터 전원을 분리한 채 PC와 Pi에서 UDP 왕복과 차량 상태 등록을 확인한다. [중앙 UDP 프로토콜과 실행 명령](picar_lab/docs/CENTRAL_UDP_KO.md)
4. 고정·규칙 신호로 실차 정지와 진입을 검증한다. DQN은 실차 관측이 준비되면 판단만 기록하는 shadow 모드부터 시작한다.

## 완료 범위

| 항목 | 결과 |
|---|---|
| 초기 재현 | 3시드 × 80회 새 학습, 비교 168회, pytest 26개 |
| 후속 실험 | 시드별 240회까지 연장, 추가 학습 480회, 비교 320회 |
| 240회 이내 선택 모델 | 시드 42/43/44의 검증 최우수 회차는 220/140/180 |
| 실시간 SUMO GUI | 1,800초, 행동 360건과 기존 교통 지표 일치 |
| UDP | 로컬 장애 시험 11개, SUMO 관측→DQN→신호 제어기→가상 차량, PC/Pi용 모터 없는 상태·ACK 도구 |
| 현장 단계 | Pi 실기기, ROS2 모터 제어, 센서 및 축소 트랙 주행 검증. Colab은 노트북 구조와 코드 셀 검사까지 수행했다. |

[초기 검증 보고서](VERIFICATION_KO.md) · [80→240회 비교 보고서](outputs/budget_check_20261103/TRAINING_BUDGET_KO.md) · [후속 비교 그래프](outputs/budget_check_20261103/evaluation/budget_comparison.png) · [SUMO 실행 영상](outputs/verify_20261103/gui_verified_seed43/playback_full.mp4)

## PC 빠른 확인

압축을 풀고 이 문서가 있는 프로젝트 루트에서 실행한다. Windows SUMO 1.28.0의 한글 SUMO_HOME 문제를 피하려면 영문 경로에 가상 환경을 둔다. 모터 없는 LAN 도구 두 개는 Python 표준 라이브러리만 필요하다.

```powershell
$venv = Join-Path $env:TEMP 'tjunction_picar_cpu'
py -3.12 -m venv $venv
$py = Join-Path $venv 'Scripts/python.exe'
& $py -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
$env:SUMO_HOME = Join-Path $venv 'Lib/site-packages/sumo'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Get-Location).Path

& $py -m scripts.check_environment
& $py -m pytest -q
& $py -m scripts.inference_demo --model outputs/budget_check_20261103/training/seed43/checkpoints/best_model.pt
& $py picar_lab/verify_lab.py
& $py outputs/budget_check_20261103/picar_policy_to_udp.py
```

마지막 명령은 저장한 SUMO 기록 두 장면을 재생해 가상 차량에 허가를 보낸다. 결과는 [정책→UDP 기록](outputs/budget_check_20261103/picar_dry_run/policy_to_udp.json)에 저장된다.

## 파일 지도

- `configs/`, `maps/`, `src/`, `scripts/`, `tests/`, `notebooks/`는 SUMO 학습·평가 프로젝트다. [기존 README](README.md)
- `outputs/budget_check_20261103/`에는 240회 모델, 학습·검증 로그, 320회 평가와 경로, 비교 보고서가 있다.
- `outputs/verify_20261103/`는 80회 모델, 초기 평가, GUI 검증이다. `outputs/revision/`, `outputs/reference/`, `outputs/revision_comparison/`은 입력 ZIP에 있던 자료다.
- `picar_lab/`에는 모터 없는 UDP 실습, 등록 예시, 하드웨어 기록 양식, 트랙 도면과 현장 체크리스트가 있다. `picar_lab/ros2_reference/`는 노션의 두 하드웨어 소스 스냅샷이다.

