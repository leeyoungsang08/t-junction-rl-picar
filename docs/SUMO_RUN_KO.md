# SUMO 실행·재개 가이드

한 개 T자형 교차로에서 **교통신호를 유지하거나 다음 방향으로 전환**하는 DQN을 학습하고,
고정식·규칙식과 동일한 차량 수요로 비교합니다. 차량 조향이나 모터를 학습하는 프로젝트는 아닙니다.

## 바로 시작: Colab

1. ZIP을 내려받고 압축을 풉니다.
2. Google Colab에서 **`notebooks/START_HERE.ipynb`**를 업로드해 엽니다.
3. 첫 셀부터 실행하고 파일 선택창에 **전체 프로젝트 ZIP**을 업로드합니다.
4. 해당 노트북 하나를 위에서부터 실행하면 설치부터 학습·평가·보고서까지 이어집니다.
   단계별로 실행하려면 `00`~`06` 개별 노트북을 사용하세요.
5. 기존 모델만 시험하려면 설치 후 노트북 마지막 **06 · 학습된 모델 시험하기**로 이동하세요.
   별도 노트북 **`06_policy_playground.ipynb`**도 제공합니다.

노트북을 바꿀 때 런타임도 바뀌었다면 00의 ZIP 업로드·설치 셀을 다시 실행합니다.
Colab 노트북은 구조와 코드 셀을 검사했습니다. 아래 통합 패키지 검증은 Windows CPU에서 실행했습니다.
작은 DQN은 GPU가 없어도 학습할 수 있습니다. RTX 3060/3080/5070 등 모델명에 코드가 의존하지 않습니다.
SUMO는 CPU에서 실행되고 PyTorch는 `configs/training.yaml`의 `device: auto/cpu/cuda`로 선택합니다.

## 이번 재학습 결과

`VERIFICATION_KO.md`는 Windows CPU에서 실행한 `verify_20261103`의 검증과 재현 명령을 설명합니다.
새 모델·로그·평가·그래프는 `outputs/verify_20261103` 아래에 있습니다.
아래 제공 결과는 입력 ZIP에 포함된 이전 실험입니다.

## 제공 결과

`DELIVERY_STATUS.md`에서 실제 실행 범위와 완료 상태를 확인하세요.
개선 모델은 `outputs/revision/seed42`, `seed43`, `seed44`에 있습니다. 각각 검증 기준으로 선택한 모델과 마지막 모델, 학습 재개 체크포인트를 제공합니다.
비교 결과와 실제 SUMO 로그는 `outputs/revision_comparison`에 있습니다.
`outputs/reference/legacy_v1.pt`는 이전 모델입니다. 예전 모델은 12개 입력, 개선 모델은 25개 입력을 사용합니다.

```bash
python -m scripts.run_training --mode revision --training-seed 42 --output outputs/new_seed42
python -m scripts.run_evaluation --model outputs/new_seed42/checkpoints/best_model.pt --mode quick --output outputs/new_comparison
```

위 명령은 시드 42의 새 모델을 빠른 평가에 사용합니다. 세 시드의 168회 비교는 각 시드의 모델을 준비한 뒤 python -m scripts.compare_revision --training-root <모델-부모-경로> --output <새-결과-경로>로 실행합니다.

## Linux / WSL2

```bash
cd T_Junction_RL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
bash scripts/setup_colab.sh
python -m maps.generate_map
python -m pytest -q
python -m scripts.smoke_test --train
python -m scripts.run_training --mode quick --output outputs/my_quick
python -m scripts.run_evaluation --model outputs/my_quick/checkpoints/best_model.pt --output outputs/my_eval --mode quick
python -m src.visualization --training outputs/my_quick --evaluation outputs/my_eval --videos
python -m scripts.generate_report --training outputs/my_quick --evaluation outputs/my_eval
```

정식 학습·평가·그래프·보고서·영상은 아래 하나의 명령으로 실행할 수 있습니다.
기존 결과를 덮어쓰지 않도록 새 출력 경로를 사용하세요.

```bash
python -m scripts.run_pipeline --mode full --output outputs/new_experiment --videos
```

## Windows PowerShell

Python 3.11/3.12와 WSL2 Ubuntu 사용을 우선 권장합니다. Windows 네이티브도 pip SUMO를 지원합니다.

```powershell
cd T_Junction_RL
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.check_environment
python -m pytest -q
python -m scripts.run_training --mode full --output outputs/my_full
```

Windows에서 `SUMO_HOME`에 한글이 포함되면 SUMO 1.28.0이
`Quitting (on unknown error).`로 종료할 수 있습니다. 이 경우 영문 경로에 가상 환경을
만들거나 설치 디렉터리의 영문 경로 별칭을 `SUMO_HOME`으로 지정하세요. 재현 예시는
`VERIFICATION_KO.md`에 있습니다.

CUDA를 사용하려면 https://pytorch.org/get-started/locally/ 에서 현재 드라이버와 맞는 명령을 선택합니다.
Colab의 정상 PyTorch를 재설치하지 마세요. CPU만 필요하면 다음을 사용할 수 있습니다.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 재개와 Drive 보관

```bash
python -m scripts.run_training --mode full --output outputs/my_full --mirror /content/drive/MyDrive/T_Junction_RL/runs/my_full
python -m scripts.resume_training --output outputs/my_full --resume outputs/my_full/checkpoints/training_checkpoint.pt
```

`resume_training --help`로 인수를 확인하세요. 모델·target·optimizer·replay·난수 상태를 복원합니다.
마지막 완료 checkpoint 이후의 에피소드는 다시 시작합니다. 실행 중 SUMO 중간 상태는 저장하지 않습니다.
Drive 복구 시 보관된 run 폴더 전체를 복사하세요. `--mirror`는 완료 체크포인트와 해당 시점의 원시 로그까지 복사합니다.
평가를 중단했다면 같은 실행 옵션으로 `--resume`을 붙이면 완료한 조합을 건너뜁니다.
신호 설정·맵·정규화를 변경한 모델을 기존 모델과 섞지 마세요.

## 학습 결과 시험

Colab: `06_policy_playground.ipynb`에서 교통량·seed·제어기를 고른 뒤 5초 단위로 진행하거나 끝까지 실행합니다.
화면의 차량 위치는 실제 TraCI 결과이며, 임의로 움직이는 데모가 아닙니다.
로컬에서는 다음 명령을 사용합니다. `sumo-gui`가 설치된 데스크톱에서는 `--gui`를 추가할 수 있습니다.

```bash
python -m scripts.play_simulation --model outputs/budget_check_20261103/training/seed43/checkpoints/best_model.pt --scenario heldout_reversal --seed 3405 --horizon 1800 --output outputs/my_play
python -m scripts.inference_demo --model outputs/budget_check_20261103/training/seed43/checkpoints/best_model.pt
python -m scripts.benchmark_device
```

## 실험 변형

```bash
python -m scripts.run_training --mode full --double-dqn --output outputs/ddqn_run
python -m scripts.run_training --mode full --reward simple --output outputs/simple_reward_run
```

실제로 재학습·평가한 모델만 비교합니다. 테스트 결과를 보고 설정을 고친 경우 기존 test는 개발 데이터로
취급하고 새로 분리한 테스트 수요를 준비해야 합니다.

## 핵심 정의

- 수요: 전체 교차로 기준 Low 240 / Medium 480 / High 780 veh/h. Poisson 발생.
- 지도: 중심 (0,0), 서·동·남 250 m, 각 진입·진출 1차로, 6 OD, U턴 없음.
- 신호: W → E → S, 녹색 10–45 s, 황색 3 s, 전적색 1 s.
- DQN: 25 → 128 → 128 → 2, 행동 선택/target 계산 모두 mask 적용.
- DQN과 규칙식 의사결정은 5 s. 고정식은 정확히 녹색 20 s를 지키기 위해 1 s 스케줄러를 사용합니다.
  동일한 초 단위 안전 제어기를 사용하며 황색·전적색은 모두 같습니다.
- `mean_waiting_s`: 완료 차량만의 전체 정차 대기시간. 이동하면 초기화되는 현재 연속 대기시간과 다릅니다.
- `cohort_delay_s`: 종료까지 모든 예약 차량의 정차 대기 및 출발 지연 / 예약 차량 수.
  종료 이후의 잔여 지연은 관측하지 않으므로 완료율과 함께 읽습니다.
- reward는 매초 비용의 합, gamma=0.99는 **5초 의사결정 단위**입니다. 마지막 5초 미만 구간도 동일 gamma를 씁니다.
- `bootstrap_truncated: true`. 시간 제한에서는 마지막 실제 관측에서 bootstrap하며, 과제 자체가 종료되면 bootstrap하지 않습니다.

## 문서

`docs/architecture.md`, `experiment_design.md`, `state_action_reward.md`,
`evaluation_protocol.md`, `future_deployment.md`를 참조하세요.
결과를 미리 가정하지 않습니다. DQN이 나쁜 시나리오도 원시 CSV와 보고서에 그대로 남습니다.
