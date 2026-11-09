# PiCar Real-to-Sim 진행 상황 (2026-11-09)

브랜치 `picar-real2sim`. 기존 `src/`·`outputs/`·KEEP/SWITCH 모델은 건드리지 않았고, 새 작업은 모두 [`picar_sim/`](picar_sim/)에 있다.
자세한 설계·파라미터·결과: [docs/PICAR_REAL2SIM_PLAN_KO.md](docs/PICAR_REAL2SIM_PLAN_KO.md) · Colab: [notebooks/08_picar_real2sim_colab.ipynb](notebooks/08_picar_real2sim_colab.ipynb)

## 한눈에 보기

| 항목 | 상태 |
|---|---|
| 실물 코스 → 미터 단위 SUMO 맵 | 완료 (외곽 순환로 + 중앙 연결로, T자 2곳). **치수는 사용자 설명 기반 가정값**, 사진·실측 미반영 |
| 관측 가능한 상태 (APPROACH/ARRIVE/IN/EXIT 이벤트 + 출구 구간 여유) | 완료. SUMO 정답값은 보상·지표에만 사용 |
| 독립 안전 제어기 (최소·최대 녹색, 점유 0 확인 후 전환, 센서·통신 결함 시 전면 STOP, 박스 막힘 방지) | 완료. SUMO와 실차 중앙 PC가 같은 클래스 사용 |
| J2(위 T자) | 같은 안전 제어기 + 고정 대기열 규칙 |
| DQN 학습 (시드 42/43/44 × 1,500 에피소드) | 완료: 순환 모드, 직접 선택 모드 |
| 짝지은 평가 (test 시나리오 3개 × 시드 10개 × nominal/noisy) | 완료, 600회 |
| 테스트 | 9개 통과 (`python -m pytest -q picar_sim/tests`) |
| 실차 중앙 PC 브리지 (UDP, shadow/active, 로그 재생) | 코드 완료. SUMO 기록 재생 시 행동 100% 일치. **실차 미검증** |
| 실측 → 보정 도구 (`calibrate.py`) | 코드 완료. 실측 데이터 없음 |
| PyTorch(GPU) 백엔드 | 코드 있음, **미실행** (작업 공간에 설치 실패 → numpy로 학습) |
| 실측 범위 랜덤화 학습 | 미실행 (실측값이 생긴 뒤 진행) |

## 결과 요약 (SUMO, 가정 파라미터)

| 모드 | 제어기 | 지연 [s] (nominal) | 최장 대기 [s] | 통과 [대/분] |
|---|---|---:|---:|---:|
| 순환 | DQN 3시드 평균 | 14.73 | 53.3 | 6.61 |
| 순환 | 대기열 규칙 | 16.03 | 50.6 | 6.13 |
| 순환 | 고정식 5 s | 21.19 | 74.7 | 5.31 |
| 직접 선택 | DQN 3시드 평균 | **13.93** | 51.9 | **6.90** |
| 직접 선택 | 대기열 규칙 | 14.40 | **41.0** | 6.35 |

- 고정식 대비 지연 약 −30%. 대기열 규칙 대비 차이는 작고, 95% CI가 0을 벗어난 경우는 일부 시나리오·시드뿐.
- 직접 선택이 DQN과 규칙 모두를 개선. DQN direct는 최장 대기가 규칙보다 약 11 s 김.
- 충돌·상충 동시 점유·정지선 위반·교착 모두 0 (600회).

## 작업 중 발견하고 고친 문제

1. SUMO 기본값(`jmStoplineGap` 1 m) 때문에 차량이 정지선 1 m 앞에 서서 ARRIVE 이벤트가 생기지 않음 → 0.02 m로 고치고 재학습.
2. 연결로가 방향마다 1대만 수용 → 6대에서 제어기와 무관하게 교착 10–30% → "다음 구간에 자리가 있을 때만 GO" 추가 후 교착 0.
3. J2를 SUMO 우선순위 교차로로 두면 SUMO 내부 교착 발생. PiCar는 스스로 양보하지 못하므로 실물 J2도 중앙 GO/STOP 필요.

## 다음 할 일

1. **실측**: 코스 가로·세로, 최소 회전반경(안쪽 차로 코너 반경 0.225 m가 가능한지), 정지선~충돌구간 거리, 저속 속도·정지오차·통과시간, UDP 왕복 지연 → `picar_sim/config/field_measurements.template.csv` 형식
2. `python -m picar_sim.calibrate --csv field.csv` → 보정 맵에서 재학습·재평가
3. 단일 변경 실험: 장기 대기 벌점 강화(직접 선택 DQN의 최장 대기 개선), 실측 범위 랜덤화
4. 실차: `real_bridge replay` → `serve --mode shadow` → 저속 `--mode active`

## 이 커밋에 없는 것
- 학습 재개용 `checkpoint.pkl`(시드당 약 20 MB)은 용량 때문에 제외. 정책(`best_policy.npz`, `last_policy.npz`)과 로그·평가 결과는 포함.
