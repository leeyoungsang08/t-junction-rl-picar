import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import yaml
from src.utils import ROOT, atomic_text, write_json


def table(frame):
    def format_cell(value):
        if pd.isna(value):
            return "미측정"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(format_cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def analyze_failures(evaluation_dir, training_dir):
    evaluation_dir, training_dir = Path(evaluation_dir), Path(training_dir)
    results = pd.read_csv(evaluation_dir / "seed_level_results.csv")
    fixed = results[results.controller == "fixed_time"].set_index(["scenario", "seed"])
    dqn = results[results.controller == "dqn"].set_index(["scenario", "seed"])
    joined = dqn.join(fixed, lsuffix="_dqn", rsuffix="_fixed")
    rows = []
    for (scenario, seed), row in joined.iterrows():
        ts = pd.read_csv(ROOT / row.episode_dir_dqn / "timeseries.csv")
        q = ts.queue_total
        n = max(1, len(q) // 4)
        rows.append({"scenario": scenario, "seed": int(seed),
                     "waiting_delta_s": row.mean_waiting_s_dqn - row.mean_waiting_s_fixed,
                     "cohort_delay_delta_s": row.cohort_delay_s_dqn - row.cohort_delay_s_fixed,
                     "worse_completed_wait": bool(row.mean_waiting_s_dqn > row.mean_waiting_s_fixed),
                     "worse_cohort_delay": bool(row.cohort_delay_s_dqn > row.cohort_delay_s_fixed),
                     "completion_bias_risk_flag": bool(row.mean_waiting_s_dqn < row.mean_waiting_s_fixed and
                                                      (row.cohort_delay_s_dqn > row.cohort_delay_s_fixed or
                                                       row.completion_rate_dqn < row.completion_rate_fixed - 0.02)),
                     "long_wait_flag": bool(row.max_consecutive_wait_s_dqn > 180),
                     "queue_growth_flag": bool(q.iloc[-n:].mean() > q.iloc[:n].mean() + 5),
                     "frequent_switch_flag": bool(row.signal_switches_dqn > 1.5 * row.signal_switches_fixed),
                     "uninserted_flag": bool(row.uninserted_vehicles_dqn > 0),
                     "incomplete_fraction": 1 - row.completion_rate_dqn,
                     "dqn_episode_dir": row.episode_dir_dqn, "fixed_episode_dir": row.episode_dir_fixed})
    failures = pd.DataFrame(rows)
    atomic_text(evaluation_dir / "failure_cases.csv", failures.to_csv(index=False))
    cases = {}
    for name, index in [("best_relative", failures.cohort_delay_delta_s.idxmin()), ("worst_relative", failures.cohort_delay_delta_s.idxmax())]:
        row = failures.loc[index]
        cases[name] = {k: v.item() if isinstance(v, np.generic) else v for k, v in row.to_dict().items()}
        directory = evaluation_dir / "case_studies" / name
        directory.mkdir(parents=True, exist_ok=True)
        source = ROOT / row.dqn_episode_dir
        decisions = pd.read_csv(source / "decisions.csv")
        ticks = pd.read_csv(source / "timeseries.csv")
        combined = decisions.merge(ticks[["time", "phase", "stage", "queue_total", "mean_current_wait_s", "cumulative_arrived"]], on="time", how="left")
        combined["scenario"] = row.scenario
        combined["seed"] = row.seed
        combined["episode_cohort_delay_delta_s"] = row.cohort_delay_delta_s
        atomic_text(directory / "decision_trace.csv", combined.to_csv(index=False))
    write_json(evaluation_dir / "case_studies.json", cases)
    train = pd.read_csv(training_dir / "logs/training.csv")
    correlation = train[["episode_reward", "cohort_delay_s"]].corr(method="spearman").iloc[0, 1]
    alignment = {"spearman_reward_vs_cohort_delay": float(correlation) if np.isfinite(correlation) else None,
                 "warning_flag": bool(np.isfinite(correlation) and correlation > 0),
                 "interpretation": "Higher reward should accompany lower delay. Mixed training scenarios confound this descriptive correlation; it is not causal evidence."}
    write_json(evaluation_dir / "reward_metric_alignment.json", alignment)
    return failures, cases, alignment


def generate_report(training_dir, evaluation_dir):
    training_dir, evaluation_dir = Path(training_dir), Path(evaluation_dir)
    results = pd.read_csv(evaluation_dir / "seed_level_results.csv")
    paired = pd.read_csv(evaluation_dir / "scenario_comparison.csv")
    status = json.loads((training_dir / "training_status.json").read_text())
    manifest = json.loads((evaluation_dir / "experiment_manifest.json").read_text())
    config = yaml.safe_load((training_dir / "experiment_config.yaml").read_text())
    best = json.loads((training_dir / "checkpoints/best_model.metadata.json").read_text())
    failures, cases, alignment = analyze_failures(evaluation_dir, training_dir)
    metrics = ["mean_waiting_s", "cohort_delay_s", "mean_queue_veh", "throughput_veh_h", "completion_rate"]
    overall = results.groupby("controller")[metrics].mean().reset_index()
    improvements = paired[(paired.controller == "dqn") & (paired.metric == "mean_waiting_s")][
        ["scenario", "n_pairs", "improvement_pct", "difference_other_minus_fixed", "ci95_low", "ci95_high"]]
    horizon = int(results.horizon_seconds.iloc[0])
    flags = int(failures.worse_completed_wait.sum())
    unseen = results[results.scenario == "unseen"].groupby("controller")[metrics].mean().reset_index()
    report = f"""# T자형 교차로 강화학습 실험 보고서

## 목적과 검증 범위
West/East/South 단일 T자형 교차로에서 DQN이 교통 수요에 맞춰 녹색을 유지·전환하는지 검증한다.
PiCar 없이 SUMO에서 실험했다. 본 보고서의 모든 표는 실제 평가 CSV에서 계산했다.
실험 범위: **{manifest['evidence_scope']}**. Colab 자체 실행은 이 패키지 제작 환경에서 수행하지 않았다.

## 실행 환경
- Python: {manifest['versions']['python']}
- SUMO: {manifest['versions']['sumo']}
- PyTorch: {manifest['versions']['torch']}
- CUDA available: {manifest['versions']['cuda_available']}
- GPU: {manifest['versions']['gpu']}
- 맵: 250 m 도로 3방향, 방향당 진입·진출 각 1차로, 6 OD, 제한속도 13.89 m/s
- 신호: W → E → S. 녹색 최소 10 s / 최대 45 s, 황색 3 s, 전적색 1 s
- 고정식: 녹색 20 s. 모든 알고리즘에 같은 안전 제어기 적용

## 상태·행동·보상·모델
상태 스키마: {config['environment'].get('observation_schema', 1)}. 입력 크기: {config['training'].get('state_dim', 12)}.
행동은 KEEP / SWITCH이며 행동 선택과 Bellman target 양쪽에 유효 행동 마스크를 적용한다.
보상 모드: {config['environment']['reward']['mode']}; Double DQN: {config['training']['double_dqn']}.
상태 특성과 비용 정의는 `docs/state_action_reward.md`를 참조한다.

## 학습과 모델 선택
- 실행 모드: {status['mode']}
- 완료 에피소드: {status['completed_episodes']} / {status['planned_episodes']}
- 회당 학습 길이: {config['training']['horizon_seconds']} s
- DQN 업데이트: {status['updates']}
- 모델 난수 시드: {config['training']['seed']} (단일 학습 seed)
- Best 모델 에피소드: {best['episode'] + 1}
- Best validation cohort delay: {best['validation_score']:.3f} s/request
- Validation은 학습·테스트와 분리한 차량 seed를 사용한다. Unseen 수요는 학습과 validation에 사용하지 않았다.
- 중단 시 SUMO의 에피소드 중간 상태는 저장하지 않는다. 마지막 완료 체크포인트부터 재개한다.

## 평가 조건
{results.scenario.nunique()}개 시나리오, 시나리오당 {results.seed.nunique()}개 paired seed,
제어기 {results.controller.nunique()}개, 총 {len(results)}회, 각 {horizon} s.
모든 제어기는 같은 수요 XML, SUMO seed, 맵과 안전 규칙을 사용한다.
학습보다 긴 평가 길이는 시간 길이에 대한 추가 일반화 검사다.
Low/Medium/High의 전체 교차로 수요는 240/480/780 veh/h이다. 실제 발생량은 Poisson 표본이라 매회 다르다.
High라는 이름 자체가 포화 상태를 보장하지 않는다. 삽입 지연과 대기행렬의 실제 측정값을 함께 확인한다.

## 전체 성능
아래는 시나리오·seed에 같은 가중치를 둔 평균이다. mean_waiting_s는 **완료 차량만**의 총 정차 대기시간이다.
cohort_delay_s는 모든 예약 차량의 정차 대기 + 출발 지연을 포함하고, 미삽입 차량은 종료까지의 출발 지연을 반영한다.

{table(overall)}

## 미학습 수요와 미완료 차량의 영향
Unseen 시나리오에서의 결과는 아래와 같다. 완료 차량 평균 대기시간이 줄더라도,
cohort delay가 크거나 완료율이 낮으면 전체 수요를 더 잘 처리했다고 결론 낼 수 없다.
특히 규칙식과 DQN의 cohort delay와 완료율을 함께 비교해야 한다.
완료 차량 대기시간 개선과 전체 지연 악화 또는 완료율 2%p 초과 감소가 함께 나타난
진단 사례는 {int(failures.completion_bias_risk_flag.sum())}/{len(failures)}이다. 이는 선택 편향 위험 신호이며 원인의 입증은 아니다.

{table(unseen)}

## 시나리오별 DQN 대 고정식
개선율 양수는 대기시간 감소, 음수는 악화다. Difference 및 95% CI 단위는 초이며 DQN−Fixed 순서다.
CI는 같은 seed의 차이에 대한 Student t 구간이다. 학습 seed 변동까지 포함한 신뢰구간은 아니다.

{table(improvements)}

## 실패와 일반화
완료 차량 평균 대기시간이 고정식보다 나쁜 사례: {flags}/{len(failures)}.
내부 충돌 이벤트 합계: {int(results.collision_events.sum())}; Teleport 합계: {int(results.teleport_events.sum())}.
미삽입 차량 합계: {int(results.uninserted_vehicles.sum())}.
최대 관측 연속 대기시간: {results.max_consecutive_wait_s.max():.1f} s.
상대적으로 가장 좋은 사례: {cases['best_relative']['scenario']} / seed {cases['best_relative']['seed']}.
상대적으로 가장 나쁜 사례: {cases['worst_relative']['scenario']} / seed {cases['worst_relative']['seed']}.
각 사례의 실제 state/action/Q/신호/대기열 기록은 case_studies 폴더에 있다.
장기 대기·대기열 증가·빈번한 전환 플래그는 진단용 기준이며 자동으로 원인을 입증하지 않는다.
Reward와 cohort delay의 Spearman 상관: {alignment['spearman_reward_vs_cohort_delay']}.

## 영상과 그래프
plots/summary_figure.png 및 PDF, 학습·손실·epsilon·validation·Q분포·제어기 비교·시간별 그래프를 생성한다.
videos의 차량 좌표와 속도는 실제 TraCI 기록이다. 비교 영상은 같은 route와 시간 축을 쓴다.
영상 표시는 현재 도로 위 차량의 연속 대기시간이며, 최종 완료 차량 지표와 정의가 다르다.
`worst_relative_case.mp4`는 DQN−고정식 cohort delay가 가장 큰 사례의 전체 시간 구간을 보여준다.

## 한계
단일 교차로·한 개 학습 seed·보수적인 단일 방향 현시 구성이다. 대규모 도로망이나 최적 제어의 입증은 아니다.
경로·차로·차량 길이·안전시간은 고정했고, 센서 오차를 학습하지 않았다.
고정 평가 종료 시 미완료 차량이 존재할 수 있으며 완료율과 cohort 지표를 함께 읽어야 한다.
방향별 집계 관측은 개별 차량의 위치와 속도를 모두 보존하지 않으므로 완전한 Markov 상태를 보장하지 않는다.
GPU/CPU 속도 비교, libsumo, Double DQN 및 simple 보상 재학습은 실제 별도 실행한 경우만 결과로 인정한다.

## 향후 외부 시스템 연결
StateAdapter → CPU Policy.predict → 독립 SignalController → 외부 신호 시스템으로 연결한다.
정책 출력은 교통신호 유지·전환 결정이며 차량 조향/가감속 명령이 아니다.
센서에서 전체 방향 차량 수와 대기시간을 관측·추정하는 방법과 지연·누락 처리는 별도 검증이 필요하다.

## 원시 결과
seed_level_results.csv, final_results.csv, scenario_comparison.csv, failure_cases.csv,
experiment_config.yaml, experiment_manifest.json, 각 episode의 tripinfo.xml/timeseries.csv/decisions.csv.
모델·수요·맵의 SHA-256과 버전은 manifest에 기록했다.

## 공식 참고자료
- https://sumo.dlr.de/docs/Downloads.php
- https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html
- https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html
- https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html
- https://gymnasium.farama.org/api/env/
- https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
    atomic_text(evaluation_dir / "final_report.md", report)
    shutil.copy2(training_dir / "logs/training.csv", evaluation_dir / "training_summary.csv")
    return evaluation_dir / "final_report.md"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--evaluation", required=True)
    a = p.parse_args()
    print(generate_report(a.training, a.evaluation))


if __name__ == "__main__":
    main()
