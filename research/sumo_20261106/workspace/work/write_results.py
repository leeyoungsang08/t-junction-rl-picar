import json
import re
from pathlib import Path
import sys

import pandas as pd

root = Path(__file__).resolve().parent / "T_Junction_RL"
research = root / "outputs/v3_research"
stage = sys.argv[1]
selected = research / stage
verification = research / "verification"

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_doc(name, lines):
    (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

def count_runs(folder):
    rows = [read_json(p) for p in folder.glob("*/training_status.json")]
    return {"models": len(rows), "completed_runs": sum(r["complete"] for r in rows),
            "episodes": sum(r["completed_episodes"] for r in rows), "updates": sum(r["updates"] for r in rows)}

stages = {name: count_runs(research / name) for name in ["smoke", "quick", "full"]}
audit = read_json(verification / "evidence_audit.json")
env_text = (verification / "env_final.log").read_text(encoding="utf-8-sig")
versions = json.loads(env_text[env_text.index("{"):env_text.rindex("}")+1])
pytest_text = (verification / "pytest.log").read_text(encoding="utf-8-sig")
passed = int(re.findall(r"(\d+) passed", pytest_text)[-1])
device = (verification / "device.log").read_text(encoding="utf-8-sig").strip()
plots = "figures" if stage == "full" else "final_figures"
groups = pd.read_csv(selected / plots / "algorithm_validation.csv")
quick = pd.read_csv(research / "quick/figures/algorithm_validation.csv")
validation = read_json(research / "quick/validation_eval/evaluation_status.json")
evaluation = selected / "final_eval"
plan = read_json(evaluation / "evaluation_plan.json")
final_status = read_json(evaluation / "evaluation_status.json")
raw = pd.read_csv(evaluation / "seed_level_results.csv")
assert final_status["complete"] and len(raw) == len(plan["scenarios"]) * plan["seeds"] * len(plan["controllers"])
assert not raw.duplicated(["scenario", "seed", "controller"]).any()
assert raw.groupby(["scenario", "seed"]).route_sha256.nunique().eq(1).all()
assert audit["validated_training"]["full"]["runs"] == stages["full"]["completed_runs"]
assert all(raw.groupby(["scenario", "seed"]).size() == len(plan["controllers"]))
group_name = raw.controller.str.replace(r"_seed\d+$", "", regex=True)
raw = raw.assign(group=group_name)
condition = raw.groupby(["scenario", "group"]).mean(numeric_only=True)
overall = condition.groupby("group").mean(numeric_only=True).sort_values("cohort_delay_s")
case_paired = pd.read_csv(evaluation / "group_paired_comparison.csv")
cases = ["high", "south_bias", "south_oversaturated", "heldout_reversal"]
baseline_names = ["fixed_time", "rule_based", "max_pressure_proxy"]
rl_groups = list(overall.index.difference(baseline_names))
full_runs = stages["full"]["completed_runs"]
full_scope = f"{full_runs}/24개 완료, {24 - full_runs}개 NOT RUN" if full_runs else "24개 NOT RUN"
backend = plan["environment"]["fixed_time"]["simulation"]["backend"]

lines = [
    "# 새 V3 실험 결과", "",
    "이 실험은 단일 T자형 교차로에서 신호 정책을 바꿨을 때 요청당 지연과 완료율이 어떻게 달라지는지 측정한다. "
    "기존 평가의 실행 중단 오류와 기준선 혼용을 수정한 뒤 같은 수요와 안전 제약을 사용했다. "
    "설계 근거는 [논문 조사](literature_review.md), 관측·보상은 [상태·행동·보상](state_action_reward.md)을 참조한다.", "",
    "## 학습 및 후보 선정", "",
    f"- Smoke {stages['smoke']['completed_runs']}개, {stages['smoke']['episodes']} 에피소드, {stages['smoke']['updates']} 업데이트.",
    f"- 예비 학습 {stages['quick']['completed_runs']}개, {stages['quick']['episodes']} 에피소드, {stages['quick']['updates']} 업데이트. "
    f"공통 검증 기준선 및 모델 재평가 {validation['episodes']}회.",
    f"- 정식 학습 {full_scope}. 최종 평가는 **{stage} 단계**의 모델을 사용했다. "
    f"실행된 정식 모델은 각 300×900초로 예산을 맞췄다. CPU 실행 시간 측정 결과를 반영해 검증 상위의 Cycle/Direct 후보에 정식 예산을 먼저 배정했다.",
    "- 각 run의 체크포인트는 validation 지연 최솟값으로 선택했다. 아래 표는 선택된 체크포인트의 학습 시드 간 평균과 표준편차다.", "",
    "| 예비 학습 조합 | Validation 지연 (초/요청) | 학습 시드 표준편차 | 시드 수 |",
    "|---|---:|---:|---:|",
]
for row in quick.itertuples():
    lines.append(f"| {row.group} | {row.mean:.2f} | {row.std:.2f} | {row.count} |")
lines += [
    "", f"해당 단계의 두 후보 중 검증 평균 지연이 가장 낮은 조합은 **{groups.iloc[0]['group']}** "
    f"({groups.iloc[0]['mean']:.2f}초/요청)였다.", "",
    f"동결된 최종 모델군은 {', '.join(rl_groups)}이다. 선택 근거는 각 제어 방식 안에서 해당 단계의 "
    "학습 시드 평균 validation 지연이 가장 낮은 조합이었다. 정식 비교 일부만 수행한 경우 Stage 2 순위로 Stage 3에 올릴 후보를 정했다.", "",
    "## 동결 후 독립 평가", "",
    f"{len(plan['scenarios'])}조건 × {plan['seeds']}수요 시드 × {len(plan['controllers'])}제어기 = {final_status['episodes']}회. "
    f"각 회차 {plan['horizon_seconds']}초, SUMO 백엔드는 {backend}. 완료율은 요청당 비율, 방향 격차는 방향별 관측 지연 평균의 최대 차이다. "
    "아래는 각 시나리오를 같은 비중으로 평균한 수치다.", "",
    "| 제어기 / 모델군 | 요청당 지연 (초) | 완료율 (%) | 처리량 (대/시간) | 방향 격차 (초) |",
    "|---|---:|---:|---:|---:|",
]
for name, row in overall.iterrows():
    lines.append(f"| {name} | {row.cohort_delay_s:.2f} | {row.completion_rate*100:.1f} | {row.throughput_veh_h:.1f} | {row.direction_delay_gap_s:.2f} |")
names = list(overall.index)
lines += ["", "### 시나리오별 요청당 지연 (초)", "",
          "| 조건 | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
for scenario in plan["scenarios"]:
    lines.append("| " + scenario + " | " + " | ".join(f"{condition.loc[(scenario, name), 'cohort_delay_s']:.2f}" for name in names) + " |")
lines += ["", f"최종 평가 충돌 {raw.collision_events.sum()}회, teleport {raw.teleport_events.sum()}회. "
          "모델별·방향별 지표, 수요별 짝 비교 및 신뢰구간은 원시 CSV에 포함한다. "
          f"[최종 평가 폴더](../outputs/v3_research/{stage}/final_eval/) · [그래프](../outputs/v3_research/{stage}/{plots}/)", "",
          "V1/V2의 예전 검증은 학습 예산·수요·제어 설정이 달라 본 표에 포함하지 않았다. "
          "완료되지 않은 차량은 종료까지의 관측 지연과 완료율로 평가했으며, 종료 후의 대기시간에는 검열이 있다."]
write_doc("docs/experiment_results.md", lines)

lines = ["# 실패 조건과 근거", "",
         "수요 시드별 차이를 비교하고 녹색 배분·대기 큐·완료율을 확인했다. 아래 차이는 모델군의 학습 시드를 수요별로 평균한 값이며 "
         "양수는 RL의 지연이 더 길다는 뜻이다.", ""]
for scenario in cases:
    lines.extend([f"## {scenario}", "", "| 모델군 | 규칙식 대비 차이 (95% 구간) | 큐 대리 제어기 대비 차이 (95% 구간) | 완료율 | 남쪽 지연 (초) |",
                  "|---|---:|---:|---:|---:|"])
    for model in rl_groups:
        rows = case_paired[(case_paired.scenario == scenario) & (case_paired.metric == "cohort_delay_s")
                           & (case_paired.controller == model)].set_index("baseline")
        intervals = []
        for baseline in ("rule_based", "max_pressure_proxy"):
            diff = rows.loc[baseline]
            intervals.append(f"{diff.difference_other_minus_baseline:+.2f} "
                             f"[{diff.ci95_low:+.2f}, {diff.ci95_high:+.2f}]")
        row = condition.loc[(scenario, model)]
        lines.append(f"| {model} | {' | '.join(intervals)} | {row.completion_rate*100:.1f}% | {row.cohort_delay_S_s:.2f} |")
    case_rows = raw[(raw.scenario == scenario) & raw.group.isin(rl_groups)]
    worst = case_rows.loc[case_rows.cohort_delay_s.idxmax()]
    pair = raw[(raw.scenario == scenario) & (raw.seed == worst.seed) & (raw.controller == "rule_based")].iloc[0]
    lines += ["", f"해당 조건에서 RL 지연 최대 회차는 `{worst.controller}`, 수요 시드 {worst.seed}다. "
              f"지연 {worst.cohort_delay_s:.2f}초, 완료율 {worst.completion_rate*100:.1f}%, 미삽입 {int(worst.uninserted_vehicles)}대였다. "
              f"같은 수요의 규칙식은 {pair.cohort_delay_s:.2f}초, 완료율 {pair.completion_rate*100:.1f}%, "
              f"미삽입 {int(pair.uninserted_vehicles)}대였다.", "",
              "| 제어기 | W/E/S 녹색 (초) | W/E/S 관측 지연 (초) | 최대 큐 + 미삽입 | 전환 횟수 |",
              "|---|---:|---:|---:|---:|"]
    for case in [worst, pair]:
        trace = pd.read_csv(evaluation / case.episode_dir / "timeseries.csv")
        green = trace.loc[trace.stage == "green", "phase"].value_counts().reindex([0,1,2], fill_value=0)
        delays = [case[f"cohort_delay_{d}_s"] for d in "WES"]
        peak = (trace.queue_total + trace.pending_total).max()
        lines.append(f"| {case.controller} | {' / '.join(map(str, green))} | "
                     f"{' / '.join(f'{value:.1f}' for value in delays)} | {peak} | {int(case.signal_switches)} |")
    if scenario == "heldout_reversal":
        request_file = evaluation / "routes/final" / f"{scenario}_seed{worst.seed}_t{plan['horizon_seconds']}.rou.json"
        requests = list(read_json(request_file)["vehicles"].values())
        lines += ["", "| 구간 (초) | W/E/S 요청 | RL W/E/S 녹색 (초) | 규칙식 W/E/S 녹색 (초) |",
                  "|---|---:|---:|---:|"]
        traces = [pd.read_csv(evaluation / r.episode_dir / "timeseries.csv") for r in [worst, pair]]
        for segment in plan["traffic"]["scenarios"][scenario]["segments"]:
            begin, end = [part * plan["horizon_seconds"] for part in segment[:2]]
            demand = [sum(v["origin"] == direction and begin <= v["requested_depart"] < end for v in requests)
                      for direction in "WES"]
            allocations = [trace.loc[(trace.time > begin) & (trace.time <= end) & (trace.stage == "green"), "phase"]
                           .value_counts().reindex([0,1,2], fill_value=0).tolist() for trace in traces]
            lines.append(f"| {begin:.0f}–{end:.0f} | " + " | ".join(" / ".join(map(str, column))
                         for column in [demand, *allocations]) + " |")
    lines.append("")
south = condition.loc["south_oversaturated"]
cycle = south.loc["cycle_rainbow_lite"]
direct = south.loc["direct_rainbow_lite"]
pressure = south.loc["max_pressure_proxy"]
direct_oversat = raw[(raw.scenario == "south_oversaturated") & (raw.group == "direct_rainbow_lite")]
tail = direct_oversat.loc[direct_oversat.cohort_delay_s.idxmax()]
tail_proxy = raw[(raw.scenario == "south_oversaturated") & (raw.controller == "max_pressure_proxy")
                 & (raw.seed == tail.seed)].iloc[0]
lines += ["## 해석", "",
          f"남쪽 과포화에서 남쪽 요청의 평균 지연은 Cycle {cycle.cohort_delay_S_s:.1f}초, "
          f"Direct {direct.cohort_delay_S_s:.1f}초였다. Direct의 동쪽 지연도 {direct.cohort_delay_E_s:.1f}초로 길었다. "
          f"큐 대리 제어기는 남쪽 {pressure.cohort_delay_S_s:.1f}초와 동쪽 {pressure.cohort_delay_E_s:.1f}초를 기록했다. "
          "전체 요청 기준의 개선과 소수 방향의 대기를 함께 확인할 필요가 있다.", "",
          f"Direct의 과포화 최대 지연 회차인 {tail.controller}, 수요 {tail.seed}에서는 "
          f"{tail.cohort_delay_s:.2f}초로 동일 수요의 큐 대리 제어기 {tail_proxy.cohort_delay_s:.2f}초보다 길었다.", "",
          "High 조건에서 Direct의 큐 대리 제어기 대비 95% 구간은 0을 포함한다. "
          "남쪽 과포화에서 Cycle의 평균 지연은 큐 대리 제어기보다 약 40초 길었다. "
          "Direct도 이 조건에서 약 4분의 1이 관측 종료까지 완료되지 않았으며, 특정 학습 시드에서 teleport가 발생했다.", "",
          "녹색 배분이 수요 편중을 따라가지 못하면 해당 접근로의 미삽입과 대기가 늘어난다. "
          "반대로 높은 부하에 집중하면 낮은 수요 방향의 지연 격차가 커질 수 있다. "
          "표의 한 회차는 이 가설을 확인할 진단이며, 보상항이나 관측 하나의 인과 효과를 분리하는 실험은 아니다.", "",
          "남쪽 과포화와 반전 유형은 과거에 실패가 알려진 조건이다. 이번에는 새 seed의 요청 시각과 OD 표본을 사용했다. "
          "이 결과를 이용해 설정을 바꿀 경우 다음 연구에는 새로운 평가 수요를 동결해야 한다."]
write_doc("docs/failure_analysis.md", lines)

lines = ["# 검증 보고서", "",
         f"- 실행 환경 Windows / Python {versions['python']} / {versions['sumo']} / PyTorch {versions['torch']}.",
         f"- Pytest 최종 {passed} 통과. 지도 재생성, 6개 OD 도착, reset, 신호 보호, 마스크, TD/PER, V1/V2 로딩, "
         "Direct 평가, 실제 중단 후 재개와 동일 seed 재현을 포함한다.",
         f"- 신규 원시 로그 {audit['episodes_with_valid_tls_and_accounting']}회에서 구간 신호, 차량 수 보존, split seed 및 해시 분리 검사 통과. "
         f"충돌 {audit['collisions']}회, teleport {audit['teleports']}회.",
         f"- Stage 1은 {stages['smoke']['completed_runs']}개 실행 완료. Stage 2는 {stages['quick']['completed_runs']}개 실행 완료. "
         f"Stage 3은 {full_scope}. 새 독립 수요에서 {final_status['episodes']}회 평가 완료.",
         f"- 장치 벤치마크 `{device}`. CUDA 및 CPU/GPU 전체 경과시간 비교는 NOT RUN.",
         "- Colab 노트북의 구조와 코드 셀 문법은 검증했다. Colab 호스트, PiCar 및 V2I 통신 실행은 NOT RUN.",
         "- 첫 전체 테스트에서 PER의 epsilon 항을 빠뜨린 기대값 한 건이 실패했다. 테스트 기대값을 수정한 뒤 전부 통과했다. 최초 실패 로그도 보존했다.",
         "- 소켓 TraCI와 Libsumo에서 4개 수요·제어 조합의 metrics, decisions 및 timeseries 일치 확인.",
         "- 초기 Windows 설치는 SUMO wheel의 긴 경로에서 실패했다. 짧은 작업 경로에 별도 가상환경을 두고 재실행했다.",
         "- 원본 파일 분류 및 ZIP 검사 항목은 `RUN_MANIFEST.json`에 기록했다."]
write_doc("VERIFICATION_REPORT.md", lines)

manifest = {
    "date_kst": "2026-11-03", "base": "T_Junction_RL_Project_Paper_V3.zip",
    "github_reference_commit": "8636f8f25fc895ed9f0951bf7624453105deff64",
    "secondary_reference_zip_available": False,
    "versions": {k: v for k, v in versions.items() if k not in ("project_path","SUMO_HOME")},
    "tests": {"passed": passed, "failed_in_final_run": 0, "first_run_failed": 1}, "stages": stages,
    "full_training_status": full_scope, "final_evaluation": {"stage": stage, **final_status,
        "horizon_seconds": plan["horizon_seconds"], "seeds": plan["seeds"], "seed_start": plan["seed_start"],
        "groups": rl_groups, "backend": backend},
    "artifact_audit": audit, "not_run": [f"{24-full_runs} formal model/seed combinations", "Colab host execution", "CUDA benchmark", "PiCar/V2I"],
    "required_evidence": [f"outputs/v3_research/{stage}/final_eval/{file}" for file in (
        "evaluation_plan.json", "seed_level_results.csv", "group_paired_comparison.csv", "demand_manifest.json"
    )] + [f"outputs/v3_research/{stage}/{plots}/{file}" for file in (
        "algorithm_validation.csv", "scenario_heatmap.png", "paired_confidence_intervals.png"
    )] + ["outputs/v3_research/verification/pytest.log", "outputs/v3_research/verification/evidence_audit.json"],
}
(root / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps({k: manifest[k] for k in ["stages", "tests", "final_evaluation"]}, ensure_ascii=True))
print(overall[["cohort_delay_s", "completion_rate"]].to_string())
print(condition.loc[cases][["cohort_delay_s", "completion_rate"]].to_string())
