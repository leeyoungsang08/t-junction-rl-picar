#!/usr/bin/env python3
"""Judge Gazebo runs against the acceptance table of docs/PICAR_GAZEBO_KO.md (section 4).

python3 check_acceptance.py <acceptance_dir>          # dirs seed*/ (normal runs) and beam*/ (beam-fault run)

Per run it reads what course.launch.py leaves in the run directory:
  metrics.json  (course_monitor)   collisions, lane departures, stop errors
  launch.log    (tee of ros2 launch) brain events: "enter J W->E legal=...", "FAULT ...", dead processes
  poses.csv     (pose_logger.py)    deadlock = every car stationary for >= 30 s
  central.jsonl (central server)    beam run: J grants issued while J supervisor was in fault
Writes summary.md / summary.json next to the runs; exit code 0 only if every check passes.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

STOP_TOL_M = 0.05          # bumper vs. stop-line far edge
BOX_GAP_M = 0.12           # stop-line far edge -> conflict box (course.yaml stopline_offset_m)
DEADLOCK_S = 30.0
MOVE_EPS_M = 0.01          # moved less than this in 1 s = stationary
STARTUP_S = 15.0           # ignore the first seconds (cars spawn at +6 s, register, wait for permits)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def read_log(path):
    if not path.exists():
        return None
    full = ANSI.sub("", path.read_text(errors="replace"))
    cut = min([i for i in (full.find("user interrupted"), full.find("signal_handler(SIGINT")) if i >= 0] or [len(full)])
    text = full[:cut]                          # deaths/tracebacks after the end-of-run SIGINT are shutdown noise
    enters = re.findall(r"enter (J2?) \w+->\w+ legal=(True|False)", text)
    return {"crossings": len(enters),
            "illegal": sum(1 for _, legal in enters if legal == "False"),
            "faults": len(re.findall(r"\]: FAULT", text)),
            "fault_lines": sorted(set(re.findall(r"\]: (FAULT[^\n]*)", text)))[:5],
            "dead": len(re.findall(r"process has died", text)),
            "tracebacks": len(re.findall(r"Traceback \(most recent call last\)", text)),
            "calib_missing": "missing: run calibrate.launch.py first" in text,
            "delays": junction_delays(text)}


def junction_delays(text):
    """Per junction: APPROACH request -> entry [s] for every crossing (brain events in the drive-node log)."""
    last, out = {}, {"J": [], "J2": []}
    for t, car, kind, jn in re.findall(r"\[(\d+\.\d+)\] \[([\w.]+)\]: (approach|enter) (J2?)\b", text):
        car = car.split(".")[-1]
        if kind == "approach":
            last[(car, jn)] = float(t)
        elif (car, jn) in last:
            out[jn].append(float(t) - last.pop((car, jn)))
    return out


def deadlock(path):
    """Longest stretch [s] in which every car was stationary, and the sim/wall time covered."""
    if not path.exists():
        return None
    by_car = defaultdict(list)
    for line in path.read_text().splitlines()[1:]:
        try:
            w, s, car, x, y = line.split(",")
            by_car[car].append((float(w), float(s), float(x), float(y)))
        except ValueError:
            continue
    if not by_car:
        return None
    t0 = min(v[0][0] for v in by_car.values())
    t1 = max(v[-1][0] for v in by_car.values())
    sims = [r[1] for v in by_car.values() for r in v]
    moving_at = defaultdict(set)              # 1-s bins in which a car moved
    for car, rows in by_car.items():
        j = 0
        for i, (w, _, x, y) in enumerate(rows):
            while rows[j][0] < w - 1.0:
                j += 1
            if ((x - rows[j][2]) ** 2 + (y - rows[j][3]) ** 2) ** 0.5 > MOVE_EPS_M:
                moving_at[int(w - t0)].add(car)
    longest = cur = 0
    for b in range(int(STARTUP_S), int(t1 - t0)):
        cur = cur + 1 if not moving_at.get(b) else 0
        longest = max(longest, cur)
    wall = t1 - t0
    return {"all_stopped_max_s": float(longest), "wall_s": round(wall, 1),
            "sim_s": round(max(sims) - min(sims), 1), "rtf": round((max(sims) - min(sims)) / wall, 2) if wall else None}


def beam_check(path, start, stop):
    """Grants for J issued while the J supervisor reported a fault, and whether the fault happened at all."""
    if not path.exists():
        return None
    fault_at, grants = {}, {}
    fault_spans, cur = [], None
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t, f = r["t"], bool((r.get("J") or {}).get("fault"))
        fault_at[round(t, 1)] = f
        if f and cur is None:
            cur = t
        elif not f and cur is not None:
            fault_spans.append((cur, t))
            cur = None
        for car, g in (r.get("reserved") or {}).items():
            if g.get("junction") == "J":
                grants.setdefault(g["grant"], (round(g["t"], 1), car))
    if cur is not None:
        fault_spans.append((cur, t))
    during = [(gid, t, car) for gid, (t, car) in grants.items() if fault_at.get(t)]
    seen = any(a <= stop + 5 and b >= start for a, b in fault_spans)
    return {"grants_during_fault": len(during), "examples": during[:5], "fault_seen_in_window": seen,
            "fault_spans_s": [(round(a, 1), round(b, 1)) for a, b in fault_spans][:5]}


def judge(run, kind, minutes, beam_window):
    m = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else None
    log = read_log(run / "launch.log")
    dl = deadlock(run / "poses.csv")
    checks = []

    def add(name, ok, value):
        checks.append({"check": name, "ok": bool(ok), "value": value})

    if m is None:
        add("metrics.json 있음", False, "없음 (course_monitor가 안 돌았거나 run 실패)")
    else:
        add("충돌 0", m["collisions"] == 0, m["collisions"])
        add("차로 이탈 0", m["lane_departures"] == 0, m["lane_departures"])
        errs = m.get("stop_errors_m") or []
        bad = [e for e in errs if abs(e) > STOP_TOL_M]
        inbox = [e for e in errs if e < -BOX_GAP_M]
        add("정지 위치 ±5 cm", errs and not bad,
            f"{len(errs)}회, 벗어남 {len(bad)}, p95 오차 = {m.get('stop_error_p95_abs_m')} m" if errs else "기록 없음")
        add("박스 안 정지 0", not inbox, len(inbox))
        if kind == "seed":
            add(f"주행 시간 ≥ {minutes} 분", m.get("elapsed_s", 0) >= minutes * 60 - 15, f"{m.get('elapsed_s')} s")
    if log is None:
        add("launch.log 있음", False, "없음")
    else:
        add("허가 없는 진입 0", log["illegal"] == 0, f"{log['illegal']} / 진입 {log['crossings']}")
        add("FAULT 0", log["faults"] == 0, log["faults"] if not log["fault_lines"] else f"{log['faults']}: {log['fault_lines']}")
        add("노드 비정상 종료 0", log["dead"] == 0 and log["tracebacks"] == 0,
            f"died {log['dead']}, traceback {log['tracebacks']}")
        add("교차로 통과 발생", log["crossings"] > 0, log["crossings"])
        if log["calib_missing"]:
            add("보정 파일 사용", False, "car_calibration.yaml 없이 실행됨")
    if dl is None:
        add("poses.csv 있음", False, "없음 (교착 판정 불가)")
    else:
        add("교착 0 (전원 30 s 이상 정지)", dl["all_stopped_max_s"] < DEADLOCK_S, f"최장 {dl['all_stopped_max_s']:.0f} s")
        add("실시간 비율 ≥ 0.9", (dl["rtf"] or 0) >= 0.9, f"RTF {dl['rtf']} (sim {dl['sim_s']} s / wall {dl['wall_s']} s)")
    if kind == "beam":
        b = beam_check(run / "central.jsonl", *beam_window)
        if b is None:
            add("central.jsonl 있음", False, "없음")
        else:
            add("빔 결함이 실제로 감지됨", b["fault_seen_in_window"], b["fault_spans_s"])
            add("빔 결함 중 신규 진입 허가 0", b["grants_during_fault"] == 0,
                f"{b['grants_during_fault']} {b['examples'] or ''}")
    perf = None
    if log and m:
        d = log["delays"]["J"]
        mins = max(m.get("elapsed_s", 0) - 6, 1) / 60
        perf = {"J_crossings": len(d), "J_per_min": round(len(d) / mins, 2),
                "J_delay_mean_s": round(sum(d) / len(d), 2) if d else None, "J_delay_max_s": round(max(d), 1) if d else None,
                "all_crossings": log["crossings"]}
    return {"run": run.name, "kind": kind, "ok": all(c["ok"] for c in checks), "checks": checks,
            "metrics": m, "deadlock": dl, "perf": perf}


def main():
    root = Path(sys.argv[1]).expanduser()
    cfg = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else {}
    minutes = float(cfg.get("minutes", 10))
    window = (float(cfg.get("beam_fault_start", 60)), float(cfg.get("beam_fault_stop", 80)))
    runs = [judge(d, "beam" if d.name.startswith("beam") else "seed", minutes, window)
            for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith(("seed", "beam"))]
    calib = Path.home() / "picar_gz_runs" / "car_calibration.yaml"
    lines = [f"# PiCar Gazebo 합격 기준 확인 — {root.name}", ""]
    if calib.exists():
        lines += ["보정 파일 `car_calibration.yaml`:", "```", calib.read_text().strip(), "```", ""]
    for r in runs:
        lines += [f"## {r['run']} — {'합격' if r['ok'] else '불합격'}", "", "| 항목 | 결과 | 값 |", "|---|---|---|"]
        lines += [f"| {c['check']} | {'OK' if c['ok'] else 'FAIL'} | {c['value']} |" for c in r["checks"]]
        lines.append("")
    groups = {}
    for r in runs:
        if r["kind"] == "seed" and r["perf"]:
            groups.setdefault(r["run"].split("_", 1)[1] if "_" in r["run"] else "rule", []).append(r)
    if len(groups) > 1 or any(len(v) > 0 for v in groups.values()):
        lines += ["## 교차로 J 성능 (APPROACH 요청 → 진입)", "",
                  "| 제어 | run 수 | 통과 [대/분] | 평균 지연 [s] | 최장 지연 [s] | 안전 항목 |", "|---|---:|---:|---:|---:|---|"]
        for g, rs in sorted(groups.items()):
            ds = [r["perf"] for r in rs]
            n = sum(p["J_crossings"] for p in ds)
            mean = sum((p["J_delay_mean_s"] or 0) * p["J_crossings"] for p in ds) / n if n else None
            lines.append(f"| {g} | {len(rs)} | {sum(p['J_per_min'] for p in ds) / len(ds):.2f} | "
                         f"{'-' if mean is None else f'{mean:.2f}'} | {max((p['J_delay_max_s'] or 0) for p in ds):.1f} | "
                         f"{'모두 OK' if all(r['ok'] for r in rs) else 'FAIL 있음'} |")
        lines.append("")
    overall = bool(runs) and all(r["ok"] for r in runs)
    lines.append(f"**종합: {'합격' if overall else '불합격'}** ({sum(r['ok'] for r in runs)}/{len(runs)} run 통과)")
    text = "\n".join(lines)
    (root / "summary.md").write_text(text + "\n")
    (root / "summary.json").write_text(json.dumps({"ok": overall, "runs": runs}, indent=1, default=str))
    print(text)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
