from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
from run_transfer import HERE, PROJECT, FULL, MODELS, OUT, sha256

root = HERE.parents[1]
deliverables = root / "outputs"
archive = deliverables / "T_Junction_Transfer_Evidence.zip"
plan = json.loads((HERE/"plan.json").read_text())
assert len(list(OUT.rglob("tripinfo.xml"))) == 432
assert len(list(OUT.rglob("timeseries.csv"))) == 432
assert len(list(OUT.rglob("decisions.csv"))) == 432
readme = """Transfer test reproduction (Python 3.11 on Windows was used)
Create a venv. Install T_Junction_RL/requirements.txt and libsumo==1.28.0.
From this extracted folder run:
  python transfer/run_transfer.py
  python transfer/analyze_transfer.py
This regenerates map and demand and overwrites the archived run files. Use a copy if you want to preserve evidence.
The supplied plan records effective horizon_seconds separately from the inherited source environment.
The replay can be opened directly in a modern browser. It uses a fixed example, not a selection based on outcomes.
"""
with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=5) as z:
    z.writestr("README.txt", readme)
    for p in (HERE/"maps", HERE/"demand", OUT):
        for f in sorted(p.rglob("*")):
            if f.is_file():
                z.write(f, Path("transfer")/f.relative_to(HERE))
    for name in ("run_transfer.py", "analyze_transfer.py", "plan.json", "episode_metrics.csv",
                 "scenario_summary.csv", "paired.csv", "demand_blocks.csv", "signal_blocks.csv",
                 "block_summary.csv", "signal_audit.csv", "run.log", "analysis.log"):
        z.write(HERE/name, Path("transfer")/name)
    for f in sorted((PROJECT/"src").glob("*.py")):
        z.write(f, Path("T_Junction_RL")/f.relative_to(PROJECT))
    for f in (PROJECT/"maps/t_junction.net.xml", PROJECT/"requirements.txt", FULL/"final_eval/evaluation_plan.json"):
        z.write(f, Path("T_Junction_RL")/f.relative_to(PROJECT))
    for c in MODELS:
        f = FULL/c/"checkpoints/best_model.pt"
        assert plan["model_hashes"][c] == sha256(f)
        z.write(f, Path("T_Junction_RL")/f.relative_to(PROJECT))
    for name in ("T_Junction_Transfer_Report_KR.md", "T_Junction_Transfer_Signal_Replay.html"):
        z.write(deliverables/name, name)
with ZipFile(archive) as z:
    assert z.testzip() is None
    print(len(z.namelist()), "verified entries,", archive.stat().st_size, "bytes")
