import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from zipfile import ZipFile

WORK = Path(__file__).resolve().parent
PROJECT = WORK.parent / "T_Junction_RL"
OUT = WORK.parents[1] / "outputs"
STAGES = {
    "smoke": (4, 3, 180, "sumo"),
    "quick": (24, 24, 600, "sumo"),
    "full": (6, 300, 900, "libsumo"),
}
PREFIX = "T_Junction_RL/outputs/v3_research/"
def rows(data):
    return csv.DictReader(io.StringIO(data.decode("utf-8")))

def audit(package, prefix):
    names = set(package.namelist())
    by_stage = {}
    for stage, (runs, episodes, horizon, backend) in STAGES.items():
        folder = prefix + stage + "/"
        statuses = [n for n in names if n.startswith(folder) and n.endswith("/training_status.json")]
        assert len(statuses) == runs
        stage_updates = 0
        for n in sorted(statuses):
            root = n.rsplit("/", 1)[0] + "/"
            status = json.loads(package.read(n))
            assert status["complete"] and status["completed_episodes"] == episodes
            training = list(rows(package.read(root+"logs/training_summary.csv")))
            assert len(training) == episodes
            assert [int(r["episode"]) for r in training] == list(range(episodes))
            assert [int(r["global_step"]) for r in training] == list(range(horizon//5, episodes*horizon//5+1, horizon//5))
            updates = list(rows(package.read(root+"logs/updates.csv")))
            assert status["updates"] == len(updates)
            assert all(int(row["update"]) == i and math.isfinite(float(row["loss"])) for i, row in enumerate(updates, 1))
            stage_updates += len(updates)
            for row in training:
                episode, seed = int(row["episode"]), int(row["seed"])
                raw = package.read(root+f"training_episodes/ep{episode:04d}/tripinfo.xml")
                header = raw.split(b"-->", 1)[0]
                assert f"by Eclipse SUMO {backend} 1.28.0".encode() in header
                assert f'<seed value="{seed}"'.encode() in header
                assert b"</tripinfos>" in raw
        by_stage[stage] = {"runs": runs, "training_episodes": runs*episodes,
                           "training_sumo_seconds": runs*episodes*horizon, "optimizer_updates": stage_updates}
    return by_stage

with ZipFile(OUT/"T_Junction_RL_Project_Validated_Final.zip") as archive:
    source = [n for n in archive.namelist() if re.match(r"T_Junction_RL/(src|scripts|maps|configs)/", n)
              and Path(n).suffix in {".py",".xml",".yaml"}]
    assert all((PROJECT / n.split("/",1)[1]).read_bytes() == archive.read(n) for n in source)
    delivery = audit(archive, PREFIX)
    print("Delivery training run and source files matched.", flush=True)
with ZipFile(OUT/"T_Junction_RL_Reproduced_Results.zip") as archive:
    reproduction = audit(archive, "")
    print("Reproduction training run matched.", flush=True)
assert delivery == reproduction
summary = {"source_files_identical_to_delivery_for_live_replay":len(source),"delivery":delivery,"reproduction":reproduction}
(WORK/"training_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
