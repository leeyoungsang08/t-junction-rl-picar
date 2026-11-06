import csv
from pathlib import Path
p=Path("R:/reproduction/T_Junction_RL/outputs/v3_research")
fields=("episode","scenario","seed")
for d in sorted((p/"colab/full").glob("*_seed*")):
 new=d/"logs/validation.csv"
 old=p/"full"/d.name/"logs/validation.csv"
 if not new.exists(): continue
 with old.open(newline="") as f:
    prior={tuple(r[k] for k in fields):r for r in csv.DictReader(f)}
 with new.open(newline="") as f:
    rows=list(csv.DictReader(f))
 diffs=sum(any(v!=prior[tuple(r[k] for k in fields)][k] for k,v in r.items() if k!="wall_seconds") for r in rows)
 print(d.name, len(rows), diffs)

