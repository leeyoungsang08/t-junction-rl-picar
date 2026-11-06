from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
here=Path(__file__).resolve().parent
workspace=here.parents[1]
fair=workspace/"work/fairness"
project=fair/"T_Junction_RL"
study=fair/"study"
target=workspace/"outputs/T_Junction_RL_Reassessment_Evidence.zip"
assert len(list((here/"runs").rglob("tripinfo.xml")))==288
assert "7 passed" in (here/"tests.log").read_text()
readme="""Reassessment evidence - Python 3.11, SUMO/libsumo 1.28.0
Install dependencies from fairness/T_Junction_RL/requirements.txt plus libsumo==1.28.0.
From the extracted root run:
 python reassessment/run_comparison.py
 python reassessment/analyze.py
Output files in reassessment/runs will be regenerated. Use a copy to retain evidence.
The observation gate is an advisory prototype, not a traffic or vehicle controller.
"""
with ZipFile(target,"w",ZIP_DEFLATED,compresslevel=5) as z:
 z.writestr("README.txt",readme)
 for folder in ("src","configs","maps","tests"):
  for f in (project/folder).rglob("*"):
   if f.is_file() and "__pycache__" not in f.parts:
    z.write(f,Path("fairness/T_Junction_RL")/f.relative_to(project))
 for f in (project/"requirements.txt",project/"pyproject.toml"):
  z.write(f,Path("fairness/T_Junction_RL")/f.name)
 for f in (project/"outputs").rglob("*"):
  if f.is_file(): z.write(f,Path("fairness/T_Junction_RL/outputs")/f.relative_to(project/"outputs"))
 for folder in ("maps","demand"):
  for f in (study/folder).rglob("*"):
   if f.is_file() and "__pycache__" not in f.parts: z.write(f,Path("fairness/study")/f.relative_to(study))
 for name in ("run_guard.py","transfer_reference.py","holdout.csv","selection.json","plan_holdout.json"):
  z.write(study/name,Path("fairness/study")/name)
 for folder in ("src","tests"):
  for name in (("observation_gate.py",) if folder=="src" else ("test_observation_gate.py",)):
   z.write(here/"reviewed_project"/folder/name,Path("observation_gate")/folder/name)
 for f in (here/"runs").rglob("*"):
  if f.is_file(): z.write(f,Path("reassessment/runs")/f.relative_to(here/"runs"))
 for name in ("run_comparison.py","analyze.py","comparison.csv","summary.csv","paired.csv","analysis.log","run.log","tests.log","inference_bench.json"):
  z.write(here/name,Path("reassessment")/name)
 for seed in (42,43,44):
  base=workspace/f"work/T_Junction_RL/outputs/v3_research/full/direct_rainbow_lite_seed{seed}"
  for name in ("training.csv","validation.csv","updates.csv"):
   z.write(base/"logs"/name,Path("training_logs")/f"seed{seed}"/name)
  for name in ("training_status.json","experiment_config.yaml"):
   z.write(base/name,Path("training_logs")/f"seed{seed}"/name)
 z.write(workspace/"outputs/T_Junction_RL_Reassessment_KR.md","T_Junction_RL_Reassessment_KR.md")
with ZipFile(target) as z:
 assert z.testzip() is None
 print(len(z.namelist()),"entries verified",target.stat().st_size,"bytes")
