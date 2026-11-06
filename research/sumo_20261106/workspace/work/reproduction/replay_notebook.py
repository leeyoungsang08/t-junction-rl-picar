import os
import json
import subprocess
import sys
from pathlib import Path

workspace = Path(__file__).absolute().parent
root = workspace / "T_Junction_RL"
notebook = root / "notebooks" / "07_paper_v3_experiments.ipynb"
cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
output = workspace.parent.parent / "outputs" / "T_Junction_RL_Reproduced_Results"
if not (root / "requirements.txt").is_file():
    raise FileNotFoundError("Extract the delivery ZIP first.")
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  PYTHONUTF8="1", PYTHONUNBUFFERED="1")
os.environ["PYTEST_ADDOPTS"] = "--basetemp=R:/reproduction_pytest"
namespace = dict(__name__="__main__", root=root, subprocess=subprocess, sys=sys, os=os,
                 display=print, shutil=__import__("shutil"))
for index in (3, 5, 7, 9, 11, 13):
    cell = "".join(cells[index]["source"])
    if index == 3:
        shell_setup = 'subprocess.run(["bash", "scripts/setup_colab.sh"], cwd=root, check=True)'
        assert cell.count(shell_setup) == 1
        cell = cell.replace(shell_setup, 'subprocess.run([sys.executable, "-m", "pip", "check"], cwd=root, check=True)')
    elif index == 5:
        for flag in ("RUN_QUICK_TRAIN", "RUN_FULL_TRAIN", "RUN_FINAL_TEST"):
            assert cell.count(f"{flag} = False") == 1
            cell = cell.replace(f"{flag} = False", f"{flag} = True")
        cell = cell.replace("JOBS = 1", "JOBS = 8").replace("FULL_GROUPS = []", 'FULL_GROUPS = ["cycle_rainbow_lite", "direct_rainbow_lite"]')
    elif index == 13:
        namespace["local_archive"] = workspace / output.name
        cell = cell.replace('"/content/T_Junction_RL_Colab_Results"', "str(local_archive)").replace("files.download(archive)", "print(archive)")
    print(f"=== Notebook cell {index} started ===", flush=True)
    exec(compile(cell, f"{notebook.name} cell {index}", "exec"), namespace)
    print(f"=== Notebook cell {index} completed ===", flush=True)
