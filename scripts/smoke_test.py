import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ROOT
from scripts.check_environment import main
from maps.generate_map import generate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--output", default="outputs/smoke_new")
    a = p.parse_args()
    main()
    generate()
    subprocess.check_call([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if a.train:
        from src.train import train
        train("smoke", a.output)
