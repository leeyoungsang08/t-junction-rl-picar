"""One command: build map -> train N seeds (parallel processes) -> paired evaluation.

python -m picar_sim.run_all --out runs/my_run --episodes 1500 --seeds 42 43 44 --parallel 3
Defaults follow picar_sim/config/course.yaml: direct phase selection, max-wait guard 20 s, 6-car validation.
Add --backend torch on Colab / a GPU PC (a 22-input MLP is small: CPU is often as fast as GPU).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from picar_sim.map_builder import build, load_config, plot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/picar_run")
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--backend", default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--control-mode", choices=["cycle", "direct"], help="default: config (direct)")
    p.add_argument("--val-seeds-per", type=int, default=4)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-seeds", type=int, default=10)
    a = p.parse_args()
    cfg = load_config()
    net, _ = build(cfg)
    plot(net, Path(net).with_name("picar_course_topview.png"), cfg)
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    pending, running = list(a.seeds), []
    while pending or running:
        while pending and len(running) < a.parallel:
            s = pending.pop(0)
            cmd = [sys.executable, "-m", "picar_sim.train", "--seed", str(s), "--episodes", str(a.episodes),
                   "--out", str(out), "--backend", a.backend, "--val-seeds-per", str(a.val_seeds_per)]
            cmd += ["--control-mode", a.control_mode] if a.control_mode else []
            cmd += ["--randomize"] if a.randomize else []
            cmd += ["--resume"] if a.resume else []
            log = open(out / "logs" / f"seed{s}.log", "w")
            running.append((s, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
            print("started seed", s, flush=True)
        for item in list(running):
            if item[1].poll() is not None:
                item[2].close()
                running.remove(item)
                print(f"seed {item[0]} finished with code {item[1].returncode}", flush=True)
        if running:
            running[0][1].wait(timeout=None) if not pending and len(running) == 1 else __import__("time").sleep(5)
    subprocess.run([sys.executable, "-m", "picar_sim.evaluate", "--runs", str(out), "--out", str(out / "evaluation"),
                    "--seeds-per", str(a.eval_seeds)], check=True, env=env)


if __name__ == "__main__":
    main()
