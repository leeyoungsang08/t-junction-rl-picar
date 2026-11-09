"""Plot validation score and episode delay per training seed.

python -m picar_sim.plot_training --runs runs/picar_v3_cycle --baselines queue_rule=17.36 fixed_time=19.6
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--baselines", nargs="*", default=[])
    p.add_argument("--out")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for log in sorted(Path(a.runs).glob("seed*/training_log.csv")):
        df = pd.read_csv(log)
        v = df[df.val_score.notna() & (df.val_score.astype(str) != "")]
        ax1.plot(v.episode, v.val_score.astype(float), label=log.parent.name)
        ax2.plot(df.episode, df.mean_delay_s.rolling(50, min_periods=1).mean(), label=log.parent.name)
    for b in a.baselines:
        name, val = b.split("=")
        ax1.axhline(float(val), ls="--", lw=1, color="k" if "queue" in name else "0.5", label=f"{name} (validation)")
    ax1.set(title="validation score (lower is better)", xlabel="episode")
    ax2.set(title="training episode mean delay [s], 50-episode mean", xlabel="episode")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out or Path(a.runs) / "training_curves.png", dpi=120)


if __name__ == "__main__":
    main()
