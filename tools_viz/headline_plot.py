import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

repo = sys.argv[1]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SER = {"picar_v18_mix_seg_norm": ("v18 (혼합 학습 + 보상 정규화)", "#2a78d6", "o"),
       "picar_v15_obsage_tail": ("v15 (실제 코스만 학습)", "#eb6834", "s")}
ko = [f.fname for f in font_manager.fontManager.ttflist if any(k in f.name for k in ("Noto Sans CJK", "NanumGothic", "Noto Sans KR"))]
if ko:
    font_manager.fontManager.addfont(ko[0]); plt.rcParams["font.family"] = font_manager.FontProperties(fname=ko[0]).get_name()
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.unicode_minus": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF)
titles = {"real": "실제 PiCar 코스 (3.46 × 2.30 m)", "large": "큰 맵 (도로 2.5배, 시뮬레이션 전용)"}
for ax, mp in zip(axes, ("real", "large")):
    d = f"{repo}/runs/final_v18_{mp}"
    p = pd.read_csv(f"{d}/paired_vs_best_rule.csv")
    s = pd.read_csv(f"{d}/summary_by_cars.csv")
    ax.set_facecolor(SURF)
    ax.axhspan(-100, 0, color="#eef4fb", zorder=0)
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    ends = {}
    for fam, (lab, col, mk) in SER.items():
        g = p[p.family == fam].sort_values("cars")
        base = [float(s[(s.cars == c) & (s.family == v)].mean_delay_s.iloc[0]) for c, v in zip(g.cars, g.vs)]
        y = 100 * g.mean_delay_s_diff.values / base
        lo = 100 * g.mean_delay_s_ci_lo.values / base
        hi = 100 * g.mean_delay_s_ci_hi.values / base
        ax.fill_between(g.cars, lo, hi, color=col, alpha=.15, lw=0, zorder=2)
        ax.plot(g.cars, y, color=col, lw=2, marker=mk, ms=8, mec=SURF, mew=2, zorder=3, label=lab)
        ends[fam] = (g.cars.values[-1], y[-1], lab.split(" ")[0])
    (x1, y1, l1), (x2, y2, l2) = ends.values()
    nudge = 7 if abs(y1 - y2) < 4 else 0
    for (x, y_, l), dy in (((x1, y1, l1), nudge if y1 >= y2 else -nudge), ((x2, y2, l2), nudge if y2 > y1 else -nudge)):
        ax.annotate(l, (x, y_), xytext=(9, dy), textcoords="offset points", va="center", fontsize=10, color=INK)
    if mp == "real":
        ax.axvline(6, color=INK2, ls=":", lw=1)
        ax.text(6.1, -33, "실차 6대", fontsize=9, color=INK2)
    ax.set_ylim(-35, 42)
    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + (1.2 if mp == "real" else 5))
    ax.text(0.98, 0.04, "0 아래 = DQN이 가장 좋은 규칙보다 빠름", transform=ax.transAxes, fontsize=9, color=INK2, ha="right")
    ax.set_title(titles[mp], fontsize=12, color=INK, loc="left")
    ax.set_xlabel("코스 위 차량 수")
    ax.grid(axis="y", color=GRID, lw=.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("평균 지연 변화 (가장 좋은 규칙 대비, %)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.995, 0.995))
fig.suptitle("DQN vs 가장 좋은 규칙: 평균 지연  (음영 = 95% 신뢰구간 · 새 테스트 seed · 충돌·위반 0)", fontsize=12, x=0.01, y=0.985, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{repo}/runs/final_v18_headline.png", dpi=150, facecolor=SURF)
