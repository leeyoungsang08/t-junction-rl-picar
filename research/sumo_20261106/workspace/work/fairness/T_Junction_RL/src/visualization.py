import argparse
import json
from pathlib import Path
import shutil
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import sumolib
from src.evaluate import run_episode
from src.inference import Policy
from src.utils import ROOT, DIRECTIONS, atomic_text, load_yaml

COLORS = {"fixed_time": "#667085", "rule_based": "#A66F3F", "dqn": "#34785B"}
ORIGIN_COLORS = {"W": "#4075A8", "E": "#B97343", "S": "#6C5B8F"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})


def network():
    return sumolib.net.readNet(str(ROOT / "maps/t_junction.net.xml"), withInternal=True)


def draw_map(ax, net=None, active=None, labels=True):
    net = net or network()
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            shape = np.array(lane.getShape())
            internal = edge.getFunction() == "internal"
            color = "#DBDED9" if internal else "#969D9C"
            ax.plot(shape[:, 0], shape[:, 1], color=color, linewidth=5 if not internal else 2, solid_capstyle="butt", zorder=1)
            if active and edge.getID() == f"{active}_in":
                ax.plot(shape[:, 0], shape[:, 1], color="#3B8763", linewidth=3, zorder=2)
    if labels:
        for d in DIRECTIONS:
            point = np.array(net.getNode(d).getCoord())
            ax.text(point[0], point[1] + (16 if d != "S" else -15), {"W": "West", "E": "East", "S": "South"}[d], ha="center", va="center")
            shape = np.array(net.getLane(f"{d}_in_0").getShape())
            at = int(len(shape) // 2)
            mid = (shape[0] + shape[-1]) / 2
            vec = (shape[-1] - shape[0]) / np.linalg.norm(shape[-1] - shape[0]) * 35
            ax.annotate("", xy=mid + vec / 2, xytext=mid - vec / 2,
                        arrowprops={"arrowstyle": "->", "color": "#27352D", "lw": 1.8})
    xs = [net.getNode(d).getCoord()[0] for d in DIRECTIONS]
    ys = [net.getNode(d).getCoord()[1] for d in DIRECTIONS]
    ax.set_xlim(min(xs) - 35, max(xs) + 35)
    ax.set_ylim(min(ys) - 35, max(ys) + 45)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def map_figures(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    net = network()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_map(ax, net)
    ax.set_title("T-junction | 3 incoming + 3 outgoing lanes | right-hand traffic")
    fig.tight_layout()
    fig.savefig(directory / "t_junction_map.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_map(ax, net, labels=False)
    for edge in net.getEdges(withInternal=True):
        if edge.getFunction() == "internal":
            for lane in edge.getLanes():
                shape = np.array(lane.getShape())
                ax.plot(shape[:, 0], shape[:, 1], color="#34785B", lw=2)
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 20)
    for x, y, label in [(-29, -8, "West"), (29, -8, "East"), (10, -29, "South")]:
        ax.text(x, y, label, ha="center", fontsize=10)
    ax.set_title("Six permitted OD connections | no U-turns")
    fig.tight_layout()
    fig.savefig(directory / "lane_connections.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, d in zip(axes, DIRECTIONS):
        draw_map(ax, net, active=d)
        ax.set_title(f"Phase {d}: one approach green")
    fig.suptitle("Cyclic order W > E > S | yellow and all-red between greens")
    fig.tight_layout()
    fig.savefig(directory / "signal_phases.png")
    plt.close(fig)


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_results(training_dir, evaluation_dir):
    training_dir, evaluation_dir = Path(training_dir), Path(evaluation_dir)
    plots = evaluation_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    map_figures(plots)
    train = pd.read_csv(training_dir / "logs/training.csv")
    updates = pd.read_csv(training_dir / "logs/updates.csv")
    validation = pd.read_csv(training_dir / "logs/validation.csv")
    results = pd.read_csv(evaluation_dir / "seed_level_results.csv")
    for column, ylabel, filename in [("episode_reward", "Episode reward", "episode_reward.png"),
                                      ("epsilon", "Exploration probability", "epsilon_decay.png")]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train.episode + 1, train[column], color="#B4BBB7", lw=1, label="Measured")
        if column == "episode_reward":
            ax.plot(train.episode + 1, train[column].rolling(10, min_periods=1).mean(), color="#34785B", lw=2, label="10-episode moving average")
        ax.set(xlabel="Training episode", ylabel=ylabel, title=ylabel)
        ax.legend()
        save_plot(fig, plots / filename)
    if len(updates):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(updates.global_step, updates.loss, alpha=0.25, color="#667085", lw=0.5)
        ax.plot(updates.global_step, updates.loss.rolling(100, min_periods=1).mean(), color="#34785B", label="100-update average")
        ax.set(xlabel="Environment decisions", ylabel="Huber loss", title="DQN optimization loss")
        ax.legend()
        save_plot(fig, plots / "training_loss.png")
    qdata = []
    for file in sorted((training_dir / "training_episodes").glob("*/decisions.csv")):
        f = pd.read_csv(file)
        if "q_keep" in f:
            qdata.append(f[["q_keep", "q_switch"]])
        else:
            new_cols = [c for c in f.columns if c.startswith("q_action_")]
            if new_cols:
                qdata.append(f[new_cols])
    if qdata:
        qs = pd.concat(qdata)
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in qs:
            ax.hist(qs[c].dropna(), bins=60, density=True, alpha=0.5, label=c)
        ax.set(xlabel="Q value (actual training decisions)", ylabel="Density", title="Q-value distribution")
        ax.legend()
        save_plot(fig, plots / "q_value_distribution.png")
    for column, ylabel in [("cohort_delay_s", "Waiting + insertion delay per request (s)"),
                            ("mean_waiting_s", "Completed-vehicle waiting time (s)"), ("mean_queue_veh", "Queue (vehicles)")]:
        grouped = validation.groupby("episode")[column].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(grouped.index + 1, grouped["mean"], yerr=grouped["std"].fillna(0), marker="o", color="#34785B", capsize=3)
        ax.set(xlabel="Training episode", ylabel=ylabel, title="Validation mean +/- seed/scenario SD")
        save_plot(fig, plots / f"validation_{column}.png")
    names = list(dict.fromkeys(results.scenario))
    controllers = list(dict.fromkeys(results.controller))
    metric_labels = {
        "mean_waiting_s": "Completed-vehicle waiting (s)", "mean_travel_s": "Completed-vehicle travel time (s)",
        "mean_time_loss_s": "Completed-vehicle time loss (s)", "mean_queue_veh": "Mean incoming queue (vehicles)",
        "completed_vehicles": "Completed vehicles", "throughput_veh_h": "Throughput (vehicles/hour)",
        "max_queue_veh": "Maximum total queue (vehicles)", "signal_switches": "Signal switches",
        "cohort_delay_s": "Cohort waiting + insertion delay (s/request)", "completion_rate": "Completion fraction"}
    for metric, label in metric_labels.items():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, controller in enumerate(controllers):
            sub = results[results.controller == controller].groupby("scenario")[metric].agg(["mean", "std"]).reindex(names)
            ax.bar(np.arange(len(names)) + (i - (len(controllers) - 1) / 2) * .25, sub["mean"], .24,
                   yerr=sub["std"].fillna(0), capsize=2, color=COLORS.get(controller), label=controller)
        ax.set_xticks(range(len(names)), names, rotation=15)
        ax.set(ylabel=label, title=f"{label} | mean +/- SD across paired seeds")
        ax.legend(ncol=len(controllers), fontsize=9)
        save_plot(fig, plots / f"comparison_{metric}.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, controller in enumerate(controllers):
        means = [results[results.controller == controller][f"mean_waiting_{d}_s"].mean() for d in DIRECTIONS]
        ax.bar(np.arange(3) + (i - 1) * .25, means, .24, label=controller, color=COLORS.get(controller))
    ax.set_xticks(range(3), DIRECTIONS)
    ax.set(ylabel="Completed-vehicle waiting (s)", title="Direction-level mean (equal seed/scenario weights)")
    ax.legend()
    save_plot(fig, plots / "direction_waiting.png")
    paired = pd.read_csv(evaluation_dir / "scenario_comparison.csv")
    improvements = paired[(paired.controller == "dqn") & (paired.metric == "mean_waiting_s")].set_index("scenario").reindex(names)
    fig, ax = plt.subplots(figsize=(8, 4))
    values = improvements.improvement_pct
    ax.bar(names, values, color=["#34785B" if v >= 0 else "#B45950" for v in values])
    ax.axhline(0, color="#333333", lw=1)
    ax.set(ylabel="Improvement over fixed time (%)", title="DQN waiting-time change | positive = lower waiting")
    ax.tick_params(axis="x", rotation=15)
    save_plot(fig, plots / "scenario_improvement.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    draw_map(axes[0, 0])
    axes[0, 0].set_title("Actual SUMO network")
    axes[0, 1].plot(train.episode + 1, train.episode_reward, color="#B4BBB7", lw=.5)
    axes[0, 1].plot(train.episode + 1, train.episode_reward.rolling(10, min_periods=1).mean(), color="#34785B")
    axes[0, 1].set(title="DQN training reward", xlabel="Episode", ylabel="Reward")
    g = results.groupby("controller").mean_waiting_s.mean().reindex(controllers)
    axes[1, 0].bar(controllers, g, color=[COLORS.get(c) for c in controllers])
    axes[1, 0].set(title="Waiting time (equal scenario/seed weights)", ylabel="Completed-vehicle waiting (s)")
    axes[1, 1].bar(names, values, color=["#34785B" if v >= 0 else "#B45950" for v in values])
    axes[1, 1].axhline(0, color="#333333", lw=1)
    axes[1, 1].set(title="DQN vs fixed time", ylabel="Waiting improvement (%)")
    axes[1, 1].tick_params(axis="x", rotation=20)
    fig.suptitle(f"T-junction signal control | {int(results.horizon_seconds.iloc[0])} s/run | {results.seed.nunique()} paired seeds/scenario", fontsize=14)
    fig.text(.5, .012, "Completed-trip averages exclude unfinished demand. Also inspect cohort delay and completion rate.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(plots / "summary_figure.png", dpi=240)
    fig.savefig(plots / "summary_figure.pdf")
    plt.close(fig)
    time_analysis(results, plots)
    return plots


def time_analysis(results, plots):
    chosen = results[(results.scenario == "high") & (results.seed == results.seed.min())]
    logs = {r.controller: pd.read_csv(ROOT / r.episode_dir / "timeseries.csv") for r in chosen.itertuples()}
    for field, ylabel in [("queue_total", "Total incoming queue (vehicles)"), ("mean_current_wait_s", "Mean consecutive wait of current vehicles (s)"),
                          ("cumulative_arrived", "Cumulative completed vehicles"), ("cumulative_departed", "Cumulative inserted vehicles")]:
        fig, ax = plt.subplots(figsize=(10, 4))
        for controller, data in logs.items():
            ax.plot(data.time, data[field], color=COLORS.get(controller), label=controller)
        ax.set(xlabel="Simulation time (s)", ylabel=ylabel, title="High traffic | same route file and seed")
        ax.legend()
        save_plot(fig, plots / f"time_{field}.png")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, d in zip(axes, DIRECTIONS):
        for controller, data in logs.items():
            ax.plot(data.time, data[f"queue_{d}"], label=controller, color=COLORS.get(controller))
        ax.set_ylabel(f"{d} queue (veh)")
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Simulation time (s)")
    save_plot(fig, plots / "time_direction_queues.png")
    fig, axes = plt.subplots(len(logs), 1, figsize=(10, 4), sharex=True, squeeze=False)
    for ax, (controller, data) in zip(axes[:, 0], logs.items()):
        vals = np.where(data.stage == "green", data.phase, 3)
        im = ax.imshow(vals[None, :], extent=[0, data.time.max(), 0, 1], aspect="auto", vmin=0, vmax=3, cmap=plt.get_cmap("Set2", 4))
        ax.set_ylabel(controller)
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("Time (s); codes W=0, E=1, S=2, clearance=3")
    save_plot(fig, plots / "signal_timeline.png")
    if len(chosen):
        route = ROOT / chosen.iloc[0].episode_dir
        manifest_file = route.parents[1] / "experiment_manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text())
            entry = next(x for x in manifest["routes"] if x["scenario"] == "high" and x["seed"] == int(chosen.iloc[0].seed))
            demand = json.loads((ROOT / entry["route"]).with_suffix(".json").read_text())
            times = [v["requested_depart"] for v in demand["vehicles"].values()]
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(times, bins=np.arange(0, demand["horizon_seconds"] + 60, 60), color="#667085")
            ax.set(xlabel="Simulation time (s)", ylabel="Scheduled vehicles / 60 s", title="Actual generated demand | high traffic")
            save_plot(fig, plots / "time_demand.png")


def render_trajectories(paths, labels, output, seconds=600, stride=3, fps=10, _frames_only=False):
    traces = [[json.loads(line) for line in Path(path).read_text().splitlines()] for path in paths]
    n = min(min(map(len, traces)), seconds)
    if n <= 0:
        raise ValueError("Empty SUMO trajectory")
    for trace in traces[1:]:
        if [f["time"] for f in trace[:n]] != [f["time"] for f in traces[0][:n]]:
            raise ValueError("Comparison video requires an identical time axis")
    fig, axes = plt.subplots(1, len(paths), figsize=(7 * len(paths), 4.8), squeeze=False)
    scatters, texts, lamps = [], [], []
    net = network()
    for ax, label in zip(axes[0], labels):
        draw_map(ax, net)
        ax.set_title(label)
        scatters.append(ax.scatter([], [], s=20, zorder=5, marker="s", edgecolors="white", linewidths=.3))
        texts.append(ax.text(.02, .03, "", transform=ax.transAxes, fontsize=9, bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"}))
        current_lamps = []
        for d in DIRECTIONS:
            pos = net.getLane(f"{d}_in_0").getShape()[-1]
            current_lamps.append(ax.scatter([pos[0]], [pos[1]], s=45, c="red", zorder=7))
        lamps.append(current_lamps)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = shutil.which("ffmpeg")
    if not encoder:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            encoder = None
    process = None
    fallback = output.with_suffix("")
    if encoder and not _frames_only:
        process = subprocess.Popen([encoder, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
                                    "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps), "-i", "-", "-an",
                                    "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE)
    else:
        fallback.mkdir(parents=True, exist_ok=True)
    failed = False
    try:
        for i in range(0, n, stride):
            for j, trace in enumerate(traces):
                f = trace[i]
                vehicles = f["vehicles"]
                points = np.array([[v["x"], v["y"]] for v in vehicles]).reshape(-1, 2)
                scatters[j].set_offsets(points)
                scatters[j].set_color([ORIGIN_COLORS[v["origin"]] for v in vehicles])
                d = DIRECTIONS[f["phase"]]
                for k, lamp in enumerate(lamps[j]):
                    lamp.set_color("#3B8763" if k == f["phase"] and f["stage"] == "green" else "#D9AA35" if k == f["phase"] and f["stage"] == "yellow" else "#B45950")
                texts[j].set_text(f"t={f['time']:4d} s | green={d if f['stage']=='green' else f['stage']}\nqueue={sum(f['queues'])} vehicles | completed={f['arrived']}\ncurrent mean consecutive wait={f['mean_current_wait_s']:.1f} s")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            if process:
                process.stdin.write(rgb.tobytes())
            else:
                from PIL import Image
                Image.fromarray(rgb).save(fallback / f"frame_{i:06d}.png")
        if process:
            process.stdin.close()
            if process.wait() != 0:
                failed = True
    except (BrokenPipeError, OSError):
        if not process:
            raise
        failed = True
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait()
        plt.close(fig)
    if failed:
        output.unlink(missing_ok=True)
        print("FFmpeg encoding failed; saving actual trajectory PNG frames instead.")
        return render_trajectories(paths, labels, output, seconds, stride, fps, _frames_only=True)
    return output if process else fallback


def create_demo_videos(model_path, evaluation_dir, scenario="high", seed=3000, seconds=600):
    evaluation_dir = Path(evaluation_dir)
    manifest = json.loads((evaluation_dir / "experiment_manifest.json").read_text())
    entry = next(x for x in manifest["routes"] if x["scenario"] == scenario and x["seed"] == seed)
    route = ROOT / entry["route"]
    import yaml
    cfg = yaml.safe_load((evaluation_dir / "experiment_config.yaml").read_text())["environment"]
    policy = Policy(model_path)
    paths = {}
    for controller in ("fixed_time", "rule_based", "dqn"):
        directory = evaluation_dir / "video_logs" / f"{scenario}_{seed}_{controller}"
        metrics = run_episode(cfg, route, seed, controller, directory, policy=policy, capture=True)
        table = pd.read_csv(evaluation_dir / "seed_level_results.csv")
        recorded = table[(table.scenario == scenario) & (table.seed == seed) & (table.controller == controller)].iloc[0]
        if abs(metrics["cohort_delay_s"] - recorded.cohort_delay_s) > 1e-8:
            raise RuntimeError("Video rerun does not reproduce evaluation metric")
        path = directory / "trajectory.jsonl"
        paths[controller] = path
        render_trajectories([path], [controller], evaluation_dir / "videos" / f"{controller}_demo.mp4", seconds=seconds)
    render_trajectories([paths["fixed_time"], paths["dqn"]], ["Fixed time", "DQN"],
                        evaluation_dir / "videos/comparison_fixed_vs_dqn.mp4", seconds=seconds)
    atomic_text(evaluation_dir / "videos/README.md", f"Actual SUMO trajectories, scenario={scenario}, seed={seed}. First {seconds} simulation seconds, 3 s/frame, 10 fps (30x). The displayed waiting metric is the consecutive wait of CURRENT vehicles; it differs from final completed-trip waiting.\n")


def create_worst_case_video(model_path, evaluation_dir):
    import yaml
    evaluation_dir = Path(evaluation_dir)
    case = json.loads((evaluation_dir / "case_studies.json").read_text())["worst_relative"]
    manifest = json.loads((evaluation_dir / "experiment_manifest.json").read_text())
    entry = next(x for x in manifest["routes"] if x["scenario"] == case["scenario"] and x["seed"] == case["seed"])
    route = ROOT / entry["route"]
    cfg = yaml.safe_load((evaluation_dir / "experiment_config.yaml").read_text())["environment"]
    policy = Policy(model_path)
    expected = pd.read_csv(evaluation_dir / "seed_level_results.csv")
    paths = []
    for controller in ["fixed_time", "dqn"]:
        directory = evaluation_dir / "video_logs" / f"worst_{controller}"
        metric = run_episode(cfg, route, case["seed"], controller, directory, policy=policy, capture=True)
        row = expected[(expected.scenario == case["scenario"]) & (expected.seed == case["seed"]) & (expected.controller == controller)].iloc[0]
        if abs(metric["cohort_delay_s"] - row.cohort_delay_s) > 1e-8:
            raise RuntimeError("Worst-case video does not reproduce evaluation")
        paths.append(directory / "trajectory.jsonl")
    label = f"{case['scenario']} / seed {case['seed']}"
    output = render_trajectories(paths, ["Fixed time | " + label, "DQN | " + label],
                                evaluation_dir / "videos/worst_relative_case.mp4",
                                seconds=cfg["simulation"]["horizon_seconds"], stride=6, fps=10)
    note = evaluation_dir / "videos/README.md"
    prior = note.read_text() if note.exists() else ""
    atomic_text(note, prior + f"\nWorst relative case: {label}, complete horizon, 6 s/frame at 10 fps (60x). Chosen by largest DQN minus fixed cohort delay.\n")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--training", required=True)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--videos", action="store_true")
    a = p.parse_args()
    print(plot_results(a.training, a.evaluation))
    if a.videos:
        from src.export_results import generate_report
        generate_report(a.training, a.evaluation)
        create_demo_videos(Path(a.training) / "checkpoints/best_model.pt", a.evaluation)
        create_worst_case_video(Path(a.training) / "checkpoints/best_model.pt", a.evaluation)
