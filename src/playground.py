from pathlib import Path
import matplotlib.pyplot as plt
from src.baselines import baseline_action
from src.environment import TJunctionEnv
from src.inference import Policy
from src.traffic_generator import generate_traffic
from src.utils import load_yaml, DIRECTIONS
from src.visualization import draw_map, ORIGIN_COLORS, render_trajectories


class Playground:
    def __init__(self, model, scenario='high', seed=3050, horizon=900, controller='dqn', output='outputs/playground'):
        self.policy = Policy(model)
        self.controller = controller
        self.output = Path(output)
        cfg = load_yaml('base.yaml')
        cfg['signal'] = self.policy.payload['signal']
        cfg['simulation']['horizon_seconds'] = horizon
        if controller == 'fixed_time':
            cfg['simulation']['decision_seconds'] = 1
        route = generate_traffic(scenario, seed, horizon, 'test', self.output / 'routes')
        self.env = TJunctionEnv(cfg, route, seed, self.output, capture=True)
        self.state, self.info = self.env.reset(seed=seed)
        self.done = False
        self.last_prediction = None

    def advance(self, seconds=5):
        until = min(self.env.time + seconds, self.env.horizon)
        while self.env.time < until and not self.done:
            if self.controller == 'dqn':
                self.last_prediction = self.policy.predict(self.info['raw_state'])
                action = self.last_prediction['action_id']
            else:
                action = baseline_action(self.controller, self.info, self.env.config)
            self.state, reward, terminated, truncated, self.info = self.env.step(action)
            self.done = terminated or truncated
            if self.last_prediction and self.controller == 'dqn':
                self.env.decisions[-1].update(q_keep=self.last_prediction['q_values'][0], q_switch=self.last_prediction['q_values'][1])
            if self.done:
                self.env.save_logs()
        return self.info

    def figure(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        draw_map(ax)
        if self.env.frames:
            f = self.env.frames[-1]
            if f['vehicles']:
                ax.scatter([v['x'] for v in f['vehicles']], [v['y'] for v in f['vehicles']],
                           c=[ORIGIN_COLORS[v['origin']] for v in f['vehicles']], s=20, marker='s', zorder=5)
        raw = self.info['raw_state']
        phase = DIRECTIONS[raw['phase']] if not raw['transitioning'] else self.env.controller.stage
        ax.set_title(f'{self.controller} | t={self.env.time}s | signal={phase} | queue={sum(self.info["queues"])} | completed={len(self.env.arrived)}')
        fig.tight_layout()
        return fig

    def video(self):
        if not self.done:
            raise RuntimeError('Finish the episode before encoding a video')
        return render_trajectories([self.output / 'trajectory.jsonl'], [self.controller], self.output / 'playback.mp4', seconds=self.env.horizon)

    def close(self):
        self.env.close()
