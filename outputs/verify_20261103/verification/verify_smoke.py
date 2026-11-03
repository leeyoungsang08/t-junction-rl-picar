from pathlib import Path
import ast
import json
import platform
import sys
import nbformat
import numpy as np
import torch

from src.dqn_agent import DQNAgent
from src.evaluate import evaluate
from src.inference import Policy
from src.state_adapter import StateAdapter
from src.utils import load_yaml

run = Path("outputs/verify_20261103")
smoke = run / "smoke"
path = smoke / "checkpoints/best_model.pt"
initial = DQNAgent({**load_yaml("training.yaml"), "state_dim": 25})
policy = Policy(path)
assert any(not torch.equal(p, q) for p, q in zip(initial.online.parameters(), policy.model.parameters()))
checkpoint = torch.load(smoke / "checkpoints/training_checkpoint.pt", weights_only=False, map_location="cpu")
agent = DQNAgent(checkpoint["config"])
agent.restore(smoke / "checkpoints/training_checkpoint.pt")
assert checkpoint["next_episode"] == 3 and agent.global_step == 108 and agent.updates == 93
assert all(torch.equal(t, checkpoint["online"][k]) for k, t in agent.online.state_dict().items())
assert all(torch.equal(t, checkpoint["target"][k]) for k, t in agent.target.state_dict().items())
assert agent.replay.size == 108 and agent.optimizer.state
print("Restored online/target/optimizer/replay/RNG after episode 3. 108 decisions, 93 updates; exported weights differ from initialization.")
cfg = load_yaml("base.yaml")
for model in [Path("outputs/reference/legacy_v1.pt"), path]:
    policy = Policy(model)
    schema = policy.adapter.schema
    layers = policy.model.layers
    print(model, "schema", schema, "input", layers[0].in_features, "device", policy.device, "metadata", policy.payload["metadata"])
    assert layers[0].in_features == len(StateAdapter(cfg["normalization"], cfg["signal"], schema).features) == (12 if schema == 1 else 25)

out = run / "resume_evaluation"
options = dict(seeds=1, horizon=120, scenarios=["medium"])
first = evaluate(path, out, **options)
log = out / "episodes/medium_3000_dqn/decisions.csv"
fixed = out / "episodes/medium_3000_fixed_time/decisions.csv"
mtime = log.stat().st_mtime_ns
fixed_mtime = fixed.stat().st_mtime_ns
second = evaluate(path, out, **options, resume=True)
assert len(second) == 3 and log.stat().st_mtime_ns == mtime
log.unlink()
third = evaluate(path, out, **options, resume=True)
assert log.is_file() and fixed.stat().st_mtime_ns == fixed_mtime
for frame in [first, second, third]:
    assert len(frame) == 3
    assert np.allclose(frame.sort_values("controller").cohort_delay_s,
                       first.sort_values("controller").cohort_delay_s)
print("Evaluation resume skipped existing episodes, repaired one missing log, reproduced cohort delay.")
for path in sorted(Path("notebooks").glob("*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    from IPython.core.inputtransformer2 import TransformerManager
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    for index, source in enumerate(code):
        compile(TransformerManager().transform_cell(source), f"{path}#{index}", "exec")
    print("Notebook", path, "schema and code compilation passed", len(code), "code cells")
print(platform.platform(), sys.version)
