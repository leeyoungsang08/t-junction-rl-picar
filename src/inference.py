import argparse
from pathlib import Path
import json
import torch
import numpy as np
from src.dqn_network import QNetwork
from src.dqn_agent import atomic_torch_save
from src.state_adapter import StateAdapter
from src.utils import atomic_text, write_json
import yaml


def export_model(agent, base_config, path, metadata):
    path = Path(path)
    adapter = StateAdapter(base_config["normalization"], base_config["signal"], base_config.get("observation_schema", 1))
    if agent.online.layers[0].in_features != len(adapter.features):
        raise ValueError("Agent and observation schema differ")
    payload = {"state_dict": {k: v.detach().cpu() for k, v in agent.online.state_dict().items()},
               "hidden": agent.config["hidden"], "normalization": base_config["normalization"],
               "signal": base_config["signal"], "metadata": metadata, "schema_version": adapter.schema}
    atomic_torch_save(payload, path)
    write_json(path.with_suffix(".metadata.json"), metadata)
    write_json(path.parent / "normalization.json", payload["normalization"])
    write_json(path.parent / "observation_schema.json", {"shape": [len(adapter.features)], "dtype": "float32", "direction_order": ["W", "E", "S"],
               "features": adapter.features, "schema_version": adapter.schema,
               "scaling": "V1: divide then clip; V2: open-ended traffic features use x/(scale+x)"})
    write_json(path.parent / "action_schema.json", {"0": "KEEP", "1": "SWITCH", "phase_order": ["W", "E", "S"]})
    if path.stem == "best_model" or not (path.parent / "model_metadata.json").exists():
        write_json(path.parent / "model_metadata.json", metadata)
    atomic_text(path.parent / "model_config.yaml", yaml.safe_dump({"hidden": agent.config["hidden"],
                "normalization": payload["normalization"], "signal": payload["signal"]}, sort_keys=False))


class Policy:
    def __init__(self, model_path, device="cpu"):
        torch.set_num_threads(1)
        self.payload = torch.load(model_path, map_location=device, weights_only=True)
        self.device = torch.device(device)
        self.adapter = StateAdapter(self.payload["normalization"], self.payload["signal"], self.payload.get("schema_version", 1))
        self.model = QNetwork(self.payload["hidden"], len(self.adapter.features)).to(self.device)
        self.model.load_state_dict(self.payload["state_dict"])
        self.model.eval()

    def predict(self, state):
        vector = self.adapter.encode(state)
        mask = self.adapter.mask(state)
        with torch.inference_mode():
            q = self.model(torch.as_tensor(vector, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        action = int(np.where(mask, q, -np.inf).argmax())
        return {"action_id": action, "action": ["KEEP", "SWITCH"][action], "q_values": q.tolist(), "valid_actions": mask.tolist()}
