import numpy as np
from src.utils import DIRECTIONS


def action_mask(green_elapsed, transitioning, config):
    if transitioning or green_elapsed < config["min_green"]:
        return np.array([True, False])
    if green_elapsed >= config["max_green"]:
        return np.array([False, True])
    return np.array([True, True])


class SignalController:
    def __init__(self, controlled_links, config):
        self.config = config
        for key in ("min_green", "max_green", "yellow", "all_red"):
            if int(config[key]) != config[key] or config[key] <= 0:
                raise ValueError(f"{key} must be a positive integer in seconds")
        if config["min_green"] > config["max_green"]:
            raise ValueError("min_green exceeds max_green")
        self.groups = []
        movements = set()
        for links in controlled_links:
            origins = {entry[0].split("_")[0] for entry in links}
            if len(origins) != 1 or not origins <= set(DIRECTIONS):
                raise ValueError("Ambiguous or empty TLS link group")
            self.groups.append(origins.pop())
            movements.update((entry[0], entry[1]) for entry in links)
        expected = {(f"{a}_in_0", f"{b}_out_0") for a in DIRECTIONS for b in DIRECTIONS if a != b}
        if movements != expected:
            raise ValueError("TLS does not control exactly the six expected OD movements")
        self.phase = 0
        self.stage = "green"
        self.green_elapsed = 0
        self.clearance_elapsed = 0
        self.switches = 0
        self.forced_switches = 0

    @property
    def transitioning(self):
        return self.stage != "green"

    def mask(self):
        return action_mask(self.green_elapsed, self.transitioning, self.config)

    def request(self, action):
        if action not in (0, 1) or not self.mask()[action]:
            raise ValueError(f"Invalid action {action}; mask={self.mask().tolist()}")
        if action == 1:
            self._switch()

    def _switch(self):
        self.stage = "yellow"
        self.clearance_elapsed = 0
        self.switches += 1

    def before_second(self):
        if self.stage == "green" and self.green_elapsed >= self.config["max_green"]:
            self._switch()
            self.forced_switches += 1
        color = {"green": "G", "yellow": "y", "all_red": "r"}[self.stage]
        return "".join(color if group == DIRECTIONS[self.phase] else "r" for group in self.groups)

    def after_second(self):
        if self.stage == "green":
            self.green_elapsed += 1
            return
        self.clearance_elapsed += 1
        if self.stage == "yellow" and self.clearance_elapsed >= self.config["yellow"]:
            self.stage = "all_red"
            self.clearance_elapsed = 0
        elif self.stage == "all_red" and self.clearance_elapsed >= self.config["all_red"]:
            self.phase = (self.phase + 1) % 3
            self.stage = "green"
            self.green_elapsed = 0
            self.clearance_elapsed = 0
