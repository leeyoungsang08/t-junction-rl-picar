import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import Policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    a = p.parse_args()
    state = {"vehicles": [12, 3, 5], "mean_waiting_seconds": [20, 5, 12], "internal_vehicles": 1,
             "phase": 0, "green_elapsed": 20, "transitioning": False,
             "queues": [4, 1, 2], "pending": [0, 0, 0], "longest_wait": [30, 9, 18],
             "recent_requests": [4, 2, 3]}
    print(json.dumps(Policy(a.model).predict(state), indent=2))
