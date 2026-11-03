"""Replay two real DQN observations through the signal FSM and a loopback UDP admission gate."""
from contextlib import ExitStack
from pathlib import Path
import csv
import json
import socket
import xml.etree.ElementTree as ET

from picar_loopback import Central, MockCar, MAX_PACKET
from src.inference import Policy
from src.signal_controller import SignalController
from src.utils import DIRECTIONS, load_yaml, write_json

run = Path("outputs/budget_check_20261103")
policy = Policy(run / "training/seed43/checkpoints/best_model.pt", device="cpu")
cfg = load_yaml("base.yaml")
links = sorted((el for el in ET.parse("maps/t_junction.net.xml").findall("connection") if el.get("tl") == "J"),
               key=lambda el: int(el.get("linkIndex")))
controlled = [((f'{el.get("from")}_{el.get("fromLane")}', f'{el.get("to")}_{el.get("toLane")}', el.get("via")),)
              for el in links]
cases = []
for traffic_seed, second, action in ((3401, 595, 0), (3405, 950, 1)):
    path = run / f"evaluation/episodes/heldout_reversal_{traffic_seed}_v2_43_240/decisions.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(r for r in csv.DictReader(stream) if int(r["decision_time"]) == second)
    state = json.loads(row["state"])
    verdict = policy.predict(state)
    assert verdict["action_id"] == int(row["action"]) == action
    assert state["phase"] == 1 and state["green_elapsed"] == 21 and not state["transitioning"]
    cases.append({"traffic_seed": traffic_seed, "time_s": second, "state": state,
                  "logged_action": int(row["action"]), "prediction": verdict})

ctl = SignalController(controlled, cfg["signal"])
ctl.phase, ctl.green_elapsed = 1, 21
ctl.request(cases[0]["prediction"]["action_id"])
assert ctl.stage == "green" and DIRECTIONS[ctl.phase] == "E"
ctl.request(cases[1]["prediction"]["action_id"])

with ExitStack() as stack:
    listener = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(1)
    cars = {}
    for name in ("E1", "S1", "S2"):
        client = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
        client.connect(listener.getsockname())
        client.settimeout(1)
        cars[name] = MockCar(client, name, name[0])
    central = Central(listener, {car.car: (car.lane, car.boot, car.channel.getsockname()) for car in cars.values()})

    def exchange(car):
        client = cars[car]
        client.poll()
        expected = central.receive_once()
        reply = json.loads(client.channel.recv(MAX_PACKET))
        assert client.accept(reply) and client.output() == expected
        return expected

    timeline = []
    for offset in range(5):
        tls = ctl.before_second()
        central.green = DIRECTIONS[ctl.phase] if ctl.stage == "green" else None
        timeline.append({"offset_sim_s": offset, "stage": ctl.stage, "tls": tls,
                         "E1": exchange("E1"), "S1": exchange("S1")})
        ctl.after_second()
    assert [r["stage"] for r in timeline] == ["yellow", "yellow", "yellow", "all_red", "green"]
    assert [r["S1"] for r in timeline] == ["STOP"] * 4 + ["GO"]
    assert all(r["E1"] == "STOP" for r in timeline)

    central.clear = False   # Emulated reservation/occupancy, pending an independent exit observation.
    while_reserved = exchange("S2")
    central.clear = True    # Emulated observation that the entire first car crossed the exit marker.
    after_exit = exchange("S2")
    assert (while_reserved, after_exit) == ("STOP", "GO")

result = {"observation_source": "Recorded SUMO decisions, validation-selected seed 43 model",
          "actuation": "mock clients only", "cases": cases, "timeline": timeline,
          "next_car_while_reserved": while_reserved, "next_car_after_emulated_exit": after_exit}
write_json(run / "picar_dry_run/policy_to_udp.json", result)
print("Two recorded DQN decisions reproduced. SWITCH -> three yellow ticks -> all red -> UDP GO for S1.")
print("S2:", while_reserved, "until simulated EXIT, then", after_exit)
