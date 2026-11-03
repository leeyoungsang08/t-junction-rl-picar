"""Dry-run a central/PiCar link on localhost. This script never opens motor/GPIO devices."""
import argparse
import csv
import json
from pathlib import Path
import socket
import time
from uuid import uuid4

from src.inference import Policy
from src.utils import atomic_text, write_json

HOST = "127.0.0.1"
LEASE_SECONDS = 0.25
MAX_PACKET = 1024


def encode(packet):
    return json.dumps(packet, separators=(",", ":")).encode("ascii")


class Central:
    def __init__(self, channel, registry):
        self.channel = channel
        self.registry = registry
        self.latest = {}
        self.green = "W"
        self.clear = True
        self.telemetry_valid = True

    def receive_once(self, deliver=True):
        raw, addr = self.channel.recvfrom(MAX_PACKET + 1)
        if len(raw) > MAX_PACKET:
            return "invalid"
        try:
            status = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            return "invalid"
        if not isinstance(status, dict):
            return "invalid"
        car, lane, boot, seq = (status.get(key) for key in ("car", "lane", "boot", "seq"))
        if not isinstance(car, str) or self.registry.get(car) != (lane, boot, addr):
            return "unregistered"
        if status.get("v") != 1 or type(seq) is not int or seq <= self.latest.get(car, 0):
            return "stale"
        if status.get("zone") not in ("approach", "stopline", "inside", "exit"):
            return "invalid"
        if any(type(status.get(key)) is not bool for key in ("obstacle", "line_ok")):
            return "invalid"
        self.latest[car] = seq
        admit = (self.telemetry_valid and self.clear and self.green == lane and status["zone"] == "stopline"
                 and status["line_ok"] and not status["obstacle"])
        verdict = {"v": 1, "car": car, "boot": boot, "ack": seq,
                   "command": "GO" if admit else "STOP", "ttl_ms": int(LEASE_SECONDS * 1000)}
        if deliver:
            self.channel.sendto(encode(verdict), addr)
        return verdict["command"]


class MockCar:
    def __init__(self, channel, car, lane):
        self.channel = channel
        self.car, self.lane, self.boot = car, lane, uuid4().hex
        self.seq = 0
        self.request = None
        self.expires = 0.0
        self.obstacle = False
        self.line_ok = True
        self.zone = "stopline"

    def poll(self):
        self.seq += 1
        self.request = self.seq, time.monotonic()
        status = {"v": 1, "car": self.car, "lane": self.lane, "boot": self.boot, "seq": self.seq,
                  "zone": self.zone, "obstacle": self.obstacle, "line_ok": self.line_ok}
        raw = encode(status)
        self.channel.send(raw)
        return raw

    def accept(self, response):
        if self.request is None:
            return False
        ack, sent = self.request
        if (response.get("v"), response.get("car"), response.get("boot"), response.get("ack")) != (
                1, self.car, self.boot, ack):
            return False
        if response.get("command") not in ("GO", "STOP") or response.get("ttl_ms") != 250:
            return False
        self.request = None
        if time.monotonic() >= sent + LEASE_SECONDS:
            return False
        self.expires = sent + LEASE_SECONDS if response["command"] == "GO" else 0.0
        return True

    def output(self):
        allow = (time.monotonic() < self.expires and not self.obstacle and self.line_ok
                 and self.zone == "stopline")
        return "GO" if allow else "STOP"


def bench():
    checks = {}
    with (socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as listener,
          socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ws,
          socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as es):
        listener.bind((HOST, 0))
        listener.settimeout(1)
        for channel in (ws, es):
            channel.connect(listener.getsockname())
            channel.settimeout(0.1)
        west, east = MockCar(ws, "CAR1", "W"), MockCar(es, "CAR2", "E")
        central = Central(listener, {c.car: (c.lane, c.boot, c.channel.getsockname()) for c in (west, east)})

        def request(car, expected):
            raw = car.poll()
            assert central.receive_once() == expected
            response = json.loads(car.channel.recv(MAX_PACKET))
            assert car.accept(response) and car.output() == expected
            return raw, response

        packet, grant = request(west, "GO")
        checks["fresh_green"] = True
        request(east, "STOP")
        checks["opposing_red"] = True
        ws.send(packet)
        assert central.receive_once() == "stale"
        assert not west.accept(grant)
        checks["duplicate_no_renewal"] = True
        west.obstacle = True
        assert west.output() == "STOP"
        request(west, "STOP")
        west.obstacle = False
        checks["obstacle_overrides_grant"] = True
        request(west, "GO")
        west.line_ok = False
        assert west.output() == "STOP"
        request(west, "STOP")
        west.line_ok = True
        checks["line_loss_overrides_grant"] = True
        central.green = None
        request(west, "STOP")
        checks["clearance_stop"] = True
        central.green, central.clear = "W", False
        request(west, "STOP")
        checks["occupancy_stop"] = True
        central.clear, central.telemetry_valid = True, False
        request(west, "STOP")
        checks["unknown_observation_stop"] = True
        central.telemetry_valid = True
        request(west, "GO")
        west.poll()
        assert central.receive_once(deliver=False) == "GO"
        try:
            ws.recv(MAX_PACKET)
            raise AssertionError("Expected a dropped reply")
        except TimeoutError:
            pass
        time.sleep(max(0.0, west.expires - time.monotonic()) + 0.01)
        assert west.output() == "STOP"
        checks["dropped_reply_lease_expires"] = True
        west.poll()
        assert central.receive_once() == "GO"
        delayed = json.loads(ws.recv(MAX_PACKET))
        time.sleep(LEASE_SECONDS + 0.01)
        assert not west.accept(delayed) and west.output() == "STOP"
        checks["late_reply_rejected"] = True
        request(west, "GO")
        checks["fresh_poll_recovers"] = True
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = Policy(args.model, device="cpu")
    assert policy.adapter.schema == 2
    lines = ["decision_time_s,logged_action,shadow_action,q_keep,q_switch"]
    with args.decisions.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            prediction = policy.predict(json.loads(row["state"]))
            assert prediction["action_id"] == int(row["action"])
            assert prediction["valid_actions"] == json.loads(row["action_mask"])
            q0, q1 = prediction["q_values"]
            lines.append(f'{row["decision_time"]},{row["action"]},{prediction["action_id"]},{q0},{q1}')
    assert len(lines) == 361
    checks = bench()
    atomic_text(args.output / "shadow_decisions.csv", "\n".join(lines) + "\n")
    result = {"transport": "real UDP on localhost", "motor_output": "mock only",
              "model_device": "cpu", "matching_shadow_decisions": len(lines) - 1,
              "lease_ms": int(LEASE_SECONDS * 1000), "checks": checks}
    write_json(args.output / "picar_loopback_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
