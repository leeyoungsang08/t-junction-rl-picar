"""PiCar <-> central PC UDP protocol v2 (standard library only).

Extends the v1 lab_status of docs/CENTRAL_UDP_KO.md with what the Real-to-Sim controller needs.

car -> central   car_status   {v, type, car, boot, seq, state, junction, lane, next, edge, line_ok, obstacle}
                 state: REGISTER (edge = start segment) | APPROACH | ARRIVE | IN | CRUISE
                 Every status gets a new seq. APPROACH/ARRIVE are re-sent every refresh_s as the request;
                 the latest seq is the request id a PERMIT must refer to.
central -> car   status_ack   {v, type, session, car, boot, ack}
central -> car   permit       {v, type, session, car, boot, req, grant, junction, ttl_ms}
car -> central   permit_ack   {v, type, car, boot, grant, result}
beam -> central  exit_event   {v, type, node, seq, junction, edge}       (no car id: the beam cannot tell)
beam -> central  beam_heartbeat {v, type, node, seq, junction, ok}
"""
import json
import re

VERSION = 2
MAX_PACKET = 1024
TOKEN = re.compile(r"[0-9a-f]{32}")
CAR_STATES = ("REGISTER", "APPROACH", "ARRIVE", "IN", "CRUISE")
ARMS = ("W", "E", "N")
JUNCTIONS = ("J", "J2")


def encode(msg):
    raw = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_PACKET:
        raise ValueError("packet too large")
    return raw


def decode(raw):
    if len(raw) > MAX_PACKET:
        return None
    try:
        msg = json.loads(raw)
    except (ValueError, UnicodeError):
        return None
    return msg if isinstance(msg, dict) and msg.get("v") == VERSION and isinstance(msg.get("type"), str) else None


def _edge_ok(e):
    return e is None or (isinstance(e, str) and len(e.split("_")) == 2 and e.split("_")[0] in ARMS
                         and e.split("_")[1] in ("in", "out"))


def valid_car_status(m):
    return (m.get("type") == "car_status" and isinstance(m.get("car"), str) and TOKEN.fullmatch(str(m.get("boot", "")))
            and type(m.get("seq")) is int and m["seq"] >= 1 and m.get("state") in CAR_STATES
            and m.get("junction") in (None, *JUNCTIONS) and m.get("lane") in (None, *ARMS)
            and _edge_ok(m.get("next")) and _edge_ok(m.get("edge"))
            and type(m.get("line_ok")) is bool and type(m.get("obstacle")) is bool
            and (m["state"] != "REGISTER" or m.get("edge") is not None)
            and (m["state"] not in ("APPROACH", "ARRIVE", "IN") or (m.get("junction") and m.get("lane") and m.get("next"))))


def valid_permit(m):
    return (m.get("type") == "permit" and TOKEN.fullmatch(str(m.get("session", ""))) and isinstance(m.get("car"), str)
            and TOKEN.fullmatch(str(m.get("boot", ""))) and type(m.get("req")) is int and type(m.get("grant")) is int
            and m.get("junction") in JUNCTIONS and type(m.get("ttl_ms")) is int and 0 < m["ttl_ms"] <= 5000)


def valid_beam(m):
    return (m.get("type") in ("exit_event", "beam_heartbeat") and isinstance(m.get("node"), str)
            and type(m.get("seq")) is int and m.get("junction") in JUNCTIONS
            and (m["type"] != "exit_event" or (_edge_ok(m.get("edge")) and m.get("edge") is not None)))
