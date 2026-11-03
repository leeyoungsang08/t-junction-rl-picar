"""Motorless PiCar status link. Standard library only, suitable for a PC and a Pi."""
import argparse
import json
from pathlib import Path
import re
import socket
import time
from uuid import uuid4

VERSION = 1
MAX_PACKET = 1024
STATES = ("APPROACH", "ARRIVE", "IN", "EXIT")
TOKEN = re.compile(r"[0-9a-f]{32}")


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def decode(raw):
    if len(raw) > MAX_PACKET:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        return None
    return value if isinstance(value, dict) else None


def valid_status(message, cars):
    car = message.get("car")
    boot = message.get("boot")
    seq = message.get("seq")
    return (message.get("type") == "lab_status" and type(message.get("v")) is int
            and message["v"] == VERSION and isinstance(car, str) and car in cars
            and message.get("lane") == cars[car]["lane"] and isinstance(boot, str)
            and TOKEN.fullmatch(boot) and type(seq) is int and 1 <= seq <= len(STATES)
            and message.get("state") == STATES[seq - 1]
            and type(message.get("line_ok")) is bool and type(message.get("obstacle")) is bool)


def serve(host, port, cars, log_path, max_messages):
    session = uuid4().hex
    latest = {}
    processed = 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server, log_path.open("a", encoding="utf-8") as log:
        server.bind((host, port))
        print(f"READY {server.getsockname()[0]}:{server.getsockname()[1]} session={session}", flush=True)
        while not max_messages or processed < max_messages:
            raw, peer = server.recvfrom(65535)
            message = decode(raw)
            if not message or not valid_status(message, cars):
                continue
            car, boot, seq = (message[key] for key in ("car", "boot", "seq"))
            prior = latest.get(car)
            if prior is None:
                result = "accepted" if seq == 1 else "out_of_order"
            elif (boot, peer) != prior[:2]:
                result = "session_conflict"
            elif seq == prior[2]["seq"]:
                result = "duplicate" if message == prior[2] else "conflict"
            elif seq < prior[2]["seq"]:
                result = "stale"
            else:
                result = "accepted" if seq == prior[2]["seq"] + 1 else "out_of_order"
            if result == "accepted":
                latest[car] = (boot, peer, message)
            ack = {"v": VERSION, "type": "lab_ack", "session": session, "car": car,
                   "boot": boot, "ack": seq, "result": result}
            event = {"received_monotonic_s": time.monotonic(), "peer": list(peer), "status": message, "ack": ack}
            line = json.dumps(event, separators=(",", ":"))
            print(line, file=log, flush=True)
            print(f"{car} {message['state']} #{seq} {result}", flush=True)
            server.sendto(encode(ack), peer)
            processed += 1


def send_sequence(host, port, car, lane):
    boot, session = uuid4().hex, None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.connect((host, port))
        for seq, state in enumerate(STATES, 1):
            message = {"v": VERSION, "type": "lab_status", "car": car, "boot": boot,
                       "seq": seq, "lane": lane, "state": state, "line_ok": True, "obstacle": False}
            for attempt in range(1, 4):
                sent = time.perf_counter()
                client.send(encode(message))
                deadline = sent + 0.35
                ack = None
                while time.perf_counter() < deadline:
                    client.settimeout(max(0.001, deadline - time.perf_counter()))
                    try:
                        raw = client.recv(65535)
                    except TimeoutError:
                        break
                    reply = decode(raw)
                    if reply and (reply.get("type"), reply.get("v"), reply.get("car"), reply.get("boot"),
                                  reply.get("ack")) == ("lab_ack", VERSION, car, boot, seq):
                        ack = reply
                        break
                if ack is None:
                    continue
                remote_session = ack.get("session")
                if (not isinstance(remote_session, str) or not TOKEN.fullmatch(remote_session)
                        or session is not None and session != remote_session):
                    raise RuntimeError("Server session changed. Restart the motorless test.")
                if ack.get("result") not in ("accepted", "duplicate"):
                    raise RuntimeError(f"Server rejected {car}/{seq}: {ack.get('result')}")
                session = remote_session
                print(json.dumps({"car": car, "state": state, "seq": seq, "attempt": attempt,
                                  "ack": ack["result"], "rtt_ms": round(1000 * (time.perf_counter() - sent), 3)}))
                break
            else:
                raise TimeoutError(f"No matching ACK for {car}/{seq}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("server", "client"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config/lab.example.json")
    parser.add_argument("--car", default="S1")
    parser.add_argument("--lane", choices=("W", "E", "S"), default="S")
    parser.add_argument("--log", type=Path, default=Path("lab_run/central_status.jsonl"))
    parser.add_argument("--max-messages", type=int, default=0, help="Bounded server run for testing")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535 or args.max_messages < 0 or args.mode == "client" and args.port == 0:
        parser.error("Invalid UDP port or server message limit.")
    try:
        if args.mode == "server":
            cfg = json.loads(args.config.read_text(encoding="utf-8"))
            serve(args.host, args.port, cfg["cars"], args.log, args.max_messages)
        else:
            send_sequence(args.host, args.port, args.car, args.lane)
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
