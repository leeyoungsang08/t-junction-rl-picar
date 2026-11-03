"""Read-only UDP link probe. Run on the PC and Pi without ROS or motor access."""
import argparse
import json
import math
import socket
import time
from uuid import uuid4

PREFIX = b"picar-probe/1/"
MAX_SIZE = 64


def listen(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        print("Listening", *sock.getsockname(), flush=True)
        while True:
            payload, peer = sock.recvfrom(65535)
            if len(payload) == len(PREFIX) + 32 and payload.startswith(PREFIX):
                sock.sendto(payload, peer)


def measure(host, port, count):
    rtt = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect((host, port))
        for _ in range(count):
            payload = PREFIX + uuid4().hex.encode("ascii")
            sent = time.perf_counter()
            deadline = sent + 0.25
            try:
                sock.send(payload)
                while time.perf_counter() < deadline:
                    sock.settimeout(max(0.001, deadline - time.perf_counter()))
                    received = sock.recv(65535)
                    now = time.perf_counter()
                    if received == payload and now <= deadline:
                        rtt.append(1000 * (now - sent))
                        break
            except (TimeoutError, ConnectionResetError, ConnectionRefusedError):
                pass
            time.sleep(max(0.0, sent + 0.1 - time.perf_counter()))
    rtt.sort()
    result = {"sent": count, "matched": len(rtt), "lost_or_late": count - len(rtt),
              "p95_rtt_ms": round(rtt[math.ceil(0.95 * len(rtt)) - 1], 3) if rtt else None,
              "deadline_ms": 250}
    print(json.dumps(result))
    return 0 if rtt else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("listen", "probe"))
    parser.add_argument("--host", default="127.0.0.1", help="IPv4 bind address or central PC IPv4")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.count <= 10000:
        parser.error("Specify a valid UDP port and 1 to 10000 probes.")
    if args.mode == "listen":
        try:
            listen(args.host, args.port)
        except KeyboardInterrupt:
            return 0
    return measure(args.host, args.port, args.count)


if __name__ == "__main__":
    raise SystemExit(main())

