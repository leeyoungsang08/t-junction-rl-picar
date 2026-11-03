"""Exercise the motorless status server with separate client processes and fault packets."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from status_link import encode

SCRIPT = Path(__file__).with_name("status_link.py")


def verify():
    with TemporaryDirectory() as temp:
        log = Path(temp) / "status.jsonl"
        server = subprocess.Popen(
            [sys.executable, str(SCRIPT), "server", "--port", "0",
             "--max-messages", "16", "--log", str(log)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        )
        try:
            ready = server.stdout.readline().split()
            assert ready[0] == "READY", ready
            host, port = ready[1].split(":")
            for car, lane in (("S1", "S"), ("W1", "W")):
                response = subprocess.check_output([sys.executable, str(SCRIPT), "client", "--host", host,
                                                   "--port", port, "--car", car, "--lane", lane], text=True)
                rows = [json.loads(line) for line in response.splitlines()]
                assert [row["state"] for row in rows] == ["APPROACH", "ARRIVE", "IN", "EXIT"]
                assert all(row["ack"] == "accepted" for row in rows)
            boot = uuid4().hex
            base = {"v": 1, "type": "lab_status", "car": "E1", "lane": "E",
                    "boot": boot, "line_ok": True, "obstacle": False}
            first = {**base, "seq": 1, "state": "APPROACH"}
            second = {**base, "seq": 2, "state": "ARRIVE"}
            third = {**base, "seq": 3, "state": "IN"}
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
                udp.connect((host, int(port)))
                udp.settimeout(0.15)
                udp.send(encode({**first, "car": "unregistered"}))
                try:
                    udp.recv(65535)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError("Unregistered car received a response")
                udp.settimeout(2)
                for packet, result in [(second, "out_of_order"), (first, "accepted"), (first, "duplicate"),
                                       ({**first, "obstacle": True}, "conflict"), (third, "out_of_order"),
                                       (second, "accepted"), (first, "stale"),
                                       ({**third, "boot": uuid4().hex}, "session_conflict")]:
                    udp.send(encode(packet))
                    ack = json.loads(udp.recv(65535))
                    assert ack["type"] == "lab_ack" and ack["result"] == result, ack
                    assert "command" not in ack and "ttl_ms" not in ack
            assert server.wait(timeout=5) == 0, server.stdout.read()
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 16 and sum(r["ack"]["result"] == "accepted" for r in rows) == 10
            print("PASS: two clients and 16 status/ACK exchanges, invalid ID ignored, order and replay checked.")
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=5)


if __name__ == "__main__":
    verify()
