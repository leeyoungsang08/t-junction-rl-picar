"""Exit beam sensors at the conflict box exits (one per exit arm) + heartbeat, standard library only.

A beam is "blocked" while a car is in it. blocked -> clear means the car's rear passed the exit line:
an exit_event for that edge is sent. The beam cannot identify the car; the central PC assigns the event
to the oldest car inside that junction whose next segment is that edge. Heartbeats prove the sensor
chain is alive; without them the central supervisor stops issuing new permits.

On the Pi:  python3 -m picar_lab.car_agent.exit_beam --junction J --pins W_out=36,E_out=38,N_out=40 \\
                --host 192.168.50.10 --port 5007
"""
import argparse
import socket
import time

from picar_lab.car_agent.protocol import VERSION, encode


class ExitBeamMonitor:
    def __init__(self, node, junction, edges):
        self.node, self.junction, self.edges = node, junction, list(edges)
        self.blocked = {e: False for e in self.edges}
        self.seq = 0

    def _msg(self, kind, **extra):
        self.seq += 1
        return {"v": VERSION, "type": kind, "node": self.node, "seq": self.seq, "junction": self.junction, **extra}

    def update(self, readings, healthy=True):
        """readings: {edge: blocked(bool)}. Returns packets (exit events then one heartbeat)."""
        out = []
        for e in self.edges:
            now_blocked = bool(readings[e])
            if self.blocked[e] and not now_blocked:
                out.append(self._msg("exit_event", edge=e))
            self.blocked[e] = now_blocked
        out.append(self._msg("beam_heartbeat", ok=bool(healthy)))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--junction", choices=("J", "J2"), required=True)
    ap.add_argument("--pins", required=True, help="edge=BOARD_pin list, e.g. W_out=36,E_out=38,N_out=40")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--period", type=float, default=0.05)
    ap.add_argument("--active-low", action="store_true", help="receiver output low when beam is blocked")
    a = ap.parse_args()
    pins = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in a.pins.split(","))
    import RPi.GPIO as GPIO                      # only needed on the Pi
    GPIO.setmode(GPIO.BOARD)
    for p in pins.values():
        GPIO.setup(p, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    mon = ExitBeamMonitor(f"beam-{a.junction}", a.junction, pins)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            while True:
                readings = {e: (GPIO.input(p) == 0) == a.active_low for e, p in pins.items()}
                for m in mon.update(readings):
                    s.sendto(encode(m), (a.host, a.port))
                time.sleep(a.period)
        finally:
            GPIO.cleanup()


if __name__ == "__main__":
    main()
