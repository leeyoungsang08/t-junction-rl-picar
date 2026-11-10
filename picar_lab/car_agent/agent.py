"""Pi-side intersection agent (standard library only, hardware independent).

Responsibilities on the car:
  * turn course markers into status events: APPROACH marker, stop line (ARRIVE when stopped), crossing (IN)
  * keep a request alive (APPROACH/ARRIVE re-sent every refresh_s; every send gets a new seq)
  * accept a PERMIT only if it names this car/boot/session and one of our recent request seqs, and only
    until request_send_time + ttl (local monotonic clock, per CENTRAL_UDP_KO.md); duplicates never extend it
  * motion gate: the car may pass the stop line only with a valid permit; line loss or obstacle always stops;
    once the stop line is crossed the car is committed and finishes crossing
The ROS2 node (ros2_gate_node.py) and the SUMO software-in-the-loop harness (picar_sim/sil.py) both drive
this class, so the logic tested in SUMO is the logic that runs on the Pi.
"""
from collections import OrderedDict
from uuid import uuid4

from picar_lab.car_agent.protocol import VERSION, valid_permit

ZONES = ("CRUISE", "APPROACH", "ARRIVE", "IN")


class CarAgent:
    def __init__(self, car, route, start_index=0, refresh_s=0.2, retransmit_s=0.15, boot=None):
        """route: cyclic list of segment edges the car drives, e.g. ["W_in", "E_out", "N_in", "W_out"]."""
        if len(route) < 2:
            raise ValueError("route needs at least two segments")
        self.car, self.route, self.idx = car, list(route), start_index
        self.boot = boot or uuid4().hex
        self.refresh_s, self.retransmit_s = refresh_s, retransmit_s
        self.seq = 0
        self.zone = "CRUISE"
        self.session = None
        self.requests = OrderedDict()      # seq -> local send time (APPROACH/ARRIVE only)
        self.permit_until = None           # local time until which the stop line may be crossed
        self.permit_grant = None
        self.committed = False
        self.last_request_sent = None
        self.unacked = {}                  # seq -> (msg, last_send_t) for REGISTER/IN (reliable)
        self.line_ok, self.obstacle = True, False
        self.outbox = []
        self.registered = False

    # ------------------------------------------------------------------ helpers
    @property
    def edge(self):
        return self.route[self.idx % len(self.route)]

    @property
    def next_edge(self):
        return self.route[(self.idx + 1) % len(self.route)]

    @property
    def junction(self):
        """Junction at the end of the current segment: *_in -> J, *_out -> J2."""
        return "J" if self.edge.endswith("_in") else "J2"

    @property
    def lane(self):
        return self.edge.split("_")[0]

    def _status(self, state, now, reliable=False, **extra):
        self.seq += 1
        msg = {"v": VERSION, "type": "car_status", "car": self.car, "boot": self.boot, "seq": self.seq,
               "state": state, "junction": None, "lane": None, "next": None, "edge": None,
               "line_ok": self.line_ok, "obstacle": self.obstacle, **extra}
        self.outbox.append(msg)
        if reliable:
            self.unacked[self.seq] = (msg, now)
        if state in ("APPROACH", "ARRIVE"):
            self.requests[self.seq] = now
            self.last_request_sent = now
            while len(self.requests) > 50:
                self.requests.popitem(last=False)
        return msg

    def _request(self, now):
        state = "ARRIVE" if self.zone == "ARRIVE" else "APPROACH"
        self._status(state, now, junction=self.junction, lane=self.lane, next=self.next_edge)

    # ------------------------------------------------------------------ inputs (sensors)
    def register(self, now):
        self._status("REGISTER", now, reliable=True, edge=self.edge)
        self.registered = True

    def on_approach_marker(self, now):
        if self.zone == "CRUISE":
            self.zone = "APPROACH"
            self._request(now)

    def on_stopped_at_stopline(self, now):
        if self.zone in ("CRUISE", "APPROACH"):
            self.zone = "ARRIVE"
            self._request(now)

    def commit(self, now):
        """The car is about to pass the stop line and cannot stop before it any more. Allowed only while the
        permit is valid; after this the car finishes crossing even if the permit expires."""
        if self.may_cross(now):
            self.committed = True
        return self.committed

    def on_crossed_stopline(self, now):
        """Front of the car passed the stop line. Legal only if committed or holding a valid permit."""
        legal = self.committed or self.may_cross(now)
        self.committed, self.zone = True, "IN"
        self._status("IN", now, reliable=True, junction=self.junction, lane=self.lane, next=self.next_edge)
        self.idx += 1                      # now driving onto the next segment
        self.requests.clear()
        self.permit_until = self.permit_grant = None
        return legal

    def on_left_box(self, now):
        """Rear passed the exit (car-side estimate; the exit beam is the authority for the central PC)."""
        if self.zone == "IN":
            self.zone, self.committed = "CRUISE", False
            self._status("CRUISE", now, edge=self.edge)

    def set_health(self, line_ok, obstacle):
        self.line_ok, self.obstacle = bool(line_ok), bool(obstacle)

    # ------------------------------------------------------------------ inputs (network)
    def on_packet(self, msg, now):
        t = msg.get("type")
        if t == "status_ack" and msg.get("car") == self.car and msg.get("boot") == self.boot:
            if self.session is None:
                self.session = msg.get("session")
            if msg.get("session") == self.session:
                self.unacked.pop(msg.get("ack"), None)
            return None
        if t == "permit" and valid_permit(msg):
            return self._on_permit(msg, now)
        return None

    def _on_permit(self, msg, now):
        result = "applied"
        if msg["car"] != self.car or msg["boot"] != self.boot:
            result = "rejected_identity"
        elif self.session is not None and msg["session"] != self.session:
            result = "rejected_session"
        elif self.zone not in ("APPROACH", "ARRIVE") or msg["junction"] != self.junction:
            result = "rejected_state"
        elif msg["req"] not in self.requests:
            result = "rejected_stale"
        else:
            until = self.requests[msg["req"]] + msg["ttl_ms"] / 1000.0
            if until <= now:
                result = "rejected_expired"
            elif self.permit_grant == msg["grant"]:
                result = "duplicate"                       # a re-sent permit never extends validity
            else:
                self.permit_until, self.permit_grant = until, msg["grant"]
        ack = {"v": VERSION, "type": "permit_ack", "car": self.car, "boot": self.boot, "grant": msg["grant"],
               "result": result}
        self.outbox.append(ack)
        return result

    # ------------------------------------------------------------------ outputs
    def may_cross(self, now):
        return (self.permit_until is not None and now < self.permit_until and self.line_ok and not self.obstacle)

    def motion_allowed(self, now, at_stopline):
        """Gate for the line-follower's cmd_vel. at_stopline: the car would pass the line in the next step."""
        if not self.line_ok or self.obstacle:
            return False
        if self.committed:
            return True
        return (not at_stopline) or self.may_cross(now)

    def tick(self, now):
        """Periodic: refresh the request, retransmit reliable messages. Returns packets to send."""
        if self.zone in ("APPROACH", "ARRIVE") and (self.last_request_sent is None
                                                    or now - self.last_request_sent >= self.refresh_s - 1e-9):
            self._request(now)
        for seq, (msg, sent) in list(self.unacked.items()):
            if now - sent >= self.retransmit_s - 1e-9:
                self.outbox.append(msg)                     # same seq: the central PC de-duplicates
                self.unacked[seq] = (msg, now)
        out, self.outbox = self.outbox, []
        return out


class MarkerDetector:
    """Classify perpendicular tape marks from the 4-channel line sensor (1 = dark tape under the sensor).

    A frame with at least 3 of 4 channels dark is a crossing tape (the lane line can cover at most 2
    sensors). One clean frame inside a band is tolerated as sensor noise, and a band needs min_full
    frames with all 4 dark (one flipped sensor on the lane line must not make a marker). The band length at the given
    speed tells thin (APPROACH marker) from thick (stop line). Thresholds are in metres and must be measured.
    """

    def __init__(self, dt=0.1, approach_max_m=0.03, stop_min_m=0.04, min_dark=3, gap_frames=1, min_full=2):
        self.dt, self.approach_max_m, self.stop_min_m = dt, approach_max_m, stop_min_m
        self.min_dark, self.gap_frames, self.min_full = min_dark, gap_frames, min_full
        self.run = 0
        self.full = 0                       # frames with all 4 dark (a flipped lane-line sensor gives 3)
        self.dist = 0.0
        self.gap, self.gap_dist = 0, 0.0

    def update(self, values, speed_mps):
        """values: 4 ints. Returns "stopline" as soon as a dark band is thicker than stop_min_m,
        "approach" when a thin band (<= approach_max_m) ends, otherwise None."""
        step = max(speed_mps, 0.0) * self.dt
        n = sum(1 for v in values if v == 1)
        if n >= self.min_dark:
            was_stop = self.dist >= self.stop_min_m and self.full >= self.min_full
            self.run += 1
            self.full += n == len(values)
            self.dist += step + self.gap_dist
            self.gap, self.gap_dist = 0, 0.0
            is_stop = self.dist >= self.stop_min_m and self.full >= self.min_full
            return "stopline" if is_stop and not was_stop else None
        if self.run and self.gap < self.gap_frames:      # one missing frame: maybe noise, wait
            self.gap += 1
            self.gap_dist += step
            return None
        kind = "approach" if self.run and self.full and self.dist <= self.approach_max_m else None
        self.run, self.full, self.dist, self.gap, self.gap_dist = 0, 0, 0.0, 0, 0.0
        return kind
