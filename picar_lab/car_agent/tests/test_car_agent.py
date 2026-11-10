"""Pi-side agent tests: no ROS2, no GPIO, no network."""
from picar_lab.car_agent.agent import CarAgent, MarkerDetector
from picar_lab.car_agent.exit_beam import ExitBeamMonitor
from picar_lab.car_agent.protocol import VERSION, decode, encode, valid_car_status

SESSION = "a" * 32


def agent_at_stopline():
    ag = CarAgent("W1", ["W_in", "E_out", "N_in", "W_out"])
    ag.register(0.0)
    ag.on_packet({"type": "status_ack", "session": SESSION, "car": "W1", "boot": ag.boot, "ack": 1}, 0.0)
    ag.on_approach_marker(1.0)
    ag.on_stopped_at_stopline(2.0)
    return ag


def permit(ag, req, grant=1, ttl_ms=500, **kw):
    return {"v": VERSION, "type": "permit", "session": SESSION, "car": ag.car, "boot": ag.boot, "req": req,
            "grant": grant, "junction": "J", "ttl_ms": ttl_ms, **kw}


def test_status_messages_are_valid_and_carry_next_segment():
    ag = agent_at_stopline()
    msgs = ag.tick(2.0)
    assert all(valid_car_status(m) for m in msgs if m["type"] == "car_status")
    arrive = [m for m in msgs if m.get("state") == "ARRIVE"][-1]
    assert (arrive["junction"], arrive["lane"], arrive["next"]) == ("J", "W", "E_out")
    assert decode(encode(arrive)) == arrive


def test_permit_ttl_counts_from_request_send_time_and_duplicates_do_not_extend():
    ag = agent_at_stopline()
    req = ag.seq                                  # ARRIVE sent at t=2.0
    assert ag.on_packet(permit(ag, req), 2.3) == "applied"
    assert ag.may_cross(2.45) and not ag.may_cross(2.5)
    assert ag.on_packet(permit(ag, req), 2.4) == "duplicate"
    assert not ag.may_cross(2.55)


def test_permit_rejections():
    ag = agent_at_stopline()
    req = ag.seq
    assert ag.on_packet(permit(ag, req, car="E1"), 2.1) == "rejected_identity"
    assert ag.on_packet(permit(ag, req, session="b" * 32), 2.1) == "rejected_session"
    assert ag.on_packet(permit(ag, req + 99), 2.1) == "rejected_stale"
    assert ag.on_packet(permit(ag, req), 2.6) == "rejected_expired"
    assert ag.on_packet(permit(ag, req, junction="J2"), 2.1) == "rejected_state"
    assert not ag.may_cross(2.1)


def test_motion_gate_stops_at_line_without_permit_and_on_line_loss():
    ag = agent_at_stopline()
    assert not ag.motion_allowed(2.1, at_stopline=True)
    assert ag.motion_allowed(2.1, at_stopline=False)
    ag.on_packet(permit(ag, ag.seq), 2.1)
    assert ag.motion_allowed(2.2, at_stopline=True)
    ag.set_health(line_ok=False, obstacle=False)
    assert not ag.motion_allowed(2.2, at_stopline=False)


def test_commit_point_and_crossing_legality():
    ag = agent_at_stopline()
    assert ag.on_crossed_stopline(2.1) is False          # crossed without permit -> reported as violation
    ag = agent_at_stopline()
    ag.on_packet(permit(ag, ag.seq), 2.1)
    assert ag.commit(2.3)
    assert ag.on_crossed_stopline(3.0) is True            # permit expired after the commit point: still legal
    assert ag.edge == "E_out" and ag.zone == "IN"


def test_requests_are_refreshed_and_reliable_messages_retransmitted():
    ag = agent_at_stopline()
    ag.tick(2.0)
    later = ag.tick(2.25)
    assert any(m.get("state") == "ARRIVE" for m in later)          # refreshed request (new seq)
    ag.on_crossed_stopline(2.3)
    first = [m for m in ag.tick(2.3) if m.get("state") == "IN"]
    again = [m for m in ag.tick(2.5) if m.get("state") == "IN"]
    assert first and again and first[0]["seq"] == again[0]["seq"]  # same seq until acknowledged


def test_marker_detector_thin_vs_thick():
    md = MarkerDetector(dt=0.1, approach_max_m=0.03, stop_min_m=0.04)
    # a band ends after two clean frames (one clean frame inside a band is tolerated as noise)
    seq = [[1, 1, 1, 1], [0, 1, 1, 0], [0, 1, 1, 0]]
    assert [md.update(v, 0.25) for v in seq] == [None, None, "approach"]
    out = [md.update([1, 1, 1, 1], 0.25) for _ in range(3)]
    assert out == [None, "stopline", None]


def test_marker_detector_tolerates_ir_noise():
    md = MarkerDetector(dt=0.05, approach_max_m=0.03, stop_min_m=0.04)
    # 5 cm stop tape at 0.25 m/s = 4 frames; one channel drops out and one frame is lost entirely
    frames = [[1, 1, 0, 1], [0, 0, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]]
    assert "stopline" in [md.update(v, 0.25) for v in frames]
    # the lane line alone (2 sensors) is never a cross tape
    md2 = MarkerDetector(dt=0.05)
    assert all(md2.update([0, 1, 1, 0], 0.25) is None for _ in range(20))


def test_exit_beam_reports_rear_leaving_and_heartbeat():
    mon = ExitBeamMonitor("beam-J", "J", ["W_out", "E_out", "N_out"])
    assert [m["type"] for m in mon.update({"W_out": True, "E_out": False, "N_out": False})] == ["beam_heartbeat"]
    out = mon.update({"W_out": False, "E_out": False, "N_out": False})
    assert [(m["type"], m.get("edge")) for m in out] == [("exit_event", "W_out"), ("beam_heartbeat", None)]
