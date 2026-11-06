import pytest
from src.signal_controller import SignalController


def tick(c, n):
    history = []
    for _ in range(n):
        history.append((c.phase, c.stage, c.before_second()))
        c.after_second()
    return history


def test_minimum_green_and_invalid_action(links, base):
    c = SignalController(links, base["signal"])
    assert c.mask().tolist() == [True, False]
    with pytest.raises(ValueError):
        c.request(1)
    tick(c, 9)
    assert not c.mask()[1]
    tick(c, 1)
    assert c.mask()[1]


def test_exact_yellow_all_red_and_next_green(links, base):
    c = SignalController(links, base["signal"])
    tick(c, 10)
    c.request(1)
    assert not c.mask()[1]
    history = tick(c, 5)
    assert [x[1] for x in history] == ["yellow", "yellow", "yellow", "all_red", "green"]
    assert history[-1][0] == 1 and c.green_elapsed == 1
    assert all(ch == "r" for ch in history[3][2])


def test_maximum_green_forces_switch_inside_decision(links, base):
    c = SignalController(links, base["signal"])
    tick(c, 45)
    assert c.mask().tolist() == [False, True]
    c.before_second()
    assert c.stage == "yellow" and c.forced_switches == 1


def test_no_conflicting_origins_and_cyclic_order(links, base):
    c = SignalController(links, base["signal"])
    starts = [0]
    prior = c.phase
    for _ in range(700):
        signal = c.before_second()
        served = {c.groups[i] for i, x in enumerate(signal) if x in "Gy"}
        assert len(served) <= 1
        assert c.green_elapsed <= base["signal"]["max_green"]
        c.after_second()
        if c.phase != prior:
            starts.append(c.phase)
            prior = c.phase
    assert starts == [i % 3 for i in range(len(starts))]
