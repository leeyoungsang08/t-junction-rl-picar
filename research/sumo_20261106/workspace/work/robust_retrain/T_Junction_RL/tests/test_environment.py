import json
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from src.environment import TJunctionEnv
from src.evaluate import run_episode
from src.utils import atomic_text, write_json


@pytest.mark.integration
def test_all_six_movements_depart_and_arrive_without_collision(base, tmp_path):
    horizon = 650
    base["simulation"]["horizon_seconds"] = horizon
    root = ET.Element("routes")
    requests = {}
    for index, (a, b) in enumerate(( (a,b) for a in "WES" for b in "WES" if a != b)):
        vid = f"v{index}"
        depart = 1 + 80 * index
        v = ET.SubElement(root, "vehicle", id=vid, depart=str(depart))
        ET.SubElement(v, "route", edges=f"{a}_in {b}_out")
        requests[vid] = {"requested_depart": depart, "origin": a}
    route = tmp_path / "six.rou.xml"
    atomic_text(route, ET.tostring(root, encoding="unicode"))
    write_json(route.with_suffix(".json"), {"horizon_seconds": horizon, "vehicles": requests})
    m = run_episode(base, route, 1000, "fixed_time", tmp_path / "run")
    assert m["departed_vehicles"] == m["completed_vehicles"] == 6
    assert m["collision_events"] == m["teleport_events"] == m["uninserted_vehicles"] == 0


@pytest.mark.integration
def test_gym_shape_truncation_and_invalid_action(base, tmp_path):
    from src.traffic_generator import generate_traffic
    base["simulation"]["horizon_seconds"] = 17
    route = generate_traffic("high", 1000, 17, directory=tmp_path)
    e = TJunctionEnv(base, route, output_dir=tmp_path / "env")
    try:
        state, info = e.reset(seed=1000)
        assert e.observation_space.contains(state)
        with pytest.raises(ValueError):
            e.step(1)
        done = False
        while not done:
            state, reward, term, done, info = e.step(int(not info["action_mask"][0]))
            assert not term and e.observation_space.contains(state)
        assert e.time == 17 and len(e.timeseries) == 17
        assert (tmp_path / "env/metrics.json").exists()
    finally:
        e.close()


@pytest.mark.integration
def test_fixed_time_preserves_exact_twenty_second_greens(base, tmp_path):
    import pandas as pd
    from src.traffic_generator import generate_traffic
    base["simulation"]["horizon_seconds"] = 200
    route = generate_traffic("medium", 3000, 200, "test", tmp_path)
    run_episode(base, route, 3000, "fixed_time", tmp_path / "fixed")
    data = pd.read_csv(tmp_path / "fixed/timeseries.csv")
    groups = data.groupby((data.stage.ne(data.stage.shift()) | data.phase.ne(data.phase.shift())).cumsum())
    sections = [(part.stage.iloc[0], len(part)) for _, part in groups]
    assert all(length == {"green": 20, "yellow": 3, "all_red": 1}[stage] for stage, length in sections[:-1])
