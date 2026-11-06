import pytest
from src.evaluate import run_episode
from src.traffic_generator import generate_traffic


@pytest.mark.integration
def test_same_route_and_sumo_seed_reproduce_metrics(base, tmp_path):
    base["simulation"]["horizon_seconds"] = 180
    route = generate_traffic("high", 3000, 180, "test", tmp_path)
    a = run_episode(base, route, 3000, "fixed_time", tmp_path / "a")
    b = run_episode(base, route, 3000, "fixed_time", tmp_path / "b")
    for key in a:
        if key != "wall_seconds":
            assert a[key] == b[key], key
