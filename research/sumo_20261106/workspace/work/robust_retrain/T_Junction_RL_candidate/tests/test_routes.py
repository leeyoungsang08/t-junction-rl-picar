import xml.etree.ElementTree as ET
import pytest
from src.traffic_generator import generate_traffic
from src.utils import load_yaml, sha256


def test_route_contents_and_reproducible_bytes(tmp_path):
    a = generate_traffic("medium", 1000, 3600, directory=tmp_path / "a")
    b = generate_traffic("medium", 1000, 3600, directory=tmp_path / "b")
    assert sha256(a) == sha256(b)
    root = ET.parse(a)
    assert len(root.findall("route")) == 6
    assert all(x.get("edges").split()[0] != x.get("edges").split()[-1] for x in root.findall("route"))
    times = [float(x.get("depart")) for x in root.findall("vehicle")]
    assert times == sorted(times) and 0 < min(times) < max(times) < 3600


def test_split_separation_and_unseen_restriction(tmp_path):
    cfg = load_yaml("scenarios.yaml")["splits"]
    sets = [set(range(v["start"], v["start"] + v["count"])) for v in cfg.values()]
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i + 1:])
    with pytest.raises(ValueError):
        generate_traffic("unseen", 1000, 100, directory=tmp_path)
    with pytest.raises(ValueError):
        generate_traffic("low", 3000, 100, "train", tmp_path)
