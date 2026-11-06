import xml.etree.ElementTree as ET
from maps.generate_map import validate_map
from src.utils import ROOT


def test_three_approaches_and_six_od():
    links = validate_map(ROOT / "maps/t_junction.net.xml")
    assert len(links) == 6
    root = ET.parse(ROOT / "maps/t_junction.net.xml")
    assert root.find("junction[@id='N']") is None
    for phase in root.findall("tlLogic/phase"):
        state = phase.get("state")
        origins = {c.get("from") for c in links if state[int(c.get("linkIndex"))] in "Gy"}
        assert len(origins) <= 1
