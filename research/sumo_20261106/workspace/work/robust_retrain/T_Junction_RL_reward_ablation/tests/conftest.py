from copy import deepcopy
import pytest
from maps.generate_map import generate
from src.utils import load_yaml


@pytest.fixture
def base():
    return deepcopy(load_yaml("base.yaml"))


@pytest.fixture
def agent_config():
    c = load_yaml("training.yaml")
    c.update(batch_size=4, learning_start=4, replay_capacity=32, target_update_steps=2, device="cpu")
    return c


@pytest.fixture(scope="session", autouse=True)
def map_ready():
    generate()


@pytest.fixture
def links():
    return [(('S_in_0', 'E_out_0', ':J_0'),), (('W_in_0', 'S_out_0', ':J_1'),),
            (('E_in_0', 'W_out_0', ':J_2'),), (('S_in_0', 'W_out_0', ':J_3'),),
            (('W_in_0', 'E_out_0', ':J_4'),), (('E_in_0', 'S_out_0', ':J_5'),)]
