from copy import deepcopy
import numpy as np
import pytest
from src.robust_observation import perturb_raw, tail_score

def example():
 return dict(vehicles=[10,20,30],queues=[8,9,7],pending=[3,5,2],recent_requests=[20,4,7],
             mean_waiting_seconds=[20,40,60],longest_wait=[10,20,30],internal_vehicles=4,
             phase=1,green_elapsed=15,transitioning=False,clearance_remaining=0)

def test_noise_reproducible_bounded_and_no_mutation():
 source=example();copy=deepcopy(source)
 a=perturb_raw(source,42,5)
 b=perturb_raw(source,42,5)
 assert a==b and source==copy
 assert a["phase"]==1 and a["green_elapsed"]==15 and not a["transitioning"]
 assert all(abs(x-y)<=max(.5,0.15*y+.5) for x,y in zip(a["vehicles"],source["vehicles"]))
 assert perturb_raw(source,42,5,False)==source

def test_tail_score_weights_high_delay_not_negative_reward():
 assert tail_score([1,2,3,4,100])==(.5*22+.5*100)
 assert tail_score([7,7,7])==7
 with pytest.raises(ValueError):tail_score([])
