from copy import deepcopy
from pre_picar_study import LimitedSensor, LatchedGuard


def raw_state():
    return {'vehicles':[4,2,0], 'queues':[2,1,0], 'pending':[99,88,77],
            'mean_waiting_seconds':[900,800,700], 'longest_wait':[900,800,700],
            'recent_requests':[99,88,77], 'internal_vehicles':42,
            'phase':0, 'green_elapsed':20, 'transitioning':False, 'clearance_remaining':0}


def test_hidden_values_do_not_affect_limited_state():
    first=raw_state(); second=deepcopy(first)
    second.update(pending=[0,0,0], mean_waiting_seconds=[0,0,0],
                  longest_wait=[0,0,0], recent_requests=[0,0,0], internal_vehicles=0)
    a,_=LimitedSensor('limited_clean',7).read(first,0)
    b,_=LimitedSensor('limited_clean',7).read(second,0)
    assert a==b
    assert a['pending']==[0,0,0] and a['internal_vehicles']==0


def test_latched_guard_ignores_true_queues_and_pending():
    a=LatchedGuard(2); b=LatchedGuard(2)
    state={'queues':[0,1,0]}
    a.set_observation(state); b.set_observation(state)
    for _ in range(2):
        a.observe_second(0,'green',[0,0,0],[0,0,0])
        b.observe_second(0,'green',[20,0,40],[7,100,4])
    assert a.ages==b.ages==(0,2,0)
    assert a.choose(0,[True,True,True],0,False,state['queues'],[0,0,0])[0]==1


def test_outage_keeps_last_counts_and_recovers():
    sensor=LimitedSensor('limited_outage',7)
    raw=raw_state(); initial,_=sensor.read(raw,11)
    # Changed traffic during outage should not be treated as a new observation.
    raw['queues']=[0,0,0]; raw['vehicles']=[0,0,0]
    held,flag=sensor.read(raw,12)
    assert flag and held['queues']==initial['queues']
    for step in (13,14,15): sensor.read(raw,step)
    assert sensor.outage_steps==4
    _,flag=sensor.read(raw,16)
    assert not flag and sensor.outage_steps==0
