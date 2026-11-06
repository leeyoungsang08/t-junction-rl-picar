import numpy as np
from fixed_intersection_evaluate import SevereSensor
from fixed_intersection_gate import choose_by_teacher_gap
import fixed_intersection_gate as gate
from fixed_intersection_watchdog import Watchdog
import pytest


def state():
    return {'vehicles':[10,5,3],'queues':[8,4,1],'pending':[999,999,999],
     'mean_waiting_seconds':[999,999,999],'longest_wait':[999,999,999],
     'recent_requests':[999,999,999],'internal_vehicles':999,'phase':0,
     'green_elapsed':20,'transitioning':False,'clearance_remaining':0}


def test_severe_sensor_is_deterministic_and_no_oracle_features():
    a=SevereSensor(70990);b=SevereSensor(70990)
    x,_=a.read(state(),0);y,_=b.read(state(),0)
    assert x==y and x['pending']==[0.,0.,0.] and x['internal_vehicles']==0
    assert np.all(np.asarray(x['queues'])<=np.asarray(x['vehicles']))


def test_severe_outage_latches_sensor_and_recovers():
    a=SevereSensor(70990);first,_=a.read(state(),11)
    change=state();change['vehicles']=[0,0,0];change['queues']=[0,0,0]
    for step in range(12,18):
        result,outage=a.read(change,step)
        assert outage and result['queues']==first['queues']
    assert a.outage_steps==6
    _,outage=a.read(change,18)
    assert not outage and a.outage_steps==0


def test_teacher_gap_gate_falls_back_and_rejects_illegal_actions():
    ref={'action_id':0,'q_values':[3.,2.9,1.], 'valid_actions':[True,True,False]}
    action,gap=choose_by_teacher_gap(ref,1,.2)
    assert action==1 and gap==pytest.approx(.1)
    action,_=choose_by_teacher_gap(ref,1,.05)
    assert action==0
    with pytest.raises(ValueError):choose_by_teacher_gap(ref,2,.5)


def test_selection_rejects_single_large_regression(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'OUT',tmp_path)
    rows=[]
    for sc in gate.SCENARIOS:
      for seed in gate.SEEDS:
       for sensor in gate.SENSORS:
        base={'scenario':sc,'seed':seed,'sensor':sensor,'route_sha256':'route',
              'completion_rate':.9,'max_consecutive_wait_s':80,'candidate_allowed':0}
        rows.append({**base,'controller':'old','cohort_delay_s':10.})
        for threshold in gate.THRESHOLDS:
            bad=(sc,seed,sensor)==(gate.SCENARIOS[0],gate.SEEDS[0],gate.SENSORS[0])
            rows.append({**base,'controller':f'anchor42_t{threshold:.2f}','cohort_delay_s':11.1 if bad else 9.})
    result=gate.select(rows,{'selected_candidates':{'anchor42':{}}})
    assert result['selected_gate'] is None
    assert all(not d['eligible'] and d['max_individual_delta']>1 for d in result['all_candidates'])


def test_watchdog_respects_signal_mask_and_transition():
    w=Watchdog(70);w.set_observation({'queues':[0,0,0]})
    for _ in range(70):w.observe_second(0,'green',[5,5,5],[5,5,5])
    assert w.request(0,[True,False,False],0,False)==0
    assert w.request(0,[True,True,True],0,True)==0
    assert w.request(0,[True,True,True],0,False)==1
    w.watch_threshold=None
    assert w.request(0,[True,True,True],0,False)==0
