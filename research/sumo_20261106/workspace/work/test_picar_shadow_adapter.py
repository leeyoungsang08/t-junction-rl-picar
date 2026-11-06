import pytest
import yaml
from pre_picar_study import OUT,LimitedSensor
from picar_shadow_adapter import ShadowAdapter


def make_adapter():
    cfg=yaml.safe_load((OUT/'inputs/config.yaml').read_text(encoding='utf8'))
    cfg['signal']['fixed_green']=15
    return ShadowAdapter(OUT/'inputs/old.pt',cfg)


def test_no_sensor_means_no_recommendation():
    a=make_adapter();a.tick(0,0,'green')
    r=a.decide(0,{'phase':0,'stage':'green','green_elapsed':10,'clearance_remaining':0})
    assert r['action_id'] is None and r['requires_independent_safe_controller']


def test_missing_frames_trigger_advisory_fallback_and_legal_action():
    a=make_adapter();a.tick(0,0,'green')
    signal={'phase':0,'stage':'green','green_elapsed':10,'clearance_remaining':0}
    assert a.decide(0,signal,{'timestamp':0,'queues':[1,0,0],'vehicles':[1,0,0]})['status']=='recommendation_only'
    for second in range(1,16):
        a.tick(second,0,'green')
        if second%5==0:
            signal['green_elapsed']=10+second
            result=a.decide(second,signal)
    assert result['fallback'] and result['missing_frames']==3
    assert result['action_id'] in (0,1,2)


def test_stale_frames_and_missing_signal_heartbeat_are_rejected():
    a=make_adapter();a.tick(0,0,'green')
    sig={'phase':0,'stage':'green','green_elapsed':10,'clearance_remaining':0}
    with pytest.raises(ValueError):a.decide(0,sig,{'timestamp':-100,'queues':[0,0,0],'vehicles':[0,0,0]})
    assert a.decide(0,sig,{'timestamp':0,'queues':[0,0,0],'vehicles':[0,0,0]})['status']=='recommendation_only'
    with pytest.raises(ValueError):a.tick(2,0,'green')


def test_shadow_features_match_clean_simulated_sensor():
    a=make_adapter();a.tick(0,0,'green');sensor=LimitedSensor('limited_clean',11)
    raw={'vehicles':[4,2,1],'queues':[2,1,0],'pending':[5,4,3],
         'mean_waiting_seconds':[20,10,5],'longest_wait':[20,10,5],
         'recent_requests':[5,4,3],'internal_vehicles':3,'phase':0,'green_elapsed':10,
         'transitioning':False,'clearance_remaining':0}
    expected,_=sensor.read(raw,0)
    got=a.decide(0,{'phase':0,'stage':'green','green_elapsed':10,'clearance_remaining':0},
                 {'timestamp':0,'queues':[2,1,0],'vehicles':[4,2,1]})['observation']
    assert got==expected
