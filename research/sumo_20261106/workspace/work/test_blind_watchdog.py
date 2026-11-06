from watchdog_study import BlindWatchdog


def test_watchdog_can_serve_undetected_direction_and_obeys_mask():
    guard=BlindWatchdog(80)
    guard.set_observation({'queues':[0,0,0]})
    for _ in range(70): guard.observe_second(0,'green',[20,10,30],[3,4,5])
    assert guard.ages==(0,0,0)  # hidden traffic was ignored
    assert guard.red_age==[0,70,70]
    assert guard.apply_watchdog(0,[True,False,False],0,False)==0
    assert guard.apply_watchdog(0,[True,True,True],0,True)==0
    assert guard.apply_watchdog(0,[True,True,True],0,False)==1
