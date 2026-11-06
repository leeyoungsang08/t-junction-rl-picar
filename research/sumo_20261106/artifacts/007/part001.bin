"""Observation-only adapter prototype. It never sends commands to hardware.

The caller supplies verified signal feedback, a 1 Hz tick, and 5 s detector
frames. Values and watchdog timeouts are assumptions until measured on PiCar.
"""
from collections import deque
from numbers import Integral
import math
import numpy as np
from src.inference import Policy
from src.service_guard import DirectionServiceGuard
from src.baselines import direct_phase_baseline


class ShadowAdapter:
    def __init__(self,model_path,config,*,max_frame_age_s=10):
        if not math.isfinite(max_frame_age_s) or max_frame_age_s<=0: raise ValueError('max_frame_age_s must be positive and finite')
        self.policy=Policy(model_path); self.config=config;self.max_frame_age_s=float(max_frame_age_s)
        expected=self.policy.payload['signal']; actual=config['signal']
        for key in ('control_mode','min_green','max_green','yellow','all_red'):
            if expected.get(key)!=actual.get(key):raise ValueError(f'signal configuration differs from checkpoint: {key}')
        if config['simulation']['decision_seconds']!=5:raise ValueError('prototype assumes a 5-second decision interval')
        self.guard=DirectionServiceGuard(80);self.latched=[0,0,0];self.frame=None
        self.last_frame_timestamp=None;self.last_decision_timestamp=None;self.last_tick_timestamp=None
        self.last_signal=None;self.outage_steps=0;self.age=np.zeros(3);self.prev=np.zeros(3);self.recent=deque(maxlen=6)

    def tick(self,at,phase,stage):
        if not math.isfinite(at) or isinstance(phase,bool) or not isinstance(phase,Integral) or phase not in (0,1,2) or stage not in ('green','yellow','all_red'):
            raise ValueError('invalid trusted signal feedback')
        if self.last_tick_timestamp is not None and abs(at-self.last_tick_timestamp-1)>1e-6:
            raise ValueError('signal ticks must be continuous at 1 Hz; reset/hold safely on loss')
        self.last_tick_timestamp=float(at); self.last_signal=(phase,stage)
        self.guard.observe_second(phase,stage,self.latched,[0,0,0])

    @staticmethod
    def _counts(values):
        if len(values)!=3 or any(isinstance(x,bool) or not isinstance(x,Integral) or x<0 for x in values):
            raise ValueError('each direction needs a nonnegative integer count')
        return np.asarray(values,dtype=float)

    def decide(self,at,signal,frame=None):
        if not math.isfinite(at): raise ValueError('invalid timestamp')
        if self.last_decision_timestamp is not None and abs(at-self.last_decision_timestamp-5)>1e-6:
            raise ValueError('decisions must be exactly five seconds apart')
        if self.last_tick_timestamp is None or not 0<=at-self.last_tick_timestamp<=1:
            raise ValueError('fresh trusted signal feedback is required')
        phase=signal['phase'];stage=signal['stage']; elapsed=float(signal['green_elapsed']);remaining=float(signal['clearance_remaining'])
        if isinstance(phase,bool) or not isinstance(phase,Integral):raise ValueError('invalid signal phase')
        phase=int(phase)
        if self.last_signal!=(phase,stage) or elapsed<0 or remaining<0 or not math.isfinite(elapsed+remaining):
            raise ValueError('inconsistent signal feedback')
        valid=False
        if frame is not None:
            ts=float(frame['timestamp']); q=self._counts(frame['queues']);v=self._counts(frame['vehicles'])
            if (q>v).any() or not math.isfinite(ts) or ts>at or at-ts>self.max_frame_age_s:
                raise ValueError('invalid or stale detector frame')
            if self.last_frame_timestamp is not None and ts<=self.last_frame_timestamp:
                raise ValueError('detector frame timestamps must increase; use None for a missing frame')
            self.frame=(q,v);self.last_frame_timestamp=ts;valid=True
        self.last_decision_timestamp=float(at)
        self.outage_steps=0 if valid else self.outage_steps+1
        if self.frame is None:
            return {'action_id':None,'status':'no_sensor_data','requires_independent_safe_controller':True}
        q,v=(x.copy() for x in self.frame)
        self.age=np.where(q>0,self.age+5,0);self.recent.append(np.maximum(v-self.prev,0));self.prev=v;self.latched=q.tolist()
        state={'vehicles':v.tolist(),'queues':q.tolist(),'pending':[0,0,0],
               'mean_waiting_seconds':self.age.tolist(),'longest_wait':self.age.tolist(),
               'recent_requests':np.sum(self.recent,axis=0).tolist(),'internal_vehicles':0,
               'phase':phase,'green_elapsed':elapsed,'transitioning':stage!='green','clearance_remaining':remaining}
        mask=self.policy.adapter.mask(state)
        if 'action_mask' in signal and not np.array_equal(mask,np.asarray(signal['action_mask'],dtype=bool)):
            raise ValueError('trusted controller mask differs from policy mask')
        fallback=self.outage_steps>=3
        if fallback:
            action=direct_phase_baseline('fixed_time',{'action_mask':mask,'raw_state':state},self.config)
            # The stale queue guard cannot supersede a prolonged-loss fallback.
        else:
            preferred=self.policy.predict(state)['action_id']
            action,_=self.guard.choose(preferred,mask,phase,stage!='green',q,[0,0,0])
        return {'action_id':int(action),'direction':'WES'[action],'fallback':fallback,
                'missing_frames':self.outage_steps,'status':'recommendation_only',
                'requires_independent_safe_controller':True,'observation':state}
