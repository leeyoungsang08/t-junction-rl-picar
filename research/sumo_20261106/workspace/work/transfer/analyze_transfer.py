from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import t
from run_transfer import HERE, OUT, DIRECTIONS, SCENARIOS, SEEDS, HORIZON, MODELS, BASELINES, DEMAND, MAPS, SOURCE_NET, sha256

def main():
    frame = pd.read_csv(HERE / "episode_metrics.csv")
    assert len(frame) == 2*len(SCENARIOS)*len(SEEDS)*6
    assert not frame.duplicated(["geometry", "scenario", "seed", "controller"]).any()
    audits, blocks, requests = [], [], []
    for (scenario, seed), group in frame.groupby(["scenario","seed"]):
        manifest = json.loads((DEMAND/f"{scenario}_{seed}.rou.json").read_text())
        expected = manifest["vehicles"]
        assert group.route_sha256.nunique() == 1
        assert group.route_sha256.iloc[0] == sha256(DEMAND/f"{scenario}_{seed}.rou.xml") == manifest["route_sha256"]
        for idx, (lo, hi, nominal, _) in enumerate(SCENARIOS[scenario]):
            counts = Counter(v["origin"] for v in expected.values() if lo <= v["requested_depart"] < hi)
            requests.append(dict(scenario=scenario,seed=seed,block=idx,start=lo,end=hi,nominal_veh_h=nominal,
                                 **{f"requests_{d}":counts[d] for d in DIRECTIONS}))
        for row in group.itertuples(index=False):
            g, c = row.geometry, row.controller
            path = OUT/g/scenario/str(seed)/c
            assert row.net_sha256 == sha256(MAPS/"skew_t.net.xml" if g == "new" else SOURCE_NET)
            trips = {n.get("id"): n.attrib for n in ET.parse(path/"tripinfo.xml").findall("tripinfo")}
            assert trips.keys() <= expected.keys() and len(trips) == row.departed_vehicles
            assert len(expected) == row.requested_vehicles and len(expected)-len(trips) == row.uninserted_vehicles
            arrivals = sum(float(v["arrival"]) >= 0 for v in trips.values())
            assert arrivals == row.completed_vehicles and abs(arrivals/len(expected) - row.completion_rate) < 1e-10
            delays = {k: (float(trips[k]["waitingTime"])+float(trips[k]["departDelay"]) if k in trips
                         else HORIZON-v["requested_depart"]) for k,v in expected.items()}
            assert abs(np.mean(list(delays.values())) - row.cohort_delay_s) < 1e-10
            for d in DIRECTIONS:
                assert abs(np.mean([delays[k] for k,v in expected.items() if v["origin"]==d]) - getattr(row,f"cohort_delay_{d}_s")) < 1e-10
            df = pd.read_csv(path/"timeseries.csv")
            assert df.time.tolist() == list(range(1,HORIZON+1))
            assert df.cumulative_arrived.iloc[-1] == arrivals and df.cumulative_departed.iloc[-1] == len(trips)
            phase, stage, states = df.phase.tolist(), df.stage.tolist(), df.tls_state.tolist()
            colors = {"green":"G","yellow":"y","all_red":"r"}
            assert all(s == "".join(colors[b] if k == DIRECTIONS[a] else "r" for k in row.tls_groups)
                       for a,b,s in zip(phase,stage,states))
            runs = [(p,s,len(list(items))) for (p,s),items in groupby(zip(phase,stage))]
            assert runs[0][:2] == (0,"green")
            assert sum(s=="yellow" for _,s,_ in runs) == row.signal_switches == df.switch_events.sum()
            for i,(p,s,dur) in enumerate(runs):
                assert 1<=dur<=({"green":45,"yellow":3,"all_red":1}[s])
                if i == len(runs)-1:
                    continue
                q, next_stage, _ = runs[i+1]
                if s=="green":
                    assert dur >=10 and q==p and next_stage=="yellow"
                elif s=="yellow":
                    assert dur==3 and q==p and next_stage=="all_red"
                else:
                    assert dur==1 and q!=p and next_stage=="green"
            decisions = pd.read_csv(path/"decisions.csv")
            assert len(decisions) == HORIZON // (1 if c=="fixed20" else 5)
            no_queue = [], []
            at_switch = []
            for dec in decisions.itertuples(index=False):
                raw, mask = json.loads(dec.state), json.loads(dec.action_mask)
                assert mask[dec.action]
                if not pd.isna(dec.requested_switch_target):
                    assert dec.action==dec.requested_switch_target and dec.decision_time < HORIZON
                    if df.stage.iloc[dec.decision_time]!="yellow":
                        raise AssertionError("requested change did not apply")
                    q = np.asarray(raw["queues"])+np.asarray(raw["pending"])
                    if not bool(dec.max_green_mask_required):
                        at_switch.append((q[dec.action] > 0, q[dec.action] == max(q[i] for i in range(3) if mask[i])))
            red_streaks = {}
            for i,d in enumerate(DIRECTIONS):
                red_streaks[d] = max((len(list(gp)) for k,gp in groupby(
                    ((df.phase!=i)|(df.stage!="green")) & (df[f"queue_{d}"]>0)) if k), default=0)
            audits.append(dict(geometry=g, scenario=scenario, seed=seed, controller=c, n_verified_seconds=HORIZON,
                voluntary_switches_measured=len(at_switch),
                voluntary_to_queued=sum(a for a,_ in at_switch), voluntary_to_largest=sum(b for _,b in at_switch),
                mean_green_run_s=np.mean([dur for _,s,dur in runs[:-1] if s=="green"]),
                clearance_pct=100*sum(s!="green" for s in stage)/HORIZON,
                **{f"max_queued_without_green_{d}_s":red_streaks[d] for d in DIRECTIONS}))
            if scenario=="balanced":
                windows = [(0,360),(360,720),(720,1080)]
            else:
                windows = [(lo,hi) for lo,hi,*_ in SCENARIOS[scenario]]
            for j,(lo,hi) in enumerate(windows):
                window=df.iloc[lo:hi]
                greens = {f"green_pct_{d}":100*((window.phase==i)&(window.stage=="green")).mean()
                          for i,d in enumerate(DIRECTIONS)}
                block_metrics={}
                for d in DIRECTIONS:
                    local_delays = [delays[k] for k,v in expected.items() if lo<=v["requested_depart"]<hi and v["origin"]==d]
                    block_metrics[f"request_delay_{d}_s"] = np.mean(local_delays) if local_delays else np.nan
                    block_metrics[f"mean_queue_{d}"] = window[f"queue_{d}"].mean()
                blocks.append(dict(geometry=g, scenario=scenario, seed=seed, controller=c, block=j,start=lo,end=hi,
                                   **greens,**block_metrics))
    audit=pd.DataFrame(audits)
    audit.to_csv(HERE/"signal_audit.csv",index=False)
    pd.DataFrame(requests).to_csv(HERE/"demand_blocks.csv",index=False)
    block=pd.DataFrame(blocks)
    block.to_csv(HERE/"signal_blocks.csv",index=False)
    ids=["geometry","scenario","seed","controller"]
    merged=frame.merge(audit,on=ids,validate="one_to_one")
    def with_rl(data,grouping):
        base=data[data.controller.isin(BASELINES)]
        rl=data[data.controller.isin(MODELS)].groupby(grouping).mean(numeric_only=True).reset_index()
        rl["controller"]="direct"
        return pd.concat([rl,base],ignore_index=True)
    dat=with_rl(merged,ids[:-1])
    idx=["geometry","scenario","controller"]
    data=dat.groupby(idx).mean(numeric_only=True).drop(columns="seed")
    data.to_csv(HERE/"scenario_summary.csv")
    blk=with_rl(block,[*ids[:-1],"block"]).groupby([*idx,"block"]).mean(numeric_only=True).drop(columns="seed")
    blk.to_csv(HERE/"block_summary.csv")
    paired=[]
    def summarize_diff(diff):
        n=len(diff)
        half=t.ppf(.975,n-1)*diff.std(ddof=1)/np.sqrt(n)
        return dict(n_pairs=n, difference=diff.mean(), lower95=diff.mean()-half, upper95=diff.mean()+half,
                    lower_count=int((diff<0).sum()))
    for (g, s), group in dat.groupby(["geometry","scenario"]):
        for m in ["cohort_delay_s","completion_rate","worst_direction_cohort_delay_s","p95_cohort_delay_s"]:
            wide=group.pivot(index="seed",columns="controller",values=m)
            for b in BASELINES:
                paired.append(dict(geometry=g,scenario=s,metric=m,comparison=f"direct_minus_{b}",
                                   **summarize_diff(wide.direct-wide[b])))
    for (s,c), group in dat.groupby(["scenario","controller"]):
        wide=group.pivot(index="seed",columns="geometry",values="cohort_delay_s")
        paired.append(dict(geometry="new-source",scenario=s,metric="cohort_delay_s",comparison=c,
                           **summarize_diff(wide["new"]-wide.source)))
    pd.DataFrame(paired).to_csv(HERE/"paired.csv",index=False)
    brief_cols=["cohort_delay_s","completion_rate","mean_queue_veh","p95_cohort_delay_s","worst_direction_cohort_delay_s",
                "cohort_delay_W_s","cohort_delay_E_s","cohort_delay_S_s","clearance_pct","signal_switches",
                "forced_switches","collision_events","teleport_events"]
    print("SCENARIO SUMMARY\n",data[brief_cols].round(3).to_string())
    print("\nMACRO NEW\n",data.loc["new"][brief_cols].groupby("controller").mean().round(4).to_string())
    print("\nPAIRED DELAY\n",pd.DataFrame(paired).query("metric=='cohort_delay_s' and geometry != 'source'").round(4).to_string(index=False))
    print("\nNEW GREEN BLOCKS\n",blk.loc["new"][["green_pct_W","green_pct_E","green_pct_S",
          "mean_queue_W","mean_queue_E","mean_queue_S","request_delay_S_s"]].round(2).to_string())
    print("\nNEW SUMMARY AUDIT\n",audit[audit.geometry=="new"].drop(columns="seed").groupby("controller").sum(numeric_only=True).to_string())
    print("\nCOUNTS\n",frame.groupby("geometry")[["requested_vehicles","collision_events","teleport_events"]].sum())
    print("Native XML recalculation and second-by-second signal checks passed:",len(frame),len(frame)*HORIZON)
if __name__=="__main__":
    main()
