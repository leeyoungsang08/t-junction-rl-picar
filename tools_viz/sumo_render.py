import sys
import numpy as np
import sumolib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager
ko=[f.fname for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
font_manager.fontManager.addfont(ko[0]); plt.rcParams["font.family"]=font_manager.FontProperties(fname=ko[0]).get_name()
from picar_sim.env import PicarEnv
from picar_sim.map_builder import load_config, maps_dir
from picar_sim.dqn import Policy
from picar_sim.observation import Encoder

def lane_poly(shape, w):
    s=np.array(shape); out=[]
    left=[];right=[]
    for i in range(len(s)):
        a=s[max(i-1,0)]; b=s[min(i+1,len(s)-1)]; d=b-a; n=np.array([-d[1],d[0]])/(np.hypot(*d)+1e-9)
        left.append(s[i]+n*w/2); right.append(s[i]-n*w/2)
    return np.array(left+right[::-1])

def render(ax, mp, cars, bias, steps, title, seed=3100):
    cfg=load_config(); cfg["signal"]["max_wait_guard_s"]=20
    env=PicarEnv(cfg, control_mode="direct", map_name=mp)
    pol=Policy("runs/picar_v18_mix_seg_norm/seed42/best_policy.npz")
    enc=Encoder(cfg["signal"]["max_green_s"], cfg["signal"]["clear_timeout_s"], pol.meta["schema"])
    obs,info=env.reset(seed=seed,cars=cars,bias=bias)
    for _ in range(steps):
        obs,r,te,tr,info=env.step(pol.act(env._obs(enc),info["mask"]))
        if te or tr: break
    k=env.k
    net=sumolib.net.readNet(str(maps_dir(mp)/"picar_course.net.xml"), withInternal=True)
    for e in net.getEdges():
        for ln in e.getLanes():
            internal=e.getID().startswith(":")
            ax.add_patch(Polygon(lane_poly(ln.getShape(), ln.getWidth()), closed=True,
                                 fc="#d9d8d4" if internal else "#8f8e8a", ec="none", zorder=1 if not internal else 0.5))
    # stop lines: end of incoming lanes to J / J2, coloured by permit state
    tls={jn:k.trafficlight.getRedYellowGreenState(jn) for jn in ("J","J2")}
    for jn in ("J","J2"):
        links=k.trafficlight.getControlledLinks(jn); st=tls[jn]
        seen=set()
        for i,lk in enumerate(links):
            if not lk: continue
            ln=lk[0][0]
            if ln in seen: continue
            seen.add(ln)
            sh=np.array(k.lane.getShape(ln)); p=sh[-1]; d=sh[-1]-sh[-2]; d/=np.hypot(*d); n=np.array([-d[1],d[0]])
            w=k.lane.getWidth(ln); c=st[i].lower()
            col={"g":"#1baa6c","y":"#e0b000","r":"#d33d3d"}.get(c,"#555")
            ax.plot([p[0]+n[0]*w/2,p[0]-n[0]*w/2],[p[1]+n[1]*w/2,p[1]-n[1]*w/2],color=col,lw=4 if mp=="real" else 3,solid_capstyle="butt",zorder=3)
        jx,jy=net.getNode(jn).getCoord()
        ax.annotate(jn+(" (DQN)" if jn=="J" else " (규칙)"),(jx,jy),fontsize=9,ha="center",va="center",zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="#999",alpha=.9))
    L=0.24; W=0.15
    for vid in k.vehicle.getIDList():
        x,y=k.vehicle.getPosition(vid); ang=np.radians(90-k.vehicle.getAngle(vid)); spd=k.vehicle.getSpeed(vid)
        c,s=np.cos(ang),np.sin(ang)
        # SUMO position is the front bumper
        pts=np.array([[0,W/2],[0,-W/2],[-L,-W/2],[-L,W/2]])@np.array([[c,s],[-s,c]])+[x,y]
        ax.add_patch(Polygon(pts,closed=True,fc="#2a78d6" if spd>0.02 else "#eb6834",ec="#1a1a1a",lw=0.6,zorder=4))
    xs=[p[0] for e in net.getEdges() for ln in e.getLanes() for p in ln.getShape()]
    ys=[p[1] for e in net.getEdges() for ln in e.getLanes() for p in ln.getShape()]
    pad=0.25; ax.set_xlim(min(xs)-pad,max(xs)+pad); ax.set_ylim(min(ys)-pad,max(ys)+pad)
    ax.set_aspect("equal"); ax.set_facecolor("#fcfcfb"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color("#ddd")
    nv=len(k.vehicle.getIDList()); stopped=sum(k.vehicle.getSpeed(v)<=0.02 for v in k.vehicle.getIDList())
    ax.set_title(f"{title} · 차량 {nv}대 (정지 {stopped}) · {env.t:.0f}초", fontsize=11, loc="left")
    env.close()

which=sys.argv[1]
if which=="real":
    fig,ax=plt.subplots(figsize=(9,6.6))
    render(ax,"real",6,"uniform",90,"SUMO · 실제 PiCar 코스 (3.46×2.30 m)")
    out="/home/claude/notion_img/sumo_real_6cars.png"
else:
    fig,ax=plt.subplots(figsize=(11,7.6))
    render(ax,"large",40,"west_heavy",150,"SUMO · 큰 맵 (도로 2.5배, 학습·스트레스 테스트 전용) · 서쪽 쏠림 수요")
    out="/home/claude/notion_img/sumo_large_40cars.png"
fig.text(0.01,0.01,"파랑 = 주행 중, 주황 = 정지·대기 / 정지선 색 = 신호 허가 (초록 GO, 노랑 황색, 빨강 STOP)",fontsize=9,color="#52514e")
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(out,dpi=130)
print(out)
