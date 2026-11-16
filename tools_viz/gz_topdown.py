import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import font_manager
ko=[f.fname for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
if ko: font_manager.fontManager.addfont(ko[0]); plt.rcParams["font.family"]=font_manager.FontProperties(fname=ko[0]).get_name()
X0,X1,Y0,Y1=-0.15,3.61,-0.15,2.45
L,W=0.24,0.17
COL=["#2a78d6","#eb6834","#1baa6c","#8a5cd1","#d6a12a","#d33d6b"]
def car_poly(x,y,yaw):
    c,s=np.cos(np.radians(yaw)),np.sin(np.radians(yaw))
    pts=np.array([[L/2,W/2],[L/2,-W/2],[-L/2,-W/2],[-L/2,W/2]])
    return pts@np.array([[c,s],[-s,c]])+[x,y]
def draw(ax, run, t, tex, highlight=(), trail=0.0, zoom=None, title=""):
    p=pd.read_csv(run+"/poses.csv")
    img=plt.imread(tex)
    ax.imshow(img, extent=(X0,X1,Y0,Y1), origin="upper", zorder=0)
    cars=sorted(p.car.unique())
    for i,c in enumerate(cars):
        q=p[p.car==c]
        if trail>0:
            tr=q[(q.sim_s>=t-trail)&(q.sim_s<=t)]
            ax.plot(tr.x,tr.y,color=COL[i%6],lw=1.5,alpha=.6,zorder=1)
        r=q.iloc[(q.sim_s-t).abs().argmin()]
        lw=2.5 if c in highlight else 1
        ax.add_patch(Polygon(car_poly(r.x,r.y,r.yaw_deg),closed=True,fc=COL[i%6],ec="#d10000" if c in highlight else "#222",lw=lw,zorder=3))
        cc,ss=np.cos(np.radians(r.yaw_deg)),np.sin(np.radians(r.yaw_deg))
        ax.plot([r.x,r.x+0.1*cc],[r.y,r.y+0.1*ss],color="white",lw=1.5,zorder=4)
        ax.annotate(c,(r.x,r.y),xytext=(0,14),textcoords="offset points",ha="center",fontsize=8,zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=.85))
    if zoom: ax.set_xlim(zoom[0],zoom[1]); ax.set_ylim(zoom[2],zoom[3])
    else: ax.set_xlim(X0,X1); ax.set_ylim(Y0,Y1)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title,fontsize=11,loc="left")
    return p
mode=sys.argv[1]
if mode=="fail":
    run="/mnt/user-data/uploads/picar_gz_runs/acceptance_20261112_005751/seed1"
    p=pd.read_csv(run+"/poses.csv"); t=67.32
    r=p[(p.car.isin(["car0","car2"]))]; r=r.iloc[[(r[r.car==c].sim_s-t).abs().idxmin() for c in ["car0","car2"]]] if False else None
    q=p[p.car.isin(["car0","car2"])]; pts=q.iloc[(q.sim_s-t).abs().argsort()[:2]]
    cx,cy=pts.x.mean(),pts.y.mean()
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={"width_ratios":[1.45,1]})
    draw(axes[0],run,t,run+"/course_texture.png",("car0","car2"),trail=4,title=f"Gazebo 기록 재구성 · 6대 · 시뮬레이션 {t:.0f}초")
    axes[0].add_patch(plt.Rectangle((cx-0.4,cy-0.32),0.8,0.64,fill=False,ec="#d10000",lw=1.5,ls="--",zorder=6))
    draw(axes[1],run,t,run+"/course_texture.png",("car0","car2"),trail=4,zoom=(cx-0.4,cx+0.4,cy-0.32,cy+0.32),title="확대: 두 차가 맞닿음 (빨간 테두리)")
    fig.suptitle("수정 전 실패 사례: 코너를 빠져나오며 마주 오는 차와 접촉 (2분 주행 중 충돌 29회, 차로 이탈 9회)",x=0.01,ha="left",fontsize=12)
    fig.tight_layout(); fig.savefig("/home/claude/notion_img/gazebo_fail_contact.png",dpi=130)
else:
    run="/mnt/user-data/uploads/picar_gz_runs/acceptance_20261117_005624/seed1_dqn"
    fig,ax=plt.subplots(figsize=(9,6.4))
    t=float(sys.argv[2]) if len(sys.argv)>2 else 300
    draw(ax,run,t,run+"/course_texture.png",trail=5,title=f"Gazebo 기록 재구성 · 가상 실차 6대 · v18 제어 · 시뮬레이션 {t:.0f}초 (꼬리선 = 최근 5초 경로)")
    fig.tight_layout(); fig.savefig("/home/claude/notion_img/gazebo_v18_6cars.png",dpi=130)
