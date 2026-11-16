import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ko=[f.fname for f in font_manager.fontManager.ttflist if "Noto Sans CJK" in f.name]
font_manager.fontManager.addfont(ko[0]); plt.rcParams["font.family"]=font_manager.FontProperties(fname=ko[0]).get_name()
run="/mnt/user-data/uploads/picar_gz_runs/acceptance_20261112_005751/seed1/"
fig,axes=plt.subplots(2,3,figsize=(11,6.2))
for row,car in enumerate(["car0","car2"]):
    c=pd.read_csv(run+f"frames_{car}.csv"); raw=np.fromfile(run+f"frames_{car}.u8",dtype=np.uint8)
    off=np.concatenate([[0],np.cumsum((c.h*c.w).values)])
    for col,t in enumerate([63.0,65.5,67.3]):
        i=int((c.t-t).abs().argmin()); h,w=int(c.h[i]),int(c.w[i])
        img=raw[off[i]:off[i]+h*w].reshape(h,w)
        ax=axes[row][col]; ax.imshow(img,cmap="gray",vmin=0,vmax=255); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{car} 카메라 · {c.t[i]:.1f}초 · 조향 {c.w_cmd[i]:+.2f}",fontsize=10,loc="left")
fig.suptitle("접촉 직전 두 차의 Gazebo 카메라 화면 (차선 추종 입력, 160×120). 위 car0, 아래 car2",x=0.01,ha="left",fontsize=11)
fig.tight_layout(); fig.savefig("/home/claude/notion_img/gazebo_fail_camera.png",dpi=130)
print(c[["t","mode","zone","edge","v","w_cmd"]].iloc[[int((c.t-67.3).abs().argmin())]])
