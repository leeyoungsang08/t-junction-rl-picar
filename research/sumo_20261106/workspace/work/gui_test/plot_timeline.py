from pathlib import Path
from itertools import groupby
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
root=Path('R:/gui_test')
out=Path.cwd()/'outputs'
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams['font.family']=font_manager.FontProperties(fname='C:/Windows/Fonts/malgun.ttf').get_name()
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(11,3.5))
items=[('trained','학습モデル'),('max_pressure_proxy','대기열 기준'),('fixed_time','20초 고정식')]
colors=['#398dc8','#ec895b','#786cc2']
for y,(name,label) in enumerate(items):
 d=pd.read_csv(root/'runs'/name/'timeseries.csv')
 for (phase,stage),g in groupby(zip(d.phase,d.stage)):
  vals=list(g)
  if 'pos' not in locals(): pos=0
  n=len(vals)
  ax.barh(y,n,left=pos,height=.52,color=colors[phase] if stage=='green' else '#aeb8c1')
  pos+=n
 assert pos==900
 del pos
ax.axvline(198,color='#333333',lw=.8,ls='--')
ax.axvline(495,color='#333333',lw=.8,ls='--')
ax.set_yticks(range(3),[x[1] for x in items]);ax.invert_yaxis()
ax.set_xlim(0,900);ax.set_xticks([0,198,495,900])
ax.set_xlabel('시뮬레이션 시간 (초)')
ax.set_title('교통량이 몰리는 방향: 남쪽 → 동쪽 → 서쪽',loc='left',fontweight='bold')
ax.legend(handles=[Patch(facecolor=colors[i],label=x+' 녹색') for i,x in enumerate(('서쪽','동쪽','남쪽'))]+[Patch(facecolor='#aeb8c1',label='황색·전적색')],ncol=4,loc='upper center',bbox_to_anchor=(.5,-.3),frameon=False)
for sp in ('top','right','left'):ax.spines[sp].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.tight_layout()
fig.savefig(out/'T_Junction_SUMO_GUI_Signal_Timeline.png',dpi=150,bbox_inches='tight')
