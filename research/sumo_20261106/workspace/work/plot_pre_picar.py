from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'outputs'
d=pd.read_csv(root/'SUMO_PiCar_전_균형선택_새시험/results.csv')
g=d.groupby(['sensor','controller']).cohort_delay_s.mean()
sensors=['limited_clean','limited_delay_miss','limited_outage']
labels=['Clean count','20% miss + delay','20% miss + delay + outage']
x=np.arange(3);w=.19
fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')
for j,(name,key,color) in enumerate([('Existing model','old','#2176ae'),('Balanced candidate 43','new43','#e68a2e'),('Queue rule','queue','#42a779')]):
    ax.bar(x+(j-1.5)*w,[g.loc[s,key] for s in sensors],w,label=name,color=color)
ax.bar(x+1.5*w,[g.loc[s,'fixed_s30'] for s in sensors],w,label='Tuned fixed plan',color='#9a79b5')
ax.set_xticks(x,labels);ax.set_ylabel('Mean cohort delay (simulation seconds)')
ax.set_title('SUMO comparison: sensor limitations and candidate 43')
ax.set_ylim(0,100)
ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True);ax.legend(ncol=2,frameon=True,framealpha=1,loc='upper right')
fig.text(.01,-.02,'2 test maps × 4 known traffic scenarios × 3 demand seeds; simulated sensor assumptions.',fontsize=9)
fig.savefig(root/'SUMO_PiCar_전_센서별_비교.png',dpi=180,bbox_inches='tight')
