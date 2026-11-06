from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]/'outputs'
df=pd.read_csv(root/'SUMO_balanceA_장시간대기_추가시험/test_results.csv')
order=['old','prior43','watch70','watch90'];names=['Existing','Prior 43','Prior 43\n+ 70s','Prior 43\n+ 90s']
avg=df.groupby('controller').cohort_delay_s.mean();worst=df.groupby('controller').max_consecutive_wait_s.max()
colors=['#2676ae','#e88d2d','#9d6db1','#43a174'];fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
for ax,values,title,ylabel in [(axes[0],[avg[x] for x in order],'Average cohort delay','Simulation seconds'),
                              (axes[1],[worst[x] for x in order],'Largest observed consecutive wait','Simulation seconds')]:
    bars=ax.bar(np.arange(4),values,color=colors)
    ax.set_xticks(np.arange(4),names);ax.set_title(title);ax.set_ylabel(ylabel)
    ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
    ax.bar_label(bars,labels=[f'{v:.1f}' for v in values],padding=3)
fig.suptitle('One T-junction: watchdog trade-off on new demand')
fig.text(.01,-.02,'72 matched conditions; observed maximum is not a guarantee of an individual waiting-time bound.',fontsize=9)
fig.savefig(root/'SUMO_balanceA_대기와지연_비교.png',dpi=180,bbox_inches='tight')
