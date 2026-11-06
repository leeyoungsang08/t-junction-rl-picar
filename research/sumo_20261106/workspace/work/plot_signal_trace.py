from pathlib import Path
import io,zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'outputs'
base=root/'SUMO_PiCar_전_균형선택_새시험'/'runs'/'balanceA'/'heldout_reversal'/'70980'/'limited_delay_miss'
fig,axes=plt.subplots(3,1,figsize=(12,7.2),sharex=True,layout='constrained',height_ratios=(1,1,1.2))
colors=['#2878b5','#e1842b','#42a16d']
for j,name in enumerate(('old','new43')):
    with zipfile.ZipFile(base/name/'raw.zip') as z:
        data=pd.read_csv(io.BytesIO(z.read('timeseries.csv')))
    data=data[data.time<=400]
    # Transparent gaps are yellow/all-red clearance stages.
    for d,label in enumerate('WES'):
        y=np.where((data.phase==d)&(data.stage=='green'),d,np.nan)
        axes[j].plot(data.time,y,drawstyle='steps-post',color=colors[d],linewidth=3,label=label)
    axes[j].set_yticks([0,1,2],['W','E','S']);axes[j].set_ylabel('Green direction')
    axes[j].set_title('Existing policy' if name=='old' else 'Balanced-selection candidate 43',loc='left')
    axes[j].grid(axis='x',alpha=.2)
    if name=='old': axes[2].plot(data.time,data.queue_total,label='Existing policy',color=colors[0])
    else: axes[2].plot(data.time,data.queue_total,label='Candidate 43',color=colors[1],alpha=.8)
axes[2].set_ylabel('Stopped vehicles');axes[2].set_xlabel('SUMO simulation time (s)');axes[2].legend(frameon=False)
axes[2].grid(alpha=.2);fig.suptitle('Observed signal and queue trace: balanceA, reversal, seed 70980')
fig.text(.01,-.02,'First 400 seconds; delay + missed detection. Gaps in green indicate yellow/all-red clearance.',fontsize=9)
fig.savefig(root/'SUMO_PiCar_새교차로_신호변화.png',dpi=180,bbox_inches='tight')
