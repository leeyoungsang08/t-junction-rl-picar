from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]/'outputs';folder=root/'SUMO_balanceA_독립시험_20261105'
df=pd.read_csv(folder/'results.csv');plan=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
means=df.groupby(['sensor','controller']).cohort_delay_s.mean()
sensors=['limited_clean','limited_delay_miss','limited_outage','severe_unseen']
labels=['Clean\ncounts','20% missed\n+ delay','20% missed + delay\n+ outage','40% missed + delay\n+ long outage']
candidate=('gated','Validated gate') if plan['gate'] else ('explore_plain44','Rejected candidate (diagnostic)')
controllers=[('old','Existing model','#2676ae'),('prior43','Earlier candidate','#e88d2d'),(candidate[0],candidate[1],'#9d6db1'),('queue','Queue rule','#43a174')]
x=np.arange(4);w=.19;fig,ax=plt.subplots(figsize=(10.5,5.6),layout='constrained')
for i,(key,name,color) in enumerate(controllers):ax.bar(x+(i-1.5)*w,[means.loc[s,key] for s in sensors],w,label=name,color=color)
ax.set_xticks(x,labels);ax.set_ylabel('Mean cohort delay (simulation seconds)');ax.set_title('One fixed T-junction: held-out demand and synthetic sensors')
ax.set_ylim(0,max(means.loc[(s,c)].item() for s in sensors for c,_,_ in controllers)*1.35)
ax.legend(frameon=True,framealpha=1,ncol=2,loc='upper left');ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)
fig.text(.01,-.02,'6 traffic scenarios × 3 demand seeds per sensor; the severe sensor condition is synthetic.',fontsize=9)
fig.savefig(root/'SUMO_balanceA_조건별_비교.png',dpi=180,bbox_inches='tight')
