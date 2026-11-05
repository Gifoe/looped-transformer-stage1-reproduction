"""Export all-depth native behavior; no model or new statistical selection."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
rows=lambda n:list(csv.DictReader((O/n).open(encoding='utf8',newline='')))
def main():
 overview=rows('overview_by_depth.csv');binding=rows('binding_overview.csv');stats=rows('question_level_statistics.csv');fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained');fig.suptitle('Fixed SHA pilot n=24; actual native K1-K4; descriptive training/development evidence',fontsize=11);K=[1,2,3,4]
 for c in ['P0','P1','P2','P3']:
  vals=[int(next(r for r in overview if r['condition']==c and int(r['K'])==k)['raw_target_match'])/24*100 for k in K];axs[0,0].plot(K,vals,'o-',label=c)
 axs[0,0].set(title='A  Native target match across option orders',ylabel='Target match (%)',xlabel='Native K',xticks=K);axs[0,0].legend(frameon=False,ncol=2)
 for c in ['P1','P2','P3']:
  vals=[int(next(r for r in binding if r['condition']==c and int(r['K'])==k)['content_equivariant'])/24*100 for k in K];axs[0,1].plot(K,vals,'o-',label=c)
 axs[0,1].set(title='B  Content equivariance relative to P0',ylabel='Questions equivariant (%)',xlabel='Native K',xticks=K);axs[0,1].legend(frameon=False)
 for metric,label in [('RMS_J_Q','Real question'),('RMS_J_N','N/A'),('RMS_J_QminusN','Q-minus-N response')]:
  rr=[next(r for r in stats if r['condition']=='placeholder' and r['metric']==metric and r['K']==str(k)) for k in K];axs[1,0].plot(K,[float(r['mean']) for r in rr],'o-',label=label)
 axs[1,0].set(title='C  Placeholder relative-response magnitude',ylabel='Question-mean RMS J (not benefit)',xlabel='Native K',xticks=K);axs[1,0].legend(frameon=False,fontsize=9)
 for c,label in [('P0','Q / original'),('Q_placeholder','Q / placeholder'),('N_original','N/A / original'),('N_placeholder','N/A / placeholder')]:
  vals=[int(next(r for r in overview if r['condition']==c and int(r['K'])==k)['raw_target_match'])/24*100 for k in K];axs[1,1].plot(K,vals,'o-',label=label)
 axs[1,1].set(title='D  Original ARC-label match without question',ylabel='Original target match (%)',xlabel='Native K',xticks=K);axs[1,1].legend(frameon=False,fontsize=9)
 for ax in axs.flat:ax.grid(alpha=.15)
 fig.savefig(O/'B_native_behavior.png',dpi=180);fig.savefig(O/'B_native_behavior.svg');plt.close(fig)
if __name__=='__main__':main()
