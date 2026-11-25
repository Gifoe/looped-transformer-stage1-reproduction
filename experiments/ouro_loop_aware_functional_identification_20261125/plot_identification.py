"""Draw editable research figures from the measured mechanism or sealed forecast."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLORS={'T1':'#0F4D92','T2':'#419284','Mixed':'#B64342'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.85,
                     'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})


def export(fig,out,name):
    for ending in ('pdf','svg','png'):
        fig.savefig(out/f'{name}.{ending}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def mechanism(data,out):
    fig, axes=plt.subplots(1,3,figsize=(12.9,3.6),layout='constrained')
    ticks=np.arange(3)
    for j,row in enumerate(data['training']):
        ax=axes[j]
        for i,(name,vals) in enumerate(row['slopes'].items()):
            ax.bar(ticks+(i-1)*.24,vals,width=.21,color=COLORS[name],edgecolor='#222222',linewidth=.45,label=name)
        ax.set_ylim(-.04,3.72)
        ax.set_xticks(ticks,['Direct','After loop','Interaction'])
        ax.set_title(f'{row["name"]}\n{row["n"]:,} eligible training questions',loc='left',fontsize=11)
        ax.text(-.12,1.02,'ab'[j],weight='bold',transform=ax.transAxes,fontsize=13)
    axes[0].set_ylabel('Mean local utility slope')
    axes[1].legend(loc='upper left',ncols=3,fontsize=8,handlelength=.9)
    subset=data['finite_arc_validation']
    ax=axes[2]
    for name,values in subset['operator'].items():
        ax.plot(subset['doses'],values['post_loop'],color=COLORS[name],marker='o',linewidth=1.65,label=name)
        ax.plot(subset['doses'],values['linear_post_loop'],color=COLORS[name],linestyle='--',linewidth=1.05)
    ax.set_xticks([.05,.1,1.0],['0.05','0.10','1.00'])
    ax.set_xscale('log')
    ax.set_xlabel('Fraction of the capped write')
    ax.set_ylabel('Mean post-loop utility gain')
    ax.set_title(f'ARC Validation\n{subset["n"]} sealed diagnostic questions',loc='left',fontsize=11)
    ax.text(-.14,1.02,'c',weight='bold',transform=ax.transAxes,fontsize=13)
    ax.text(.04,.94,'Solid: measured\nDashed: effective-write linearization',ha='left',va='top',transform=ax.transAxes,fontsize=8)
    export(fig,out,'early_mechanism')


def spectrum(plan,out):
    domains=plan['source_order']
    fig,ax=plt.subplots(1,2,figsize=(11.4,4),layout='constrained')
    for i,key in enumerate(('loop','excess')):
        for source in domains:
            values=np.asarray(plan['spectra'][source][key]['top_singular'])
            ax[i].plot(np.arange(1,len(values)+1),values/values[0],label=source.replace('_',' '),marker='.',linewidth=1.35)
        ax[i].set(yscale='log',xlabel='Singular component',ylabel='Singular value / first value',
                  title=('Ordinary function matrix','Loop contrast matrix')[i])
    ax[1].legend(loc='best',fontsize=8)
    export(fig,out,'functional_spectra')
    fig,ax=plt.subplots(1,2,figsize=(10.5,4.1),sharey=True,layout='constrained')
    names=[n.replace('_',' ') for n in domains]
    for j,key in enumerate(('loop','excess')):
        heat=ax[j].imshow(plan['alignment'][key],cmap='RdBu',vmin=-1,vmax=1,interpolation='nearest')
        ax[j].set(xticks=range(len(names)),yticks=range(len(names)),yticklabels=names,
                  title=('Ordinary','Loop contrast')[j]); ax[j].set_xticklabels(names,rotation=43,ha='right')
    fig.colorbar(heat,ax=ax,label='Signed matrix cosine',shrink=.83)
    export(fig,out,'cross_source_alignment')


def forecast(audit,out):
    rows=next(r['rows'] for r in audit['confirmation'] if r['depth']==4 and r['dose']==1. and r['scope']=='leave_source_out')
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.9),sharey=True,layout='constrained')
    for ax,key,label in zip(axes,('ordinary_functional','loop_and_absolute'),('Ordinary projected score','Joint loop score'),strict=True):
        for target in sorted(set(row['target'] for row in rows)):
            selected=[r for r in rows if r['target']==target]
            ax.scatter([r['predicted'][key] for r in selected],[r['actual']['scaled_loop'] for r in selected],
                       label=target.replace('_',' '),s=25,alpha=.85)
        ax.set_xlabel(label)
        ax.axhline(0,color='#767676',linewidth=.7)
    axes[0].set_ylabel('Held-out finite post-loop gain / scale')
    axes[1].legend(fontsize=8,loc='best')
    export(fig,out,'forecast_vs_observed')


def direction_contrast(rows, out):
    full_dose = [row for row in rows if row["dose"] == 1.]
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.35), sharey=True, layout="constrained")
    endpoints = ("Post-loop utility", "Loop interaction", "Candidate accuracy (pp)")
    for endpoint, (ax, title) in enumerate(zip(axes, endpoints, strict=True)):
        multiplier = 100 if endpoint == 2 else 1
        for j, row in enumerate(full_dose):
            mean = multiplier * row["mean"][endpoint]
            lo, hi = [multiplier * v for v in row["domain_question_ci95"][endpoint]]
            ax.errorbar(mean, j, xerr=[[mean - lo], [hi - mean]], fmt="o",
                        color="#175286", markersize=5, capsize=3, linewidth=1.4)
        ax.axvline(0, color="#676767", lw=.8)
        ax.set(yticks=range(len(full_dose)), yticklabels=[f'K{r["depth"]}' for r in full_dose],
               title=title, xlabel="Loop-contrast axis − ordinary axis")
    axes[0].invert_yaxis()
    export(fig, out, "paired_direction_contrast")


def official_results(document, out):
    primary = document["primary"]
    rows = {}
    for row in document["tasks"]:
        key = row["task"]
        if key not in rows and primary[key] in row["paired"]["native"]:
            rows[key] = row
    if set(rows) != primary.keys():
        raise ValueError("Final result figure requires every official primary endpoint")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True, layout="constrained")
    for ax, references in zip(axes, (("native", "fixed", "adaptive"), ("t1", "rank8")), strict=True):
        positions = np.arange(len(rows))
        for j, name in enumerate(references):
            results = [row["paired"][name][primary[key]] for key, row in rows.items()]
            gain = 100 * np.array([r["change"] for r in results])
            interval = 100 * np.array([r["ci95"] for r in results]).T
            offset = (j - (len(references) - 1) / 2) * .20
            color = ("#175286", "#2E8576", "#A54341")[j]
            ax.hlines(positions + offset, interval[0], interval[1], lw=1., color=color)
            ax.plot(gain, positions + offset, "o", ms=3.5, color=color, label=name)
        ax.axvline(0, color="#676767", lw=.8)
        ax.set(yticks=positions, yticklabels=[r.replace("_", " ") for r in rows], xlabel="New − reference (pp)")
        ax.legend(ncols=len(references), fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.12))
    axes[0].invert_yaxis()
    export(fig, out, "official_k4_performance")


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observed',type=Path,default=Path('EARLY_MECHANISM.json'))
    p.add_argument('--forecast',type=Path)
    p.add_argument('--audit',type=Path)
    p.add_argument('--paired-directions',type=Path)
    p.add_argument('--official',type=Path)
    p.add_argument('--out',type=Path,default=Path('figures'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    mechanism(json.loads(args.observed.read_text(encoding='utf-8')),args.out)
    if args.forecast:
        spectrum(json.loads(args.forecast.read_text(encoding='utf-8')),args.out)
    if args.audit:
        forecast(json.loads(args.audit.read_text(encoding='utf-8')),args.out)
    if args.paired_directions:
        direction_contrast(json.loads(args.paired_directions.read_text(encoding='utf-8')), args.out)
    if args.official:
        official_results(json.loads(args.official.read_text(encoding='utf-8')), args.out)
