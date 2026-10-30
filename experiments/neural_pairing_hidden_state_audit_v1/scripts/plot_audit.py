"""Publication exports from completed reanalyses only; no simulated measurements."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def export(fig, name):
    fig.tight_layout(pad=1.5)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / 'figures' / (name + '.' + ext), dpi=300, facecolor='white')
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.3,
        'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    data = load('published_paired_comparisons.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    colors = ['#0F4D92', '#42949E']
    for j, side in enumerate(['quill', 'ember']):
        rows = [next(r for r in data if r['run'] == run and r['exclusion'] == 'exact_input_overlap'
            and r['task'] == 'all' and r['arm'] == 'pair/' + side and r['comparator'] == 'pair/zero/' + side)
            for run in ['nl23', 'nl27', 'nl28']]
        x = np.arange(3) + (j - .5) * .16
        y = np.array([float(r['delta_pp']) for r in rows])
        lo = np.array([float(r['official_conservative_ci_low_pp']) for r in rows])
        hi = np.array([float(r['official_conservative_ci_high_pp']) for r in rows])
        axes[0].errorbar(x, y, yerr=[y - lo, hi - y], fmt='o', capsize=4, color=colors[j], label=side.title())
    axes[0].axhline(0, color='#767676', lw=1, ls='--')
    axes[0].set(xticks=np.arange(3), xticklabels=['nl23', 'nl27', 'nl28'],
        ylabel='Link ON minus CUT (percentage points)', title='A  Same-checkpoint communication; n = 3,134')
    axes[0].legend()
    strata = [r for r in load('disagreement_analysis.csv') if r['run'] == 'nl23'
              and r['stratum'] in ['only_quill_correct', 'only_ember_correct']]
    labels, rates = [], []
    for group in ['only_quill_correct', 'only_ember_correct']:
        for side in ['quill', 'ember']:
            row = next(r for r in strata if r['stratum'] == group and r['receiver'] == side)
            rate = row['damage_rate'] or row['rescue_rate']
            labels.append(('Only Quill correct' if group == 'only_quill_correct' else 'Only Ember correct')
                          + '\n' + side.title() + (' damaged' if row['damage_rate'] else ' rescued'))
            rates.append(float(rate) * 100)
    axes[1].barh(np.arange(4), rates, color=['#B64342', '#42949E', '#42949E', '#B64342'], edgecolor='black', lw=.7)
    for i, r in enumerate(rates):
        axes[1].text(r + 1, i, f'{r:.1f}%', va='center', fontsize=10)
    axes[1].set(yticks=np.arange(4), yticklabels=labels, xlim=(0, 94),
        xlabel='Fraction within link-cut stratum (%)', title='B  nl23 disagreement outcomes')
    axes[1].invert_yaxis()
    fig.suptitle('Reanalysis of published records; no fresh checkpoint inference', fontsize=13)
    export(fig, 'published_reasoning_audit')
    rows = [r for r in load('published_payload_results.csv') if r['run'] == 'nl33w'
            and r['game'] in ['relay_word', 'relay_number', 'merge_sum', 'lookup']]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    games = ['relay_word', 'relay_number', 'merge_sum', 'lookup']
    for ax, side in zip(axes, ['quill', 'ember']):
        for j, (cond, color) in enumerate([('link', '#0F4D92'), ('cut', '#767676'), ('shuffle', '#B64342')]):
            y = [float(next(r for r in rows if r['game'] == game and r['receiver'] == side)[cond + '_ce_episode_mean']) for game in games]
            ax.bar(np.arange(4) + (j - 1) * .23, y, .23, color=color, edgecolor='black', lw=.6, label=cond.upper())
        ax.set(title=side.title() + ' receiver', xticks=np.arange(4), xticklabels=['Word', 'Number', 'Sum', 'Lookup'])
    axes[0].set_ylabel('Mean payload cross-entropy (nats/token)')
    axes[1].legend()
    fig.suptitle('Published nl33w payload records; 8 episodes/receiver (sum: 16)', fontsize=13)
    export(fig, 'published_transfer_audit')


if __name__ == '__main__':
    main()
