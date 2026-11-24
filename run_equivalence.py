"""Equivalence (TOST) and robustness analyses for the sensor-sizing claims.

Adds three things the first submission lacked:
  1. pre-specified-margin equivalence tests (TOST) on the per-subject paired
     MAE differences, so "not significant" becomes "equivalent within +/-D"
     where the data support it;
  2. a success-conditioned robustness check: the sizing law re-estimated on
     the windows where the baseline pipeline actually succeeds (|err|<=5 bpm),
     answering "is the sizing law an artifact of a failing baseline?";
  3. per-activity sizing tables.

Margins are pre-specified as design-irrelevant fractions of each cohort's MAE
scale: 2.0 bpm for Wrist (~6% of 34 bpm) and 0.5 bpm for BIDMC (~6% of 8.7 bpm).
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
MARGIN_WRIST = 2.0
MARGIN_BIDMC = 0.5
SUCCESS_BPM = 5.0


def per_subject(rows, cfg):
    acc = {}
    for r in rows:
        acc.setdefault(r['subject'], []).append(r['err'][cfg])
    return {s: float(np.mean(v)) for s, v in acc.items()}


def tost(d, margin):
    d = np.asarray(d, float)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    t1 = (m + margin) / se          # H0: mean <= -margin
    t2 = (m - margin) / se          # H0: mean >= +margin
    p1 = float(stats.t.sf(t1, df))
    p2 = float(stats.t.cdf(t2, df))
    tc = stats.t.ppf(0.95, df)
    return {'n_subjects': n, 'mean_diff': float(m), 'sd_diff': float(sd),
            'ci90': [float(m - tc * se), float(m + tc * se)],
            'p_lower': p1, 'p_upper': p2,
            'p_tost': max(p1, p2),
            'equivalent_at_margin': bool(max(p1, p2) < ALPHA),
            'margin_bpm': margin}


def paired(rows, cfg, ref):
    a, b = per_subject(rows, cfg), per_subject(rows, ref)
    subs = sorted(set(a) & set(b))
    return [a[s] - b[s] for s in subs]


def mae(rows, cfg):
    return float(np.mean([r['err'][cfg] for r in rows]))


def main():
    w = json.load(open(os.path.join(HERE, 'wrist_windows_cache.json')))
    b = json.load(open(os.path.join(HERE, 'bidmc_windows_cache.json')))
    out = {'protocol': {'alpha': ALPHA, 'margin_wrist_bpm': MARGIN_WRIST,
                        'margin_bidmc_bpm': MARGIN_BIDMC,
                        'success_threshold_bpm': SUCCESS_BPM,
                        'wrist_sizing_reference': 'ch_acc_gyro (native 256 Hz, 16-bit)',
                        'bidmc_sizing_reference': 'native (125 Hz, 16-bit)',
                        'test': 'two one-sided t-tests on per-subject paired MAE '
                                'differences; equivalence iff both p < alpha '
                                '(equivalently 90% CI inside +/-margin)'}}

    ch = ['ch_acc_wide', 'ch_acc_ln', 'ch_gyro', 'ch_acc_gyro', 'ch_acc_gyro_mag']
    out['channel_equivalence_wrist'] = {
        c: tost(paired(w, c, 'ch_full'), MARGIN_WRIST) for c in ch}

    szw = ['sz_fs64', 'sz_fs32', 'sz_fs16', 'sz_b8', 'sz_b10', 'sz_b12']
    out['sizing_equivalence_wrist'] = {
        c: tost(paired(w, c, 'ch_acc_gyro'), MARGIN_WRIST) for c in szw}
    szb = ['fs64', 'fs32', 'fs16', 'b8', 'b10', 'b12']
    out['sizing_equivalence_bidmc'] = {
        c: tost(paired(b, c, 'native'), MARGIN_BIDMC) for c in szb}

    # success-conditioned robustness: does the sizing law survive where the
    # baseline pipeline works?
    rob = {}
    for name, rows, ref, cfgs in (
            ('wrist', w, 'ch_acc_gyro', ['ch_acc_gyro', 'sz_fs32', 'sz_fs16', 'sz_b8']),
            ('bidmc', b, 'native', ['native', 'fs32', 'fs16', 'b8'])):
        good = [r for r in rows if r['err'][ref] <= SUCCESS_BPM]
        block = {'n_windows_all': len(rows), 'n_windows_success': len(good),
                 'success_fraction': len(good) / len(rows), 'configs': {}}
        for c in cfgs:
            block['configs'][c] = {
                'mae_on_success_subset': mae(good, c),
                'mae_all': mae(rows, c),
                'success_fraction': float(np.mean(
                    [r['err'][c] <= SUCCESS_BPM for r in rows]))}
        rob[name] = block
    out['success_conditioned_robustness'] = rob

    act = {}
    for a in sorted({r['act'] for r in w}):
        rows = [r for r in w if r['act'] == a]
        act[a] = {'n_windows': len(rows),
                  **{c: mae(rows, c) for c in ['ch_acc_gyro'] + szw}}
    out['sizing_by_activity_wrist'] = act

    with open(os.path.join(HERE, 'equivalence.json'), 'w') as fh:
        json.dump(out, fh, indent=1)

    print('== channel equivalence (wrist, margin %.1f) ==' % MARGIN_WRIST)
    for c, v in out['channel_equivalence_wrist'].items():
        print('%-16s d=%+.2f ci90[%+.2f,%+.2f] p_tost=%.3f equiv=%s'
              % (c, v['mean_diff'], v['ci90'][0], v['ci90'][1], v['p_tost'],
                 v['equivalent_at_margin']))
    print('== sizing equivalence wrist ==')
    for c, v in out['sizing_equivalence_wrist'].items():
        print('%-10s d=%+.2f p_tost=%.3f equiv=%s'
              % (c, v['mean_diff'], v['p_tost'], v['equivalent_at_margin']))
    print('== sizing equivalence bidmc (margin %.1f) ==' % MARGIN_BIDMC)
    for c, v in out['sizing_equivalence_bidmc'].items():
        print('%-8s d=%+.3f p_tost=%.4f equiv=%s'
              % (c, v['mean_diff'], v['p_tost'], v['equivalent_at_margin']))
    print('== success-conditioned ==')
    for k, v in rob.items():
        print(k, 'success n=%d/%d' % (v['n_windows_success'], v['n_windows_all']))
        for c, cc in v['configs'].items():
            print('   %-12s mae_success=%.2f mae_all=%.2f succ=%.3f'
                  % (c, cc['mae_on_success_subset'], cc['mae_all'],
                     cc['success_fraction']))


if __name__ == '__main__':
    main()
