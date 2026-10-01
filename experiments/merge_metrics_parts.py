"""Merge the part files bench_year_resolution.py writes when METRICS_PART is set, into the
single metrics.npz, big_rivers_daily.npz and summary.json the figures read.
Usage: merge_metrics_parts.py <dir> [part names, default every summary_*.json in the dir]
"""
import os, sys, glob, json
import numpy as np

D = sys.argv[1].rstrip('/')
parts = sys.argv[2:] or sorted(os.path.basename(f)[len('summary_'):-len('.json')]
                               for f in glob.glob(D + '/summary_*.json'))
if not parts:
    sys.exit('no part files in ' + D)
print('parts:', ', '.join(parts))

summary, res, series = None, {}, {}
for p in parts:
    s = json.load(open('%s/summary_%s.json' % (D, p)))
    if summary is None:
        summary = dict(s)
    else:
        assert s['eval_cells'] == summary['eval_cells'], (p, 'a different evaluation set')
        summary['k'].update(s['k'])
    m = np.load('%s/metrics_%s.npz' % (D, p))
    b = np.load('%s/big_rivers_daily_%s.npz' % (D, p))
    for src, dst in ((m, res), (b, series)):
        for key in src.files:
            if key not in dst:
                dst[key] = src[key]
    print('  %-6s k = %s' % (p, ','.join(sorted(s['k'], key=int))))

summary['k'] = {k: summary['k'][k] for k in sorted(summary['k'], key=int)}
np.savez_compressed(D + '/metrics.npz', **res)
np.savez_compressed(D + '/big_rivers_daily.npz', **series)
json.dump(summary, open(D + '/summary.json', 'w'), indent=2)
ks = sorted(summary['k'], key=int)
print('merged %d iteration counts: %s' % (len(ks), ','.join(ks)))
for k in ks:
    v = summary['k'][k]
    print('k=%-3s KGE median %.5f  5th pct %.4f  NSE median %.5f  max|bias| %.4f'
          % (k, v['kge_median'], v['kge_p5'], v['nse_median'], v['bias_absmax']))
chosen = [k for k in ks if summary['k'][k]['kge_p5'] >= 0.99]
print('first k with 5th-pct KGE >= 0.99:', chosen[0] if chosen else 'none in this list')
