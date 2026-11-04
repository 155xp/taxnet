#!/usr/bin/env python3
"""Show where the net disagrees with the reference: python bench/errors.py PACK [PRED] [-n N] [--cls REF:NET]"""
import collections, json, sys, bisect
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/ref')
from spec import CLASSES
args, skip = [], False
for a in sys.argv[1:]:
    if skip: skip = False; continue
    if a in ('-n', '--cls'): skip = True; continue
    args.append(a)
base = args[0]
pred = open(args[1] if len(args) > 1 else '/tmp/taxnet_pred.bin', 'rb').read()
N = int(sys.argv[sys.argv.index('-n') + 1]) if '-n' in sys.argv else 12
filt = sys.argv[sys.argv.index('--cls') + 1].split(':') if '--cls' in sys.argv else None
src = open(base + '.src', 'rb').read(); lbl = open(base + '.lbl', 'rb').read()
idx = json.load(open(base + '.idx.json'))['files']
offs = [f[1] for f in idx]
LET = '.kTbfiNSCo#'
byfile = collections.Counter(); kinds = collections.Counter(); shown = collections.Counter()
i, n = 0, len(src)
runs = []
while i < n:
    if pred[i] != lbl[i]:
        j = i
        while j < n and pred[j] != lbl[j] and lbl[j] == lbl[i] and pred[j] == pred[i]: j += 1
        runs.append((i, j)); i = j
    else: i += 1
for s, e in runs:
    f = bisect.bisect_right(offs, s) - 1
    byfile[idx[f][0]] += e - s
    k = (CLASSES[lbl[s]], CLASSES[pred[s]]); kinds[k] += e - s
print('error runs:', len(runs), ' bytes:', sum(e - s for s, e in runs))
print('top files:', byfile.most_common(8))
for (s, e) in runs:
    k = (CLASSES[lbl[s]], CLASSES[pred[s]])
    if filt and list(k) != filt: continue
    if shown[k] >= N: continue
    shown[k] += 1
    f = bisect.bisect_right(offs, s) - 1
    a, b = max(offs[f], s - 50), min(offs[f] + idx[f][2], e + 30)
    seg = src[a:b]
    print('--- %s -> %s  %s @%d' % (k[0], k[1], idx[f][0], s - offs[f]))
    print('   src ' + repr(seg.decode('utf-8', 'replace')))
    t = lambda arr: ''.join(LET[c] if 32 <= src[a+i] < 127 or src[a+i] in (9,10) else '?' for i, c in enumerate(arr[a:b]))
    print('   ref ' + t(lbl).replace('\n', ' ')); print('   net ' + t(pred).replace('\n', ' '))
