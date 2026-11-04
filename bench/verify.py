#!/usr/bin/env python3
"""Check that the compiled bit-sliced kernel computes exactly what the weights say.

    python bench/verify.py LANG PACK [N_FILES]

Runs the dense numpy executor (model/numpy_ref.py) on a deterministic sample of files from
PACK and compares it byte for byte with the kernel's predictions (PACK.pred, written by
`taxnet LANG --eval PACK -o PACK.pred`) and with the reference labels.
"""
import json, os, random, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))
import numpy_ref  # noqa: E402

lang, base = sys.argv[1], sys.argv[2]
nfiles = int(sys.argv[3]) if len(sys.argv) > 3 else 200
idx = json.load(open(base + ".idx.json"))["files"]
src, lbl, pred = (open(base + ext, "rb").read() for ext in (".src", ".lbl", ".pred"))
rng = random.Random(1234)
small = [f for f in idx if f[2] <= 16384]          # the dense executor is slow; keep files short
sample = rng.sample(small, min(nfiles, len(small)))
net = numpy_ref.Net(os.path.join(ROOT, "weights", lang + ".tnw"))
sh = net.shapes()
print("%s: dense layers fwd %s, bwd %s; %d dense parameters" % (lang, sh["fwd_layers"], sh["bwd_layers"], sh["dense_params"]))
total = diff_kernel = diff_ref = 0
t0 = time.time()
sample.sort(key=lambda f: f[2])
for i in range(0, len(sample), 64):                       # batches of similar length
    group = sample[i:i + 64]
    outs = net.run([src[o:o + n] for _, o, n in group])
    for (_, o, n), y in zip(group, outs):
        y = y.tobytes()
        total += n
        diff_kernel += sum(a != b for a, b in zip(y, pred[o:o + n]))
        diff_ref += sum(a != b for a, b in zip(y, lbl[o:o + n]))
print("%d files, %d bytes in %.0f s: numpy vs kernel: %d bytes differ; numpy vs reference labels: %d differ"
      % (len(sample), total, time.time() - t0, diff_kernel, diff_ref))
