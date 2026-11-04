#!/usr/bin/env python3
"""Run a taxnet weight file the plain way: dense float32 matrices and numpy matmuls.

    python model/numpy_ref.py weights/c.tnw FILE...        -> classes, one digit per byte

This is the reference executor. It knows nothing about bits, gates or chunks: the weight
file is unpacked into dense layer matrices and every byte goes through

    x_t = E[byte_t]
    f_t = layers_fwd(x_t, f_{t-1})        each layer: a = clip(W @ [x_t; f_{t-1}; f_t] + b, 0, 1)
    b_t = layers_bwd(x_t, f_t, b_{t+1})   each layer: a = clip(W @ [x_t; f_t; b_{t+1}; b_t] + b, 0, 1)
    y_t = argmax(W_head @ b_t + c_head)

Many files run side by side as the batch dimension. bench/verify.py compares this against
the compiled kernel byte for byte.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxnet  # noqa: E402

KX, KF, KFP, KB, KBN = range(5)


def layers_of(units, same_kind):
    """Group units into layers: a unit sits one layer above the same-step units it reads."""
    level = []
    for _, _, terms in units:
        deps = [level[j] for k, j, _ in terms if k == same_kind]
        level.append(1 + max(deps) if deps else 0)
    out = {}
    for i, lv in enumerate(level):
        out.setdefault(lv, []).append(i)
    return [out[k] for k in sorted(out)]


class Net:
    def __init__(self, path):
        w = taxnet.load(path)
        self.lang = w["lang"]
        self.E = np.array(w["emb"], dtype=np.float32)                     # 256 x F
        nf_, nu = len(w["feats"]), len(w["cells"]["f"])
        nb = len(w["cells"]["b"])
        self.F, self.NU, self.NB = nf_, nu, nb
        self.h0 = np.array(w["h0"], dtype=np.float32)
        # forward input vector: [x (F) ; f_prev (NU) ; f_cur (NU)]
        self.fwd = []
        for idx in layers_of(w["cells"]["f"], KF):
            W = np.zeros((len(idx), nf_ + 2 * nu), np.float32)
            b = np.zeros(len(idx), np.float32)
            for r, i in enumerate(idx):
                _, bias, terms = w["cells"]["f"][i]
                b[r] = bias
                for k, j, wt in terms:
                    W[r, {KX: 0, KFP: nf_, KF: nf_ + nu}[k] + j] += wt
            self.fwd.append((np.array(idx), W, b))
        # backward input vector: [x (F) ; f (NU) ; b_next (NB) ; b_cur (NB)]
        self.bwd = []
        for idx in layers_of(w["cells"]["b"], KB):
            W = np.zeros((len(idx), nf_ + nu + 2 * nb), np.float32)
            b = np.zeros(len(idx), np.float32)
            for r, i in enumerate(idx):
                _, bias, terms = w["cells"]["b"][i]
                b[r] = bias
                for k, j, wt in terms:
                    W[r, {KX: 0, KF: nf_, KBN: nf_ + nu, KB: nf_ + nu + nb}[k] + j] += wt
            self.bwd.append((np.array(idx), W, b))
        # head: logit_0 = 0.5, logit_k = class unit k-1
        self.Wh = np.zeros((11, nb), np.float32)
        self.ch = np.zeros(11, np.float32)
        self.ch[0] = 0.5
        for k, ui in enumerate(w["classes"]):
            self.Wh[k + 1, ui] = 1.0

    def shapes(self):
        return {"fwd_layers": [W.shape for _, W, _ in self.fwd], "bwd_layers": [W.shape for _, W, _ in self.bwd],
                "head": self.Wh.shape, "dense_params": sum(W.size + b.size for _, W, b in self.fwd + self.bwd)
                + self.Wh.size + self.ch.size + self.E.size}

    def run(self, seqs):
        """seqs: list of bytes. Returns list of uint8 class arrays."""
        B, T = len(seqs), max(len(s) for s in seqs)
        lens = np.array([len(s) for s in seqs])
        buf = np.zeros((B, T), np.uint8)
        for i, s in enumerate(seqs):
            buf[i, :len(s)] = np.frombuffer(s, np.uint8)
        F, NU, NB = self.F, self.NU, self.NB
        fvals = np.zeros((T, NU, B), np.float32)
        v = np.zeros((F + 2 * NU, B), np.float32)
        v[F:F + NU] = self.h0[:, None]
        for t in range(T):
            v[:F] = self.E[buf[:, t]].T
            v[F + NU:] = 0
            for idx, W, b in self.fwd:
                v[F + NU + idx] = np.clip(W @ v + b[:, None], 0, 1)
            fvals[t] = v[F + NU:]
            v[F:F + NU] = v[F + NU:]                                     # f_t becomes f_{t-1}
        out = np.zeros((B, T), np.uint8)
        u = np.zeros((F + NU + 2 * NB, B), np.float32)
        for t in range(T - 1, -1, -1):
            u[:F] = self.E[buf[:, t]].T
            u[F:F + NU] = fvals[t]
            u[F + NU:F + NU + NB] *= (t < lens - 1)[None, :]            # b_{t+1} = 0 past each end
            u[F + NU + NB:] = 0
            for idx, W, b in self.bwd:
                u[F + NU + NB + idx] = np.clip(W @ u + b[:, None], 0, 1)
            logits = self.Wh @ u[F + NU + NB:] + self.ch[:, None]
            out[:, t] = np.argmax(logits, axis=0)
            u[F + NU:F + NU + NB] = u[F + NU + NB:]                     # b_t becomes b_{t+1}
        return [out[i, :lens[i]] for i in range(B)]


if __name__ == "__main__":
    net = Net(sys.argv[1])
    files = sys.argv[2:]
    res = net.run([open(f, "rb").read() for f in files])
    for r in res:
        sys.stdout.write("".join("0123456789a"[c] for c in r))
