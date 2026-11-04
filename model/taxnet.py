"""taxnet: a bidirectional recurrent network whose weights are written by hand.

Architecture (one network per language, same shape for all three):

    x_t  = E[byte_t]                                   embedding, 256 x F, entries 0/1
    f_t  = cell_fwd(x_t, f_{t-1})                      forward recurrent cell
    b_t  = cell_bwd(x_t, f_t, b_{t+1})                 backward recurrent cell
    y_t  = argmax(W_head . b_t + c_head)               linear head over the class units

Each cell is a stack of layers; every unit in it is a neuron

    u = clip( sum_i w_i * input_i + bias, 0, 1 )

with integer weights and bias. A unit's inputs are embedding features of the current
byte, forward units of the previous step (f_{t-1}), backward units of the next step
(b_{t+1}), or units already computed in the same step. Integer weights on 0/1 inputs
make every pre-activation an integer, so every unit is exactly 0 or 1 and the network
can be evaluated in float32, in int8, or bit-sliced, with identical results.

Nothing here is trained. The builders in lang_*.py write each neuron by hand; the keyword
tables are a trie of "this byte continues that prefix" neurons whose wiring comes
straight from the spec word lists.
"""
import json
import struct

# reference kinds
X, F, FP, B, BN = "x", "f", "f-", "b", "b+"
KIND_CODE = {X: 0, F: 1, FP: 2, B: 3, BN: 4}


class Ref:
    __slots__ = ("kind", "name")

    def __init__(self, kind, name):
        self.kind, self.name = kind, name

    @property
    def prev(self):
        assert self.kind == F, "only forward units have a previous step"
        return Ref(FP, self.name)

    @property
    def next(self):
        assert self.kind == B, "only backward units have a next step"
        return Ref(BN, self.name)

    def __repr__(self):
        return "%s:%s" % (self.kind, self.name)


class Unit:
    __slots__ = ("name", "cell", "terms", "bias")

    def __init__(self, name, cell, terms, bias):
        self.name, self.cell, self.terms, self.bias = name, cell, terms, bias


class Net:
    def __init__(self, lang):
        self.lang = lang
        self.feats = {}          # feature name -> frozenset of byte values
        self.units = {}          # unit name -> Unit
        self.order = {F: [], B: []}
        self.h0 = {}             # forward unit -> value before the first byte
        self.classes = []        # backward unit for each class id >= 1 (class 0 is the default)

    # ---- embedding -------------------------------------------------------------------
    def x(self, name, chars=None):
        if chars is not None:
            vals = frozenset(c if isinstance(c, int) else ord(c) for c in chars)
            assert self.feats.get(name, vals) == vals, name
            self.feats[name] = vals
        assert name in self.feats, name
        return Ref(X, name)

    def eq(self, ch):
        return self.x("eq_" + (ch if ch.isalnum() or ch == "_" else "%02x" % ord(ch)), [ch])

    # ---- neurons ---------------------------------------------------------------------
    def unit(self, cell, name, terms, bias):
        assert name not in self.units, "duplicate unit " + name
        merged = {}
        for ref, w in terms:
            if cell == F:
                assert ref.kind in (X, F, FP), (name, ref)
                assert ref.kind != F or self.units[ref.name].cell == F, (name, ref)
            else:
                assert ref.kind in (X, F, B, BN), (name, ref)
            if ref.kind in (F, B):
                assert ref.name in self.units, "%s uses %s before it is defined" % (name, ref)
            key = (ref.kind, ref.name)
            merged[key] = merged.get(key, 0) + w
        terms = [(Ref(k, n), w) for (k, n), w in merged.items() if w != 0]
        self.units[name] = Unit(name, cell, terms, bias)
        self.order[cell].append(name)
        return Ref(cell, name)

    def AND(self, cell, name, *pos, neg=()):
        """1 iff every `pos` input is 1 and every `neg` input is 0."""
        return self.unit(cell, name, [(r, 1) for r in pos] + [(r, -1) for r in neg], 1 - len(pos))

    def OR(self, cell, name, *pos, neg=()):
        """1 iff some `pos` input is 1 and every `neg` input is 0."""
        k = max(1, len(pos))
        return self.unit(cell, name, [(r, 1) for r in pos] + [(r, -k) for r in neg], 0)

    def ANDOR(self, cell, name, all_of, any_of, neg=()):
        """1 iff every `all_of` input is 1, some `any_of` input is 1, every `neg` is 0."""
        k = len(any_of)
        assert k >= 1
        terms = [(r, k) for r in all_of] + [(r, 1) for r in any_of] + [(r, -k) for r in neg]
        return self.unit(cell, name, terms, -k * len(all_of))

    def NOT(self, cell, name, a):
        return self.unit(cell, name, [(a, -1)], 1)

    # ---- trie of words ---------------------------------------------------------------
    def trie(self, prefix, words, start, minimize=True):
        """Word recognizer. `words` maps word -> set of tags. `start` is a same-step forward
        unit that is 1 on a byte that may begin a word. Returns {tag: unit} where unit is 1
        on the last byte read so far iff the word read since `start` carries that tag.

        Each trie node is one neuron: node(t) = eq_c(t) AND (parent(t-1) OR ...). With
        minimize=True nodes with the same byte and identical futures are shared (a DAWG),
        which keeps the "...Error"/"..._t" tails from being duplicated."""
        root = {"ch": None, "tags": frozenset(), "kids": {}}
        for w, tags in sorted(words.items()):
            n = root
            for ch in w:
                n = n["kids"].setdefault(ch, {"ch": ch, "tags": frozenset(), "kids": {}})
            n["tags"] = n["tags"] | frozenset(tags)
        canon, nodes = {}, []   # signature -> id ; id -> (ch, tags, kid ids)

        def visit(n):
            kids = tuple(sorted((c, visit(k)) for c, k in n["kids"].items()))
            sig = (n["ch"], n["tags"], kids) if minimize else id(n)
            if sig not in canon:
                canon[sig] = len(nodes)
                nodes.append((n["ch"], n["tags"], kids))
            return canon[sig]

        root_id = visit(root)
        parents = {i: set() for i in range(len(nodes))}
        for i, (_, _, kids) in enumerate(nodes):
            for _, k in kids:
                parents[k].add(i)
        # emit units in an order where same-step dependencies (only on `start`) are fine
        refs = {}
        for i in range(len(nodes)):
            if i == root_id:
                continue
            ch = nodes[i][0]
            ins = [start if p == root_id else Ref(FP, "%s%d" % (prefix, p)) for p in sorted(parents[i])]
            k = len(ins)
            refs[i] = self.unit(F, "%s%d" % (prefix, i), [(self.eq(ch), k)] + [(r, 1) for r in ins], -k)
        out = {}
        tags = sorted({t for _, ts, _ in nodes for t in ts})
        for t in tags:
            members = [refs[i] for i, (_, ts, _) in enumerate(nodes) if t in ts]
            out[t] = self.OR(F, "%s_is_%s" % (prefix, t), *members)
        return out

    # ---- inspection ------------------------------------------------------------------
    def stats(self):
        nz = sum(1 for u in self.units.values() for _, w in u.terms) + \
            sum(1 for u in self.units.values() if u.bias != 0)
        emb = sum(len(v) for v in self.feats.values())
        return {"features": len(self.feats), "fwd_units": len(self.order[F]),
                "bwd_units": len(self.order[B]), "weights_nonzero": nz, "embedding_nonzero": emb,
                "head": 2 * len(self.classes), "params_nonzero": nz + emb + 2 * len(self.classes)}

    # ---- serialization ---------------------------------------------------------------
    def to_json(self):
        return {
            "format": "taxnet-1", "lang": self.lang,
            "embedding": {k: sorted(v) for k, v in self.feats.items()},
            "h0": self.h0,
            "cells": {cell: [{"name": n, "bias": self.units[n].bias,
                              "w": [[r.kind, r.name, w] for r, w in self.units[n].terms]}
                             for n in self.order[cell]] for cell in (F, B)},
            "head": {"class_units": self.classes},
        }

    def save(self, path):
        """Binary weights: what the executors load."""
        feats = list(self.feats)
        fidx = {n: i for i, n in enumerate(feats)}
        idx = {cell: {n: i for i, n in enumerate(self.order[cell])} for cell in (F, B)}
        out = bytearray(b"TAXNET01")
        put = lambda fmt, *v: out.extend(struct.pack("<" + fmt, *v))
        lang = self.lang.encode()
        put("I", len(lang)); out.extend(lang)
        put("I", len(feats))
        for n in feats:
            nb = n.encode(); put("I", len(nb)); out.extend(nb)
        emb = bytearray(256 * len(feats))
        for n, vals in self.feats.items():
            for v in vals:
                emb[v * len(feats) + fidx[n]] = 1
        out.extend(emb)
        for cell in (F, B):
            put("I", len(self.order[cell]))
            for n in self.order[cell]:
                u = self.units[n]
                nb = n.encode(); put("I", len(nb)); out.extend(nb)
                put("iI", u.bias, len(u.terms))
                for r, w in u.terms:
                    i = fidx[r.name] if r.kind == X else idx[F if r.kind in (F, FP) else B][r.name]
                    put("BIi", KIND_CODE[r.kind], i, w)
        put("I", len(self.order[F]))
        out.extend(bytes(self.h0.get(n, 0) for n in self.order[F]))
        put("I", len(self.classes))
        for n in self.classes:
            put("I", idx[B][n])
        with open(path, "wb") as f:
            f.write(out)


def load(path):
    """Parse a binary weight file into plain Python structures."""
    data = open(path, "rb").read()
    assert data[:8] == b"TAXNET01"
    pos = 8

    def get(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v

    def s():
        nonlocal pos
        (n,) = get("I")
        v = data[pos:pos + n].decode()
        pos += n
        return v

    lang = s()
    (nf,) = get("I")
    feats = [s() for _ in range(nf)]
    emb = [list(data[pos + b * nf: pos + (b + 1) * nf]) for b in range(256)]
    pos += 256 * nf
    cells = {}
    for cell in (F, B):
        (nu,) = get("I")
        units = []
        for _ in range(nu):
            name = s()
            bias, nt = get("iI")
            terms = [get("BIi") for _ in range(nt)]
            units.append((name, bias, terms))
        cells[cell] = units
    (n0,) = get("I")
    h0 = list(data[pos:pos + n0]); pos += n0
    (nc,) = get("I")
    classes = [get("I")[0] for _ in range(nc)]
    assert pos == len(data)
    return {"lang": lang, "feats": feats, "emb": emb, "cells": cells, "h0": h0, "classes": classes}


def interpret(w, data):
    """Unit-by-unit evaluation straight from a loaded weight file (slow; for debugging)."""
    fw, bw = w["cells"][F], w["cells"][B]
    nf, nb = len(fw), len(bw)
    prev = list(w["h0"])
    fvals = []
    for byte in data:
        x = w["emb"][byte]
        cur = [0] * nf
        for i, (_, bias, terms) in enumerate(fw):
            s = bias
            for kind, j, wt in terms:
                s += wt * (x[j] if kind == 0 else cur[j] if kind == 1 else prev[j])
            cur[i] = min(1, max(0, s))
        fvals.append(cur)
        prev = cur
    out = bytearray(len(data))
    nxt = [0] * nb
    for t in range(len(data) - 1, -1, -1):
        x = w["emb"][data[t]]
        cur = [0] * nb
        for i, (_, bias, terms) in enumerate(bw):
            s = bias
            for kind, j, wt in terms:
                s += wt * (x[j] if kind == 0 else fvals[t][j] if kind == 1 else cur[j] if kind == 3 else nxt[j])
            cur[i] = min(1, max(0, s))
        nxt = cur
        best, bv = 0, 0.5   # class 0 has logit 0.5, class k has logit = its unit
        for k, ui in enumerate(w["classes"]):
            if cur[ui] > bv:
                best, bv = k + 1, cur[ui]
        out[t] = best
    return out
