#!/usr/bin/env python3
"""Compile a taxnet weight file into a bit-sliced C kernel.

    python kernel/codegen.py weights/c.tnw kernel/gen/net_c.h

Every value in taxnet is 0 or 1, so a batch of 256 independent byte streams can be run
at once with one bit per stream: a neuron's activation for all 256 streams is a single
256-bit vector, and a neuron clip(sum w_i x_i + b, 0, 1) becomes a few AND/OR/ANDNOT
instructions. This file derives that boolean form from the weights *and proves it*: for
each neuron it finds the partition of inputs into must-be-on (A), at-least-one-on (O) and
must-be-off (N) sets and checks the inequalities that make

    [sum_i w_i x_i + b >= 1]  ==  AND(A) & OR(O) & ~OR(N)    for every 0/1 input.

A neuron whose weights fail every proof is compiled from its full truth table (or the
compile stops). The generated code contains no other logic: change a weight, regenerate,
and the kernel changes with it. Each header records the SHA-256 of the weight file it came from
(`taxnet <lang> --info` prints it).
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
import taxnet  # noqa: E402

KX, KF, KFP, KB, KBN = range(5)


def prove(terms, bias):
    """terms: [(key, w)]. Returns ('const', v) or ('aon', A, O, N) proven exact, else None."""
    pos = [(k, w) for k, w in terms if w > 0]
    neg = [(k, w) for k, w in terms if w < 0]
    if not terms:
        return ("const", 1 if bias >= 1 else 0)
    N = [k for k, _ in neg]
    maxneg = max((w for _, w in neg), default=None)
    cands = []
    if pos:
        ws = sorted({w for _, w in pos})
        cands.append(([k for k, _ in pos], []))            # all positive are required
        cands.append(([], [k for k, _ in pos]))            # any positive suffices
        if len(ws) == 2:                                   # heavy ones required, light ones any
            cands.append(([k for k, w in pos if w == ws[1]], [k for k, w in pos if w == ws[0]]))
    else:
        cands.append(([], []))
    wt = dict(terms)
    for A, O in cands:
        sA = sum(wt[k] for k in A)
        sO = sum(wt[k] for k in O)
        ok = sA + bias + (min(wt[k] for k in O) if O else 0) >= 1          # F=1 fires
        if A:
            ok &= sA - min(wt[k] for k in A) + sO + bias <= 0              # an A off: silent
        if O:
            ok &= sA + bias <= 0                                           # every O off: silent
        if N:
            ok &= sA + sO + maxneg + bias <= 0                             # an N on: silent
        if ok:
            return ("aon", A, O, N)
    if not pos and bias <= 0:
        return ("const", 0)
    return None


def truth_table_expr(terms, bias, name_of):
    keys = [k for k, _ in terms]
    if len(keys) > 10:
        raise SystemExit("neuron with %d inputs has no AND/OR form" % len(keys))
    mins = []
    for m in range(1 << len(keys)):
        s = bias + sum(w for i, (_, w) in enumerate(terms) if m >> i & 1)
        if s >= 1:
            mins.append("(" + " & ".join(name_of(k) if m >> i & 1 else "~" + name_of(k)
                                         for i, k in enumerate(keys)) + ")")
    return " | ".join(mins) if mins else "ZERO"


def set_expr(vals, lo, hi, cache, emit):
    """Boolean expression (over nibble one-hot planes) for 'byte in vals'."""
    vals = frozenset(vals)
    if len(vals) == 256:
        return "ONES"
    if not vals:
        return "ZERO"
    if len(vals) > 128:
        return "~(" + set_expr(frozenset(range(256)) - vals, lo, hi, cache, emit) + ")"
    terms = []
    for h in range(16):
        ls = frozenset(v & 15 for v in vals if v >> 4 == h)
        if not ls:
            continue
        if len(ls) == 16:
            terms.append(hi[h])
            continue
        if ls not in cache:
            if len(ls) > 8:
                e = "~(" + " | ".join(lo[l] for l in sorted(set(range(16)) - ls)) + ")"
            else:
                e = " | ".join(lo[l] for l in sorted(ls))
            cache[ls] = emit("V", e)
        terms.append("(%s & %s)" % (hi[h], cache[ls]))
    return " | ".join(terms)


def generate(w, sym):
    feats, emb = w["feats"], w["emb"]
    fw, bw = w["cells"]["f"], w["cells"]["b"]
    fname = [u[0] for u in fw]
    bname = [u[0] for u in bw]

    # which forward units must persist across steps, which the backward cell reads
    persist = sorted({j for _, _, terms in fw for k, j, _ in terms if k == KFP})
    saved = sorted({j for _, _, terms in bw for k, j, _ in terms if k == KF})
    bnext = sorted({j for _, _, terms in bw for k, j, _ in terms if k == KBN})
    fx = sorted({j for _, _, terms in fw for k, j, _ in terms if k == KX})
    bx = sorted({j for _, _, terms in bw for k, j, _ in terms if k == KX})
    pidx = {j: i for i, j in enumerate(persist)}
    sidx = {j: i for i, j in enumerate(saved)}
    nidx = {j: i for i, j in enumerate(bnext)}
    fset = {i: frozenset(b for b in range(256) if emb[b][i]) for i in range(len(feats))}
    stats = {"aon": 0, "tt": 0, "const": 0}

    def feature_block(used, indent):
        lines = []
        tmp = [0]

        def emit(ty, e):
            tmp[0] += 1
            v = "t%d" % tmp[0]
            lines.append("%s%s %s = %s;" % (indent, ty, v, e))
            return v
        lines.append(indent + "const V b0 = in[0], b1 = in[1], b2 = in[2], b3 = in[3], "
                              "b4 = in[4], b5 = in[5], b6 = in[6], b7 = in[7];")
        lines.append(indent + "(void)b0; (void)b1; (void)b2; (void)b3; (void)b4; (void)b5; (void)b6; (void)b7;")
        lo = ["lo%d" % i for i in range(16)]
        hi = ["hi%d" % i for i in range(16)]
        for name, (q0, q1, q2, q3) in (("lo", ("b0", "b1", "b2", "b3")), ("hi", ("b4", "b5", "b6", "b7"))):
            # 2-bit decodes of each half of the nibble, then the 16 nibble values
            lines.append("%sconst V U %s_a0 = ~(%s | %s), %s_a1 = %s & ~%s, %s_a2 = %s & ~%s, %s_a3 = %s & %s;"
                         % (indent, name, q0, q1, name, q0, q1, name, q1, q0, name, q0, q1))
            lines.append("%sconst V U %s_b0 = ~(%s | %s), %s_b1 = %s & ~%s, %s_b2 = %s & ~%s, %s_b3 = %s & %s;"
                         % (indent, name, q2, q3, name, q2, q3, name, q3, q2, name, q2, q3))
            for v in range(16):
                lines.append("%sconst V U %s%d = %s_a%d & %s_b%d;" % (indent, name, v, name, v & 3, name, v >> 2))
        cache = {}
        for i in used:
            lines.append("%sconst V x%d = %s; /* %s */" % (indent, i, set_expr(fset[i], lo, hi, cache, emit), feats[i]))
        return lines

    def neuron(name, bias, terms, ref_name, indent, var):
        keyed = [((k, j), wt) for k, j, wt in terms]
        p = prove(keyed, bias)
        if p is None:
            stats["tt"] += 1
            return "%sconst V %s = %s; /* %s (truth table) */" % (indent, var, truth_table_expr(keyed, bias, ref_name), name)
        if p[0] == "const":
            stats["const"] += 1
            return "%sconst V %s = %s; /* %s */" % (indent, var, "ONES" if p[1] else "ZERO", name)
        stats["aon"] += 1
        _, A, O, N = p
        parts = [ref_name(k) for k in A]
        if O:
            parts.append("(" + " | ".join(ref_name(k) for k in O) + ")" if len(O) > 1 else ref_name(O[0]))
        expr = " & ".join(parts) if parts else "ONES"
        if N:
            nexpr = " | ".join(ref_name(k) for k in N)
            expr = ("~(%s)" % nexpr) if not parts else "%s & ~(%s)" % (expr, nexpr)
        return "%sconst V %s = %s; /* %s */" % (indent, var, expr, name)

    L = []
    L.append("/* Generated by kernel/codegen.py from the %s weight file. Do not edit. */" % w["lang"])
    L.append("#ifndef U\n#define U __attribute__((unused))\n#endif")
    L.append("#define %s_NPERSIST %d" % (sym, len(persist)))
    L.append("#define %s_NSAVE %d" % (sym, len(saved)))
    L.append("#define %s_NNEXT %d" % (sym, len(bnext)))
    L.append("#define %s_NFWD %d" % (sym, len(fw)))
    L.append("#define %s_NBWD %d" % (sym, len(bw)))
    L.append("static const char *const %s_persist_names[] = {%s};" % (sym, ", ".join('"%s"' % fname[j] for j in persist)))
    L.append("static const unsigned char %s_h0[%d] = {%s};" % (sym, max(1, len(persist)), ", ".join(str(w["h0"][j]) for j in persist) or "0"))

    # forward step
    L.append("static inline void %s_fwd(const V *restrict in, V *restrict st, V *restrict save) {" % sym)
    L.extend(feature_block(fx, "    "))

    def fref(key):
        k, j = key
        if k == KX:
            return "x%d" % j
        if k == KF:
            return "f%d" % j
        if k == KFP:
            return "st[%d]" % pidx[j]
        raise AssertionError(key)
    # The state is updated in place: unit i's new value overwrites st[i] right after the last
    # unit that reads its old value (or right after i itself, if that is later). The trie is
    # emitted children-first, so its nodes are written as soon as they are computed.
    last_reader = {}
    for r, (_, _, terms) in enumerate(fw):
        for k, j, _ in terms:
            if k == KFP:
                last_reader[j] = max(last_reader.get(j, -1), r)
    write_at = {}
    for j in persist:
        write_at.setdefault(max(j, last_reader.get(j, -1)), []).append(j)
    for i, (name, bias, terms) in enumerate(fw):
        L.append(neuron(name, bias, terms, fref, "    ", "f%d" % i))
        if i in sidx:
            L.append("    save[%d] = f%d;" % (sidx[i], i))
        for j in write_at.get(i, []):
            L.append("    st[%d] = f%d;" % (pidx[j], j))
    L.append("}")

    # backward step: emits the class index as 4 bit-planes
    L.append("static inline void %s_bwd(const V *restrict in, const V *restrict save, V *restrict nx, V *restrict cls) {" % sym)
    L.extend(feature_block(bx, "    "))

    def bref(key):
        k, j = key
        if k == KX:
            return "x%d" % j
        if k == KF:
            return "save[%d]" % sidx[j]
        if k == KB:
            return "g%d" % j
        if k == KBN:
            return "nx[%d]" % nidx[j]
        raise AssertionError(key)
    for i, (name, bias, terms) in enumerate(bw):
        L.append(neuron(name, bias, terms, bref, "    ", "g%d" % i))
    for j in bnext:
        L.append("    nx[%d] = g%d;" % (nidx[j], j))
    for bit in range(4):
        members = ["g%d" % u for c, u in enumerate(w["classes"]) if (c + 1) >> bit & 1]
        L.append("    cls[%d] = %s;" % (bit, " | ".join(members) if members else "ZERO"))
    L.append("}")
    h = hashlib.sha256("\n".join(L).encode()).hexdigest()[:16]
    L.insert(1, "#define %s_HASH \"%s\"" % (sym, h))
    return "\n".join(L) + "\n", stats


def main():
    src, dst = sys.argv[1], sys.argv[2]
    w = taxnet.load(src)
    sym = w["lang"].upper()
    code, stats = generate(w, sym)
    with open(src, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    code = code.replace("#define %s_HASH" % sym, "#define %s_WEIGHTS_SHA256 \"%s\"\n#define %s_HASH" % (sym, digest, sym), 1)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w") as f:
        f.write(code)
    print("%s: %d fwd + %d bwd neurons -> %d proven AND/OR forms, %d truth tables, %d constants"
          % (src, len(w["cells"]["f"]), len(w["cells"]["b"]), stats["aon"], stats["tt"], stats["const"]))


if __name__ == "__main__":
    main()
