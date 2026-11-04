"""taxnet-py: the hand-written Python network (reference: CPython 3.13 `tokenize`).

Python 3.12+ tokenizes inside f-strings: a replacement field `{...}` holds ordinary code
(which may hold further strings), `:` starts a format spec (literal text, which may hold
nested `{...}` fields, which may have specs of their own), `{{`/`}}` are literal braces.
The forward cell tracks three levels of strings: level 1 opens in top-level code, level
l+1 inside a level-l field. For each level it keeps the quote character, triple-quoting,
the f and raw prefixes, the f-string mode (literal / field / spec) with the number of
open specs, the bracket depth inside a field, and the escape and closing-quote counters.
"""
import itertools

from common import ALL, DIGITS, HIGH, LETTERS, P, bom, class_units, word_classes, words
from taxnet import B, F, Net, Ref

LEVELS = 3


def prefixes():
    """String prefixes CPython accepts, every case variant: r u b br rb f fr rf."""
    out = {}
    for p in ("r", "u", "b", "br", "rb", "f", "fr", "rf"):
        for cs in itertools.product(*[(c, c.upper()) for c in p]):
            out["".join(cs)] = {"pfx"} | ({"pfxf"} if "f" in p else set()) | ({"pfxr"} if "r" in p else set())
    return out


def build():
    n = Net("py")
    A = lambda name, *a, **k: n.AND(F, name, *a, **k)
    O = lambda name, *a, **k: n.OR(F, name, *a, **k)
    AO = lambda name, all_of, any_of, neg=(): n.ANDOR(F, name, all_of, any_of, neg)

    # ---- embedding ------------------------------------------------------------------
    x = n.x
    nl, cr = x("nl", "\n"), x("cr", "\r")
    hws = x("hws", " \t\f\r\v")
    anyws = x("anyws", " \t\f\r\v\n")
    sptab = x("sptab", " \t")
    idc = x("idc", list(LETTERS + DIGITS + "_") + HIGH)
    dig = x("dig", DIGITS)
    punct = x("punct", [b for b in ALL if b < 128 and chr(b) not in LETTERS + DIGITS + "_ \t\f\r\v\n"])
    sq, dq, anyq = x("sq", "'"), x("dq", '"'), x("anyq", "'\"")
    bsl, hash_, at, dot_ = x("bsl", "\\"), x("hash", "#"), x("at", "@"), x("dot", ".")
    lbr, rbr, colon, capN, lpar = x("lbrace", "{"), x("rbrace", "}"), x("colon", ":"), x("N", "N"), x("lpar", "(")
    opn, cls = x("open", "([{"), x("close", ")]}")
    zero, xX, oObB, eE, jJ, sign = x("zero", "0"), x("xX", "xX"), x("oObB", "oObB"), x("eE", "eE"), x("jJ", "jJ"), x("sign", "+-")
    decd, hexd = x("decd", DIGITS + "_"), x("hexd", DIGITS + "_abcdefABCDEF")

    bom_any, bom_whole = bom(n)

    # ---- forward cell ---------------------------------------------------------------
    L = range(1, LEVELS + 1)
    p = lambda name, l: P("%s%d" % (name, l))          # previous-step unit of level l
    u = {}                                                # same-step units by (name, level)

    for l in L:     # the matching quote of each level's open string
        u["mq", l] = O("mq%d" % l, A("mq_d%d" % l, p("qd", l), dq), A("mq_s%d" % l, sq, neg=[p("qd", l)]))
    in_p = lambda l: [p(s, l) for s in ("o1", "o2", "sc", "lit", "fld", "spc")]

    # top-level code: no level-1 string open (an empty '' just closed counts), no comment
    C0 = O("C0", A("C0_plain", neg=in_p(1) + [P("com"), bom_any]), A("C0_after_empty", p("o2", 1), neg=[u["mq", 1]]))

    def level(l, ctx):
        """All units of string level `l`; `ctx` = byte read in the code that opens it."""
        mq = u["mq", l]
        first = A("first%d" % l, p("o1", l), neg=[mq])                    # 1st byte after a lone quote
        rS = O("rS%d" % l, p("sc", l), A("rS_first%d" % l, first, neg=[p("isf", l)]))
        nstart = A("nstart%d" % l, p("bN", l), lbr)                       # the { of \N{
        rL = O("rL%d" % l, A("rL_lit%d" % l, p("lit", l), neg=[p("lb", l), p("nesc", l), nstart]),
               A("rL_first%d" % l, first, p("isf", l)))
        escbr = A("escbr%d" % l, p("lb", l), lbr)                          # 2nd brace of {{
        fopen = A("fopen%d" % l, p("lb", l), neg=[lbr])                    # { at t-1 opened a field
        rP = O("rP%d" % l, p("spc", l))
        if l < LEVELS:   # a string of the next level may be open inside this level's field
            inner_close = A("inner_close%d" % l, p("o2", l + 1), neg=[u["mq", l + 1]])
            rF = O("rF%d" % l, A("rF_fld%d" % l, p("fld", l), neg=in_p(l + 1) + [P("com")]), fopen, inner_close)
        else:
            rF = O("rF%d" % l, A("rF_fld%d" % l, p("fld", l), neg=[P("com")]), fopen)
        cq = O("cq%d" % l, A("cq_single%d" % l, mq, neg=[p("esc", l), p("tq", l)]),
               A("cq_triple%d" % l, mq, p("tq", l), p("r2", l), neg=[p("esc", l)]))
        closeS = A("closeS%d" % l, rS, cq)
        closeL = A("closeL%d" % l, rL, cq)
        open_ = A("open%d" % l, ctx, anyq)
        A("o1%d" % l, open_)
        o2 = A("o2%d" % l, p("o1", l), mq)
        tqopen = A("tqopen%d" % l, p("o2", l), mq)
        AO("esc%d" % l, [bsl], [rS, rL, rP], neg=[p("esc", l)])
        rq = AO("rq%d" % l, [p("tq", l), mq], [rS, rL], neg=[p("esc", l)])
        A("r1%d" % l, rq, neg=[p("r1", l), p("r2", l)])
        A("r2%d" % l, rq, p("r1", l))
        sc = O("sc%d" % l, A("sc_keep%d" % l, rS, neg=[closeS]), A("sc_triple%d" % l, tqopen, neg=[p("isf", l)]))
        A("bN%d" % l, rL, p("esc", l), capN, neg=[p("raw", l)])
        rN = O("rN%d" % l, p("nesc", l))
        O("nesc%d" % l, nstart, A("nesc_keep%d" % l, p("nesc", l), neg=[rbr]))
        lb = A("lb%d" % l, rL, lbr)
        # field / spec structure. sd<k> = at least k format specs are open (fields nest
        # inside specs: f"{x:{w}}" ; their specs nest again: f"{x:{w:{v}}}")
        fclose = A("fclose%d" % l, rF, rbr, neg=[p("da", l)])
        fspec = A("fspec%d" % l, rF, colon, neg=[p("da", l)])
        sfopen = A("sfopen%d" % l, rP, lbr)
        sclose = A("specclose%d" % l, rP, rbr)
        stext = A("spectext%d" % l, rP, neg=[lbr, rbr])
        O("sd1%d" % l, fspec, A("sd1_keep%d" % l, p("sd1", l), neg=[A("sd1_drop%d" % l, sclose, neg=[p("sd2", l)])]))
        O("sd2%d" % l, A("sd2_inc%d" % l, fspec, p("sd1", l)),
          A("sd2_keep%d" % l, p("sd2", l), neg=[A("sd2_drop%d" % l, sclose, neg=[p("sd3", l)])]))
        O("sd3%d" % l, A("sd3_inc%d" % l, fspec, p("sd2", l)), A("sd3_keep%d" % l, p("sd3", l), neg=[sclose]))
        # bracket depth inside a field (thermometer, 4 deep)
        inc = A("inc%d" % l, rF, opn)
        dec = A("dec%d" % l, rF, cls, p("da", l))
        O("da%d" % l, inc, A("da_keep%d" % l, p("da", l), neg=[A("da_drop%d" % l, dec, neg=[p("db", l)])]))
        O("db%d" % l, A("db_inc%d" % l, inc, p("da", l)),
          A("db_keep%d" % l, p("db", l), neg=[A("db_drop%d" % l, dec, neg=[p("dc", l)])]))
        O("dc%d" % l, A("dc_inc%d" % l, inc, p("db", l)),
          A("dc_keep%d" % l, p("dc", l), neg=[A("dc_drop%d" % l, dec, neg=[p("dd", l)])]))
        O("dd%d" % l, A("dd_inc%d" % l, inc, p("dc", l)), A("dd_keep%d" % l, p("dd", l), neg=[dec]))
        fld = O("fld%d" % l, A("fld_hold%d" % l, p("fld", l), neg=[rF]), A("fld_keep%d" % l, rF, neg=[fclose, fspec]),
                sfopen)
        spc = O("spc%d" % l, fspec, stext, A("spc_from_fld%d" % l, fclose, p("sd1", l)),
                A("spc_from_spc%d" % l, sclose, p("sd2", l)))
        lit = O("lit%d" % l, A("lit_keep%d" % l, rL, neg=[closeL]), escbr, nstart, rN,
                A("lit_triple%d" % l, tqopen, p("isf", l)), A("lit_from_fld%d" % l, fclose, neg=[p("sd1", l)]),
                A("lit_from_spc%d" % l, sclose, neg=[p("sd2", l)]))
        inside = O("in%d" % l, open_, o2, sc, lit, fld, spc)
        for flag, cond in (("qd", dq), ("isf", P("k_is_pfxf")), ("raw", P("k_is_pfxr"))):
            O("%s%d" % (flag, l), A("%s_set%d" % (flag, l), open_, cond),
              A("%s_keep%d" % (flag, l), p(flag, l), inside, neg=[open_]))
        O("tq%d" % l, tqopen, A("tq_keep%d" % l, p("tq", l), inside, neg=[open_]))
        # bytes that belong to this level's string tokens
        m_str = O("m_str%d" % l, open_, o2, tqopen, rS, A("rL_text%d" % l, rL, neg=[lbr]), escbr, nstart, rN, stext)
        u.update({("rF", l): rF, ("open", l): open_, ("fopen", l): fopen, ("lb", l): lb,
                  ("sfopen", l): sfopen, ("specclose", l): sclose, ("m_str", l): m_str})

    level(1, C0)
    for l in L[1:]:
        level(l, u["rF", l - 1])
    code = O("code", C0, *[u["rF", l] for l in L])

    # comments run to the end of the line (a CRLF's \r is not part of them)
    com = O("com", A("com_new", code, hash_), A("com_keep", P("com"), neg=[nl, cr]))

    # numbers (tokenize's grammar): 0x.. 0o.. 0b.., digits[.digits][e[+-]digits][j], .digits..
    nstart = A("nstart", code, dig, neg=[P("wc"), P("num"), P("d1")])
    A("nz", nstart, zero)
    nint = O("nint", nstart, A("nint_more", P("nint"), decd))
    nhex = O("nhex", A("nhex_pfx", P("nz"), xX), A("nhex_more", P("nhex"), hexd))
    nob = O("nob", A("nob_pfx", P("nz"), oObB), A("nob_more", P("nob"), decd))
    nlead = A("nfrac_lead", code, P("d1"), dig)                       # .5
    nfrac = O("nfrac", A("nfrac_dot", P("nint"), dot_), A("nfrac_more", P("nfrac"), decd), nlead)
    nexp0 = AO("nexp0", [eE], [P("nint"), P("nfrac")])
    nexps = A("nexps", P("nexp0"), sign)
    nexp = O("nexp", AO("nexp_first", [dig], [P("nexp0"), P("nexps")]), A("nexp_more", P("nexp"), decd))
    nj = AO("nj", [jJ], [P("nint"), P("nfrac"), P("nexp")])
    num = O("num", nint, nhex, nob, nfrac, nexp0, nexps, nexp, nj)

    wc = A("wc", code, idc, neg=[num])
    wstart = A("wstart", wc, neg=[P("wc")])

    # member access: the last token was '.' (not '...'); a continuation backslash is whitespace
    # dots group into '...' greedily from the left: e1 e2 e3 e1 e2 e3 ...
    d1 = A("d1", code, dot_, neg=[num])
    A("e1", d1, neg=[P("e1"), P("e2")])
    A("e2", d1, P("e1"))
    e3 = A("e3", d1, P("e2"))                                             # ends a '...'
    O("dot", A("dot_new", d1, neg=[e3]), AO("dot_keep", [P("dot")], [anyws, bsl]))
    wdot = O("wdot", A("wdot_new", wstart, P("dot")), A("wdot_keep", wc, P("wdot")))

    vocab = {}
    for w in words("py_keywords"):
        vocab.setdefault(w, set()).add("kw")
    for w in words("py_builtins"):
        vocab.setdefault(w, set()).add("bi")
    for w, tags in prefixes().items():
        vocab.setdefault(w, set()).update(tags)
    fin = n.trie("k", vocab, wstart)

    # decorators: '@' first on a physical line that does not continue a backslash line and
    # is outside any brackets, then a dotted name: '@' NAME ('.' NAME)*
    O("bsc", A("bsc_new", C0, bsl), A("bsc_cr", P("bsc"), cr))
    O("ls", A("ls_nl", nl, neg=[P("bsc")]), AO("ls_keep", [P("ls")], [hws, bom_any]))
    pinc = A("pinc", C0, opn)
    pdec = A("pdec", C0, cls, P("pd1"))
    DEPTH = 12                                        # thermometer code: pd<k> = depth >= k
    for k in range(1, DEPTH + 1):
        grow = pinc if k == 1 else A("pd_inc%d" % k, pinc, P("pd%d" % (k - 1)))
        drop = A("pd_drop%d" % k, pdec, neg=[P("pd%d" % (k + 1))] if k < DEPTH else [])
        O("pd%d" % k, grow, A("pd_keep%d" % k, P("pd%d" % k), neg=[drop]))
    deco_at = A("deco_at", C0, at, P("ls"), neg=[P("pd1")])
    dname = AO("dname", [wc], [P("dexp"), P("dname")])                   # a name in it
    ddot = AO("ddot", [code, dot_], [P("dname"), P("dafter")])            # the '.' after a name
    O("dexp", deco_at, ddot, A("dexp_ws", P("dexp"), hws))                # waiting for a name
    O("dafter", AO("dafter_ws", [hws], [P("dname"), P("dafter")]))        # blanks after a name
    deco = O("deco", deco_at, dname, ddot)

    # ---- backward cell --------------------------------------------------------------
    bA = lambda name, *a, **k: n.AND(B, name, *a, **k)
    bO = lambda name, *a, **k: n.OR(B, name, *a, **k)
    nx = lambda name, ref: n.OR(B, "nx_" + name, ref)
    nx_wc = nx("wc", wc)
    nx_open = nx("open", O("open_any", *[u["open", l] for l in L]))
    nx_fopen = {l: nx("fopen%d" % l, u["fopen", l]) for l in L}
    nx_nlead = nx("nlead", nlead)
    lp = bO("lp", bA("lp_here", code, lpar), bA("lp_skip", sptab, Ref(B, "lp").next))
    w = word_classes(n, wc, {"kw": fin["kw"], "bi": fin["bi"]}, nx_wc.next,
                     extra_end=[("fn", lp.next), ("pfx", (fin["pfx"], nx_open.next))])

    lb_op = [bA("lb_op%d" % l, u["lb", l], nx_fopen[l].next) for l in L]           # { that opened a field
    lb_str = [bA("lb_str%d" % l, u["lb", l], neg=[nx_fopen[l].next]) for l in L]   # first { of {{
    c_str = bO("c_str_any", *[u["m_str", l] for l in L], *lb_str, w["pfx"])
    c_num = bO("c_num_any", num, bA("dot_num", d1, nx_nlead.next, neg=[e3]))
    c_bi = bA("c_bi_any", w["bi"], neg=[wdot])
    c_op = bO("c_op_any", bA("code_punct", code, punct, neg=[bsl]), *lb_op,
              *[u["sfopen", l] for l in L], *[u["specclose", l] for l in L])
    class_units(n, [("com", com), ("str", c_str), ("meta", deco), ("num", c_num), ("kw", w["kw"]),
                    ("builtin", c_bi), ("func", w["fn"]), ("ident", wc), ("op", c_op)], suppress=bom_whole)
    n.h0["ls"] = 1
    return n
