"""taxnet-go: the hand-written Go network (reference: the standard library's go/scanner)."""
from common import ALL, DIGITS, HIGH, LETTERS, P, bom, class_units, word_classes, words
from taxnet import B, F, Net, Ref


def build():
    n = Net("go")
    fwd = lambda name, *a, **k: n.AND(F, name, *a, **k)
    fwd_or = lambda name, *a, **k: n.OR(F, name, *a, **k)

    # ---- embedding ----------------------------------------------------------------------
    x = n.x
    nl = x("nl", "\n")
    anyws = x("anyws", " \t\r\n")
    sptab = x("sptab", " \t")
    idc = x("idc", list(LETTERS + DIGITS + "_") + HIGH)
    dig = x("dig", DIGITS)
    decd = x("decd", DIGITS + "_")                 # decimal digits and separators
    hexd = x("hexd", DIGITS + "_abcdefABCDEF")
    punct = x("punct", [b for b in ALL if b < 128 and chr(b) not in LETTERS + DIGITS + "_ \t\r\n"])
    slash_, star, q2, q1, bq = x("slash", "/"), x("star", "*"), x("q2", '"'), x("q1", "'"), x("bq", "`")
    bsl, dot_, lpar = x("bsl", "\\"), x("dot", "."), x("lpar", "(")
    zero, xX, oObB, eE, pP, imag = x("zero", "0"), x("xX", "xX"), x("oObB", "oObB"), x("eE", "eE"), x("pP", "pP"), x("i", "i")
    expd = x("expd", DIGITS + "_+-")

    # ---- forward cell ---------------------------------------------------------------------
    bom_any, bom_whole = bom(n)
    cm = fwd("cm", neg=[P("bcom"), P("lcom"), P("str"), P("raw"), P("rune"), bom_any])

    copen_l = fwd("copen_l", cm, P("slash"), slash_)
    copen_b = fwd("copen_b", cm, P("slash"), star)
    lcont = fwd("lcont", P("lcom"), neg=[nl])                  # // runs to the newline
    lcom = fwd_or("lcom", copen_l, lcont)
    bclose = fwd("bclose", P("bcom"), P("bstar"), slash_)
    bcont = fwd("bcont", P("bcom"), neg=[bclose])
    fwd_or("bcom", copen_b, bcont)
    fwd("bstar", P("bcom"), star)
    m_com = fwd_or("m_com", P("bcom"), lcont, copen_l, copen_b)

    # "..." strings, `...` raw strings, '...' runes
    sclose = fwd("sclose", P("str"), q2, neg=[P("esc")])
    seol = fwd("seol", P("str"), nl, neg=[P("esc")])
    scont = fwd("scont", P("str"), neg=[sclose, seol])
    sopen = fwd("sopen", cm, q2)
    fwd_or("str", sopen, scont)
    rclose = fwd("rclose", P("raw"), bq)
    rcont = fwd("rcont", P("raw"), neg=[rclose])
    ropen = fwd("ropen", cm, bq)
    fwd_or("raw", ropen, rcont)
    qclose = fwd("qclose", P("rune"), q1, neg=[P("esc")])
    qeol = fwd("qeol", P("rune"), nl, neg=[P("esc")])
    qcont = fwd("qcont", P("rune"), neg=[qclose, qeol])
    qopen = fwd("qopen", cm, q1)
    fwd_or("rune", qopen, qcont)
    n.ANDOR(F, "esc", [bsl], [P("str"), P("rune")], neg=[P("esc")])
    m_str = fwd_or("m_str", sopen, scont, sclose, ropen, rcont, rclose, qopen, qcont, qclose)

    slash = fwd("slash", cm, slash_, neg=[copen_l])

    # numbers, following go/scanner.scanNumber
    nstart = fwd("nstart", cm, dig, neg=[P("wc"), P("num"), P("d1")])   # .5 starts at the dot
    nz = fwd("nz", nstart, zero)                                          # leading 0
    ndec = fwd_or("ndec", nstart, fwd("ndec_more", P("ndec"), decd), fwd("ndec_pfx", P("nz"), oObB))
    nhex = fwd_or("nhex", fwd("nhex_pfx", P("nz"), xX), fwd("nhex_more", P("nhex"), hexd))
    nlead = fwd("nfrac_lead", cm, P("d1"), dig)                             # .5
    nfrac = fwd_or("nfrac", fwd("nfrac_dot", P("ndec"), dot_), fwd("nfrac_more", P("nfrac"), decd), nlead)
    nhfrac = fwd_or("nhfrac", fwd("nhfrac_dot", P("nhex"), dot_), fwd("nhfrac_more", P("nhfrac"), hexd))
    nexp0 = fwd_or("nexp0", n.ANDOR(F, "nexp0_e", [eE], [P("ndec"), P("nfrac")]),
                   n.ANDOR(F, "nexp0_p", [pP], [P("nhex"), P("nhfrac")]))
    nexp = fwd_or("nexp", fwd("nexp_first", P("nexp0"), expd), fwd("nexp_more", P("nexp"), decd))
    nimag = n.ANDOR(F, "nimag", [imag], [P("ndec"), P("nfrac"), P("nhex"), P("nhfrac"), P("nexp")])
    num = fwd_or("num", ndec, nhex, nfrac, nhfrac, nexp0, nexp, nimag)

    wc = fwd("wc", cm, idc, neg=[num])
    wstart = fwd("wstart", wc, neg=[P("wc")])

    # member access: last token was '.' (not '...')
    # dots group into '...' greedily from the left: e1 e2 e3 e1 e2 e3 ...
    d1 = fwd("d1", cm, dot_, neg=[num])
    fwd("e1", d1, neg=[P("e1"), P("e2")])
    fwd("e2", d1, P("e1"))
    e3 = fwd("e3", d1, P("e2"))                                           # ends a '...'
    dot = fwd_or("dot", fwd("dot_new", d1, neg=[e3]), fwd("dot_keep", P("dot"), anyws))
    wdot = fwd_or("wdot", fwd("wdot_new", wstart, P("dot")), fwd("wdot_keep", wc, P("wdot")))

    vocab = {}
    for w in words("go_keywords"):
        vocab.setdefault(w, set()).add("kw")
    for w in words("go_types"):
        vocab.setdefault(w, set()).add("ty")
    for w in words("go_builtins"):
        vocab.setdefault(w, set()).add("bi")
    fin = n.trie("k", vocab, wstart)

    # ---- backward cell ----------------------------------------------------------------
    bwd = lambda name, *a, **k: n.AND(B, name, *a, **k)
    bwd_or = lambda name, *a, **k: n.OR(B, name, *a, **k)
    nx = lambda name, ref: n.OR(B, "nx_" + name, ref)
    nx_wc = nx("wc", wc)
    nx_copen = nx("copen", fwd_or("copen_any", copen_l, copen_b))
    nx_nlead = nx("nlead", nlead)
    lp = bwd_or("lp", bwd("lp_here", cm, lpar), bwd("lp_skip", sptab, Ref(B, "lp").next))
    w = word_classes(n, wc, {"kw": fin["kw"], "ty": fin["ty"], "bi": fin["bi"]}, nx_wc.next,
                     extra_end=[("fn", lp.next)])

    c_com = bwd_or("c_com_any", m_com, bwd("slash_com", slash, nx_copen.next))
    c_num = bwd_or("c_num_any", num, bwd("dot_num", d1, nx_nlead.next, neg=[e3]))
    c_type = bwd("c_type_any", w["ty"], neg=[wdot])
    c_bi = bwd("c_bi_any", w["bi"], neg=[wdot])
    c_op = bwd("c_op_any", cm, punct)
    class_units(n, [("com", c_com), ("str", m_str), ("num", c_num), ("kw", w["kw"]),
                    ("type", c_type), ("builtin", c_bi), ("func", w["fn"]), ("ident", wc),
                    ("op", c_op)], suppress=bom_whole)
    return n
