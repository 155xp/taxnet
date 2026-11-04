"""taxnet-c: the hand-written C network (reference: clang -std=gnu17 raw lexer)."""
from common import ALL, DIGITS, HIGH, LETTERS, P, bom, class_units, word_classes, words
from taxnet import B, F, Net, Ref


def build():
    n = Net("c")
    fwd = lambda name, *a, **k: n.AND(F, name, *a, **k)
    fwd_or = lambda name, *a, **k: n.OR(F, name, *a, **k)

    # ---- embedding: which bytes light up which input feature ---------------------------
    x = n.x
    nl = x("nl", "\n")
    eol = x("eol", "\n\r")                       # ends // comments and unterminated literals
    hws = x("hws", " \t\f\v\r")                  # horizontal whitespace
    anyws = x("anyws", " \t\f\v\r\n")
    sptab = x("sptab", " \t")
    idc = x("idc", list(LETTERS + DIGITS + "_$") + HIGH)
    ppn = x("ppn", list(LETTERS + DIGITS + "_.") + HIGH)  # pp-number body
    dig = x("dig", DIGITS)
    expo = x("expo", "eEpP")
    sign = x("sign", "+-")
    punct = x("punct", [b for b in ALL if b not in set(map(ord, LETTERS + DIGITS + "_$ \t\f\v\r\n")) and b < 128])
    slash_, star, hash_, q2, q1 = x("slash", "/"), x("star", "*"), x("hash", "#"), x("q2", '"'), x("q1", "'")
    bsl, dot_, lt, gt, minus_, lpar = x("bsl", "\\"), x("dot", "."), x("lt", "<"), x("gt", ">"), x("minus", "-"), x("lpar", "(")

    # ---- forward cell -------------------------------------------------------------------
    bom_any, bom_whole = bom(n)
    # cm: this byte is read in code (not inside a comment, literal or header name)
    cm = fwd("cm", neg=[P("bcom"), P("lcom"), P("str"), P("chr"), P("hdr"), bom_any])

    # comments. `slash` (below) marks a code '/' that may open one.
    copen_l = fwd("copen_l", cm, P("slash"), slash_)          # 2nd byte of //
    copen_b = fwd("copen_b", cm, P("slash"), star)            # 2nd byte of /*
    lend = fwd("lend", P("lcom"), eol, neg=[P("bs")])          # // ends at an unspliced newline
    lcont = fwd("lcont", P("lcom"), neg=[lend])
    lcom = fwd_or("lcom", copen_l, lcont)
    bclose = fwd("bclose", P("bcom"), P("bstar"), slash_)
    bcont = fwd("bcont", P("bcom"), neg=[bclose])
    bcom = fwd_or("bcom", copen_b, bcont)
    fwd("bstar", P("bcom"), star)
    m_com = fwd_or("m_com", P("bcom"), lcont, copen_l, copen_b)

    # backslash-newline splices (whitespace may sit between them; clang accepts that)
    bs = fwd_or("bs", fwd("bs_new", bsl, neg=[P("bcom")]), fwd("bs_keep", P("bs"), hws))
    cbs_new, cbs_keep = fwd("cbs_new", cm, bsl), fwd("cbs_keep", P("cbs"), hws)
    cbs = fwd_or("cbs", cbs_new, cbs_keep)
    spl_nl = fwd("spl_nl", nl, P("cbs"))                    # newline that ends a code splice

    # "..." and '...' literals; `esc` = the previous byte was an unescaped backslash
    sclose = fwd("sclose", P("str"), q2, neg=[P("esc")])
    seol = fwd("seol", P("str"), eol, neg=[P("esc"), P("bs")])
    scont = fwd("scont", P("str"), neg=[sclose, seol])
    sopen = fwd("sopen", cm, q2)
    fwd_or("str", sopen, scont)
    cclose = fwd("cclose", P("chr"), q1, neg=[P("esc")])
    ceol = fwd("ceol", P("chr"), eol, neg=[P("esc"), P("bs")])
    ccont = fwd("ccont", P("chr"), neg=[cclose, ceol])
    copen = fwd("copen", cm, q1)
    fwd_or("chr", copen, ccont)
    n.ANDOR(F, "esc", [bsl], [P("str"), P("chr")], neg=[P("esc")])
    m_str = fwd_or("m_str", sopen, scont, sclose, copen, ccont, cclose)

    slash = fwd("slash", cm, slash_, neg=[copen_l])

    # pp-numbers: digit (or .digit) then [0-9a-zA-Z_.] and e+ e- p+ p-
    nstart = fwd("nstart", cm, dig, neg=[P("wc"), P("num")])
    ncont = fwd("ncont", P("num"), ppn)
    nsign = fwd("nsign", P("nume"), sign)
    num = fwd_or("num", nstart, ncont, nsign)
    fwd("nume", num, expo)

    # identifier bytes in code
    wc = fwd("wc", cm, idc, neg=[num])
    wstart = fwd("wstart", wc, neg=[P("wc")])

    # member access: the last token was '.' or '->' (not '...')
    # dots group into '...' greedily from the left: e1 e2 e3 e1 e2 e3 ...
    d1 = fwd("d1", cm, dot_, neg=[num])
    fwd("e1", d1, neg=[P("e1"), P("e2")])
    fwd("e2", d1, P("e1"))
    e3 = fwd("e3", d1, P("e2"))                                           # ends a '...'
    fwd("arrow_minus", cm, minus_)
    dot = fwd_or("dot", fwd("dot_new", d1, neg=[e3]), fwd("dot_arrow", P("arrow_minus"), gt),
                 fwd("dot_keep", P("dot"), anyws))
    wdot = fwd_or("wdot", fwd("wdot_new", wstart, P("dot")), fwd("wdot_keep", wc, P("wdot")))

    # keywords, types, builtins, directive names and literal prefixes: one trie
    kws, types, bis = set(words("c_keywords")), set(words("c_types")), set(words("c_builtins"))
    vocab = {}
    for w in kws:
        vocab.setdefault(w, set()).add("kwty" if w in types else "kw")
    for w in types - kws:
        vocab.setdefault(w, set()).add("ty")
    for w in bis:
        vocab.setdefault(w, set()).add("bi")
    for w in ("include", "include_next", "import"):
        vocab.setdefault(w, set()).add("inc")
    for w in ("L", "u", "U", "u8"):
        vocab.setdefault(w, set()).add("pfxs")
    for w in ("L", "u", "U"):
        vocab.setdefault(w, set()).add("pfxc")
    fin = n.trie("k", vocab, wstart)

    # logical line start: only whitespace / comments / pending '\' or '/' since a newline
    slash_fail = fwd("slash_fail", P("slash"), neg=[copen_l, copen_b])
    cbs_fail = fwd("cbs_fail", P("cbs"), neg=[hws, nl])
    ls_ok = fwd("ls_ok", P("ls"), neg=[slash_fail, cbs_fail])
    nl_reset = fwd("nl_reset", nl, neg=[P("bs")])
    nl_in_bcom = fwd("nl_in_bcom", nl, P("bcom"))
    ls_keep = n.ANDOR(F, "ls_keep", [ls_ok], [anyws, fwd("cbsl", cm, bsl), m_com, slash, bom_any])
    fwd_or("ls", nl_reset, nl_in_bcom, ls_keep)

    # preprocessor: '#' first on a logical line, then the directive name
    hashdir = fwd("hashdir", ls_ok, cm, hash_)
    fwd_or("dhash", hashdir, fwd("dhash_keep", P("dhash"), hws))
    dword = n.ANDOR(F, "dword", [wc], [P("dhash"), P("dword")])
    incstart = fwd("incstart", P("dword"), P(fin["inc"].name), neg=[wc])
    n.ANDOR(F, "inc", [hws], [incstart, P("inc")])
    hopen = n.ANDOR(F, "hopen", [cm, lt], [incstart, P("inc")])
    hcont = fwd("hcont", P("hdr"), neg=[gt, eol])
    fwd_or("hdr", hopen, hcont)
    m_hdr = fwd_or("m_hdr", hopen, hcont, fwd("hclose", P("hdr"), gt))

    # ---- backward cell: things only the bytes to the right can decide -----------------
    bwd = lambda name, *a, **k: n.AND(B, name, *a, **k)
    bwd_or = lambda name, *a, **k: n.OR(B, name, *a, **k)
    nx = lambda name, ref: n.OR(B, "nx_" + name, ref)          # copied, read one step left
    nx_wc = nx("wc", wc)
    nx_copen = nx("copen", fwd_or("copen_any", copen_l, copen_b))
    nx_nstart = nx("nstart", nstart)
    nx_sopen = nx("sopen", sopen)
    nx_chopen = nx("chopen", copen)
    nlah = bwd_or("nlah", nl, bwd("nlah_skip", hws, Ref(B, "nlah").next))
    # clang starts a token that directly follows a splice at the splice's backslash, so the
    # splice bytes (backslash, trailing blanks, newline) belong to the next token
    m_spl = bwd_or("m_spl", spl_nl, n.ANDOR(B, "spl_head", [nlah.next], [cbs_new, cbs_keep]))
    # `(` ahead: spaces/tabs, then a `(` token (which may start with attached splices)
    lpa = bwd_or("lpa", bwd("lpa_here", cm, lpar), bwd("lpa_spl", m_spl, Ref(B, "lpa").next))
    lp = bwd_or("lp", lpa, bwd("lp_skip", sptab, Ref(B, "lp").next))
    w = word_classes(n, wc, {"kw": fin["kw"], "kwty": fin["kwty"], "ty": fin["ty"], "bi": fin["bi"]},
                     nx_wc.next,
                     extra_end=[("fn", lp.next), ("ps", (fin["pfxs"], nx_sopen.next)),
                                ("pc", (fin["pfxc"], nx_chopen.next))])

    c_com = bwd_or("c_com_any", m_com, bwd("slash_com", slash, nx_copen.next))
    c_str = bwd_or("c_str_any", m_str, m_hdr, w["ps"], w["pc"])
    c_num = bwd_or("c_num_any", num, bwd("dot_num", d1, nx_nstart.next, neg=[e3]))
    c_meta = bwd_or("c_meta_any", hashdir, dword)
    c_type = bwd_or("c_type_any", w["kwty"], bwd("ty_nodot", w["ty"], neg=[wdot]))
    c_bi = bwd("c_bi_any", w["bi"], neg=[wdot])
    splice = bwd("splice_bs", cm, bsl, nlah.next)
    c_op = bwd("c_op_any", cm, punct, neg=[splice])
    class_units(n, [("com", c_com), ("str", c_str), ("num", c_num), ("meta", c_meta),
                    ("type", c_type), ("kw", w["kw"]), ("builtin", c_bi), ("func", w["fn"]),
                    ("ident", wc), ("op", c_op)], copy=m_spl, suppress=bom_whole)
    n.h0["ls"] = 1
    return n
