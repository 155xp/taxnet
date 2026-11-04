"""Pieces shared by the three language networks."""
import string
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ref"))
from spec import CLASSES, words  # noqa: E402,F401

from taxnet import B, F, Ref  # noqa: E402

LETTERS = string.ascii_letters
DIGITS = string.digits
HIGH = list(range(128, 256))       # UTF-8 bytes: only ever valid inside identifiers in code
ALL = list(range(256))


def P(name):
    """The value forward unit `name` had one step earlier."""
    return Ref("f-", name)


def bom(n):
    """A UTF-8 byte order mark at the very start of a file is whitespace and changes no
    state. Forward units mark its three bytes; the backward unit `bom` confirms the whole
    sequence EF BB BF is there before any of them is called whitespace."""
    n.unit(F, "sof", [], 0)                       # 1 only before the first byte (via h0)
    b1 = n.AND(F, "bom1", P("sof"), n.x("xEF", [0xEF]))
    b2 = n.AND(F, "bom2", P("bom1"), n.x("xBB", [0xBB]))
    b3 = n.AND(F, "bom3", P("bom2"), n.x("xBF", [0xBF]))
    n.h0["sof"] = 1
    any_ = n.OR(F, "bom_any", b1, b2, b3)
    whole = n.OR(B, "bom", b3, n.ANDOR(B, "bom_head", [Ref(B, "bom").next], [b1, b2]))
    return any_, whole


def class_units(n, conds, copy=None, suppress=None):
    """One-hot output units in priority order. `conds` is [(class_name, unit), ...]; each
    class unit fires when its condition holds and no earlier class fired, so at most one
    class unit is ever 1 and the head's argmax is exact. Class 0 (ws) is the default.
    Where `copy` is 1 the byte takes the class of the byte to its right instead; where
    `suppress` is 1 no class unit fires (the byte is ws)."""
    fired = [suppress] if suppress is not None else []   # "no earlier class fired"
    out = {}
    for cls, cond in conds:
        name = "class_" + cls
        if copy is not None:
            cond = n.OR(B, name + "_cond", n.AND(B, name + "_copy", copy, Ref(B, name).next),
                        n.AND(B, name + "_own", cond, neg=[copy]))
        out[cls] = n.AND(B, name, cond, neg=fired)
        fired.append(cond)
    n.classes = [out[c].name if c in out else None for c in CLASSES[1:]]
    missing = [c for c, u in zip(CLASSES[1:], n.classes) if u is None]
    if missing:   # a class this language never emits: give it a constant-zero unit
        for c in missing:
            u = n.unit(B, "class_" + c, [], 0)
            n.classes[CLASSES.index(c) - 1] = u.name
    return out


def word_classes(n, wc, fin, nx_wc, extra_end=()):
    """Backward propagation of word-level decisions. `fin[tag]` is the forward trie output
    at the word's last byte; this spreads it over every byte of the word.
    Returns {tag: backward unit}."""
    wend = n.AND(B, "wend", wc, neg=[nx_wc])
    out = {}
    for tag, cond in list(fin.items()) + list(extra_end):
        name = "w_" + tag
        end = n.AND(B, name + "_end", wend, *cond) if isinstance(cond, tuple) else n.AND(B, name + "_end", wend, cond)
        out[tag] = n.OR(B, name, end, n.AND(B, name + "_carry", wc, Ref(B, name).next))
    return out
