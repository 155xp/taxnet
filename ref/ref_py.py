"""Python reference labeler: CPython 3.13's own `tokenize` + the rules in spec.py."""
import io
import tokenize as T

from spec import BUILTIN, COM, FUNC, IDENT, KW, META, NUM, OP, STR, Reject, words

KEYWORDS = frozenset(words("py_keywords"))
BUILTINS = frozenset(words("py_builtins"))
STRINGISH = {T.STRING, T.FSTRING_START, T.FSTRING_MIDDLE, T.FSTRING_END}
WS_TOKENS = {T.NL, T.NEWLINE, T.INDENT, T.DEDENT}


def label(data):
    if b"\0" in data:
        raise Reject("nul")
    if b"\r" in data.replace(b"\r\n", b""):
        raise Reject("lone-cr")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        raise Reject("not-utf8")
    enc, _ = T.detect_encoding(io.BytesIO(data).readline)
    if enc not in ("utf-8", "utf-8-sig") and not data.isascii():
        raise Reject("encoding-" + enc)

    # tokenize reports (row, col) with col counted in code points of the decoded line;
    # map those to byte offsets. readline() splits on b"\n", so lines do too.
    bom = 3 if data.startswith(b"\xef\xbb\xbf") else 0
    starts = [bom]
    i = data.find(b"\n")
    while i >= 0:
        starts.append(i + 1)
        i = data.find(b"\n", i + 1)
    decoded = {}

    def off(rc):
        row, col = rc
        if row > len(starts):
            return len(data)
        base = starts[row - 1]
        if row not in decoded:
            end = starts[row] if row < len(starts) else len(data)
            raw = data[base:end]
            decoded[row] = None if raw.isascii() else raw.decode("utf-8")
        line = decoded[row]
        return min(len(data), base + (col if line is None else len(line[:col].encode("utf-8"))))

    toks = []
    try:
        for t in T.tokenize(io.BytesIO(data).readline):
            if t.type == T.ERRORTOKEN:
                raise Reject("errortoken")
            if t.type in (T.ENCODING, T.ENDMARKER, T.DEDENT) or not t.string:
                continue
            s, e = off(t.start), off(t.end)
            if e > s:
                toks.append((t.type, t.string, s, e))
    except (T.TokenError, SyntaxError) as ex:
        raise Reject("tokenize: %s" % (ex,))

    lab = bytearray(len(data))
    n = len(toks)
    for i, (ty, st, s, e) in enumerate(toks):
        if ty == T.COMMENT:
            c = COM
        elif ty == T.NUMBER:
            c = NUM
        elif ty in STRINGISH:
            c = STR
            # tokenize sizes an FSTRING_MIDDLE by its *unescaped* text, so the second
            # brace of every `{{` / `}}` escape is left outside all tokens. It is string text.
            if ty == T.FSTRING_MIDDLE and st[-1:] in ("{", "}") and data[e:e + 1] == st[-1].encode():
                lab[e] = STR
        elif ty == T.NAME:
            j = i - 1
            while j >= 0 and toks[j][0] in WS_TOKENS:
                j -= 1
            after_dot = j >= 0 and toks[j][0] == T.OP and toks[j][1] == "."
            if st in KEYWORDS:
                c = KW
            elif st in BUILTINS and not after_dot:
                c = BUILTIN
            elif (i + 1 < n and toks[i + 1][0] == T.OP and toks[i + 1][1] == "("
                  and not data[e:toks[i + 1][2]].strip(b" \t")):
                c = FUNC
            else:
                c = IDENT
        elif ty == T.OP:
            c = OP
        else:
            continue  # NEWLINE / NL / INDENT: whitespace
        if lab[s] == META:
            continue  # already claimed by a decorator
        if ty == T.OP and st == "@":
            # A decorator is an `@` that starts a logical line.
            j = i - 1
            while j >= 0 and toks[j][0] in (T.NL, T.COMMENT, T.INDENT, T.DEDENT):
                j -= 1
            if j < 0 or toks[j][0] == T.NEWLINE:
                lab[s:e] = bytes([META]) * (e - s)
                k = i + 1
                if k < n and toks[k][0] == T.NAME:
                    while True:
                        _, _, ks, ke = toks[k]
                        lab[ks:ke] = bytes([META]) * (ke - ks)
                        if (k + 2 < n and toks[k + 1][0] == T.OP and toks[k + 1][1] == "."
                                and toks[k + 2][0] == T.NAME):
                            _, _, ds, de = toks[k + 1]
                            lab[ds:de] = bytes([META]) * (de - ds)
                            k += 2
                        else:
                            break
                continue
        lab[s:e] = bytes([c]) * (e - s)
    return lab
