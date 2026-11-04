"""C reference labeler: clang 19's raw lexer (libclang clang_tokenize, -std=gnu17) + spec.py rules."""
import re

import clang.cindex as ci

from spec import BUILTIN, COM, FUNC, IDENT, KW, META, NUM, OP, STR, TYPE, Reject, words

ci.Config.set_library_file("/usr/lib/llvm-19/lib/libclang-19.so.1")
KEYWORDS = frozenset(words("c_keywords"))
TYPES = frozenset(words("c_types"))
BUILTINS = frozenset(words("c_builtins"))
INCLUDES = {b"include", b"include_next", b"import"}
ARGS = ["-std=gnu17", "-nostdinc", "-ferror-limit=1", "-Wno-everything"]
_index = None

K_PUNCT, K_KEYWORD, K_IDENT, K_LITERAL, K_COMMENT = range(5)
# A token that directly follows a backslash-newline splice starts at the backslash, so its
# raw text can contain splices; classify tokens by their spelling with splices removed.
SPLICE = re.compile(rb"\\[ \t\f\v]*\r?\n")


def spell(data, s, e):
    t = data[s:e]
    return SPLICE.sub(b"", t) if b"\\" in t else t
KIND = {"PUNCTUATION": K_PUNCT, "KEYWORD": K_KEYWORD, "IDENTIFIER": K_IDENT,
        "LITERAL": K_LITERAL, "COMMENT": K_COMMENT}


def unspliced_newline(text):
    """True if `text` (whitespace between tokens) holds a newline that is not a line splice."""
    i = text.find(b"\n")
    while i >= 0:
        j = i - 1
        while j >= 0 and text[j] in b" \t\f\v\r":
            j -= 1
        if j < 0 or text[j] != 0x5C:  # no backslash before it
            return True
        i = text.find(b"\n", i + 1)
    return False


def tokens(data):
    global _index
    if _index is None:
        _index = ci.Index.create()
    tu = _index.parse("t.c", args=ARGS, unsaved_files=[("t.c", data)],
                      options=ci.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)
    out = []
    for t in tu.get_tokens(extent=tu.cursor.extent):
        ext = t.extent
        out.append((KIND[t.kind.name], ext.start.offset, ext.end.offset))
    return out


def label(data):
    if b"\0" in data:
        raise Reject("nul")
    if b"\r" in data.replace(b"\r\n", b""):
        raise Reject("lone-cr")
    toks = tokens(data)
    lab = bytearray(len(data))
    n = len(toks)
    at_line_start = True   # only whitespace/comments since the last unspliced newline
    prev_end = 0
    i = 0
    while i < n:
        kind, s, e = toks[i]
        if unspliced_newline(data[prev_end:s]):
            at_line_start = True
        text = spell(data, s, e)
        if kind == K_COMMENT:
            lab[s:e] = bytes([COM]) * (e - s)
            if text.startswith(b"/*") and b"\n" in data[s:e]:
                at_line_start = True
            prev_end = e
            i += 1
            continue
        line_start, at_line_start = at_line_start, False
        prev_end = e
        if kind == K_PUNCT and text == b"#" and line_start:
            # preprocessor directive: `#` and the directive name are meta
            lab[s:e] = bytes([META])
            i += 1
            if i < n and toks[i][0] in (K_IDENT, K_KEYWORD) and not unspliced_newline(data[e:toks[i][1]]):
                _, ns, ne = toks[i]
                lab[ns:ne] = bytes([META]) * (ne - ns)
                prev_end = ne
                i += 1
                if spell(data, ns, ne) in INCLUDES and i < n and spell(data, toks[i][1], toks[i][2]) == b"<" \
                        and not unspliced_newline(data[ne:toks[i][1]]):
                    # `#include <header>`: the header name is one string token
                    j = i
                    while j < n and not unspliced_newline(data[toks[j - 1][2]:toks[j][1]]) \
                            and spell(data, toks[j][1], toks[j][2]) != b">":
                        j += 1
                    if j < n and spell(data, toks[j][1], toks[j][2]) == b">" \
                            and not unspliced_newline(data[toks[j - 1][2]:toks[j][1]]):
                        hs, he = toks[i][1], toks[j][2]
                        lab[hs:he] = bytes([STR]) * (he - hs)
                        prev_end = he
                        i = j + 1
            continue
        if kind == K_LITERAL:
            c = NUM if text[:1].isdigit() or text[:1] == b"." else STR
        elif kind == K_PUNCT:
            c = STR if text[:1] in (b"'", b'"') else OP  # unterminated literal runs to end of line
        elif kind == K_KEYWORD:
            c = TYPE if text.decode() in TYPES else KW
        else:
            name = text.decode("utf-8", "replace")
            after_dot = i > 0 and spell(data, toks[i - 1][1], toks[i - 1][2]) in (b".", b"->")
            if name in TYPES and not after_dot:
                c = TYPE
            elif name in BUILTINS and not after_dot:
                c = BUILTIN
            elif i + 1 < n and spell(data, toks[i + 1][1], toks[i + 1][2]) == b"(" \
                    and not data[e:toks[i + 1][1]].strip(b" \t"):
                c = FUNC
            else:
                c = IDENT
        lab[s:e] = bytes([c]) * (e - s)
        i += 1
    return lab
