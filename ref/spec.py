"""The highlighting spec shared by the reference labelers and the network builder.

Every byte of a source file gets exactly one class. The reference labelers derive the
classes from each language's *official* tokenizer (CPython's `tokenize`, Go's
`go/scanner`, clang's raw lexer via libclang) plus the small set of rules below, which
operate on the token stream, never on characters:

  ws       bytes not inside any token (spaces, newlines, line continuations)
  kw       keyword, as the official tokenizer defines it
  type     C type keywords and standard typedef names; Go predeclared types
  builtin  Python builtins; Go predeclared constants and functions; C NULL/true/false/EOF/std*
  func     identifier whose next token is `(` with only spaces/tabs between them
  ident    any other identifier
  num      numeric literal
  str      string / char / rune literal, prefixes included (C `#include <...>` too)
  com      comment
  op       operator or punctuation
  meta     C preprocessor `#` + directive name; Python decorator `@` + dotted name

`type` and `builtin` only apply when the previous token is not `.` (or `->` in C), so
attribute names like `obj.len` stay identifiers.
"""
import os

CLASSES = ["ws", "kw", "type", "builtin", "func", "ident", "num", "str", "com", "op", "meta"]
WS, KW, TYPE, BUILTIN, FUNC, IDENT, NUM, STR, COM, OP, META = range(len(CLASSES))

SPEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spec")


def words(name):
    with open(os.path.join(SPEC_DIR, name + ".txt")) as f:
        return [w for w in (line.strip() for line in f) if w and not w.startswith("#")]


class Reject(Exception):
    """The file is not valid input for this language's official tokenizer."""
