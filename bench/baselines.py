#!/usr/bin/env python3
"""Speed of other highlighters / tokenizers on the same sample taxnet is timed on.

    python bench/baselines.py           (run pinned to one core: taskset -c 0 ...)

Builds ~/.cache/taxnet/packs/bench_{c,go,py} (an 8 MB sample of each test pack: whole
files, fixed seed) and times, single-threaded:
  pygments    lexer.get_tokens over the decoded text (Pygments' C/Go/Python lexers)
  tree-sitter parser.parse (parse only, no highlight query)
  tokenize    CPython's tokenize (Python only; the reference for taxnet-py)
go/scanner and clang's tokenizer are timed by bench/gobench and bench/clangbench.c.
"""
import io
import json
import os
import random
import sys
import time
import tokenize

CACHE = os.path.expanduser("~/.cache/taxnet/packs")
SAMPLE = 8 << 20


def make_sample(lang):
    base = os.path.join(CACHE, "test_" + lang)
    idx = json.load(open(base + ".idx.json"))["files"]
    src, lbl = open(base + ".src", "rb").read(), open(base + ".lbl", "rb").read()
    rng = random.Random(7)
    rng.shuffle(idx)
    out_files, s, l, off = [], [], [], 0
    for path, o, n in idx:
        if off >= SAMPLE:
            break
        s.append(src[o:o + n]), l.append(lbl[o:o + n])
        out_files.append([path, off, n])
        off += n
    dst = os.path.join(CACHE, "bench_" + lang)
    open(dst + ".src", "wb").write(b"".join(s))
    open(dst + ".lbl", "wb").write(b"".join(l))
    json.dump({"files": out_files, "rejected": []}, open(dst + ".idx.json", "w"))
    return [x for x in s]


def timeit(fn, files, reps=2):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for f in files:
            fn(f)
        best = min(best, time.perf_counter() - t)
    return sum(len(f) for f in files) / best / 1e6


def main():
    from pygments.lexers import CLexer, GoLexer, PythonLexer
    import tree_sitter_c, tree_sitter_go, tree_sitter_python
    from tree_sitter import Language, Parser
    lexers = {"c": CLexer(), "go": GoLexer(), "py": PythonLexer()}
    grammars = {"c": tree_sitter_c, "go": tree_sitter_go, "py": tree_sitter_python}
    for lang in ("c", "go", "py"):
        files = make_sample(lang)
        texts = [f.decode("utf-8", "replace") for f in files]
        lex = lexers[lang]

        def pyg(t):
            for _ in lex.get_tokens(t):
                pass
        r = {"pygments": timeit(pyg, texts, reps=1)}
        parser = Parser(Language(grammars[lang].language()))
        r["tree-sitter parse"] = timeit(lambda f: parser.parse(f), files)
        if lang == "py":
            def tok(f):
                for _ in tokenize.tokenize(io.BytesIO(f).readline):
                    pass
            r["tokenize"] = timeit(tok, files, reps=1)
        print("%-3s %5.1f MB  " % (lang, sum(map(len, files)) / 1e6) +
              "  ".join("%s %.2f M chars/s" % kv for kv in r.items()), flush=True)


if __name__ == "__main__":
    main()
