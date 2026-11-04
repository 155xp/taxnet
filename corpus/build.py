#!/usr/bin/env python3
"""Label every corpus file with its language's reference labeler and write packs.

    python corpus/build.py [dev|test] [py|go|c]

Writes, per (split, lang), into $TAXNET_CACHE/packs (default ~/.cache/taxnet/packs):
    {split}_{lang}.src        every accepted file's bytes, concatenated
    {split}_{lang}.lbl        one reference class byte per source byte
    {split}_{lang}.idx.json   [[path, offset, length], ...] and the rejected files

dev  = sources that ship with this machine (CPython 3.13 stdlib + Debian site-packages,
       the Go 1.24 standard library, C headers + Linux kernel headers + nvidia module).
test = the held-out repos fetched by corpus/fetch.sh. The weights were written
       without looking at these.
test2 = a second held-out set (corpus/fetch.sh test2), evaluated once; its errors then
       prompted the last round of fixes.
test3 = a third held-out set, fetched after those fixes and evaluated once.
"""
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ref"))
CACHE = os.environ.get("TAXNET_CACHE", os.path.expanduser("~/.cache/taxnet"))
PACKS = os.path.join(CACHE, "packs")

ROOTS = {
    "dev": {
        "py": ["/usr/lib/python3.13", "/usr/lib/python3/dist-packages"],
        "go": ["/usr/share/go-1.24/src"],
        "c": ["/usr/include", "/usr/src/linux-headers-6.12.96+deb13-common",
              "/usr/src/nvidia-current-550.163.01"],
    },
    "test": {lang: [os.path.join(CACHE, "corpus", "test", lang)] for lang in ("py", "go", "c")},
    "test2": {lang: [os.path.join(CACHE, "corpus", "test2", lang)] for lang in ("py", "go", "c")},
    "test3": {lang: [os.path.join(CACHE, "corpus", "test3", lang)] for lang in ("py", "go", "c")},
}
EXTS = {"py": (".py",), "go": (".go",), "c": (".c", ".h")}
MAX_BYTES = 16 << 20
# C headers on this machine include C++ (CUDA, clang, ...). Keep the C corpus C.
CXX = re.compile(rb"^\s*(namespace\b|template\s*<|class\s+\w+\s*[:{]|using\s+namespace\b)|\bstd::|"
                 rb"\b(static_cast|reinterpret_cast|const_cast|dynamic_cast)\s*<|\bextern\s+\"C\+\+\"",
                 re.M)


def find_files(split, lang):
    seen, out = set(), []
    for root in ROOTS[split][lang]:
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x not in ("c++", "__pycache__", ".git"))
            for f in sorted(files):
                p = os.path.join(d, f)
                if not f.endswith(EXTS[lang]) or not os.path.isfile(p) or os.path.islink(p):
                    continue
                rp = os.path.realpath(p)
                if rp in seen:
                    continue
                seen.add(rp)
                size = os.path.getsize(p)
                if 0 < size <= MAX_BYTES:
                    out.append(p)
    return out


def _label_one(args):
    lang, path = args
    import spec
    mod = __import__("ref_" + lang)
    with open(path, "rb") as f:
        data = f.read()
    if lang == "c" and CXX.search(data):
        return path, "c++", None
    try:
        return path, "OK", bytes(mod.label(data))
    except spec.Reject as e:
        return path, str(e), None


def label_files(lang, paths):
    if lang == "go":
        listf = os.path.join(PACKS, "_go_list.txt")
        outf = os.path.join(PACKS, "_go_labels.bin")
        with open(listf, "w") as f:
            f.write("".join(p + "\n" for p in paths))
        exe = os.path.join(PACKS, "_refgo")
        subprocess.run(["go", "build", "-o", exe, "."], cwd=os.path.join(REPO, "ref", "go"), check=True)
        status = subprocess.run([exe, "-spec", os.path.join(REPO, "spec"), "-list", listf, "-out", outf],
                                check=True, capture_output=True, text=True).stdout.splitlines()
        with open(outf, "rb") as f:
            blob = f.read()
        pos, out = 0, []
        for p, st in zip(paths, status):
            if st == "OK":
                n = os.path.getsize(p)
                out.append((p, "OK", blob[pos:pos + n]))
                pos += n
            else:
                out.append((p, st[4:], None))
        assert pos == len(blob)
        os.remove(listf), os.remove(outf)
        return out
    with mp.Pool(min(20, os.cpu_count())) as pool:
        return pool.map(_label_one, [(lang, p) for p in paths], chunksize=4)


def build(split, lang):
    os.makedirs(PACKS, exist_ok=True)
    paths = find_files(split, lang)
    results = label_files(lang, paths)
    base = os.path.join(PACKS, "%s_%s" % (split, lang))
    idx, rejects, off = [], [], 0
    with open(base + ".src", "wb") as fs, open(base + ".lbl", "wb") as fl:
        for p, st, lab in results:
            if lab is None:
                rejects.append([p, st])
                continue
            with open(p, "rb") as f:
                data = f.read()
            assert len(data) == len(lab), p
            fs.write(data)
            fl.write(lab)
            idx.append([p, off, len(data)])
            off += len(data)
    with open(base + ".idx.json", "w") as f:
        json.dump({"files": idx, "rejected": rejects}, f)
    print("%-4s %-3s %6d files %8.1f MB   rejected %d" % (split, lang, len(idx), off / 1e6, len(rejects)))
    return base


if __name__ == "__main__":
    splits = [a for a in sys.argv[1:] if a in ("dev", "test", "test2", "test3")] or ["dev", "test"]
    langs = [a for a in sys.argv[1:] if a in ("py", "go", "c")] or ["py", "go", "c"]
    for s in splits:
        for l in langs:
            build(s, l)
