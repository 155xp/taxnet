<img src="assets/icon.svg" width="64" height="64" alt="">

# taxnet

Neural networks that syntax-highlight **C, Go and Python**, with every weight written by
hand. Nothing was trained: there is no dataset, no optimizer, no gradient. Each network is
a bidirectional recurrent net of threshold neurons, `clip(Σ wᵢxᵢ + b, 0, 1)`, whose integer
weights are chosen one neuron at a time in `model/lang_{c,go,py}.py`.

| | C | Go | Python | total |
|---|---|---|---|---|
| **parameters (nonzero)** | 3,678 | 1,374 | 3,975 | **9,027** |
| neurons (forward + backward) | 890 + 89 | 262 + 38 | 1,091 + 46 | |
| **accuracy, every corpus** | 100.000000% | 100.000000% | 100.000000% | **0 wrong of 1,241,484,343 bytes** |
| speed, one core | 326 M chars/s | 591 M chars/s | 238 M chars/s | |
| speed, 16 threads | up to 2.5 G chars/s | up to 4.4 G chars/s | up to 1.5 G chars/s | |

For scale: [lexnet](https://huggingface.co/alpindale/lexnet), the hand-weighted C
highlighter that prompted this, has 984 parameters, covers C only, reports no accuracy
and runs at 11 M chars/s.

## What "accurate" means here

Each byte gets one of 11 classes: `ws kw type builtin func ident num str com op meta`.
The reference is each language's **own tokenizer**, plus a few token-level rules:

- **Python:** CPython 3.13 `tokenize`, f-string internals included.
- **Go:** the standard library's `go/scanner`.
- **C:** clang 19's raw lexer via libclang `clang_tokenize`, `-std=gnu17`.

The rules (see `ref/spec.py`):

- Keywords come from the official keyword sets: 132 clang gnu17 keywords (extracted and
  cross-checked against clang's `TokenKinds.def`), Go's 25, and Python's 35.
- `type` and `builtin` come from the predeclared or standard names in `spec/`, unless the
  previous token is `.` (so `obj.len` stays an identifier).
- `func` is an identifier whose next token is `(`, with only spaces or tabs between them.
- `meta` covers the C `#directive` name and Python `@decorator` dotted names.
- In C, `#include <...>` header names count as `str`.

The labelers never see the networks and the networks never see the labelers. Accuracy is
the fraction of bytes whose class matches.

## Corpora and methodology

| corpus | what | C | Go | Python | when it was used |
|---|---|---|---|---|---|
| dev | sources on this machine: /usr/include + kernel headers, Go 1.24 stdlib, CPython 3.13 stdlib + site-packages | 116.4 MB | 79.2 MB | 29.0 MB | while writing the weights |
| test | 22 repos (sqlite, git, cpython, vim, curl, ... / hugo, terraform, prometheus, ... / django, sympy, pandas, scikit-learn) | 115.4 MB | 73.7 MB | 88.7 MB | once, after dev reached 100% |
| test2 | 20 new repos (postgres, openssl, systemd, ... / moby, consul, containerd, ... / scipy, ansible, black, pytest, ...) | 147.7 MB | 158.0 MB | 47.9 MB | once, after the test fixes |
| test3 | 20 more (FFmpeg, glibc, php-src, ruby, neovim, raylib / gitea, traefik, minio, rclone, ... / numpy, sqlalchemy, sphinx, twisted, ...) | 248.7 MB | 66.3 MB | 70.3 MB | once, after the test2 fixes |

Every repo is pinned to a commit in `corpus/{test,test2,test3}_repos.lock`. Each held-out
set was scored exactly once before anything it revealed was fixed:

| first look | C | Go | Python | what the errors were |
|---|---|---|---|---|
| test | 100% ¹ | 100% | 99.999998% (2 B) | `@` starting a continuation line inside brackets is matrix multiplication, not a decorator |
| test2 | 100% | 99.999996% (6 B) | 99.999848% (73 B) | UTF-8 BOMs (Go); black's PEP 701 torture tests (f-strings nested three deep), a CRLF comment, `@lambda` decorator, `.1.real` |
| test3 | 99.999999% (2 B) | 100% | 100% | `case 64 ...127:`: the `.` before `127` closes an ellipsis, it does not start a number |

¹ The first C run showed 84 errors. All of them were labels built before a reference bug
had been fixed on dev (splice-prefixed tokens like `\`⏎`void` were read with the splice
still in their spelling). Relabelled with the fixed reference, the same weights score 100%.

After those fixes the final networks score 100.000000% on all four corpora.

## The network

```
x_t = E[byte_t]                      byte embedding, 256 x F, entries 0/1
f_t = cell_fwd(x_t, f_{t-1})         forward recurrent cell (9-12 layers per step)
b_t = cell_bwd(x_t, f_t, b_{t+1})    backward recurrent cell (5-8 layers per step)
y_t = argmax(W_head b_t + c)         one class per byte
```

Every neuron is `clip(Σ wᵢ·inputᵢ + b, 0, 1)` with integer weights. Its inputs are
embedding features of the current byte, forward neurons of the previous step, backward
neurons of the next step, or neurons already computed this step. Integer weights on 0/1
inputs keep every activation exactly 0 or 1.

The forward cell is the lexer state:

- comments, strings and chars
- C line splices, preprocessor directives and `#include <...>`
- Go raw strings and runes
- Python's three nesting levels of f-strings (quote, triple, raw/f flags, literal/field/spec mode, bracket depth, spec nesting)
- each language's number grammar
- the member-access `.`, and decorators

The backward cell supplies the lookahead a left-to-right lexer can't have:

- is this word followed by `(`?
- does the word end here, and what is it?
- is this `/` the start of a comment?
- is this `{` an f-string field or the first half of `{{`?
- does this splice belong to the next token?

Keywords, builtins and string prefixes are recognized by a **trie of neurons**: one neuron
per (deduplicated) trie node, firing when its character arrives right after its parent fired.
That's most of the parameter count, because Python has 141 builtins and clang has 132
keywords.

`weights/*.json` lists every neuron with its named inputs and weights. `weights/*.tnw` is
the binary weight file the executors load.

## Why it's fast

Every activation is a bit, so 256 independent byte streams run at once, one per bit of an
AVX2 register: a neuron's output for all 256 streams is one 256-bit vector.
`kernel/codegen.py` compiles each neuron's weights into AND/OR/ANDNOT instructions. It
**proves the translation exact** first: it checks the inequalities that make the threshold
neuron equal to `AND(A) & OR(O) & ~OR(N)` on every 0/1 input. Of the 2,416 neurons, 2,411
pass and the other 5 are constants, so no truth-table fallbacks are needed.

The runtime cuts input into chunks of about 2 KB. It prefers newlines where a top-level
statement starts, so the guess about the lexer state there is usually right. Up to 256
chunks run in lockstep per core.

A chunk that doesn't start a file begins from a guessed state (after a newline in plain
code). Chunks are then validated left to right. A chunk counts only if it started from
exactly its predecessor's validated end state; otherwise it is re-run from the right state.
Speculative re-runs of the following chunks let a long comment or docstring resolve in a
round or two. The result is exact whatever the chunking, and one large file still spreads
over every core.

Single core, same 8 MB sample of held-out files per language (`make bench`):

| chars/s, 1 core | C | Go | Python |
|---|---|---|---|
| **taxnet** | **326 M** | **591 M** | **238 M** |
| the language's own tokenizer | 80 M (clang `clang_tokenize`) | 206 M (`go/scanner`) | 19.5 M (`tokenize`) |
| tree-sitter, parse only | 10.7 M | 16.2 M | 14.9 M |
| Pygments | 1.9 M | 3.6 M | 1.6 M |

Machine: i7-13700KF (8 P + 8 E cores, AVX2), one P-core pinned. The multi-thread numbers
vary run to run with the corpus mix and other load on the machine.

## It really is the network

`model/numpy_ref.py` runs a weight file the plain way. It builds dense float32 layer
matrices (1.78 M / 181 K / 2.55 M entries, almost all zero) and does `clip(W @ v + b, 0, 1)`
per layer, per byte, then an argmax head. It knows nothing about bits, gates or chunks.
`make verify` compares it with the compiled kernel on 60 files per language: **0 bytes
differ**, and both match the reference labels.

## Run it

```sh
make                                   # weights -> kernels -> ./taxnet (needs gcc, python3)
./taxnet py some_file.py               # ANSI-highlighted output
./taxnet c  --classes some_file.c      # one class digit per byte
./taxnet go --info                     # neuron counts + sha256 of the weight file it was compiled from
```

The full evaluation needs Go 1.24, clang/libclang 19 and a venv with `numpy`,
`clang==19.1.7`, `pygments` and `tree-sitter` (`PY=...`). Run `make corpus && make eval
verify bench`. The labelled corpora (about 1.2 GB) live in `~/.cache/taxnet`.

## Layout

```
spec/            word lists: keywords, types, builtins per language
ref/             reference labelers (tokenize / go/scanner / libclang) + the class rules
model/           the hand-written networks, weight-file format, dense numpy executor
weights/         c.tnw go.tnw py.tnw (+ .json listing every neuron)
kernel/          codegen (weights -> proven bit-sliced C) and the runtime
corpus/          corpus download (pinned repos) and labelling
bench/           error inspector, numpy-vs-kernel check, baseline benchmarks
```

## Limits

- **Python** f-strings nest at most three string levels deep, and format specs three deep.
  Bracket depth inside an f-string field saturates at 4, and at top level (used for
  decorators) at 12.
- **C** follows clang's gnu17 lexing and treats whole files as C: C++ files were filtered
  out of the corpora.
- **All three:** bytes ≥ 0x80 in code are treated as identifier bytes. That's right for
  valid UTF-8 identifiers, which is all the corpora contain.
- **Benchmarks:** the speed numbers are from one machine.
