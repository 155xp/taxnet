/* taxnet runtime: runs the bit-sliced kernels generated from the weight files.
 *
 *   taxnet <lang> FILE                 highlight FILE to the terminal (ANSI colours)
 *   taxnet <lang> --classes FILE       one class digit per input byte
 *   taxnet <lang> --eval PACK [-t N]   accuracy against PACK.lbl, plus timing
 *   taxnet <lang> --bench PACK [-t N] [-r R]
 *
 * Execution model. Input is cut into chunks (normally ending right after a newline).
 * 256 chunks run in lockstep, one per bit of a 256-bit vector: every neuron's value for
 * all 256 chunks is one AVX2 register and each neuron costs a few bitwise instructions.
 * A chunk other than the first of its file starts from a guessed state (the state after a
 * newline in plain code). After a round, any chunk whose true start state (the end state
 * of its predecessor) differs from its guess is re-run from the true state, until nothing
 * changes. The backward cell resets at a newline, so chunks need nothing from their
 * right-hand neighbour; a chunk cut inside a very long line instead carries `ext` bytes of
 * lookahead it processes but does not output.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint64_t V __attribute__((vector_size(32)));
typedef char C32 __attribute__((vector_size(32)));
typedef uint8_t B32 __attribute__((vector_size(32)));
typedef uint16_t H16 __attribute__((vector_size(32)));
#define ZERO ((V){0, 0, 0, 0})
#define ONES ((V){~0ULL, ~0ULL, ~0ULL, ~0ULL})
#define LANES 256
#define TILE 32

#include "gen/net_c.h"
#if __has_include("gen/net_go.h")
#include "gen/net_go.h"
#define HAVE_GO 1
#endif
#if __has_include("gen/net_py.h")
#include "gen/net_py.h"
#define HAVE_PY 1
#endif

/* ---- jobs and scratch ------------------------------------------------------------ */
typedef struct {
    const uint8_t *src;
    int len;            /* bytes whose classes we keep */
    int ext;            /* extra lookahead bytes processed after them */
    uint8_t *out;
    const uint8_t *init; /* start state, one byte (0/1) per persistent forward unit */
    uint8_t *end;       /* filled with the state after src[len-1] (may be NULL) */
} Job;

typedef struct {
    V *in, *save, *cls;
    size_t cap;
} Scratch;

static void reserve(Scratch *s, size_t T, int nsave) {
    if (T <= s->cap) return;
    free(s->in), free(s->save), free(s->cls);
    s->cap = T + T / 4 + 64;
    s->in = aligned_alloc(64, s->cap * 8 * sizeof(V));
    s->save = aligned_alloc(64, s->cap * (size_t)(nsave ? nsave : 1) * sizeof(V));
    s->cls = aligned_alloc(64, s->cap * 4 * sizeof(V));
}

/* 64x64 bit-matrix transpose: bit j of a[i] <-> bit i of a[j] */
static inline void transpose64(uint64_t a[64]) {
    uint64_t m = 0x00000000FFFFFFFFULL;
    for (int j = 32; j; j >>= 1, m ^= m << j)
        for (int k = 0; k < 64; k = ((k | j) + 1) & ~j) {
            uint64_t t = ((a[k] >> j) ^ a[k | j]) & m;
            a[k] ^= t << j;
            a[k | j] ^= t;
        }
}

/* 8x8 byte-matrix transpose: byte j of a[i] <-> byte i of a[j] */
static inline void transpose8x8(uint64_t a[8]) {
    for (int i = 0; i < 4; i++) {
        uint64_t t = ((a[i] >> 32) ^ a[i + 4]) & 0x00000000FFFFFFFFULL;
        a[i] ^= t << 32, a[i + 4] ^= t;
    }
    for (int i = 0; i < 8; i += 4)
        for (int k = 0; k < 2; k++) {
            uint64_t t = ((a[i + k] >> 16) ^ a[i + k + 2]) & 0x0000FFFF0000FFFFULL;
            a[i + k] ^= t << 16, a[i + k + 2] ^= t;
        }
    for (int i = 0; i < 8; i += 2) {
        uint64_t t = ((a[i] >> 8) ^ a[i + 1]) & 0x00FF00FF00FF00FFULL;
        a[i] ^= t << 8, a[i + 1] ^= t;
    }
}

/* bytes of 256 lanes -> 8 bit-planes per step. Lanes past their end read NUL. */
static void transpose_in(const Job *J, int n, int t0, int tn, V *in) {
    uint64_t buf[TILE][LANES / 8] __attribute__((aligned(32)));
    for (int g = 0; g < LANES / 8; g++)
        for (int b = 0; b < TILE; b += 8) {
            uint64_t a[8];
            for (int i = 0; i < 8; i++) {
                int l = g * 8 + i;
                long avail = l < n ? (long)J[l].len + J[l].ext - (t0 + b) : 0;
                if (avail >= 8) memcpy(&a[i], J[l].src + t0 + b, 8);
                else {
                    a[i] = 0;
                    if (avail > 0) memcpy(&a[i], J[l].src + t0 + b, (size_t)avail);
                }
            }
            transpose8x8(a);
            for (int j = 0; j < 8; j++) buf[b + j][g] = a[j];
        }
    for (int t = 0; t < tn; t++) {
        uint32_t m[8][8];
        for (int g = 0; g < 8; g++) {
            B32 v;
            memcpy(&v, (const uint8_t *)buf[t] + g * 32, 32);
            for (int k = 7; k >= 0; k--) {
                m[k][g] = (uint32_t)__builtin_ia32_pmovmskb256((C32)v);
                v = (B32)((H16)v << 1);
            }
        }
        for (int k = 0; k < 8; k++) {
            V p = ZERO;
            for (int q = 0; q < 4; q++) p[q] = (uint64_t)m[k][2 * q] | (uint64_t)m[k][2 * q + 1] << 32;
            in[(size_t)t * 8 + k] = p;
        }
    }
}

/* class bit-planes -> one class byte per lane per step, 64 steps x 64 lanes at a time */
static void transpose_out(const Job *J, int n, int T, const V *cls) {
    uint64_t A[4][64];
    for (int t0 = 0; t0 < T; t0 += 64) {
        int tn = T - t0 < 64 ? T - t0 : 64;
        for (int q = 0; q < 4 && q * 64 < n; q++) {
            for (int k = 0; k < 4; k++) {
                for (int i = 0; i < tn; i++) A[k][i] = cls[(size_t)(t0 + i) * 4 + k][q];
                for (int i = tn; i < 64; i++) A[k][i] = 0;
                transpose64(A[k]);
            }
            for (int l = 0; l < 64; l++) {
                int lane = q * 64 + l;
                if (lane >= n || t0 >= J[lane].len) continue;
                int m = J[lane].len - t0 < 64 ? J[lane].len - t0 : 64;
                uint8_t *o = J[lane].out + t0;
                for (int j = 0; j < m; j += 8) {
                    uint64_t v = __builtin_ia32_pdep_di(A[0][l] >> j, 0x0101010101010101ULL) |
                                 __builtin_ia32_pdep_di(A[1][l] >> j, 0x0101010101010101ULL) << 1 |
                                 __builtin_ia32_pdep_di(A[2][l] >> j, 0x0101010101010101ULL) << 2 |
                                 __builtin_ia32_pdep_di(A[3][l] >> j, 0x0101010101010101ULL) << 3;
                    memcpy(o + j, &v, m - j < 8 ? (size_t)(m - j) : 8);
                }
            }
        }
    }
}

typedef struct {
    const char *name;
    int npersist;
    const unsigned char *h0;
    const char *weights_sha256;
    int nfwd, nbwd;
    void (*batch)(Job *, int, Scratch *);
    uint8_t *spec;      /* guessed state for chunks that start after a newline */
    uint8_t *spec_mid;  /* ... and for chunks cut after a space inside a long line */
} Lang;

static int cmp_int(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }

/* start-state planes: lanes sharing an init array are set by broadcast */
static void init_planes(const Job *J, int n, int np, V *st) {
    const uint8_t *shared[8];
    V mask[8];
    int ns = 0;
    uint64_t own[4] = {0, 0, 0, 0};
    for (int l = 0; l < n; l++) {
        int k = 0;
        while (k < ns && shared[k] != J[l].init) k++;
        if (k == ns && ns < 8) shared[ns] = J[l].init, mask[ns++] = ZERO;
        if (k < ns) mask[k][l >> 6] |= 1ULL << (l & 63);
        else own[l >> 6] |= 1ULL << (l & 63);
    }
    for (int p = 0; p < np; p++) {
        V v = ZERO;
        for (int k = 0; k < ns; k++)
            if (shared[k][p]) v |= mask[k];
        st[p] = v;
    }
    for (int l = 0; l < n; l++)
        if (own[l >> 6] >> (l & 63) & 1)
            for (int p = 0; p < np; p++) st[p][l >> 6] |= (uint64_t)J[l].init[p] << (l & 63);
}

#ifdef TAXNET_PROFILE
static double prof[6];
static long prof_steps, prof_bytes;
static double tick(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#define PROF(i) do { double _t = tick(); prof[i] += _t - _pt; _pt = _t; } while (0)
#define PROF0 double _pt = tick()
#define PROF_COUNT(T, J, n) do { prof_steps += (T); for (int _i = 0; _i < (n); _i++) prof_bytes += (J)[_i].len + (J)[_i].ext; } while (0)
#else
#define PROF_COUNT(T, J, n) do {} while (0)
#define PROF(i) do {} while (0)
#define PROF0 do {} while (0)
#endif
#define DEFINE_LANG(SYM)                                                                  \
    static void SYM##_batch(Job *J, int n, Scratch *S) {                                  \
        PROF0;                                                                            \
        int T = 0;                                                                        \
        for (int i = 0; i < n; i++)                                                       \
            if (J[i].len + J[i].ext > T) T = J[i].len + J[i].ext;                         \
        reserve(S, (size_t)T, SYM##_NSAVE);                                               \
        PROF_COUNT(T, J, n);                                                              \
        static __thread V st[SYM##_NPERSIST];                                             \
        init_planes(J, n, SYM##_NPERSIST, st);                                            \
        int cap[LANES], ncap = 0, ci = 0;                                                 \
        for (int l = 0; l < n; l++)                                                       \
            if (J[l].end) cap[ncap++] = (J[l].len - 1) << 8 | l;                          \
        qsort(cap, ncap, sizeof(int), cmp_int);                                           \
        PROF(0);                                                                          \
        for (int t0 = 0; t0 < T; t0 += TILE) {                                            \
            int tn = T - t0 < TILE ? T - t0 : TILE;                                       \
            transpose_in(J, n, t0, tn, S->in + (size_t)t0 * 8);                           \
            PROF(1);                                                                      \
            for (int t = t0; t < t0 + tn; t++) {                                          \
                SYM##_fwd(S->in + (size_t)t * 8, st, S->save + (size_t)t * SYM##_NSAVE);  \
                while (ci < ncap && cap[ci] >> 8 == t) {                                  \
                    int l = cap[ci++] & 255;                                              \
                    for (int p = 0; p < SYM##_NPERSIST; p++)                              \
                        J[l].end[p] = (uint8_t)((st[p][l >> 6] >> (l & 63)) & 1);         \
                }                                                                         \
            }                                                                             \
            PROF(2);                                                                      \
        }                                                                                 \
        V nx[SYM##_NNEXT > 0 ? SYM##_NNEXT : 1];                                          \
        for (int i = 0; i < (SYM##_NNEXT > 0 ? SYM##_NNEXT : 1); i++) nx[i] = ZERO;       \
        for (int t = T - 1; t >= 0; t--)                                                  \
            SYM##_bwd(S->in + (size_t)t * 8, S->save + (size_t)t * SYM##_NSAVE, nx,       \
                      S->cls + (size_t)t * 4);                                            \
        PROF(3);                                                                          \
        transpose_out(J, n, T, S->cls);                                                   \
        PROF(4);                                                                          \
    }

DEFINE_LANG(C)
#ifdef HAVE_GO
DEFINE_LANG(GO)
#endif
#ifdef HAVE_PY
DEFINE_LANG(PY)
#endif

static Lang LANGS[] = {
    {"c", C_NPERSIST, C_h0, C_WEIGHTS_SHA256, C_NFWD, C_NBWD, C_batch, NULL, NULL},
#ifdef HAVE_GO
    {"go", GO_NPERSIST, GO_h0, GO_WEIGHTS_SHA256, GO_NFWD, GO_NBWD, GO_batch, NULL, NULL},
#endif
#ifdef HAVE_PY
    {"py", PY_NPERSIST, PY_h0, PY_WEIGHTS_SHA256, PY_NFWD, PY_NBWD, PY_batch, NULL, NULL},
#endif
};

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---- whole files ------------------------------------------------------------------ */
#define CHUNK 2048          /* target chunk size */
#define CHUNK_MAX 16384     /* past this a chunk is cut mid-line ...  */
#define OVERLAP 1024        /* ... with this much lookahead */

/* Where to end the chunk that starts at `pos`. Any cut gives exact output (a chunk whose
 * guessed start state was wrong is re-run), so this only decides how often that happens:
 * prefer a newline whose next line starts in column 0 with something that looks like code,
 * where the true state is almost always the guess. */
static int starts(const uint8_t *s, size_t n, const char *w) {
    size_t k = strlen(w);
    return n >= k && !memcmp(s, w, k);
}

/* 2: the line almost surely starts in plain top-level code; 1: probably; 0: no idea */
static int code_line(const Lang *L, const uint8_t *s, size_t n) {
    uint8_t c = n ? s[0] : 0;
    if (L->name[0] == 'c') {
        if (c == '#' || c == '}') return 2;
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
    if (L->name[0] == 'g') {
        if (starts(s, n, "func ") || starts(s, n, "type ") || starts(s, n, "var ") || starts(s, n, "const ") || c == '}')
            return 2;
        return c >= 'a' && c <= 'z';
    }
    if (starts(s, n, "import ") || starts(s, n, "from ")) return 2;
    size_t k = 0;                       /* Python: look past the indentation */
    while (k < n && k < 64 && (s[k] == ' ' || s[k] == '\t')) k++;
    const uint8_t *t = s + k;
    size_t m = n - k;
    if (starts(t, m, "def ") || starts(t, m, "class ") || starts(t, m, "async def ") || starts(t, m, "@") ||
        starts(t, m, "elif ") || starts(t, m, "else:") || starts(t, m, "try:") || starts(t, m, "except") ||
        starts(t, m, "finally:"))
        return 2;
    return k == 0 && ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c == '#');
}

static int spliced(const uint8_t *s, size_t pos, size_t nl) {   /* s[nl] == '\n' */
    size_t k = nl;
    while (k > pos && (s[k - 1] == ' ' || s[k - 1] == '\t' || s[k - 1] == '\r')) k--;
    return k > pos && s[k - 1] == '\\';
}

static size_t cut(const Lang *L, const uint8_t *s, size_t pos, size_t n, size_t *ext, int *mid) {
    size_t lo = pos + CHUNK, hi = pos + 4 * CHUNK < n ? pos + 4 * CHUNK : n, first = 0, weak = 0;
    *mid = 0;
    for (size_t i = lo - 1; i + 1 < hi; i++) {
        const uint8_t *q = memchr(s + i, '\n', hi - 1 - i);
        if (!q) break;
        i = (size_t)(q - s);
        if (spliced(s, pos, i)) continue;
        if (!first) first = i + 1;
        int k = code_line(L, s + i + 1, n - i - 1);
        if (k == 2) return i + 1;
        if (k == 1 && !weak) weak = i + 1;
    }
    if (weak) return weak;
    if (first) return first;
    /* a very long line: cut after a blank, guess "code, mid-line", and give the chunk
       lookahead so its last bytes see what follows */
    size_t e = lo;
    while (e < hi && s[e - 1] != ' ' && s[e - 1] != '\t') e++;
    if (e >= hi) e = hi;
    if (e < n) *ext = n - e < OVERLAP ? n - e : OVERLAP, *mid = 1;
    return e;
}

typedef struct {            /* one run of a chunk from one start state */
    const uint8_t *init;
    uint8_t *end, *out;     /* `out` is the chunk's real output only for the first run */
} Cand;

typedef struct {
    const uint8_t *src;
    int len, ext;
    uint8_t *out;
    int file;
    int ncand, capcand, nrun;   /* candidates made; how many of them have run */
    Cand *cand;
    int valid;              /* index of the validated candidate, or -1 */
} Chunk;

typedef struct {            /* one scheduled run */
    int chunk, cand;
} Task;

/* ---- a small persistent thread pool: parallel-for over items --------------------------- */
typedef struct Pool {
    void (*fn)(struct Pool *, void *, int, Scratch *);
    void *ctx;
    int nitems, chunk;
    atomic_int cursor, done;
    int nthreads, generation, quit;
    pthread_mutex_t mu;
    pthread_cond_t go, finished;
} Pool;

static void pool_work(Pool *P, Scratch *S) {
    for (;;) {
        int lo = atomic_fetch_add(&P->cursor, P->chunk);
        if (lo >= P->nitems) break;
        int hi = lo + P->chunk < P->nitems ? lo + P->chunk : P->nitems;
        for (int i = lo; i < hi; i++) P->fn(P, P->ctx, i, S);
    }
}

static void *pool_main(void *arg) {
    Pool *P = arg;
    Scratch S = {0};
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (P->generation == seen && !P->quit) pthread_cond_wait(&P->go, &P->mu);
        if (P->quit) { pthread_mutex_unlock(&P->mu); break; }
        seen = P->generation;
        pthread_mutex_unlock(&P->mu);
        pool_work(P, &S);
        if (atomic_fetch_add(&P->done, 1) + 1 == P->nthreads) {
            pthread_mutex_lock(&P->mu);
            pthread_cond_signal(&P->finished);
            pthread_mutex_unlock(&P->mu);
        }
    }
    free(S.in), free(S.save), free(S.cls);
    return NULL;
}

static Scratch main_scratch;

/* run fn(i) for i in [0, n) on every thread, `chunk` items at a time */
static void pool_for(Pool *P, int n, int chunk, void (*fn)(Pool *, void *, int, Scratch *), void *ctx) {
    P->fn = fn, P->ctx = ctx, P->nitems = n, P->chunk = chunk;
    atomic_store(&P->cursor, 0);
    atomic_store(&P->done, 0);
    pthread_mutex_lock(&P->mu);
    P->generation++;
    pthread_cond_broadcast(&P->go);
    pthread_mutex_unlock(&P->mu);
    pool_work(P, &main_scratch);             /* the calling thread works too */
    pthread_mutex_lock(&P->mu);
    while (atomic_load(&P->done) < P->nthreads) pthread_cond_wait(&P->finished, &P->mu);
    pthread_mutex_unlock(&P->mu);
}

static int task_len_desc(const void *a, const void *b, void *ctx) {
    const Chunk *c = ctx;
    const Chunk *x = &c[((const Task *)a)->chunk], *y = &c[((const Task *)b)->chunk];
    return (y->len + y->ext) - (x->len + x->ext);
}

typedef struct { int rounds; long reruns, chunks; } RunStats;

static int add_cand(Chunk *c, const uint8_t *init, int np, int direct) {
    for (int k = 0; k < c->ncand; k++)
        if (c->cand[k].init == init || !memcmp(c->cand[k].init, init, np)) return -1;
    if (c->ncand == c->capcand) c->cand = realloc(c->cand, (c->capcand = c->capcand * 2 + 1) * sizeof(Cand));
    Cand *k = &c->cand[c->ncand];
    k->init = init;
    k->end = malloc(np);
    k->out = direct ? c->out : malloc(c->len);
    return c->ncand++;
}

typedef struct {
    const Lang *L;
    int nf, np, K;
    const uint8_t **src;
    const size_t *len;
    uint8_t **out;
    size_t **cuts;          /* per file: end, ext, mid triples from the cutter */
    int *ncuts, *first, *done_upto;
    Chunk *ch;
    Task *tasks;            /* tasks of the current round */
    int ntasks;
    Task **ftasks;          /* per file: tasks scheduled for the next round */
    int *nftasks, *capftasks;
    uint8_t ***fowned;      /* per file: start states allocated for re-runs */
    int *nfowned;
} Run;

static void do_cut(Pool *P, void *ctx, int f, Scratch *S) {
    Run *R = ctx;
    (void)P, (void)S;
    size_t pos = 0, n = R->len[f], cap = 16, k = 0;
    size_t *c = malloc(cap * 3 * sizeof(size_t));
    while (pos < n) {
        size_t end, ext = 0;
        int mid = 0;
        if (n - pos <= CHUNK + CHUNK / 2) end = n;
        else end = cut(R->L, R->src[f], pos, n, &ext, &mid);
        if (k == cap) c = realloc(c, (cap *= 2) * 3 * sizeof(size_t));
        c[3 * k] = end, c[3 * k + 1] = ext, c[3 * k + 2] = (size_t)mid;
        k++;
        pos = end;
    }
    R->cuts[f] = c, R->ncuts[f] = (int)k;
}

static void do_fill(Pool *P, void *ctx, int f, Scratch *S) {
    Run *R = ctx;
    (void)P, (void)S;
    size_t pos = 0;
    int prev_mid = 0;
    for (int k = 0; k < R->ncuts[f]; k++) {
        size_t end = R->cuts[f][3 * k], ext = R->cuts[f][3 * k + 1];
        Chunk *c = &R->ch[R->first[f] + k];
        *c = (Chunk){R->src[f] + pos, (int)(end - pos), (int)ext, R->out[f] + pos, f, 0, 0, 0, NULL, -1};
        add_cand(c, pos == 0 ? R->L->h0 : prev_mid ? R->L->spec_mid : R->L->spec, R->np, 1);
        prev_mid = (int)R->cuts[f][3 * k + 2];
        pos = end;
    }
    free(R->cuts[f]);
}

static void do_batch(Pool *P, void *ctx, int b, Scratch *S) {
    Run *R = ctx;
    (void)P;
    Job batch[LANES];
    int lo = b * LANES, n = R->ntasks - lo < LANES ? R->ntasks - lo : LANES;
    for (int i = 0; i < n; i++) {
        Task *t = &R->tasks[lo + i];
        Chunk *c = &R->ch[t->chunk];
        Cand *k = &c->cand[t->cand];
        batch[i] = (Job){c->src, c->len, c->ext, k->out, k->init, k->end};
    }
    R->L->batch(batch, n, S);
}

static void push_task(Run *R, int f, int chunk, int cand) {
    if (R->nftasks[f] == R->capftasks[f])
        R->ftasks[f] = realloc(R->ftasks[f], (R->capftasks[f] = R->capftasks[f] * 2 + 8) * sizeof(Task));
    R->ftasks[f][R->nftasks[f]++] = (Task){chunk, cand};
}

/* validate one file left to right, then schedule its next runs */
static void do_walk(Pool *P, void *ctx, int f, Scratch *S) {
    Run *R = ctx;
    (void)P, (void)S;
    Chunk *ch = R->ch;
    int np = R->np, i = R->done_upto[f], end = R->first[f + 1], K = R->K;
    for (; i < end; i++) {
        Chunk *c = &ch[i];
        const uint8_t *want = i == R->first[f] ? R->L->h0 : ch[i - 1].cand[ch[i - 1].valid].end;
        int k = 0;
        while (k < c->ncand && memcmp(c->cand[k].init, want, np)) k++;
        if (k == c->ncand) break;
        c->valid = k;
        if (c->cand[k].out != c->out) memcpy(c->out, c->cand[k].out, c->len);
    }
    R->done_upto[f] = i;
    if (i == end) return;
    /* (a) the first unvalidated chunk from its true start state, (b) its successors from
       that same state, (c) later chunks from where their predecessor's latest run ended */
    uint8_t *st = malloc(np);
    memcpy(st, ch[i - 1].cand[ch[i - 1].valid].end, np);
    R->fowned[f] = realloc(R->fowned[f], (R->nfowned[f] + 1) * sizeof(uint8_t *));
    R->fowned[f][R->nfowned[f]++] = st;
    for (int j = i; j < end && j <= i + K; j++) {
        int k = add_cand(&ch[j], st, np, 0);
        if (k >= 0) push_task(R, f, j, k);
    }
    for (int j = i + K + 1; j < end; j++) {
        Chunk *p = &ch[j - 1];
        int k = add_cand(&ch[j], p->cand[p->nrun - 1].end, np, 0);
        if (k >= 0) push_task(R, f, j, k);
    }
}

/* Highlight `nf` buffers with `nthreads` threads.
 *
 * Every chunk keeps the runs ("candidates") made of it, each from some start state. A
 * file is validated left to right: chunk i is final once one of its candidates started from
 * the end state of chunk i-1's validated candidate (chunk 0 from the file's start state).
 * Each round runs, for every file, (a) the first unvalidated chunk from its true start
 * state, (b) the next few chunks from that same state (a long comment or string usually
 * continues), and (c) every later chunk from the end state its predecessor's latest run
 * reached. (a) guarantees progress; (b) and (c) usually make the next round the last. */
static RunStats run_files(const Lang *L, int nf, const uint8_t **src, const size_t *len, uint8_t **out,
                          int nthreads) {
    double t_start = now();
    int np = L->npersist, dbg = getenv("TAXNET_DEBUG") != NULL;
    Pool P = {0};
    P.nthreads = nthreads > 1 ? nthreads - 1 : 0;
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.go, NULL);
    pthread_cond_init(&P.finished, NULL);
    pthread_t th[256];
    for (int i = 0; i < P.nthreads; i++) pthread_create(&th[i], NULL, pool_main, &P);

    Run R = {.L = L, .nf = nf, .np = np, .K = 8, .src = src, .len = len, .out = out};
    R.cuts = malloc(nf * sizeof *R.cuts);
    R.ncuts = malloc(nf * sizeof(int));
    R.first = malloc((nf + 1) * sizeof(int));
    R.done_upto = malloc(nf * sizeof(int));
    R.ftasks = calloc(nf, sizeof *R.ftasks);
    R.nftasks = calloc(nf, sizeof(int));
    R.capftasks = calloc(nf, sizeof(int));
    R.fowned = calloc(nf, sizeof *R.fowned);
    R.nfowned = calloc(nf, sizeof(int));
    pool_for(&P, nf, 16, do_cut, &R);
    size_t nc = 0;
    for (int f = 0; f < nf; f++) R.first[f] = (int)nc, R.done_upto[f] = (int)nc, nc += R.ncuts[f];
    R.first[nf] = (int)nc;
    R.ch = malloc((nc ? nc : 1) * sizeof(Chunk));
    pool_for(&P, nf, 16, do_fill, &R);
    if (dbg) fprintf(stderr, "chunking: %.3f s, %zu chunks\n", now() - t_start, nc);

    Task *tasks = malloc((nc ? nc : 1) * sizeof(Task));
    int nt = 0;
    for (size_t i = 0; i < nc; i++) tasks[nt++] = (Task){(int)i, 0};
    RunStats rs = {0, 0, (long)nc};
    for (;;) {
        qsort_r(tasks, nt, sizeof(Task), task_len_desc, R.ch);
        R.tasks = tasks, R.ntasks = nt;
        pool_for(&P, (nt + LANES - 1) / LANES, 1, do_batch, &R);
        if (rs.rounds++) rs.reruns += nt;
        if (dbg) fprintf(stderr, "round %d: %d runs (t=%.3f)\n", rs.rounds - 1, nt, now() - t_start);
        for (int i = 0; i < nt; i++) R.ch[tasks[i].chunk].nrun = R.ch[tasks[i].chunk].ncand;
        pool_for(&P, nf, 16, do_walk, &R);
        nt = 0;
        for (int f = 0; f < nf; f++) nt += R.nftasks[f];
        if (!nt) break;
        tasks = realloc(tasks, nt * sizeof(Task));
        nt = 0;
        for (int f = 0; f < nf; f++) {
            memcpy(tasks + nt, R.ftasks[f], R.nftasks[f] * sizeof(Task));
            nt += R.nftasks[f], R.nftasks[f] = 0;
        }
        if (R.K < 128) R.K *= 2;
    }
    free(tasks);
    pthread_mutex_lock(&P.mu);
    P.quit = 1;
    pthread_cond_broadcast(&P.go);
    pthread_mutex_unlock(&P.mu);
    for (int i = 0; i < P.nthreads; i++) pthread_join(th[i], NULL);
    for (size_t i = 0; i < nc; i++) {
        for (int k = 0; k < R.ch[i].ncand; k++) {
            free(R.ch[i].cand[k].end);
            if (R.ch[i].cand[k].out != R.ch[i].out) free(R.ch[i].cand[k].out);
        }
        free(R.ch[i].cand);
    }
    for (int f = 0; f < nf; f++) {
        for (int k = 0; k < R.nfowned[f]; k++) free(R.fowned[f][k]);
        free(R.fowned[f]), free(R.ftasks[f]);
    }
    free(R.ch), free(R.cuts), free(R.ncuts), free(R.first), free(R.done_upto), free(R.ftasks);
    free(R.nftasks), free(R.capftasks), free(R.fowned), free(R.nfowned);
    return rs;
}

static uint8_t *state_after(const Lang *L, const char *text) {
    uint8_t *st = malloc(L->npersist), out[16];
    Job j = {(const uint8_t *)text, (int)strlen(text), 0, out, L->h0, st};
    Scratch S = {0};
    L->batch(&j, 1, &S);
    free(S.in), free(S.save), free(S.cls);
    return st;
}

static void init_lang(Lang *L) {
    L->spec = state_after(L, "\n");        /* a chunk starting on a fresh line */
    L->spec_mid = state_after(L, "\n= ");  /* a chunk cut after a blank inside a line */
}

/* ---- I/O and CLI ------------------------------------------------------------------- */
static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { perror(path); exit(1); }
    fclose(f);
    return b;
}

static const char *NAMES[11] = {"ws", "kw", "type", "builtin", "func", "ident", "num", "str", "com", "op", "meta"};
static const char *ANSI[11] = {"", "\x1b[1;35m", "\x1b[36m", "\x1b[33m", "\x1b[1;34m", "",
                               "\x1b[38;5;208m", "\x1b[32m", "\x1b[2;3;37m", "\x1b[90m", "\x1b[95m"};

/* a pack: PACK.src, PACK.lbl, PACK.idx.json (we only need the [path, off, len] triples) */
typedef struct { uint8_t *src, *lbl; size_t n; int nf; size_t *off, *len; } Pack;

static Pack load_pack(const char *base) {
    Pack p = {0};
    char path[4096];
    size_t nl;
    snprintf(path, sizeof path, "%s.src", base);
    p.src = read_file(path, &p.n);
    snprintf(path, sizeof path, "%s.lbl", base);
    p.lbl = read_file(path, &nl);
    if (nl != p.n) { fprintf(stderr, "label size mismatch\n"); exit(1); }
    snprintf(path, sizeof path, "%s.idx.json", base);
    size_t ni;
    char *js = (char *)read_file(path, &ni);
    js[ni] = 0;
    char *s = strstr(js, "\"files\"");
    int cap = 1024;
    p.off = malloc(cap * sizeof(size_t)), p.len = malloc(cap * sizeof(size_t));
    char *r = strstr(js, "\"rejected\"");
    for (char *q = strchr(s, '['); q && (!r || q < r);) {
        char *e = strstr(q, "\", ");   /* end of path string */
        if (!e || (r && e > r)) break;
        size_t off, len;
        if (sscanf(e + 3, "%zu, %zu", &off, &len) != 2) break;
        if (p.nf == cap) cap *= 2, p.off = realloc(p.off, cap * sizeof(size_t)), p.len = realloc(p.len, cap * sizeof(size_t));
        p.off[p.nf] = off, p.len[p.nf] = len, p.nf++;
        q = strchr(e + 3, ']');
    }
    free(js);
    return p;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: taxnet <c|go|py> [--info|--classes|--eval|--bench] FILE|PACK [-t threads] [-r reps] [-o pred]\n");
        return 2;
    }
    Lang *L = NULL;
    for (size_t i = 0; i < sizeof LANGS / sizeof *LANGS; i++)
        if (!strcmp(argv[1], LANGS[i].name)) L = &LANGS[i];
    if (!L) { fprintf(stderr, "unknown language %s\n", argv[1]); return 2; }
    init_lang(L);
    int mode = 0, threads = 1, reps = 1;
    const char *path = NULL, *pred_path = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--info")) {
            printf("taxnet-%s: %d forward + %d backward neurons, compiled from weights/%s.tnw (sha256 %s)\n",
                   L->name, L->nfwd, L->nbwd, L->name, L->weights_sha256);
            return 0;
        }
        if (!strcmp(argv[i], "--classes")) mode = 1;
        else if (!strcmp(argv[i], "--eval")) mode = 2;
        else if (!strcmp(argv[i], "--bench")) mode = 3;
        else if (!strcmp(argv[i], "-t")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-r")) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-o")) pred_path = argv[++i];
        else path = argv[i];
    }
    if (mode <= 1) {
        size_t n;
        uint8_t *src = read_file(path, &n), *out = malloc(n + 1);
        const uint8_t *s = src;
        uint8_t *o = out;
        run_files(L, 1, &s, &n, &o, threads);
        for (size_t i = 0; i < n; i++) {
            if (mode == 1) putchar("0123456789a"[out[i]]);
            else {
                if (i == 0 || out[i] != out[i - 1]) fputs("\x1b[0m", stdout), fputs(ANSI[out[i]], stdout);
                putchar(src[i]);
            }
        }
        if (mode == 0) fputs("\x1b[0m", stdout);
        return 0;
    }
    Pack p = load_pack(path);
    const uint8_t **srcs = malloc(p.nf * sizeof *srcs);
    uint8_t **outs = malloc(p.nf * sizeof *outs);
    uint8_t *out = malloc(p.n + 1);
    for (int i = 0; i < p.nf; i++) srcs[i] = p.src + p.off[i], outs[i] = out + p.off[i];
    double best = 1e30;
    RunStats rs = {0};
    for (int r = 0; r < reps; r++) {
        double t = now();
        rs = run_files(L, p.nf, srcs, p.len, outs, threads);
        t = now() - t;
        if (t < best) best = t;
    }
#ifdef TAXNET_PROFILE
    fprintf(stderr, "profile (s): init %.3f  transpose_in %.3f  fwd %.3f  bwd %.3f  transpose_out %.3f;  %ld steps x 256 lanes, %.1f%% of lane-steps used, fwd %.0f ns/step\n",
            prof[0], prof[1], prof[2], prof[3], prof[4], prof_steps, 100.0 * prof_bytes / (prof_steps * 256.0), prof[2] / prof_steps * 1e9);
#endif
    printf("%s  %d files  %.1f MB  %d threads  %.3f s  -> %.1f M chars/s  (%.2f ns/char/thread; "
           "%ld chunks, %ld re-run in %d extra rounds)\n",
           L->name, p.nf, p.n / 1e6, threads, best, p.n / best / 1e6, best * threads / p.n * 1e9,
           rs.chunks, rs.reruns, rs.rounds - 1);
    if (mode == 2) {
        long conf[11][11] = {{0}}, wrong = 0, nonws = 0, nonws_wrong = 0;
        for (size_t i = 0; i < p.n; i++) {
            conf[p.lbl[i]][out[i]]++;
            if (p.lbl[i] != out[i]) wrong++;
            if (p.lbl[i] != 0) { nonws++; if (p.lbl[i] != out[i]) nonws_wrong++; }
        }
        printf("accuracy %.6f%%  (%ld wrong of %zu bytes); non-ws %.6f%% (%ld wrong of %ld)\n",
               100.0 * (p.n - wrong) / p.n, wrong, p.n, 100.0 * (nonws - nonws_wrong) / nonws, nonws_wrong, nonws);
        for (int a = 0; a < 11; a++)
            for (int b = 0; b < 11; b++)
                if (a != b && conf[a][b]) printf("   ref %-7s net %-7s %ld\n", NAMES[a], NAMES[b], conf[a][b]);
        FILE *f = fopen(pred_path ? pred_path : "/tmp/taxnet_pred.bin", "wb");
        fwrite(out, 1, p.n, f);
        fclose(f);
    }
    return 0;
}
