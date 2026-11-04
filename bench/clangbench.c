/* clang's tokenizer (libclang clang_tokenize, as the C reference uses it) on a taxnet pack. */
#include <clang-c/Index.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    char path[4096];
    snprintf(path, sizeof path, "%s.src", argv[1]);
    FILE *f = fopen(path, "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = malloc(n); fread(src, 1, n, f); fclose(f);
    snprintf(path, sizeof path, "%s.idx.json", argv[1]);
    f = fopen(path, "rb"); fseek(f, 0, SEEK_END); long m = ftell(f); fseek(f, 0, SEEK_SET);
    char *js = malloc(m + 1); fread(js, 1, m, f); js[m] = 0; fclose(f);
    const char *args[] = {"-std=gnu17", "-nostdinc", "-ferror-limit=1", "-Wno-everything"};
    CXIndex idx = clang_createIndex(0, 0);
    double ttok = 0, tall = 0;
    long ntok = 0;
    char *r = strstr(js, "\"rejected\"");
    for (char *q = strchr(strstr(js, "\"files\""), '['); q && q < r;) {
        char *e = strstr(q, "\", ");
        if (!e || e > r) break;
        long off, len;
        if (sscanf(e + 3, "%ld, %ld", &off, &len) != 2) break;
        q = strchr(e + 3, ']');
        double t0 = now();
        struct CXUnsavedFile uf = {"t.c", src + off, (unsigned long)len};
        CXTranslationUnit tu;
        clang_parseTranslationUnit2(idx, "t.c", args, 4, &uf, 1, CXTranslationUnit_SkipFunctionBodies, &tu);
        double t1 = now();
        CXFile file = clang_getFile(tu, "t.c");
        CXSourceRange range = clang_getRange(clang_getLocationForOffset(tu, file, 0), clang_getLocationForOffset(tu, file, (unsigned)len));
        CXToken *toks; unsigned nt;
        clang_tokenize(tu, range, &toks, &nt);
        for (unsigned i = 0; i < nt; i++) {
            CXSourceRange ext = clang_getTokenExtent(tu, toks[i]);
            unsigned so, eo;
            clang_getSpellingLocation(clang_getRangeStart(ext), NULL, NULL, NULL, &so);
            clang_getSpellingLocation(clang_getRangeEnd(ext), NULL, NULL, NULL, &eo);
            ntok += clang_getTokenKind(toks[i]) + (so < eo);
        }
        clang_disposeTokens(tu, toks, nt);
        double t2 = now();
        clang_disposeTranslationUnit(tu);
        ttok += t2 - t1, tall += now() - t0;
    }
    printf("clang raw tokenizer: %.1f M chars/s (tokenize only), %.1f M chars/s (incl. TU setup)\n", n / ttok / 1e6, n / tall / 1e6);
    return ntok == 0;
}
