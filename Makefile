# taxnet: hand-written neural networks that syntax-highlight C, Go and Python.
#
#   make            build the weight files, compile them to bit-sliced kernels, build ./taxnet
#   make corpus     fetch the held-out repos and label every corpus with the reference tokenizers
#   make eval       accuracy of all three networks on every corpus
#   make verify     dense numpy executor vs compiled kernel, byte for byte
#   make bench      single-core speed of taxnet and of the baselines on the same sample
# building weights and kernels needs only the Python standard library; the evaluation
# needs the venv described in the README
PY ?= $(shell test -x $(HOME)/.cache/taxnet/venv/bin/python && echo $(HOME)/.cache/taxnet/venv/bin/python || echo python3)
PACKS ?= $(HOME)/.cache/taxnet/packs
LANGS = c go py
SPLITS = dev test test2 test3

taxnet: kernel/taxnet.c $(LANGS:%=kernel/gen/net_%.h)
	$(CC) -O3 -march=native -Wall -o $@ kernel/taxnet.c -lpthread

weights/%.tnw: model/lang_%.py model/taxnet.py model/common.py spec/*.txt
	$(PY) -I model/build.py $*

kernel/gen/net_%.h: weights/%.tnw kernel/codegen.py
	$(PY) -I kernel/codegen.py $< $@

corpus:
	./corpus/fetch.sh test && ./corpus/fetch.sh test2 && ./corpus/fetch.sh test3
	$(PY) -I corpus/build.py dev test test2 test3

eval: taxnet
	@for l in $(LANGS); do for s in $(SPLITS); do printf "%-3s %-6s" $$l $$s; \
	  ./taxnet $$l --eval $(PACKS)/$${s}_$$l -t 16 -o $(PACKS)/$${s}_$$l.pred | grep accuracy; done; done

verify: taxnet
	@for l in $(LANGS); do ./taxnet $$l --eval $(PACKS)/dev_$$l -t 16 -o $(PACKS)/dev_$$l.pred > /dev/null; \
	  $(PY) -I bench/verify.py $$l $(PACKS)/dev_$$l 60; done

bench: taxnet
	taskset -c 0 $(PY) -I bench/baselines.py
	cd bench/gobench && go build -o /tmp/taxnet-gobench . && taskset -c 0 /tmp/taxnet-gobench $(PACKS)/bench_go
	$(CC) -O2 -I/usr/lib/llvm-19/include -o /tmp/taxnet-clangbench bench/clangbench.c -L/usr/lib/llvm-19/lib -lclang -Wl,-rpath,/usr/lib/llvm-19/lib
	taskset -c 0 /tmp/taxnet-clangbench $(PACKS)/bench_c
	@for l in $(LANGS); do taskset -c 0 ./taxnet $$l --bench $(PACKS)/bench_$$l -t 1 -r 5; done

.PHONY: corpus eval verify bench
