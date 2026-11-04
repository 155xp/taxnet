#!/usr/bin/env python3
"""Build the hand-written networks and write their weight files.

    python model/build.py            -> weights/{c,go,py}.tnw  (+ .json listing every neuron)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(os.path.dirname(HERE), "weights")


def build(lang):
    net = __import__("lang_" + lang).build()
    os.makedirs(OUT, exist_ok=True)
    net.save(os.path.join(OUT, lang + ".tnw"))
    with open(os.path.join(OUT, lang + ".json"), "w") as f:
        json.dump(net.to_json(), f, indent=0)
    st = net.stats()
    print("%-3s %s" % (lang, " ".join("%s=%s" % kv for kv in st.items())))
    return net


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["c", "go", "py"]:
        build(lang)
