"""Reproduce the generation-identity bug in the archived workspace."""
import os, sys, tempfile
import numpy as np
from joblib import Memory

state = {"v": 11}

def build(tmp):
    mem = Memory(location=tmp, verbose=0)
    @mem.cache
    def make(x):
        return np.array([state["v"], x], dtype=np.int64)
    return mem, make

# Repro 1: delete then recompute
with tempfile.TemporaryDirectory() as tmp:
    mem, make = build(tmp)
    old = make.call_and_shelve(0)
    print("old.get()", old.get())
    old.clear()
    state["v"] = 22
    make.call_and_shelve(0)
    try:
        got = old.pin().get()
        print("REPRO1 UNEXPECTED: old.pin().get() =", got)
    except KeyError as exc:
        print("REPRO1 expected KeyError:", type(exc).__name__, exc)

# Repro 2: hold the old result, force a recompute at same path
state["v"] = 11
with tempfile.TemporaryDirectory() as tmp:
    mem, make = build(tmp)
    old = make.call_and_shelve(0)
    h = old.pin()
    print("h.get()", h.get())
    state["v"] = 22
    make.call(0)
    print("h.get() after replace", h.get())
    try:
        got = old.pin().get()
        print("REPRO2 old.pin().get() =", got)
    except KeyError as exc:
        print("REPRO2 KeyError:", type(exc).__name__, exc)
