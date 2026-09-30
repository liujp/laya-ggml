#!/usr/bin/env python3
"""Benchmark the laya HTTP service: latency, throughput, and the CLI baseline.

Usage:
    ./bench_http.py --url http://127.0.0.1:8100 --requests 50 --concurrency 1 4 8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# The model, vocabulary and question set are all derived from a checkpoint, so
# none of them live in the repo -- point these at your own via env vars.
MODEL = os.environ.get("LAYA_MODEL", "model.gguf")
TOKENIZER = os.environ.get("LAYA_TOKENIZER", "tokenizer.json")
QUESTIONS = os.environ.get("LAYA_QUESTIONS", "examples/triage.json")

# Support tickets of varying length, so the server sees varied sequence lengths.
# These are plain strings; the server falls back to the questions it loaded.
STATES = [
    "I was charged twice for my subscription, please refund",
    "Your API has been returning 502 for the last two hours and our checkout is down, "
    "we need this escalated immediately",
    "Quick question: does the annual plan include the audit log, and how far back "
    "does the retention go for the team tier?",
    "cancel my account",
]


def post(url: str, state: str) -> dict:
    payload = json.dumps({"state": state}).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def bench_concurrent(url: str, n: int, conc: int) -> dict:
    """Fire n requests with `conc` in flight, measuring wall time and per-request latency."""
    latencies: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    counter = {"i": 0}
    start_barrier = threading.Barrier(conc)

    def worker():
        start_barrier.wait()
        while True:
            with lock:
                i = counter["i"]
                if i >= n:
                    return
                counter["i"] = i + 1
            t0 = time.perf_counter()
            try:
                post(url, STATES[i % len(STATES)])
            except Exception as e:  # noqa: BLE001 - report, don't crash the bench
                with lock:
                    errors.append(repr(e)[:80])
                return
            dt = (time.perf_counter() - t0) * 1000
            with lock:
                latencies.append(dt)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = (time.perf_counter() - t0) * 1000

    return {
        "concurrency": conc,
        "requests": len(latencies),
        "errors": len(errors),
        "wall_ms": wall,
        "mean_ms": statistics.mean(latencies) if latencies else 0,
        "p95_ms": (statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 5 else 0),
        "qps": len(latencies) / (wall / 1000) if wall > 0 else 0,
        "err": errors[:2],
    }


def bench_cli(n: int) -> dict:
    """Baseline: one full CLI invocation per request (python start + tokenizer + model)."""
    cmd = [sys.executable, os.path.join(HERE, "laya_predict.py"),
           "--model", MODEL, "--tokenizer", TOKENIZER,
           "--questions", QUESTIONS, STATES[0]]
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run(cmd, capture_output=True, text=True, cwd=HERE, timeout=300)
        ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": statistics.mean(ts), "qps": 1000 / statistics.mean(ts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--requests", type=int, default=50)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--cli", type=int, default=0, help="also time N CLI calls as baseline")
    args = ap.parse_args()

    url = args.url.rstrip("/") + "/v1/predict"

    # Warm up: first call pays CUDA graph capture.
    post(url, STATES[0])
    print(f"target: {url}\n")

    print(f"{'conc':>5}{'reqs':>6}{'mean ms':>10}{'p95 ms':>9}{'QPS':>9}{'err':>5}")
    print("-" * 44)
    for c in args.concurrency:
        r = bench_concurrent(url, args.requests, c)
        print(f"{r['concurrency']:>5}{r['requests']:>6}{r['mean_ms']:>10.1f}"
              f"{r['p95_ms']:>9.1f}{r['qps']:>9.1f}{r['errors']:>5}")
        if r["err"]:
            print("      ", r["err"])

    if args.cli:
        print()
        c = bench_cli(args.cli)
        print(f"CLI baseline ({args.cli} calls): {c['mean_ms']:.0f} ms/call, {c['qps']:.2f} QPS")


if __name__ == "__main__":
    main()
