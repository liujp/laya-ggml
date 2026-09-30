#!/usr/bin/env python3
"""Compare this engine against the PyTorch implementation.

Optional: needs `torch`, `transformers` and `laya` installed, plus a PyTorch
checkpoint to compare against. The engine itself needs none of that.

    python3 validate.py \
        --model model.gguf \
        --reference /path/to/checkpoint \
        --questions examples/triage.json \
        --cases cases.jsonl

`--cases` is JSONL with {"text": ..., "slots": {"<question>": "<label>"}} per
line; only questions whose label is present are compared.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from laya_predict import LayaTokenizer, LayaEngine, build_sequence  # noqa: E402

QTYPES = {"choice": 0, "score": 1, "noul": 2}


def option_keys(q):
    """Answer keys for a question, in the order the model scores them."""
    t = q.get("type", q.get("t"))
    crit = q.get("criteria", q.get("crit"))
    if t == "choice":
        return list(crit.keys()) if isinstance(crit, dict) else list(crit or [])
    if t == "score":
        return [str(i) for i in range(len(crit or []))]
    return ["false", "true"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("LAYA_MODEL", "model.gguf"),
                    help="GGUF model to test")
    ap.add_argument("--tokenizer", default=os.environ.get("LAYA_TOKENIZER", "tokenizer.json"))
    ap.add_argument("--questions", default=os.environ.get("LAYA_QUESTIONS",
                                                          "examples/triage.json"))
    ap.add_argument("--reference", default=None,
                    help="PyTorch checkpoint dir; omit to skip the comparison")
    ap.add_argument("--cases", default=None,
                    help="JSONL of labelled cases; omit for a smoke test on one text")
    ap.add_argument("--text", default="I was charged twice for my subscription, please refund")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    questions = json.load(open(args.questions))
    tok = LayaTokenizer(args.tokenizer)

    if args.cases:
        cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    else:
        cases = [{"text": args.text, "slots": {}}]
    if not cases:
        ap.error("no cases to run")
    cases = cases[:args.limit]

    # --- PyTorch side
    pt_answers = None
    if args.reference:
        import torch
        import laya
        from laya.common import build_sequence as pt_build, QTYPES as PT_QTYPES
        from transformers import AutoTokenizer

        pt_tok = AutoTokenizer.from_pretrained(os.path.join(args.reference, "tokenizer"))
        agent = laya.load(args.reference, device="cpu")
        pt_answers = []
        for c in cases:
            row = {}
            for name, q in questions.items():
                if name not in c["slots"]:
                    continue
                q_internal = {"t": q["type"], "ins": q["instructions"]}
                if "criteria" in q:
                    q_internal["crit"] = q["criteria"]
                ids, markers = pt_build(pt_tok, c["text"], q_internal, 1024, 256)
                ans = agent.predict({"body": c["text"]}, {name: q})["answers"][name]
                row[name] = ans
            pt_answers.append(row)

    # --- engine side
    env = dict(os.environ)
    if os.environ.get("GGML_LIB_DIR"):
        env["LD_LIBRARY_PATH"] = os.environ["GGML_LIB_DIR"]

    engine_bin = os.path.join(HERE, "build", "laya-infer")
    match = total = 0
    engine_ms, pt_ms = [], []
    mismatches = []

    with LayaEngine(engine_bin, args.model, daemon=True) as engine:
        # First call pays model load + CUDA graph capture; don't count it.
        warm = [{"input_ids": build_sequence(tok, "warmup", questions[next(iter(questions))])[0],
                 "marker_pos": build_sequence(tok, "warmup", questions[next(iter(questions))])[1],
                 "qtype": 0}]
        engine(warm)

        for ci, c in enumerate(cases):
            items, names = [], []
            for name, q in questions.items():
                if args.reference and name not in c["slots"]:
                    continue
                ids, markers = build_sequence(tok, c["text"], q)
                items.append({"input_ids": ids, "marker_pos": markers,
                              "qtype": QTYPES[q["type"]]})
                names.append(name)

            t0 = time.perf_counter()
            outs = engine(items)
            engine_ms.append((time.perf_counter() - t0) * 1000)

            for name, out in zip(names, outs):
                q = questions[name]
                keys = option_keys(q)
                c_choice = keys[out["choice"]] if out["choice"] < len(keys) else "?"
                if pt_answers is None:
                    continue          # nothing to compare against
                pt = pt_answers[ci].get(name)
                if pt is None:
                    continue
                total += 1
                pt_choice = pt.get("choice", pt.get("noul"))
                if str(pt_choice) == str(c_choice):
                    match += 1
                else:
                    mismatches.append(f"  {name}: engine={c_choice} pytorch={pt_choice}")

    print()
    if total:
        print(f"Choice agreement: {match}/{total} ({match / total:.1%})")
    for m in mismatches[:10]:
        print(m)
    print(f"engine avg: {sum(engine_ms) / len(engine_ms):.0f} ms over {len(cases)} case(s)")
    if pt_ms:
        print(f"PyTorch avg: {sum(pt_ms) / len(pt_ms):.0f} ms")


if __name__ == "__main__":
    main()
