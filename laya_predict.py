#!/usr/bin/env python3
"""Self-contained Laya prediction: Python tokenizer + C inference engine.

This is the final user-facing interface. No torch, no transformers at runtime —
only the lightweight `tokenizers` package (< 5 MB) for BPE tokenization.

Usage:
    # Single question
    ./laya_predict.py --model model.gguf --tokenizer tokenizer.json \
        --state '上个月各门店的GMV分别是多少' \
        --question '{"type":"choice","instructions":"分析类型?","criteria":{"single_value":"取值","trend":"趋势","ranking":"排名"}}'

    # All questions from a file
    ./laya_predict.py --model model.gguf --tokenizer tokenizer.json \
        --questions examples/triage.json 'I was charged twice'

    # Pipe mode (one JSON per line on stdin)
    echo '{"state":"上个月各门店的GMV","questions":{...}}' | ./laya_predict.py --model ... --tokenizer ...

Dependencies: tokenizers (pip install tokenizers), plus the compiled laya-infer binary.
No torch, no transformers, no CUDA runtime needed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

# Try the fast `tokenizers` library first (pure Rust, < 5MB, no torch dependency)
try:
    from tokenizers import Tokenizer
    HAS_FAST_TOKENIZER = True
except ImportError:
    HAS_FAST_TOKENIZER = False

# Fallback: transformers AutoTokenizer (needs torch)
if not HAS_FAST_TOKENIZER:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("need `pip install tokenizers` (lightweight) or `pip install transformers torch` (heavy)")

QTYPES = {"choice": 0, "score": 1, "noul": 2}

# ---- tokenizer wrapper

class LayaTokenizer:
    def __init__(self, path):
        if HAS_FAST_TOKENIZER:
            self.tok = Tokenizer.from_file(path)
            self.mask_token_id = 4   # ModernBERT [MASK]
            self.sep_token_id = 1    # [SEP]
            self.cls_token_id = 2    # [CLS] = BOS
            self.mask_token = "[MASK]"
        else:
            d = os.path.dirname(path)
            self.tok = AutoTokenizer.from_pretrained(d)
            self.mask_token_id = self.tok.mask_token_id
            self.sep_token_id = self.tok.sep_token_id or 1
            self.cls_token_id = self.tok.cls_token_id or 2
            self.mask_token = self.tok.mask_token

    def encode(self, text, add_special_tokens=False):
        if HAS_FAST_TOKENIZER:
            return self.tok.encode(text, add_special_tokens=add_special_tokens).ids
        else:
            return self.tok.encode(text, add_special_tokens=add_special_tokens)


def render_options(q):
    t = q.get("type", q.get("t"))
    crit = q.get("criteria", q.get("crit"))
    if t == "choice":
        return [f"{k}: {v}" if v else str(k) for k, v in crit.items()]
    elif t == "score":
        return [f"level {i}: {c}" for i, c in enumerate(crit)]
    else:
        return ["false: no, the statement does not hold", "true: yes, the statement holds"]


def build_sequence(tok, state, q, max_len=1024, head_max_len=256):
    """Simplified build_sequence: [CLS] type instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]"""
    t = q.get("type", q.get("t"))
    ins = q.get("instructions", q.get("ins", ""))
    opts = render_options(q)

    head_text = f"{t} question: {ins}"
    head_ids = tok.encode(head_text, add_special_tokens=False)

    opt_ids = []
    for o in opts:
        o_ids = tok.encode(" " + o, add_special_tokens=False)[:48]
        opt_ids.append([tok.mask_token_id] + o_ids)
    if not opt_ids:
        raise ValueError(f"question has no options (type={t!r}); "
                         "choice/score require criteria")

    if isinstance(state, dict):
        state_text = " ".join(f"{k}: {v}" for k, v in state.items())
    elif isinstance(state, list):
        state_text = " ".join(turn.get("content", "") for turn in state)
    else:
        state_text = str(state)
    state_ids = tok.encode(state_text, add_special_tokens=False)

    # Budget: head (type+ins+opts) gets head_max_len, state gets the rest
    head_all = head_ids + [tok.sep_token_id]
    for oid in opt_ids:
        head_all += oid
    head_all += [tok.sep_token_id]

    state_budget = max_len - len(head_all) - 2  # -2 for CLS and final SEP
    if state_budget < 1:
        raise ValueError(f"question head ({len(head_all)} tokens) does not fit "
                         f"max_len={max_len}")
    # Truncating here used to be silent: the caller got a confident answer built
    # from a prefix of the input. Refusing is slower but never wrong.
    if len(state_ids) > state_budget:
        raise ValueError(f"state is {len(state_ids)} tokens but only {state_budget} "
                         f"fit after the question head (max_len={max_len}); "
                         "shorten the input or raise max_len")
    state_ids = state_ids[:state_budget]

    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for oid in opt_ids:
        markers.append(len(ids))
        ids += oid
    ids += [tok.sep_token_id] + state_ids + [tok.sep_token_id]

    return ids, markers


# ---- C engine caller

def engine_env(bin_path):
    """Environment for spawning the engine.

    The binary is built with RPATH $ORIGIN and links the ggml built alongside it,
    so normally nothing needs adding. GGML_LIB_DIR is an escape hatch for builds
    that deliberately link a different ggml.
    """
    env = dict(os.environ)
    if os.environ.get("GGML_LIB_DIR"):
        env["LD_LIBRARY_PATH"] = os.environ["GGML_LIB_DIR"]
    return env


def pack_items(items):
    """Concatenate several slots into one packed request.

    The engine runs them as a single sequence separated by a block-diagonal
    attention mask, so one forward pass answers every slot instead of one pass
    per slot. Marker indices are relative to their own slot and have to be
    shifted into the concatenated coordinate system.
    """
    kmax = max(len(it["marker_pos"]) for it in items)
    input_ids, marker_pos = [], []
    seg_len, seg_qtype, seg_k = [], [], []
    for it in items:
        base = len(input_ids)
        seg_len.append(len(it["input_ids"]))
        seg_qtype.append(it["qtype"])
        seg_k.append(len(it["marker_pos"]))
        input_ids.extend(it["input_ids"])
        marker_pos.extend(m + base for m in it["marker_pos"])
        marker_pos.extend([0] * (kmax - len(it["marker_pos"])))  # padding slots
    return {"packed": {"input_ids": input_ids, "marker_pos": marker_pos,
                       "seg_len": seg_len, "seg_qtype": seg_qtype,
                       "seg_k": seg_k, "kmax": kmax}}


class LayaEngine:
    """Runs the C engine as a long-lived daemon, submitting every slot at once.

    Both parts matter, and they fix different costs:
      * daemon  - the 630 ms model load happens once per process, not once per slot
      * batch   - one round trip for all slots, and slots that share a shape reuse
                  the cached graph instead of paying another CUDA graph capture
    """

    def __init__(self, bin_path, model_path, daemon=True, extra_env=None,
                 timeout=30.0):
        self.bin_path = bin_path
        self.model_path = model_path
        self.extra_env = extra_env or {}
        # A wedged daemon must not take the request down with it: without a
        # timeout a hung engine blocks the thread forever and the pool never
        # gets its worker back.
        self.timeout = timeout
        self.proc = None
        self.restarts = 0
        if daemon:
            self.proc = self._spawn()

    def _spawn(self):
        env = engine_env(self.bin_path)
        env["LAYA_DAEMON"] = "1"
        env.update(self.extra_env)
        return subprocess.Popen(
            [self.bin_path, self.model_path], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
            text=True, bufsize=1,
        )

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _respawn(self):
        """Replace a dead or wedged daemon. Loses the warm graph, but a request
        that fails is strictly better than a service that stays down."""
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self.restarts += 1
        self.proc = self._spawn()

    def __call__(self, items):
        """items: [{"input_ids", "marker_pos", "qtype"}] -> list of raw results."""
        # Packing turns N forward passes into one. Falls back to the item-wise
        # path for a single slot, where there is nothing to merge.
        payload = json.dumps(pack_items(items) if len(items) > 1
                             else {"items": items}, separators=(",", ":"))
        if self.proc is None:
            env = engine_env(self.bin_path)
            env.update(self.extra_env)
            p = subprocess.run([self.bin_path, self.model_path], input=payload,
                               capture_output=True, text=True, env=env, timeout=300)
            if p.returncode != 0:
                raise RuntimeError(f"laya-infer failed: {p.stderr[:200]}")
            return json.loads(p.stdout.strip())["results"]

        if not self.alive():
            self._respawn()

        # The whole exchange runs on one thread with a single deadline. Timing out
        # only the read is not enough: a stalled daemon stops draining its stdin,
        # the pipe buffer fills, and the *write* blocks first -- which no read
        # timeout can catch. select() is no help either, since a TextIOWrapper
        # pre-reads into its own buffer and makes the descriptor look unready.
        box = []

        def exchange():
            try:
                self.proc.stdin.write(payload + "\n")
                self.proc.stdin.flush()
                box.append(self.proc.stdout.readline())
            except Exception as e:            # noqa: BLE001 - reported below
                box.append(("ERR", e))

        th = threading.Thread(target=exchange, daemon=True)
        th.start()
        th.join(self.timeout)

        if th.is_alive():
            self._respawn()
            raise RuntimeError(f"engine timed out after {self.timeout}s")

        if not box:
            self._respawn()
            raise RuntimeError("engine daemon exited unexpectedly")
        got = box[0]
        if isinstance(got, tuple):            # write failed: pipe gone
            self._respawn()
            raise RuntimeError(f"engine pipe error: {got[1]}")
        if not got:                           # EOF: daemon died mid-request
            self._respawn()
            raise RuntimeError("engine daemon exited unexpectedly")
        return json.loads(got)["results"]

    def close(self):
        if self.proc:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def predict(engine, tok, state, questions, cfg):
    """Build every slot, submit them in one call, then format each answer."""
    items, metas = [], []
    for name, q in questions.items():
        t = q.get("type", q.get("t"))
        qtype = QTYPES.get(t, 0)
        ids, markers = build_sequence(tok, state, q, cfg.get("max_len", 1024),
                                      cfg.get("head_max_len", 256))
        items.append({"input_ids": ids, "marker_pos": markers, "qtype": qtype})
        metas.append((name, q, t))

    # Group identical shapes together: the engine caches one graph, so
    # interleaving different lengths would rebuild it on every slot.
    order = sorted(range(len(items)),
                   key=lambda i: (len(items[i]["input_ids"]),
                                  len(items[i]["marker_pos"]),
                                  items[i]["qtype"]))
    ordered = [items[i] for i in order]
    outs = engine(ordered)
    by_slot = [None] * len(items)
    for pos, out in zip(order, outs):
        by_slot[pos] = out

    results = {}
    for (name, q, t), out in zip(metas, by_slot):
        keys = list(q.get("criteria", q.get("crit", {})).keys()) if t == "choice" else None
        choice_idx = out["choice"]
        if t == "choice" and keys:
            results[name] = {"choice": keys[choice_idx] if choice_idx < len(keys) else "?",
                             "probs": dict(zip(keys, out["probs"])) if keys else out["probs"],
                             "confidence": out["confidence"]}
        elif t == "noul":
            results[name] = {"noul": out["probs"][1] if len(out["probs"]) > 1 else 0.0,
                             "confidence": out["confidence"]}
        elif t == "score":
            p = out["probs"]
            # Keep the per-level probabilities: the Jev wire format reports them.
            results[name] = {"score": round(sum(i * p[i] for i in range(len(p))), 4),
                             "probs": p,
                             "confidence": out["confidence"]}
        else:
            results[name] = out
    return results


def to_jev_answers(results, questions):
    """Rewrite our results into laya's / Jev `answers` shape.

    Wire compatibility with `laya-serve` matters more than our own convenience:
    a client written against the Jev decision API can then point at this server
    unchanged. Three differences from our internal shape:

      * `probs` is called `probabilities`
      * every answer carries an explicit `type`
      * noul confidence is `max(p_true, 1 - p_true)`, not the entropy-based value
        the engine reports (over two options the two disagree, e.g. p=0.99 gives
        0.99 vs 0.919)
    """
    answers = {}
    for name, ans in results.items():
        q = questions.get(name, {})
        t = q.get("type", q.get("t"))
        crit = q.get("criteria", q.get("crit"))
        probs = ans.get("probs")

        if t == "choice":
            out = {"type": "choice", "choice": ans.get("choice"),
                   "probabilities": {k: round(float(v), 4) for k, v in (probs or {}).items()}}
        elif t == "score":
            out = {"type": "score", "score": ans.get("score"),
                   "legend": {str(i): c for i, c in enumerate(crit or [])},
                   "probabilities": {str(i): round(float(v), 4)
                                     for i, v in enumerate(probs or [])}}
        else:
            noul = float(ans.get("noul", 0.0))
            out = {"type": "noul", "noul": round(noul, 4),
                   "confidence": round(max(noul, 1.0 - noul), 4)}
            answers[name] = out
            continue

        out["confidence"] = round(float(ans.get("confidence", 0.0)), 4)
        answers[name] = out
    return answers


def main():
    ap = argparse.ArgumentParser(description="Laya inference: Python tokenizer + C engine")
    ap.add_argument("--model", required=True, help="path to .gguf model file")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer.json")
    ap.add_argument("--engine", default=None, help="path to laya-infer binary (auto-detect)")
    ap.add_argument("--config", default=None, help="path to rl_agent_config.json")
    # Mirrors laya's own CLI: the request text is a positional argument and
    # --questions points at a file. --state/--schema are kept as aliases.
    ap.add_argument("text", nargs="*", help="the request text")
    ap.add_argument("--state", help="input state text (alias for the positional)")
    ap.add_argument("--question", help="single question JSON")
    ap.add_argument("--schema", help="path to schema JSON (all questions)")
    ap.add_argument("--questions", dest="questions_file",
                    help="path to questions JSON (laya-style alias for --schema)")
    ap.add_argument("--pipe", action="store_true", help="read JSON lines from stdin")
    ap.add_argument("--no-daemon", action="store_true",
                    help="spawn a fresh process per call (slow; for debugging)")
    args = ap.parse_args()

    # Auto-detect engine binary
    engine_bin = args.engine or os.path.join(os.path.dirname(__file__), "build", "laya-infer")
    if not os.path.exists(engine_bin):
        sys.exit(f"engine not found: {engine_bin}\nBuild it: cd ggml && cmake -B build && cmake --build build")

    tok = LayaTokenizer(args.tokenizer)

    # Positional text wins; --state stays for scripts already written against it.
    state = args.state or (" ".join(args.text) if args.text else None)

    # Load config
    cfg = {}
    if args.config:
        cfg = json.load(open(args.config))
    else:
        # Try to find config next to model
        cfg_path = os.path.join(os.path.dirname(args.model), "rl_agent_config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path))

    with LayaEngine(engine_bin, args.model, daemon=not args.no_daemon) as engine:
        if args.pipe:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                result = predict(engine, tok, req["state"], req["questions"], cfg)
                print(json.dumps(result, ensure_ascii=False))
        elif state:
            qfile = args.questions_file or args.schema
            if qfile:
                questions = json.load(open(qfile))
            elif args.question:
                questions = {"q": json.loads(args.question)}
            else:
                sys.exit("need --questions/--schema or --question")
            t0 = time.perf_counter()
            result = predict(engine, tok, state, questions, cfg)
            dt = (time.perf_counter() - t0) * 1000
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print(f"\n({dt:.0f} ms, {len(questions)} slots)", file=sys.stderr)
        else:
            ap.print_help()


if __name__ == "__main__":
    main()
