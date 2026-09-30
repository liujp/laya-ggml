#!/usr/bin/env python3
"""HTTP service around the ggml engine daemon.

The CLI pays ~1.7 s per call before any real work happens: Python start-up,
loading the 34 MB tokenizer, and loading the 1.3 GB model. This service pays all
of that once at boot, so a request costs one forward pass plus HTTP overhead.

A worker is one long-lived `laya-infer` daemon. Each one is single-threaded and
handles one request at a time, so concurrent requests queue for a free worker.
Workers are health-checked and respawned, because a dead daemon used to take the
service down permanently.

Usage:
    ./laya_server.py --model model.gguf --tokenizer tokenizer.json --port 8100

    curl -s localhost:8100/v1/systemone -H 'Content-Type: application/json' \
        -d '{"state":{"message":"I was charged twice"},"questions":{...}}'

Questions can come with the request, or be loaded once with
`--questions examples/triage.json`.

Set LAYA_API_KEY to require `Authorization: Bearer <key>` on /v1/systemone.
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from laya_predict import (LayaEngine, LayaTokenizer, build_sequence, predict,
                          to_jev_answers)

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="laya ggml service", version="1.0")

# Populated by main() before uvicorn starts.
STATE: Dict[str, Any] = {}

log = logging.getLogger("laya")

# Max characters of `state` we will tokenize. build_sequence truncates to the
# model's max_len, which would silently drop the tail of long inputs.
MAX_STATE_CHARS = 4000


class Metrics:
    """Counters for /metrics. Lock-protected: requests run on a thread pool."""

    def __init__(self):
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.timeouts = 0
        self.worker_restarts = 0
        self.total_ms = 0.0
        self.slots = 0

    def record(self, ms: float, slots: int, ok: bool = True, timeout: bool = False):
        with self._lock:
            self.requests += 1
            self.total_ms += ms
            self.slots += slots
            if timeout:
                self.timeouts += 1
            if not ok:
                self.errors += 1

    def snapshot(self):
        with self._lock:
            n = self.requests
            return {
                "requests": n,
                "errors": self.errors,
                "timeouts": self.timeouts,
                "worker_restarts": self.worker_restarts,
                "mean_ms": round(self.total_ms / n, 2) if n else 0.0,
                "slots_total": self.slots,
            }


class WorkerPool:
    """Fixed set of engine daemons; `worker()` lends one out at a time.

    A daemon reads one request line and writes one result line, so two threads
    sharing one would interleave their payloads. Handing workers out exclusively
    is what keeps that from happening. A worker that died or timed out is
    replaced on the way out.
    """

    def __init__(self, bin_path: str, model_path: str, n: int, cpu: bool = False,
                 timeout: float = 30.0):
        self._bin = bin_path
        self._model = model_path
        self._cpu = cpu
        self._timeout = timeout
        self._q: "queue.Queue[LayaEngine]" = queue.Queue()
        for _ in range(n):
            self._q.put(self._new())
        self.size = n

    def _new(self):
        return LayaEngine(self._bin, self._model, daemon=True,
                          extra_env={"LAYA_CPU": "1"} if self._cpu else None,
                          timeout=self._timeout)

    @contextmanager
    def worker(self):
        w = self._q.get()
        try:
            if not w.alive():
                w._respawn()
            yield w
        finally:
            # A wedged worker is replaced rather than handed to the next request.
            if not w.alive():
                w._respawn()
            self._q.put(w)

    def restarts(self):
        return sum(w.restarts for w in list(self._q.queue))

    def alive(self):
        return sum(1 for w in list(self._q.queue) if w.alive())

    def close(self):
        while not self._q.empty():
            try:
                self._q.get_nowait().close()
            except queue.Empty:
                break


class PredictRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Dict[str, str]]]
    questions: Optional[Dict[str, Any]] = None
    # `schema` is the natural name on the wire but shadows BaseModel.schema.
    schema_name: Optional[str] = Field(None, alias="schema")


class PredictResponse(BaseModel):
    results: Dict[str, Any]
    slots: int
    ms: float


def check_key(authorization: Optional[str], x_api_key: Optional[str] = None):
    """Accept `Authorization: Bearer <key>` (laya-serve's scheme) and, for older
    clients, `X-API-Key`. Constant-time: a plain == leaks the key byte by byte."""
    expected = os.environ.get("LAYA_API_KEY")
    if not expected:
        return
    presented = None
    if authorization and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()
    elif x_api_key:
        presented = x_api_key
    if not presented or not hmac.compare_digest(presented, expected):
        raise HTTPException(401, "bad or missing credentials")


def state_too_long(state) -> bool:
    if isinstance(state, str):
        return len(state) > MAX_STATE_CHARS
    if isinstance(state, dict):
        return len(" ".join(f"{k}: {v}" for k, v in state.items())) > MAX_STATE_CHARS
    if isinstance(state, list):
        return len(" ".join(str(t.get("content", "")) for t in state)) > MAX_STATE_CHARS
    return False


def resolve_questions(req: PredictRequest) -> Dict[str, Any]:
    """Explicit `questions` win; otherwise use the schema loaded at boot."""
    if req.questions:
        for name, q in req.questions.items():
            t = q.get("type", q.get("t"))
            if t not in ("choice", "score", "noul"):
                raise HTTPException(400, f"question {name!r}: type must be one of "
                                         f"choice/score/noul, got {t!r}")
            if t in ("choice", "score") and not q.get("criteria", q.get("crit")):
                raise HTTPException(400, f"question {name!r}: {t} requires criteria")
        return req.questions

    schemas = STATE.get("schemas") or {}
    if req.schema_name:
        if req.schema_name not in schemas:
            raise HTTPException(404, f"unknown schema {req.schema_name!r}; "
                                     f"known: {list(schemas)}")
        return schemas[req.schema_name]
    if "default" in schemas:
        return schemas["default"]
    raise HTTPException(400, "provide `questions`, or start the server with --schema")


@app.exception_handler(Exception)
def unhandled(request: Request, exc: Exception):
    """Log server-side faults but don't leak internals to the client.

    HTTPException is handled by FastAPI's own handler and never reaches this one.
    Note log.exception() is useless here: the handler runs outside an active
    except block, so it prints "NoneType: None" instead of the real error.
    """
    log.error("unhandled error on %s: %r", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/health")
def health():
    pool: WorkerPool = STATE["pool"]
    return {
        "ok": pool.alive() > 0,
        "model": STATE.get("model"),
        "workers": f"{pool.alive()}/{pool.size}",
        "schemas": list((STATE.get("schemas") or {}).keys()),
    }


@app.get("/metrics")
def metrics():
    m: Metrics = STATE["metrics"]
    out = m.snapshot()
    out["worker_restarts"] = STATE["pool"].restarts()
    return out


@app.get("/v1/questions")
def list_questions():
    """Names of the question sets loaded at startup, and their question keys."""
    return {k: list(v.keys()) for k, v in (STATE.get("schemas") or {}).items()}


@app.get("/v1/schemas")
def list_schemas():
    """Deprecated alias for `/v1/questions`."""
    return list_questions()


def run_predict(req: PredictRequest, authorization: Optional[str] = None,
                x_api_key: Optional[str] = None):
    """Shared body for both routes: validate, run, record metrics."""
    check_key(authorization, x_api_key)

    questions = resolve_questions(req)
    if state_too_long(req.state):
        raise HTTPException(400, f"state too long (>{MAX_STATE_CHARS} chars); "
                                 "truncating it would silently drop content")

    tok: LayaTokenizer = STATE["tok"]
    cfg = STATE.get("cfg", {})
    metrics: Metrics = STATE["metrics"]

    t0 = time.perf_counter()
    timed_out = False
    try:
        with STATE["pool"].worker() as engine:
            results = predict(engine, tok, req.state, questions, cfg)
    except ValueError as e:
        # Raised by build_sequence when a slot does not fit max_len.
        dt = (time.perf_counter() - t0) * 1000
        metrics.record(dt, len(questions), ok=False)
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        dt = (time.perf_counter() - t0) * 1000
        timed_out = "timed out" in str(e)
        metrics.record(dt, len(questions), ok=False, timeout=timed_out)
        log.error("engine failure: %s", e)
        raise HTTPException(503, f"engine unavailable: {e}")
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        metrics.record(dt, len(questions), ok=False)
        log.exception("predict failed")
        raise HTTPException(500, "internal error")

    dt = (time.perf_counter() - t0) * 1000
    metrics.record(dt, len(questions))
    log.info("predict slots=%d ms=%.1f", len(questions), dt)
    return results, questions, dt


@app.post("/v1/systemone")
def systemone(req: PredictRequest,
              authorization: Optional[str] = Header(None)):
    """laya-serve compatible route: Jev-shaped `{model, answers, usage}`.

    A client written against `laya-serve` (or the Jev decision API generally)
    can point its baseUrl here and keep working -- the request body is already
    `{state, questions}`.
    """
    results, questions, dt = run_predict(req, authorization)
    return {
        "model": STATE.get("model_name"),
        "answers": to_jev_answers(results, questions),
        # Nothing is generated: one forward pass answers every question. So
        # `input_tokens` is the total sequence length and `output_tokens` the
        # number of decisions returned.
        "usage": {
            "input_tokens": _count_tokens(req, questions),
            "output_tokens": len(questions),
        },
    }


@app.post("/v1/predict", response_model=PredictResponse)
def do_predict(req: PredictRequest,
               authorization: Optional[str] = Header(None),
               x_api_key: Optional[str] = Header(None)):
    """Our own shape, kept for clients already written against it."""
    results, questions, dt = run_predict(req, authorization, x_api_key)
    return PredictResponse(results=results, slots=len(questions), ms=round(dt, 2))


def _count_tokens(req, questions) -> int:
    """Total tokens the request will consume, for the usage block."""
    tok: LayaTokenizer = STATE["tok"]
    cfg = STATE.get("cfg", {})
    total = 0
    for name, q in questions.items():
        ids, _ = build_sequence(tok, req.state, q, cfg.get("max_len", 1024),
                                cfg.get("head_max_len", 256))
        total += len(ids)
    return total


def main():
    ap = argparse.ArgumentParser(description="HTTP service for the ggml Laya engine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default=None, help="laya-infer binary (auto-detect)")
    ap.add_argument("--config", default=None, help="rl_agent_config.json")
    ap.add_argument("--schema", action="append", default=[],
                    help="questions JSON; repeatable. first one becomes the default")
    ap.add_argument("--questions", action="append", default=[], dest="questions_files",
                    help="laya-style alias for --schema")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--workers", type=int, default=1,
                    help="engine daemons; each holds the model, so this costs memory")
    ap.add_argument("--cpu", action="store_true", help="run workers on CPU")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds before a request abandons and restarts its worker")
    ap.add_argument("--log", default=None, help="log file (default: stderr)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        filename=args.log,
    )

    engine_bin = args.engine or os.path.join(HERE, "build", "laya-infer")
    if not os.path.exists(engine_bin):
        sys.exit(f"engine not found: {engine_bin}")

    t0 = time.perf_counter()
    tok = LayaTokenizer(args.tokenizer)
    schemas = {}
    for i, path in enumerate(args.schema + args.questions_files):
        name = os.path.splitext(os.path.basename(path))[0]
        schemas[name] = json.load(open(path))
        if i == 0:
            schemas["default"] = schemas[name]

    cfg = {}
    cfg_path = args.config or os.path.join(os.path.dirname(args.model), "rl_agent_config.json")
    if cfg_path and os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))

    pool = WorkerPool(engine_bin, args.model, args.workers, cpu=args.cpu,
                      timeout=args.timeout)
    boot = (time.perf_counter() - t0) * 1000

    STATE.update(tok=tok, cfg=cfg, schemas=schemas, pool=pool,
                 model=args.model, metrics=Metrics(),
                 model_name=os.path.splitext(os.path.basename(args.model))[0])

    print(f"[laya] boot {boot:.0f} ms | {args.workers} worker(s) "
          f"({'cpu' if args.cpu else 'gpu'}) | schemas={list(schemas)}", file=sys.stderr)
    print(f"[laya] listening on {args.host}:{args.port}", file=sys.stderr)

    try:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
