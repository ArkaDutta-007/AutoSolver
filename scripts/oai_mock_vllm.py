#!/usr/bin/env python3
"""A stand-in for `vllm serve`, so oai_pipeline.py runs end to end on a CPU.

Speaks what the pipeline uses -- /v1/models, /tokenize, /v1/chat/completions
with n and ignore_eos -- and answers from the problems' reference proofs, so
the Lean checks downstream are real. Per sample, by a fixed hash:

  plain    the reference proof, a proof that cannot compile, a reply cut off at
           max_tokens, a reply that circles then fails, or no code at all
  repair   a fix, or the same failed answer again (Lean then reports the same
           error: a `repeat`)
  hint     the proof shown in the prompt

--better moves the odds toward the reference proof and away from the failure
modes: the selftest's stand-in for a tuned model.

    python3 oai_mock_vllm.py --pairs data/pairs.jsonl data/minif2f.jsonl --port 8011
"""

import argparse
import hashlib
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import oai_sample as S

NAME = re.compile(r"theorem\s+([^\s(:{\[⦃]+)")
TACTICS = ["norm_num", "omega", "simp", "decide", "linarith", "nlinarith [sq_nonneg (x - y)]"]


def pick(key, odds):
    u = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10000 / 10000
    for outcome, p in odds:
        if u < p:
            return outcome
        u -= p
    return odds[-1][0]


def answer(probs, messages, k, better):
    users = [m["content"] for m in messages if m["role"] == "user"]
    p = next((probs[n] for n in NAME.findall(users[0]) if n in probs), None)
    if p is None:
        return "I do not recognise this theorem.", "stop"
    key = f"{p['name']}:{k}:{len(messages)}:{better}"
    head = f"{S.opens(p)}theorem {p['name']} {p['sig']} := "
    right = (f"```lean4\n{S.reference(p)}\n```" if S.has_proof(p) else
             f"```lean4\n{head}by\n  {TACTICS[int(key.encode().hex(), 16) % len(TACTICS)]}\n```")
    wrong = f"```lean4\n{head}by\n  exact no_such_lemma_{k} _\n```"
    if "A verified proof of this theorem" in users[0]:
        return f"<think>\nInduct, then close the goal.\n</think>\n\n{right}", "stop"
    if len(users) > 1:                                   # a repair turn
        last = [m["content"] for m in messages if m["role"] == "assistant"][-1]
        odds = [("fix", .6), ("same", .2), ("wrong", .2)] if better else \
               [("fix", .3), ("same", .5), ("wrong", .2)]
        out = pick(key, odds)
        return {"fix": right, "same": last, "wrong": wrong}[out], "stop"
    odds = [("right", .7), ("wrong", .2), ("cut", .04), ("loop", .03), ("none", .03)] if better \
        else [("right", .35), ("wrong", .35), ("cut", .1), ("loop", .1), ("none", .1)]
    out = pick(key, odds)
    if out == "cut":
        return "<think>\n" + "Consider the case split once more. " * 60, "length"
    if out == "loop":
        return ("<think>\n" + "Let me reconsider whether the bound holds for every n here.\n" * 30
                + f"</think>\n\n{wrong}"), "stop"
    if out == "none":
        return "<think>\nNo idea.\n</think>\nI could not find a proof.", "stop"
    return f"<think>\nThe direct route works.\n</think>\n\n{right if out == 'right' else wrong}", "stop"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--served-model-name", default="Goedel-LM/Goedel-Prover-V2-32B")
    ap.add_argument("--model-len", type=int, default=16384)
    ap.add_argument("--tok-per-s", type=float, default=2000, help="simulated decode speed")
    ap.add_argument("--better", action="store_true")
    a = ap.parse_args()
    probs = {}
    for path in a.pairs:
        for line in open(path, encoding="utf-8"):
            p = json.loads(line)
            probs[p["name"]] = p

    class H(BaseHTTPRequestHandler):
        def send(self, body, code=200):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                return self.send({"object": "list", "data": [{"id": a.served_model_name}]})
            self.send({"error": "not found"}, 404)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                count = sum(len(m["content"]) for m in req["messages"]) // 3 + 10
                return self.send({"count": count, "max_model_len": a.model_len})
            if self.path != "/v1/chat/completions" or req.get("model") != a.served_model_name:
                return self.send({"error": "bad request"}, 400)
            n, cap = req.get("n", 1), req["max_tokens"]
            if req.get("ignore_eos"):                     # the decode-speed bench
                time.sleep(cap / a.tok_per_s)
                outs = [("x " * cap, "length")] * n
            else:
                outs = [answer(probs, req["messages"], k, a.better) for k in range(n)]
            return self.send({"choices": [{"index": i, "finish_reason": f,
                                           "message": {"role": "assistant", "content": t}}
                                          for i, (t, f) in enumerate(outs)],
                              "usage": {"completion_tokens": cap * n if req.get("ignore_eos")
                                        else sum(len(t) // 3 for t, _ in outs)}})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
