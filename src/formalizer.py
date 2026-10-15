"""Client for the tuned specialist formalizer, served by vLLM on a GPU server.

ROLE SPLIT
----------
The 27B orchestrator keeps every job it had -- planning, tools, journal. This
client covers exactly one subroutine: informal claim -> Lean signature, the
step measured at 28% (T2) with 77% of failures systematic. The specialist is a
7B LoRA-tuned on 9,566 contamination-free Mathlib (docstring, signature) pairs,
and it takes over `derive_signature` ONLY while FORMALIZER_BASE is set -- and
it keeps the job only if it clears the 210-item benchmark by the 7-point
detection threshold. Anything less is a null result and the 27B path stays.

WHY THIS PATH HAS NO SCHEMA AND NO THINK FLAG
---------------------------------------------
The model is trained to emit exactly one signature line. Greedy decode, stop at
newline, done. The think/format budget collision that produced six separate
silent failures in this project cannot occur on a path with no reasoning budget
and no grammar constraint.

PROMPT is byte-identical to scripts/train_formalizer.py -- tests assert it --
because a drifted prompt is a silent train/inference distribution shift, and
this project's whole history is silent shifts found the expensive way.
"""

import requests

from . import config

# Canonical copy: scripts/train_formalizer.py defines the original.
PROMPT = (
    "Translate the mathematical claim into one Lean 4 theorem signature that "
    "elaborates against Mathlib.\n\n"
    "CLAIM: {informal}\n\n"
    "Answer with ONLY the signature, one line, no proof and no commentary:\n"
    "theorem <name> <binders> : <conclusion>"
)


def available() -> bool:
    return bool(config.FORMALIZER_BASE)


def _body(statement: str, think_off: bool) -> dict:
    """Request body, with thinking explicitly disabled.

    THE TRAINING PROMPT AND THE SERVING PROMPT MUST MATCH.
    The 27B adapter was trained against a template rendered with
    `enable_thinking=False`, which pre-fills an EMPTY CLOSED `<think></think>`
    block so the answer starts immediately. vLLM's chat endpoint applies the
    template with its defaults, which leaves that block OPEN -- the model then
    emits reasoning prose into a 220-token budget stopped at the first newline,
    and every derivation scores as a failure. Observed exactly that before this
    was added; it is the seventh time a think budget has silently collided with
    a structured-output budget in this project.
    """
    body = {"model": config.FORMALIZER_MODEL,
            "messages": [{"role": "user",
                          "content": PROMPT.format(informal=str(statement)[:1500])}],
            "temperature": 0, "max_tokens": 220, "stop": ["\n"]}
    if think_off:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def derive(statement: str, timeout: int = 120) -> str:
    """One signature line from the specialist. Empty string on any failure.

    Temperature 0: the specialist's value is that this call is REPEATABLE.
    The caller (`prover._derive_once`) runs the identical downstream path --
    extraction, rename-probe typecheck, instance repair -- so the specialist
    is judged by exactly the gates the 27B is judged by.
    """
    if not available():
        return ""
    try:
        r = requests.post(
            f"{config.FORMALIZER_BASE.rstrip('/')}/chat/completions",
            json=_body(statement, think_off=True), timeout=timeout)
        if r.status_code == 400:      # template takes no enable_thinking kwarg
            r = requests.post(
                f"{config.FORMALIZER_BASE.rstrip('/')}/chat/completions",
                json=_body(statement, think_off=False), timeout=timeout)
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"]).strip()
    except Exception:
        return ""
