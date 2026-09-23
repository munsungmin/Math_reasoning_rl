"""CPU-only adapter for the pinned Limit-of-RLVR evaluator. JSON stdin/stdout."""

import contextlib
import itertools
import json
import multiprocessing
import os
import re
import signal
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_ROOT = Path(
    os.environ.get("LIMIT_RLVR_ROOT", str(REPO_ROOT / "ext/limit-of-RLVR"))
)
EVALUATOR_DIR = EVALUATOR_ROOT / "math/examples/math_eval"
for filename in ("parser.py", "grader.py"):
    if not (EVALUATOR_DIR / filename).is_file():
        raise FileNotFoundError(
            f"MATH evaluator missing: {EVALUATOR_DIR / filename}. Set LIMIT_RLVR_ROOT to the limit-of-RLVR checkout directory."
        )
sys.path.insert(0, str(EVALUATOR_DIR))
from grader import math_equal
from parser import extract_answer, strip_string

VERSION = "limit-79c348f-explicit-v4"


def normalize(value):
    value = value.strip().rstrip(".")
    if len(value) > 1 and value[0] == value[-1] and value[0] in "_*":
        value = value[1:-1]
    value = re.sub(r"(?:°|\^\{?\\circ\}?| degrees?)\.?$", "", value)
    return strip_string(value)


def final_answer(text):
    """Require an explicit final answer; never credit an arbitrary last number."""
    xml = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if xml:
        value = xml[-1].strip()
        return (
            normalize(value)
            if "\\boxed" not in value
            else normalize(extract_answer(value, "math", False))
        )
    boxes = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 1
        for end in range(match.end(), len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            if depth == 0:
                boxes.append(text[match.end() : end])
                break
    if boxes:
        return normalize(boxes[-1])
    answers = re.findall(
        r"(?:^|\n)\s*(?:Final\s+)?Answer\s*:\s*([^\n]+)|(?:[Tt]he|[Ff]inal) answer is\s*([^\n]+)",
        text,
    )
    if answers:
        return normalize(next(x for x in answers[-1] if x).strip())
    # Recover mathematical expressions in an explicit concluding paragraph.
    # Do not scan arbitrary intermediate work or credit a final stray number.
    conclusions = list(re.finditer(r"(?:^|\n)\s*(?:Therefore|Hence|Thus|So),?\s", text))
    if conclusions:
        tail = text[conclusions[-1].end() :]
        if "```" in tail:
            return ""
        exprs = re.findall(
            r"\$\$?(.*?)\$\$?|\\\((.*?)\\\)|\\\[(.*?)\\\]|\\begin\{align\*?\}(.*?)\\end\{align\*?\}",
            tail,
            re.S,
        )
        if exprs:
            value = next((x for x in exprs[-1] if x), "").strip()
            if not value:
                return ""
            if "=" in value:
                value = value.rsplit("=", 1)[-1].lstrip("& ")
            return normalize(value)
        fraction = re.findall(r"\bis\s+(\d+/\d+)\s+of\b", tail)
        if fraction and len(set(fraction)) == 1:
            return normalize(fraction[-1])
    return ""


def equal(pred, gold):
    return bool(math_equal(pred, gold, include_percentage=False, timeout=True))


def equivalent(pred, gold):
    if not pred:
        return False
    # A scalar plus/minus answer denotes BOTH roots, not either one. Only this
    # representation permits unordered matching; coordinate order stays intact.
    marker = "\\pm" if "\\pm" in gold else "±" if "±" in gold else None
    if marker and gold.count(marker) == 1 and not gold.startswith(("(", "[")):
        expected = [gold.replace(marker, sign) for sign in ["+", "-"]]
        if marker in pred and pred.count(marker) == 1:
            actual = [pred.replace(marker, sign) for sign in ["+", "-"]]
        elif not pred.startswith(("(", "[")):
            actual = (
                pred[1:-1] if pred.startswith("{") and pred.endswith("}") else pred
            ).split(",")
        else:
            actual = []
        if len(actual) == 2:
            return any(
                all(equal(a, b) for a, b in zip(actual, order))
                for order in itertools.permutations(expected)
            )
        return False
    return equal(pred, gold)


def grade(item):
    prediction = final_answer(item["completion"])
    gold = item["answer"].split("####")[-1].strip()
    return {
        "prediction": prediction,
        "correct": equivalent(prediction, gold),
        "extraction_status": "explicit" if prediction else "no_final_answer",
        "grader_version": VERSION,
    }


def safe_grade(item):
    def expired(signum, frame):
        raise TimeoutError("Per-answer grading exceeded 5 seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 5)
    try:
        return grade(item)
    except Exception as error:
        for child in multiprocessing.active_children():
            child.terminate()
            child.join(timeout=1)
        return {
            "prediction": "",
            "correct": False,
            "extraction_status": "grader_timeout"
            if isinstance(error, TimeoutError)
            else "grader_error",
            "grader_version": VERSION,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    items = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        results = [safe_grade(i) for i in items]
    print(json.dumps(results))


if __name__ == "__main__":
    main()
