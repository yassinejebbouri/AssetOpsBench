"""Post-hoc TTA (time-to-accuracy) evaluator.

Reads one or more benchmark JSONL files, uses an LLM judge to grade each
record's ``final_answer`` against its ``characteristic_form``, and writes
a ``_scored.jsonl`` sibling with added fields:

    accuracy_status     "Pass" | "Fail" | "Error" | "Skipped"
    accuracy_reasoning  short free-text justification from the judge
    tta_seconds         total_wall_time_s when accuracy_status == "Pass", else None

"Skipped" means the record had no ``final_answer`` to judge (e.g., status
was ``error`` in the benchmark run). "Error" means the judge itself
failed (rate-limit, parse error after retries, etc.).

The judge call is intentionally post-hoc so it doesn't pollute the
hardware metrics captured during the benchmark runs themselves.

Run from the repo root:
    uv run python src/benchmarking/evaluate_tta.py benchmarking_fmsr_mcp.jsonl
    uv run python src/benchmarking/evaluate_tta.py \\
        benchmarking_fmsr_mcp.jsonl benchmarking_fmsr_direct.jsonl
    uv run python src/benchmarking/evaluate_tta.py \\
        --model openai/llama-3.3-70b-versatile benchmarking_fmsr_mcp.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm import LiteLLMBackend  # noqa: E402

# default judge -- same family as the agents being evaluated so behaviour is consistent
DEFAULT_JUDGE_MODEL = os.environ.get(
    "BENCHMARK_JUDGE_MODEL",
    os.environ.get("BENCHMARK_MODEL_ID", "watsonx/meta-llama/llama-3-2-90b-vision-instruct"),
)

_JUDGE_PROMPT = """\
You are evaluating whether an AI agent's answer satisfies the expected \
criteria for an industrial asset operations question.

Question:
{question}

Expected answer criteria (characteristic form):
{characteristic_form}

Agent's final answer:
{agent_answer}

Decide Pass or Fail:
- Pass: the agent's answer satisfies the criteria. For deterministic \
criteria that enumerate exact expected items, the answer must reference \
those items. For descriptive criteria, the answer must address the \
described requirements.
- Fail: the answer is wrong, missing key expected content, off-topic, \
or the agent returned an error message instead of substantive content.

Respond with a single JSON object on one line, nothing else:
{{"status": "Pass", "reasoning": "<one sentence>"}}
or
{{"status": "Fail", "reasoning": "<one sentence>"}}
"""

_JUDGE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> dict:
    """Extract {status, reasoning} from an LLM judge response."""
    text = raw.strip()
    # strip code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).lstrip("json").strip()
    # try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "status" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # try regex-extract the first JSON-looking block
    for m in _JUDGE_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "status" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    # fallback: keyword match
    low = text.lower()
    if "pass" in low and "fail" not in low:
        return {"status": "Pass", "reasoning": "Fallback keyword match."}
    if "fail" in low:
        return {"status": "Fail", "reasoning": "Fallback keyword match."}
    return {"status": "Error", "reasoning": f"Unparseable judge response: {text[:200]}"}


def _judge_record(llm: LiteLLMBackend, record: dict, max_retries: int = 2) -> dict:
    """Run the LLM judge on one record and return {status, reasoning}.

    Returns status="Skipped" for records with no substantive final_answer.
    Returns status="Error" only if the judge itself fails after retries.
    """
    answer = (record.get("final_answer") or "").strip()
    question = record.get("scenario_text_full") or record.get("scenario_text") or ""
    characteristic = record.get("characteristic_form") or ""

    if not answer or not characteristic:
        return {
            "status": "Skipped",
            "reasoning": "No final_answer or no characteristic_form on record.",
        }

    prompt = _JUDGE_PROMPT.format(
        question=question,
        characteristic_form=characteristic,
        agent_answer=answer,
    )

    last_err = None
    for _ in range(max_retries + 1):
        try:
            raw = llm.generate(prompt)
            parsed = _parse_judge_response(raw)
            if parsed.get("status") in ("Pass", "Fail"):
                return parsed
            last_err = parsed.get("reasoning", "parse failure")
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(2.0)

    return {
        "status": "Error",
        "reasoning": f"Judge failed after {max_retries + 1} attempts: {last_err}",
    }


def evaluate_file(
    in_path: str,
    out_path: str | None,
    llm: LiteLLMBackend,
    force: bool = False,
) -> tuple[int, int, int, int]:
    """Score every record in a JSONL file. Returns (n_pass, n_fail, n_skip, n_err)."""
    if out_path is None:
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + "_scored.jsonl"))

    # resume: skip records that already have accuracy_status in the output file
    done_ids: set[tuple[int, int]] = set()
    if os.path.exists(out_path) and not force:
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "accuracy_status" in r:
                        done_ids.add((int(r["scenario_id"]), int(r["run_id"])))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        if done_ids:
            print(f"  Resume: {len(done_ids)} records already scored, will skip.")

    records: list[dict] = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n_pass = n_fail = n_skip = n_err = 0

    file_mode = "a" if (done_ids and not force) else "w"
    with open(out_path, file_mode) as out_f:
        for i, rec in enumerate(records):
            key = (int(rec.get("scenario_id", -1)), int(rec.get("run_id", -1)))
            if key in done_ids:
                continue

            print(f"  [{i+1}/{len(records)}] id={rec.get('scenario_id')} "
                  f"run={rec.get('run_id')} status={rec.get('status')} -> judging...",
                  end=" ", flush=True)

            judgment = _judge_record(llm, rec)
            status = judgment["status"]

            if status == "Pass":
                tta = rec.get("total_wall_time_s")
                n_pass += 1
            else:
                tta = None
                if status == "Fail":
                    n_fail += 1
                elif status == "Skipped":
                    n_skip += 1
                else:
                    n_err += 1

            enriched = {
                **rec,
                "accuracy_status": status,
                "accuracy_reasoning": judgment.get("reasoning", ""),
                "tta_seconds": tta,
            }
            out_f.write(json.dumps(enriched) + "\n")
            out_f.flush()

            print(f"{status}")

    return n_pass, n_fail, n_skip, n_err


def main():
    parser = argparse.ArgumentParser(description="Post-hoc LLM-judge TTA evaluator")
    parser.add_argument("files", nargs="+",
                        help="JSONL file(s) from run_direct_agent.py / run_mcp.py to score.")
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge LLM model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Re-score even if _scored.jsonl already has the record.")
    args = parser.parse_args()

    llm = LiteLLMBackend(args.model)
    print(f"Judge model: {args.model}\n")

    grand = {"Pass": 0, "Fail": 0, "Skipped": 0, "Error": 0}
    for in_path in args.files:
        if not os.path.exists(in_path):
            print(f"Missing {in_path}, skipping.")
            continue
        print(f"Scoring {in_path} ...")
        n_pass, n_fail, n_skip, n_err = evaluate_file(in_path, None, llm, force=args.force)
        grand["Pass"] += n_pass
        grand["Fail"] += n_fail
        grand["Skipped"] += n_skip
        grand["Error"] += n_err
        total_eval = n_pass + n_fail
        acc = (n_pass / total_eval) if total_eval > 0 else 0.0
        print(f"  -> Pass={n_pass} Fail={n_fail} Skipped={n_skip} Error={n_err} "
              f"(accuracy over Pass+Fail = {acc:.1%})\n")

    total_graded = grand["Pass"] + grand["Fail"]
    print("=" * 60)
    print("TTA EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Pass    : {grand['Pass']}")
    print(f"  Fail    : {grand['Fail']}")
    print(f"  Skipped : {grand['Skipped']}  (no final_answer / no characteristic_form)")
    print(f"  Error   : {grand['Error']}    (judge failed)")
    if total_graded:
        print(f"  Overall accuracy: {grand['Pass'] / total_graded:.1%}")


if __name__ == "__main__":
    main()
