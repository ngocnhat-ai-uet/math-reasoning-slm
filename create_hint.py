import os
import json
import time
import argparse
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

# INPUT_PATH = "data/train/bespoke_math_dataset.parquet"
# OUTPUT_PATH = "data/train/hints.parquet"
# CHECKPOINT_PATH = "data/train/hints_checkpoint.jsonl"
# BATCH_DIR = "data/train/hint_batches"
INPUT_PATH = "data/train/omni_math_dataset_gt2000.parquet"
OUTPUT_PATH = "data/train/omni_hints.parquet"
CHECKPOINT_PATH = "data/train/omni_hints_checkpoint.jsonl"
BATCH_DIR = "data/train/omni_hint_batches"

MODEL = "gpt-5.4-nano"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1800"))
PROMPT_CACHE_KEY = "hint_generation_v3_three_level"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
BATCH_POLL_INTERVAL = int(os.getenv("BATCH_POLL_INTERVAL", "30"))
MAX_ACTIVE_BATCHES = int(os.getenv("MAX_ACTIVE_BATCHES", "4"))
MAX_RETRIES = 5


# =========================
# PROMPT
# =========================

HINT_GENERATION_PROMPT = r"""
You are generating three levels of hints for a math problem.

Create:
1. concise_hint: a minimal direction-only hint.
2. detailed_method_hint: a method-level scaffold without computed intermediate results.
3. detailed_scaffold_hint: a stronger solution-faithful scaffold, but still without the final answer.

Core goal:
- Follow the reasoning path used in the provided solution; do not replace it with a different or easier method.
- Test whether a student/model can understand and continue from that solution path.
- The hints must increase in helpfulness:
  concise_hint < detailed_method_hint < detailed_scaffold_hint.

Global rules:
- Do NOT reveal or paraphrase the final answer, final option, final value, final expression, boxed result, or any content from a final-answer statement.
- Do NOT identify the correct multiple-choice option directly or indirectly.
- Do NOT copy sentences from the solution.
- Do NOT give away information that allows shortcut guessing.
- Return the hints in the same language as the question.
- Return only valid JSON.

Formatting rules:
- Use plain ASCII math whenever possible; avoid TeX/LaTeX unless necessary.
- Do NOT use TeX delimiters, invisible/control characters, unusual Unicode formatting, or special separators.
- Prefer ASCII forms rather than TeX/Unicode, for example: sqrt(x) rather than \sqrt{x}; a/b rather than \frac{a}{b}; x^2 rather than x^{2}; x_1 rather than x_{1}; <=/>=/!= rather than \leq/\geq/\neq; theta/pi rather than \theta/\pi; * rather than \times or \cdot; degrees rather than ^\circ; prod/sum rather than \prod/\sum.
- Prefer verbal descriptions over long symbolic formulas.
- Avoid long products, long summations, or heavily nested formulas unless essential.

Rules for concise_hint:
- State only the central idea that unlocks the solution: the main approach, method, theorem, representation, substitution, invariant, or strategy used in the solution.
- It must summarize the solution direction at a high level, not list the solution steps.
- It must be direction-only and faithful to the solution path.
- Prefer verbal method descriptions over formulas.
- Do NOT execute the method: no calculations, derived equations, intermediate results, or step-by-step actions.
- Maximum length: 1 short sentence. Use 2 short sentences only if absolutely necessary.

Rules for detailed_method_hint:
- Return detailed_method_hint as a JSON list of concise method-scaffold points.
- It should reveal the method path, but must not execute the reasoning.
- Describe the method path, not the execution path.
- Stop once the correct setup or reasoning framework is identified; leave the actual solving/evaluation to the student.
- Use the minimum number of points needed to reflect the solution path.
- Usually use 2-5 points; for unusually complex solutions, use up to 7.
- Each point should describe one method choice, representation, transformation idea, or reasoning direction.
- You may mention definitions, formulas, substitutions, transformations, cases, invariants, or proof strategies used in the solution.
- Prefer verbal method descriptions over explicit formulas; if a formula is needed, write it in compact ASCII plain text.
- Do NOT carry out calculations, substitutions, simplifications, algebraic solving, or proof steps.
- Do NOT state results obtained after a step: derived equations, expressions, values, ratios, parameters, simplified forms, or numerical checkpoints.
- Do NOT use phrases like "this gives", "which yields", "so the equation becomes", or "you should get" followed by a mathematical result.
- Do NOT include answer-choice matching or final algebraic solving.

Rules for detailed_scaffold_hint:
- Return detailed_scaffold_hint as a JSON list of concise scaffold points.
- It should be stronger and more explicit than detailed_method_hint, but still not a complete solution.
- Use the minimum number of points needed to help the student follow the solution path.
- Usually use 3-6 points; for unusually complex solutions, use up to 8.
- Each point should describe one concrete mathematical action or non-final reasoning checkpoint.
- It may include explicit setup, formulas, substitutions, or non-final equations from the solution path.
- If formulas are used, write them in compact ASCII plain text.
- It may mention intermediate expressions or equations only if they are not decisive and do not make the remaining work trivial.
- Do NOT include solved variable values, final selected options, final simplified expressions, decisive final equations, final computation, final conclusion, or final option matching.
- Do NOT merge the entire solution into one long point.

Style rules for both detailed hints:
- Keep each point short, action-oriented, and not paragraph-length.
- Do NOT split trivial algebra into many micro-steps.
- For simple arithmetic or direct-definition problems, keep the hints minimal; do not add unnecessary scaffold points.
- Avoid vague advice such as "solve carefully", "simplify the expression", or "continue the calculation"; specify the mathematical direction without completing the step.

Question:
{question}

Solution:
{solution}

Return JSON with exactly these keys.
The example only illustrates the JSON structure, not the number of points.

{{
  "concise_hint": "...",
  "detailed_method_hint": [
    "concise method-scaffold point",
    "concise method-scaffold point",
    "concise method-scaffold point"
  ],
  "detailed_scaffold_hint": [
    "stronger but non-final scaffold point",
    "stronger but non-final scaffold point",
    "stronger but non-final scaffold point"
  ]
}}
""".strip()


# =========================
# PROMPT SPLIT
# =========================


PROMPT_QS_BLOCK = """Question:
{question}

Solution:
{solution}"""

QUESTION_SOLUTION_TEMPLATE = """Question:
{question}

Solution:
{solution}

Return output as valid JSON only."""


def build_prompt_parts(question: str, solution: str) -> tuple[str, str]:
    """Keep HINT_GENERATION_PROMPT stable, but send row data separately.

    The system prompt is identical for all rows, so the Responses API can reuse
    prompt caching for the long instruction prefix. The user input only contains
    the row-specific question and solution.
    """
    before, after = HINT_GENERATION_PROMPT.split(PROMPT_QS_BLOCK)
    system_prompt = f"{before.rstrip()}\n\n{after.lstrip()}".strip()
    system_prompt = system_prompt.replace("{{", "{").replace("}}", "}")
    user_input = QUESTION_SOLUTION_TEMPLATE.format(
        question=question,
        solution=solution,
    )
    return system_prompt, user_input


SYSTEM_PROMPT, _ = build_prompt_parts("{question}", "{solution}")


HINT_JSON_SCHEMA = {
    "type": "json_schema",
    "name": "math_hints",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "concise_hint": {"type": "string"},
            "detailed_method_hint": {
                "type": "array",
                "items": {"type": "string"},
            },
            "detailed_scaffold_hint": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["concise_hint", "detailed_method_hint", "detailed_scaffold_hint"],
    },
}


def responses_body(input_text: str, instructions: str = SYSTEM_PROMPT) -> dict:
    body = {
        "model": MODEL,
        "instructions": instructions,
        "input": input_text,
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": {"format": HINT_JSON_SCHEMA},
        "prompt_cache_key": PROMPT_CACHE_KEY,
    }
    return body


# =========================
# JSON PARSE
# =========================


def safe_json_loads(text: str):
    text = text.strip()

    # Remove markdown fences if model accidentally returns them
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    return json.loads(text)


def normalize_hint_list(value, field_name: str) -> list[str]:
    if isinstance(value, str):
        value = [line.strip() for line in value.split("\n") if line.strip()]

    if not isinstance(value, list):
        raise ValueError(f"Invalid {field_name}.")

    normalized = [str(x).strip() for x in value if str(x).strip()]
    if not normalized:
        raise ValueError(f"Empty {field_name}.")

    return normalized


def normalize_hint_obj(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("Output is not a JSON object.")

    concise = str(obj.get("concise_hint", "")).strip()
    detailed_method = normalize_hint_list(
        obj.get("detailed_method_hint", []),
        "detailed_method_hint",
    )
    detailed_scaffold = normalize_hint_list(
        obj.get("detailed_scaffold_hint", []),
        "detailed_scaffold_hint",
    )

    if not concise:
        raise ValueError("Missing concise_hint.")

    return {
        "concise_hint": concise,
        "detailed_method_hint": detailed_method,
        "detailed_scaffold_hint": detailed_scaffold,
    }


USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
)


def as_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def as_int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage(value) -> dict:
    data = as_dict(value)
    usage = as_dict(data.get("usage")) if data else as_dict(getattr(value, "usage", None))
    if not usage:
        return {}

    input_details = as_dict(
        usage.get("input_tokens_details")
        or usage.get("prompt_tokens_details")
    )

    parsed = {
        "input_tokens": as_int_or_none(
            usage.get("input_tokens") or usage.get("prompt_tokens")
        ),
        "cached_input_tokens": as_int_or_none(input_details.get("cached_tokens")),
        "output_tokens": as_int_or_none(
            usage.get("output_tokens") or usage.get("completion_tokens")
        ),
        "total_tokens": as_int_or_none(usage.get("total_tokens")),
    }
    return {key: value for key, value in parsed.items() if value is not None}


def usage_record_fields(usage: dict) -> dict:
    return {key: usage[key] for key in USAGE_FIELDS if key in usage}


def format_usage(usage: dict) -> str:
    input_tokens = usage.get("input_tokens") or 0
    cached_tokens = usage.get("cached_input_tokens") or 0
    cache_ratio = cached_tokens / input_tokens if input_tokens else 0
    return (
        f"input_tokens={input_tokens}, "
        f"cached_input_tokens={cached_tokens}, "
        f"cache_ratio={cache_ratio:.1%}, "
        f"output_tokens={usage.get('output_tokens', 0)}, "
        f"total_tokens={usage.get('total_tokens', 0)}"
    )


def hint_checkpoint_record(index: str, hint_obj: dict, usage: dict | None = None) -> dict:
    record = {
        "index": index,
        "status": "ok",
        "concise_hint": hint_obj["concise_hint"],
        "detailed_method_hint": json.dumps(
            hint_obj["detailed_method_hint"],
            ensure_ascii=False,
        ),
        "detailed_scaffold_hint": json.dumps(
            hint_obj["detailed_scaffold_hint"],
            ensure_ascii=False,
        ),
        "model": MODEL,
        "hint_valid": True,
    }
    if usage:
        record.update(usage_record_fields(usage))
    return record


# =========================
# API CALL
# =========================


def call_openai_responses(client: OpenAI, question: str, solution: str) -> dict:
    system_prompt, user_input = build_prompt_parts(question, solution)
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(**responses_body(user_input, system_prompt))

            content = response.output_text or ""
            obj = safe_json_loads(content)
            hint_obj = normalize_hint_obj(obj)
            usage = extract_usage(response)
            if usage:
                print(f"Usage: {format_usage(usage)}")
                hint_obj["_usage"] = usage
            return hint_obj

        except Exception as e:
            last_error = repr(e)
            wait = min(2 ** attempt, 30)
            time.sleep(wait)

    raise RuntimeError(f"Failed after retries: {last_error}")


def extract_responses_output_text(body: dict) -> str:
    if body.get("output_text"):
        return str(body["output_text"])

    parts = []
    for output in body.get("output", []) or []:
        for content in output.get("content", []) or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if text:
                    parts.append(str(text))
    return "\n".join(parts)


# =========================
# CHECKPOINT
# =========================


def load_done_indices(path: str) -> set:
    done = set()
    if not Path(path).exists():
        return done

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if (
                    obj.get("status") == "ok"
                    and obj.get("concise_hint")
                    and obj.get("detailed_method_hint")
                    and obj.get("detailed_scaffold_hint")
                ):
                    done.add(obj["index"])
            except Exception:
                continue

    return done


def append_jsonl(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def checkpoint_to_parquet():
    records = []
    if Path(CHECKPOINT_PATH).exists():
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if (
                    obj.get("status") == "ok"
                    and obj.get("concise_hint")
                    and obj.get("detailed_method_hint")
                    and obj.get("detailed_scaffold_hint")
                ):
                    records.append(obj)

    if not records:
        raise RuntimeError("No successful hint records found in checkpoint.")

    out_df = pd.DataFrame(records)
    out_df = out_df.drop_duplicates("index", keep="last")
    optional_cols = [field for field in USAGE_FIELDS if field in out_df.columns]
    out_df = out_df[
        [
            "index",
            "concise_hint",
            "detailed_method_hint",
            "detailed_scaffold_hint",
            "model",
            "hint_valid",
        ]
        + optional_cols
    ]
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


# =========================
# BATCH API
# =========================


def make_batch_request(row) -> dict:
    index = str(row["index"])
    _, user_input = build_prompt_parts(row["question"], row["solution"])
    return {
        "custom_id": index,
        "method": "POST",
        "url": "/v1/responses",
        "body": responses_body(user_input),
    }


def write_batch_input(rows: list, batch_no: int) -> Path:
    Path(BATCH_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(BATCH_DIR) / f"batch_input_{int(time.time())}_{batch_no:04d}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_batch_request(row), ensure_ascii=False) + "\n")
    return path


def create_batch(client: OpenAI, input_path: Path):
    with open(input_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    return client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source": "create_hint.py", "model": MODEL},
    )


def wait_for_batch(client: OpenAI, batch_id: str, poll_interval: int):
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        if counts:
            print(
                f"Batch {batch.id}: {batch.status} "
                f"({counts.completed}/{counts.total} completed, {counts.failed} failed)"
            )
        else:
            print(f"Batch {batch.id}: {batch.status}")

        if batch.status in terminal_statuses:
            return batch
        time.sleep(poll_interval)


def format_batch_status(batch) -> str:
    counts = getattr(batch, "request_counts", None)
    if counts:
        return (
            f"Batch {batch.id}: {batch.status} "
            f"({counts.completed}/{counts.total} completed, {counts.failed} failed)"
        )
    return f"Batch {batch.id}: {batch.status}"


def wait_for_active_batches(client: OpenAI, active: dict[str, object], poll_interval: int) -> list:
    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    finished = []

    while not finished:
        for batch_id in list(active):
            batch = client.batches.retrieve(batch_id)
            print(format_batch_status(batch))
            if batch.status in terminal_statuses:
                finished.append(batch)
                del active[batch_id]

        if not finished and active:
            time.sleep(poll_interval)

    return finished


def download_file(client: OpenAI, file_id: str, output_path: Path):
    response = client.files.content(file_id)
    response.write_to_file(output_path)
    return output_path


def parse_batch_output(output_path: Path):
    usage_totals = {field: 0 for field in USAGE_FIELDS}
    usage_count = 0

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            index = str(result.get("custom_id", ""))
            error = result.get("error")
            response = result.get("response") or {}

            if error:
                append_jsonl(
                    CHECKPOINT_PATH,
                    {
                        "index": index,
                        "status": "error",
                        "error": json.dumps(error, ensure_ascii=False),
                        "model": MODEL,
                    },
                )
                continue

            if response.get("status_code") != 200:
                append_jsonl(
                    CHECKPOINT_PATH,
                    {
                        "index": index,
                        "status": "error",
                        "error": json.dumps(response, ensure_ascii=False),
                        "model": MODEL,
                    },
                )
                continue

            try:
                body = response.get("body") or {}
                usage = extract_usage(body)
                if usage:
                    usage_count += 1
                    for field in USAGE_FIELDS:
                        usage_totals[field] += usage.get(field, 0)

                content = extract_responses_output_text(body)
                hint_obj = normalize_hint_obj(safe_json_loads(content))
                append_jsonl(
                    CHECKPOINT_PATH,
                    hint_checkpoint_record(index, hint_obj, usage),
                )
            except Exception as e:
                append_jsonl(
                    CHECKPOINT_PATH,
                    {
                        "index": index,
                        "status": "error",
                        "error": repr(e),
                        "model": MODEL,
                    },
                )

    if usage_count:
        print(
            f"Parsed usage for {usage_count} responses: "
            f"{format_usage(usage_totals)}"
        )


def parse_batch_error_file(client: OpenAI, file_id: str, batch_id: str):
    error_path = Path(BATCH_DIR) / f"{batch_id}_errors.jsonl"
    download_file(client, file_id, error_path)
    with open(error_path, "r", encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            append_jsonl(
                CHECKPOINT_PATH,
                {
                    "index": str(result.get("custom_id", "")),
                    "status": "error",
                    "error": json.dumps(result.get("error") or result, ensure_ascii=False),
                    "model": MODEL,
                },
            )


def process_completed_batch(client: OpenAI, batch):
    Path(BATCH_DIR).mkdir(parents=True, exist_ok=True)

    if batch.output_file_id:
        output_path = Path(BATCH_DIR) / f"{batch.id}_output.jsonl"
        download_file(client, batch.output_file_id, output_path)
        parse_batch_output(output_path)

    if batch.error_file_id:
        parse_batch_error_file(client, batch.error_file_id, batch.id)


def chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# =========================
# MAIN
# =========================


def load_pending_rows(limit: int | None = None):
    df = pd.read_parquet(INPUT_PATH)

    required_cols = {"index", "question", "solution"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    done = load_done_indices(CHECKPOINT_PATH)
    print(f"Already done: {len(done)}")

    rows = []
    for _, row in df.iterrows():
        index = str(row["index"])
        if index in done:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break

    print(f"Pending: {len(rows)}")
    return rows


def run_sync(client: OpenAI, limit: int | None = None):
    rows = load_pending_rows(limit=limit)
    for row in rows:
        index = str(row["index"])
        try:
            hint_obj = call_openai_responses(client, row["question"], row["solution"])
            append_jsonl(
                CHECKPOINT_PATH,
                hint_checkpoint_record(index, hint_obj, hint_obj.get("_usage")),
            )
        except Exception as e:
            append_jsonl(
                CHECKPOINT_PATH,
                {
                    "index": index,
                    "status": "error",
                    "error": repr(e),
                    "model": MODEL,
                },
            )
    checkpoint_to_parquet()


def run_batch(
    client: OpenAI,
    batch_size: int,
    wait: bool,
    poll_interval: int,
    limit: int | None = None,
    max_active_batches: int = MAX_ACTIVE_BATCHES,
):
    rows = load_pending_rows(limit=limit)
    if not rows:
        checkpoint_to_parquet()
        return

    submitted = []
    active = {}
    failed_batches = []

    for batch_no, batch_rows in enumerate(chunked(rows, batch_size), start=1):
        while wait and max_active_batches > 0 and len(active) >= max_active_batches:
            for completed in wait_for_active_batches(client, active, poll_interval):
                process_completed_batch(client, completed)
                if completed.status != "completed":
                    failed_batches.append((completed.id, completed.status))

        input_path = write_batch_input(batch_rows, batch_no)
        batch = create_batch(client, input_path)
        submitted.append(batch.id)
        print(f"Submitted {batch.id} with {len(batch_rows)} requests from {input_path}")

        if wait:
            active[batch.id] = batch

    if wait:
        while active:
            for completed in wait_for_active_batches(client, active, poll_interval):
                process_completed_batch(client, completed)
                if completed.status != "completed":
                    failed_batches.append((completed.id, completed.status))

        checkpoint_to_parquet()
        if failed_batches:
            details = ", ".join(f"{batch_id}:{status}" for batch_id, status in failed_batches)
            raise RuntimeError(f"Some batches did not complete: {details}")
    else:
        print("Submitted batch IDs:")
        for batch_id in submitted:
            print(batch_id)


def retrieve_batch(client: OpenAI, batch_id: str):
    batch = client.batches.retrieve(batch_id)
    print(f"Batch {batch.id}: {batch.status}")
    if batch.status != "completed":
        print("Batch is not completed yet; no output parsed.")
        return
    process_completed_batch(client, batch)
    checkpoint_to_parquet()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "sync", "retrieve"], default="batch")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--poll-interval", type=int, default=BATCH_POLL_INTERVAL)
    parser.add_argument("--max-active-batches", type=int, default=MAX_ACTIVE_BATCHES)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--batch-id")
    args = parser.parse_args()

    client = OpenAI()

    if args.mode == "sync":
        run_sync(client, limit=args.limit)
    elif args.mode == "retrieve":
        if not args.batch_id:
            raise ValueError("--batch-id is required for --mode retrieve")
        retrieve_batch(client, args.batch_id)
    else:
        run_batch(
            client=client,
            batch_size=args.batch_size,
            wait=not args.no_wait,
            poll_interval=args.poll_interval,
            limit=args.limit,
            max_active_batches=args.max_active_batches,
        )


if __name__ == "__main__":
    main()
