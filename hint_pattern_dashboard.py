#!/usr/bin/env python
"""Build a static dashboard for comparing hint-condition outcome patterns."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REASON_CAN_NOT_EXTRACT = "can_not_extract"

DEFAULT_RUN_DIRS = {
    "nohint": "tn01_qwen3_1.7b_base_omni_gt2000_nohint_pv1",
    "concise": "tn01_qwen3_1.7b_base_omni_gt2000_concise_pv1",
    "detailmethod": "tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1",
    "detailscaffold": "tn01_qwen3_1.7b_base_omni_gt2000_detailscaffold_pv1",
}

RUN_LABELS = {
    "nohint": "N: nohint",
    "concise": "C: concise",
    "detailmethod": "M: detailmethod",
    "detailscaffold": "S: detailscaffold",
}

RUN_ABBR = {
    "nohint": "N",
    "concise": "C",
    "detailmethod": "M",
    "detailscaffold": "S",
}

DEFAULT_RUN_ORDER = ["nohint", "concise", "detailmethod", "detailscaffold"]
DEFAULT_RUNS_ROOT = Path("experiments/TN01_base_inference/runs")
DEFAULT_OUTPUT_DIR = Path("experiments/TN01_base_inference/derived")
DEFAULT_MAX_ANSWER_CHARS = 1000

PATTERNS_FILENAME = "qwen3_1.7b_omni_gt2000_hint_patterns.jsonl"
SUMMARY_FILENAME = "qwen3_1.7b_omni_gt2000_hint_pattern_summary.json"
DASHBOARD_FILENAME = "qwen3_1.7b_omni_gt2000_hint_pattern_dashboard.html"


@dataclass(frozen=True)
class RunSpec:
    name: str
    label: str
    abbr: str
    path: Path


@dataclass(frozen=True)
class Status:
    numeric: int | None
    ui: str


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc


def prediction_status(row: dict[str, Any] | None) -> Status:
    if row is None:
        return Status(numeric=None, ui="M")
    if row.get("reason") == REASON_CAN_NOT_EXTRACT:
        return Status(numeric=-1, ui="X")
    return Status(numeric=1, ui="1") if row.get("is_correct") == 1 else Status(numeric=0, ui="0")


def parse_run_order(value: str) -> list[str]:
    names = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not names:
        raise ValueError("Run order must contain at least one run name.")

    unknown = [name for name in names if name not in DEFAULT_RUN_DIRS]
    if unknown:
        valid = ", ".join(DEFAULT_RUN_DIRS)
        raise ValueError(f"Unknown run name(s): {', '.join(unknown)}. Valid names: {valid}")

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate run name(s): {', '.join(duplicates)}")

    return names


def resolve_run_specs(runs_root: Path, run_order: list[str]) -> list[RunSpec]:
    return [
        RunSpec(
            name=name,
            label=RUN_LABELS[name],
            abbr=RUN_ABBR[name],
            path=runs_root / DEFAULT_RUN_DIRS[name],
        )
        for name in run_order
    ]


def load_predictions(run_path: Path) -> dict[str, dict[str, Any]]:
    prediction_path = run_path / "prediction.jsonl"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(prediction_path):
        qid = str(row.get("question_idx", ""))
        if not qid:
            continue
        rows[qid] = row
    return rows


def load_generations(run_path: Path, include_output: bool) -> dict[str, dict[str, Any]]:
    generation_path = run_path / "generations.jsonl"
    if not generation_path.exists():
        return {}

    keep_fields = {
        "run_id",
        "question_idx",
        "question",
        "hint",
        "label",
        "output_token_length",
        "finish_reason",
        "model_output",
    }

    rows: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(generation_path):
        qid = str(row.get("question_idx", ""))
        if not qid:
            continue
        rows[qid] = {key: row.get(key) for key in keep_fields if key in row}
    return rows


def first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return ""


def truncate_text(text: str, max_chars: int | None) -> tuple[str, bool, int]:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text, False, len(text)
    return text[:max_chars].rstrip() + "\n\n[truncated]", True, len(text)


def build_dashboard_records(
    run_specs: list[RunSpec],
    predictions_by_run: dict[str, dict[str, dict[str, Any]]],
    generations_by_run: dict[str, dict[str, dict[str, Any]]] | None = None,
    include_output: bool = False,
    max_answer_chars: int | None = DEFAULT_MAX_ANSWER_CHARS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generations_by_run = generations_by_run or {}
    all_qids = sorted(
        {
            qid
            for rows in list(predictions_by_run.values()) + list(generations_by_run.values())
            for qid in rows
        }
    )

    pattern_counter: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    run_status_counts = {
        spec.name: {"1": 0, "0": 0, "X": 0, "M": 0}
        for spec in run_specs
    }

    for qid in all_qids:
        pattern_parts: list[str] = []
        statuses: dict[str, int | None] = {}
        ui_statuses: dict[str, str] = {}
        run_details: dict[str, dict[str, Any]] = {}
        hints: dict[str, str] = {}
        question = ""
        label = ""

        for spec in run_specs:
            pred = predictions_by_run.get(spec.name, {}).get(qid)
            generation = generations_by_run.get(spec.name, {}).get(qid, {})
            status = prediction_status(pred)
            pattern_parts.append(status.ui)
            statuses[spec.name] = status.numeric
            ui_statuses[spec.name] = status.ui
            run_status_counts[spec.name][status.ui] += 1

            label = first_text(label, pred.get("label") if pred else None, generation.get("label"))
            question = first_text(question, generation.get("question"))
            hint = first_text(generation.get("hint"))
            if hint:
                hints[spec.name] = hint

            raw_extracted_answer = first_text(pred.get("extracted_answer") if pred else None)
            extracted_answer, answer_truncated, answer_full_chars = truncate_text(
                raw_extracted_answer,
                max_answer_chars,
            )

            detail = {
                "run_id": first_text(pred.get("run_id") if pred else None, generation.get("run_id")),
                "status": status.numeric,
                "ui_status": status.ui,
                "extracted_answer": extracted_answer,
                "extracted_answer_truncated": answer_truncated,
                "extracted_answer_full_chars": answer_full_chars,
                "reason": first_text(pred.get("reason") if pred else None),
                "output_token_length": (pred or generation).get("output_token_length"),
                "finish_reason": first_text((pred or generation).get("finish_reason")),
                "last_box_source": first_text(pred.get("last_box_source") if pred else None),
                "think_type": first_text(pred.get("think_type") if pred else None),
                "missing": pred is None,
            }
            if "model_output" in generation:
                detail["model_output"] = generation.get("model_output")
            run_details[spec.name] = detail

        pattern = "".join(pattern_parts)
        pattern_counter[pattern] += 1
        rows.append(
            {
                "question_idx": qid,
                "pattern": pattern,
                "statuses": statuses,
                "ui_statuses": ui_statuses,
                "label": label,
                "question": question,
                "hints": hints,
                "runs": run_details,
            }
        )

    total = len(rows)
    pattern_summary = [
        {
            "pattern": pattern,
            "count": count,
            "percent": (count / total) if total else 0.0,
            "x_count": pattern.count("X"),
            "has_can_not_extract": "X" in pattern,
            "statuses": {spec.name: pattern[idx] for idx, spec in enumerate(run_specs)},
        }
        for pattern, count in pattern_counter.most_common()
    ]

    pattern_question_total = sum(pattern_counter.values())
    reference_total_questions = len(predictions_by_run.get(run_specs[0].name, {})) if run_specs else total
    per_run_question_counts = {
        spec.name: len(predictions_by_run.get(spec.name, {}))
        for spec in run_specs
    }
    all_runs_same_question_count = all(
        count == reference_total_questions for count in per_run_question_counts.values()
    )
    can_not_extract_questions = sum(1 for row in rows if "X" in row["pattern"])
    all_can_not_extract_questions = pattern_counter.get("X" * len(run_specs), 0)
    patterns_with_can_not_extract = sum(1 for pattern in pattern_counter if "X" in pattern)

    per_run = {}
    for spec in run_specs:
        prediction_rows = predictions_by_run.get(spec.name, {})
        correct = sum(1 for row in prediction_rows.values() if prediction_status(row).numeric == 1)
        can_not_extract = sum(1 for row in prediction_rows.values() if prediction_status(row).numeric == -1)
        total_predictions = len(prediction_rows)
        per_run[spec.name] = {
            "label": spec.label,
            "abbr": spec.abbr,
            "path": str(spec.path),
            "total_predictions": total_predictions,
            "correct": correct,
            "accuracy": (correct / total_predictions) if total_predictions else 0.0,
            "can_not_extract": can_not_extract,
            "status_counts": run_status_counts[spec.name],
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": total,
        "reference_total_questions": reference_total_questions,
        "per_run_question_counts": per_run_question_counts,
        "total_pattern_groups": len(pattern_counter),
        "pattern_question_total": pattern_question_total,
        "pattern_total_matches_reference": pattern_question_total == reference_total_questions,
        "pattern_total_matches_joined_questions": pattern_question_total == total,
        "all_runs_same_question_count": all_runs_same_question_count,
        "can_not_extract_questions": can_not_extract_questions,
        "all_can_not_extract_questions": all_can_not_extract_questions,
        "patterns_with_can_not_extract": patterns_with_can_not_extract,
        "run_order": [spec.name for spec in run_specs],
        "run_labels": {spec.name: spec.label for spec in run_specs},
        "run_abbr": {spec.name: spec.abbr for spec in run_specs},
        "per_run": per_run,
        "pattern_counts": pattern_summary,
        "include_output": include_output,
        "max_answer_chars": max_answer_chars,
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def render_dashboard_html(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    payload = json.dumps(
        {"summary": summary, "rows": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    return DASHBOARD_TEMPLATE.replace("__DASHBOARD_DATA__", payload)


def write_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "patterns": output_dir / PATTERNS_FILENAME,
        "summary": output_dir / SUMMARY_FILENAME,
        "dashboard": output_dir / DASHBOARD_FILENAME,
    }
    write_jsonl(paths["patterns"], rows)
    write_json(paths["summary"], summary)
    paths["dashboard"].write_text(render_dashboard_html(rows, summary), encoding="utf-8")
    return paths


def build_from_files(
    *,
    runs_root: Path,
    output_dir: Path,
    run_order: list[str],
    include_output: bool,
    max_answer_chars: int | None,
) -> dict[str, Path]:
    run_specs = resolve_run_specs(runs_root, run_order)
    predictions_by_run = {spec.name: load_predictions(spec.path) for spec in run_specs}
    generations_by_run = {spec.name: load_generations(spec.path, include_output) for spec in run_specs}
    rows, summary = build_dashboard_records(
        run_specs,
        predictions_by_run,
        generations_by_run,
        include_output,
        max_answer_chars,
    )
    return write_outputs(output_dir=output_dir, rows=rows, summary=summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static hint-pattern dashboard.")
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT), help="Directory containing run folders.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for derived outputs.")
    parser.add_argument(
        "--run-order",
        default=",".join(DEFAULT_RUN_ORDER),
        help="Comma-separated run names. Default: nohint,concise,detailmethod,detailscaffold",
    )
    parser.add_argument(
        "--include-output",
        action="store_true",
        help="Embed raw model_output in the HTML detail panel. This makes the file much larger.",
    )
    parser.add_argument(
        "--max-answer-chars",
        type=int,
        default=DEFAULT_MAX_ANSWER_CHARS,
        help="Max chars to embed for extracted_answer previews. Use 0 to keep full extracted answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_order = parse_run_order(args.run_order)
    paths = build_from_files(
        runs_root=Path(args.runs_root),
        output_dir=Path(args.output_dir),
        run_order=run_order,
        include_output=args.include_output,
        max_answer_chars=args.max_answer_chars,
    )
    print(f"Patterns JSONL: {paths['patterns']}")
    print(f"Summary JSON: {paths['summary']}")
    print(f"Dashboard HTML: {paths['dashboard']}")


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3-1.7b Hint Pattern Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #65717e;
      --line: #d9dee5;
      --blue: #1f6feb;
      --green: #1a7f37;
      --red: #b42318;
      --amber: #a15c00;
      --shadow: 0 1px 3px rgba(16, 24, 40, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 14px;
      line-height: 1.45;
    }
    header {
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 18px 24px 14px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 {
      margin: 0 0 10px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 180px 160px;
      gap: 10px;
      align-items: center;
    }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 9px 10px;
      font: inherit;
      min-width: 0;
    }
    button {
      cursor: pointer;
      background: #f2f5f9;
    }
    button:hover { border-color: #9aa6b2; }
    main {
      padding: 18px 24px 28px;
      display: grid;
      gap: 16px;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      min-height: 84px;
    }
    .card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .card .value {
      margin-top: 6px;
      font-size: 24px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(520px, 1.4fr);
      gap: 16px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 15px;
    }
    .table-wrap { overflow: auto; max-height: 640px; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid #edf0f4;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f7f9fc;
      color: #384452;
      font-size: 12px;
      text-align: left;
      text-transform: uppercase;
    }
    tr.clickable { cursor: pointer; }
    tr.clickable:hover td { background: #f4f8ff; }
    tr.selected td { background: #eaf2ff; }
    .pattern {
      font-family: Consolas, "Courier New", monospace;
      font-weight: 700;
      font-size: 15px;
      white-space: nowrap;
    }
    .status {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      height: 22px;
      border-radius: 4px;
      color: #fff;
      font-weight: 700;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
    }
    .s1 { background: var(--green); }
    .s0 { background: var(--red); }
    .sX { background: var(--amber); }
    .sM { background: #6b7280; }
    .muted { color: var(--muted); }
    .question-cell {
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }
    .detail {
      padding: 14px;
      display: grid;
      gap: 12px;
    }
    .detail h3 {
      margin: 0;
      font-size: 16px;
    }
    .detail-block {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 280px;
      overflow: auto;
    }
    .run-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
    }
    .run-box {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fff;
    }
    .run-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      margin-bottom: 8px;
    }
    .kv {
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 5px 8px;
      font-size: 13px;
    }
    .notice {
      padding: 9px 14px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      background: #fbfcfe;
    }
    @media (max-width: 920px) {
      .toolbar, .grid { grid-template-columns: 1fr; }
      header, main { padding-left: 14px; padding-right: 14px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Qwen3-1.7b Hint Pattern Dashboard</h1>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search question, id, label, answer, hint">
      <select id="patternFilter"></select>
      <button id="clearFilters" type="button">Clear filters</button>
    </div>
  </header>
  <main>
    <div id="cards" class="cards"></div>
    <div class="grid">
      <section>
        <h2>Pattern Groups</h2>
        <div class="table-wrap">
          <table>
            <thead id="patternHead"></thead>
            <tbody id="patternBody"></tbody>
          </table>
        </div>
      </section>
      <section>
        <h2 id="questionTitle">Questions</h2>
        <div class="table-wrap">
          <table>
            <thead id="questionHead"></thead>
            <tbody id="questionBody"></tbody>
          </table>
        </div>
        <div id="questionNotice" class="notice"></div>
      </section>
    </div>
    <section>
      <h2>Question Detail</h2>
      <div id="detail" class="detail muted">Select a question to inspect it.</div>
    </section>
  </main>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById("dashboard-data").textContent);
    const summary = data.summary;
    const rows = data.rows;
    const runOrder = summary.run_order;
    let selectedPattern = "";
    let selectedQuestion = "";

    const fmtInt = new Intl.NumberFormat();
    const fmtPct = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }

    function statusBadge(status) {
      return `<span class="status s${esc(status)}">${esc(status)}</span>`;
    }

    function searchable(row) {
      const answers = runOrder.map(name => row.runs[name]?.extracted_answer || "").join(" ");
      const hints = Object.values(row.hints || {}).join(" ");
      return `${row.question_idx} ${row.pattern} ${row.label} ${row.question} ${answers} ${hints}`.toLowerCase();
    }

    function filteredRows() {
      const query = document.getElementById("search").value.trim().toLowerCase();
      return rows.filter(row => {
        if (selectedPattern && row.pattern !== selectedPattern) return false;
        if (query && !searchable(row).includes(query)) return false;
        return true;
      });
    }

    function renderCards() {
      const cards = [
        ["Total questions", fmtInt.format(summary.total_questions)],
        ["Pattern counted", fmtInt.format(summary.pattern_question_total ?? 0)],
        ["Reference total", fmtInt.format(summary.reference_total_questions ?? summary.total_questions)],
        ["Total check", summary.pattern_total_matches_reference ? "OK" : "Mismatch"],
        ["Pattern groups", fmtInt.format(summary.total_pattern_groups ?? summary.pattern_counts.length)],
        ["Questions with X", fmtInt.format(summary.can_not_extract_questions ?? 0)],
        ["Patterns with X", fmtInt.format(summary.patterns_with_can_not_extract ?? 0)],
        ["XXXX questions", fmtInt.format(summary.all_can_not_extract_questions ?? 0)],
        ["Top pattern", summary.pattern_counts[0]?.pattern || ""],
        ["Raw outputs embedded", summary.include_output ? "Yes" : "No"]
      ];
      for (const name of runOrder) {
        const run = summary.per_run[name];
        cards.push([run.label, `${fmtPct.format(run.accuracy)} correct, ${fmtInt.format(run.can_not_extract)} X`]);
      }
      document.getElementById("cards").innerHTML = cards.map(([label, value]) => `
        <div class="card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>
      `).join("");
    }

    function renderPatternFilter() {
      const select = document.getElementById("patternFilter");
      select.innerHTML = `<option value="">All patterns</option>` + summary.pattern_counts.map(item =>
        `<option value="${esc(item.pattern)}">${esc(item.pattern)} (${fmtInt.format(item.count)})</option>`
      ).join("");
      select.value = selectedPattern;
    }

    function renderPatternTable() {
      document.getElementById("patternHead").innerHTML = `
        <tr>
          <th style="width: 86px;">Pattern</th>
          <th style="width: 84px;">Count</th>
          <th style="width: 84px;">Percent</th>
          <th style="width: 58px;">X</th>
          ${runOrder.map(name => `<th style="width: 52px;">${esc(summary.run_abbr[name])}</th>`).join("")}
        </tr>`;
      document.getElementById("patternBody").innerHTML = summary.pattern_counts.map(item => `
        <tr class="clickable ${selectedPattern === item.pattern ? "selected" : ""}" data-pattern="${esc(item.pattern)}">
          <td class="pattern">${esc(item.pattern)}</td>
          <td>${fmtInt.format(item.count)}</td>
          <td>${fmtPct.format(item.percent)}</td>
          <td>${fmtInt.format(item.x_count || 0)}</td>
          ${runOrder.map(name => `<td>${statusBadge(item.statuses[name])}</td>`).join("")}
        </tr>
      `).join("");
      document.querySelectorAll("#patternBody tr").forEach(tr => {
        tr.addEventListener("click", () => {
          selectedPattern = tr.dataset.pattern === selectedPattern ? "" : tr.dataset.pattern;
          document.getElementById("patternFilter").value = selectedPattern;
          renderAll();
        });
      });
    }

    function renderQuestionTable() {
      const visible = filteredRows();
      const display = visible.slice(0, 500);
      document.getElementById("questionTitle").textContent =
        selectedPattern ? `Questions: ${selectedPattern}` : "Questions";
      document.getElementById("questionHead").innerHTML = `
        <tr>
          <th style="width: 150px;">Question ID</th>
          <th style="width: 82px;">Pattern</th>
          <th>Question</th>
          <th style="width: 160px;">Label</th>
          ${runOrder.map(name => `<th style="width: 52px;">${esc(summary.run_abbr[name])}</th>`).join("")}
        </tr>`;
      document.getElementById("questionBody").innerHTML = display.map(row => `
        <tr class="clickable ${selectedQuestion === row.question_idx ? "selected" : ""}" data-qid="${esc(row.question_idx)}">
          <td>${esc(row.question_idx)}</td>
          <td class="pattern">${esc(row.pattern)}</td>
          <td><div class="question-cell">${esc(row.question || "(question text not found)")}</div></td>
          <td>${esc(row.label)}</td>
          ${runOrder.map(name => `<td>${statusBadge(row.ui_statuses[name])}</td>`).join("")}
        </tr>
      `).join("");
      const limited = visible.length > display.length ? ` Showing first ${fmtInt.format(display.length)}.` : "";
      document.getElementById("questionNotice").textContent =
        `${fmtInt.format(visible.length)} matching question(s).${limited}`;
      document.querySelectorAll("#questionBody tr").forEach(tr => {
        tr.addEventListener("click", () => {
          selectedQuestion = tr.dataset.qid;
          renderQuestionTable();
          renderDetail();
        });
      });
    }

    function renderDetail() {
      const row = rows.find(item => item.question_idx === selectedQuestion);
      const target = document.getElementById("detail");
      if (!row) {
        target.className = "detail muted";
        target.textContent = "Select a question to inspect it.";
        return;
      }
      target.className = "detail";
      const hints = Object.entries(row.hints || {}).map(([name, hint]) =>
        `<div class="detail-block"><strong>${esc(summary.run_labels[name])}</strong>\n${esc(hint)}</div>`
      ).join("");
      const runBoxes = runOrder.map(name => {
        const run = row.runs[name] || {};
        const answerNote = run.extracted_answer_truncated
          ? ` <span class="muted">(preview, ${fmtInt.format(run.extracted_answer_full_chars)} chars total)</span>` : "";
        const output = run.model_output
          ? `<div class="muted" style="margin-top:8px;">Full solution</div><div class="detail-block">${esc(run.model_output)}</div>` : "";
        return `<div class="run-box">
          <div class="run-title">${statusBadge(run.ui_status || "M")} ${esc(summary.run_labels[name])}</div>
          <div class="kv">
            <div class="muted">Answer</div><div>${esc(run.extracted_answer)}${answerNote}</div>
            <div class="muted">Reason</div><div>${esc(run.reason || (run.missing ? "missing" : ""))}</div>
            <div class="muted">Tokens</div><div>${esc(run.output_token_length ?? "")}</div>
            <div class="muted">Finish</div><div>${esc(run.finish_reason)}</div>
            <div class="muted">Box</div><div>${esc(run.last_box_source)}</div>
            <div class="muted">Think</div><div>${esc(run.think_type)}</div>
          </div>
          ${output}
        </div>`;
      }).join("");
      target.innerHTML = `
        <h3>${esc(row.question_idx)} <span class="pattern">${esc(row.pattern)}</span></h3>
        <div><strong>Label:</strong> ${esc(row.label)}</div>
        <div class="detail-block">${esc(row.question || "(question text not found)")}</div>
        ${hints ? `<h3>Hints</h3>${hints}` : ""}
        <h3>Runs</h3>
        <div class="run-grid">${runBoxes}</div>
      `;
    }

    function renderAll() {
      renderCards();
      renderPatternFilter();
      renderPatternTable();
      renderQuestionTable();
      renderDetail();
    }

    document.getElementById("search").addEventListener("input", () => {
      selectedQuestion = "";
      renderQuestionTable();
      renderDetail();
    });
    document.getElementById("patternFilter").addEventListener("change", event => {
      selectedPattern = event.target.value;
      selectedQuestion = "";
      renderAll();
    });
    document.getElementById("clearFilters").addEventListener("click", () => {
      selectedPattern = "";
      selectedQuestion = "";
      document.getElementById("search").value = "";
      renderAll();
    });

    renderAll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
