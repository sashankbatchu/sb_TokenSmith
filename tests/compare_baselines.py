#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BASELINES: List[Dict[str, Any]] = [
    {
        "name": "recursive_sections",
        "label": "Recursive Sections",
        "overrides": {
            "chunk_mode": "recursive_sections",
            "enable_metadata_scoring": False,
            "page_to_chunk_map_path": "index/sections/textbook_index_page_to_chunk_map.json",
        },
    },
    {
        "name": "semantic_sections",
        "label": "Semantic Sections",
        "overrides": {
            "chunk_mode": "semantic_sections",
            "enable_metadata_scoring": False,
            "page_to_chunk_map_path": "index/semantic_sections/textbook_index_page_to_chunk_map.json",
        },
    },
    {
        "name": "semantic_metadata",
        "label": "Semantic + Metadata Reranking",
        "overrides": {
            "chunk_mode": "semantic_sections",
            "enable_metadata_scoring": True,
            "page_to_chunk_map_path": "index/semantic_sections/textbook_index_page_to_chunk_map.json",
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TokenSmith benchmark baselines.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--benchmarks", default="tests/benchmarks.yaml", help="Path to benchmarks YAML.")
    parser.add_argument("--output-dir", default="tests/results/baseline_compare", help="Directory for comparison outputs.")
    parser.add_argument("--benchmark-ids", default="", help="Comma-separated benchmark IDs to run.")
    parser.add_argument("--only-with-intent", action="store_true", help="Run only benchmarks that explicitly define an intent field.")
    parser.add_argument("--intent-values", default="", help="Comma-separated intent values to run, for example: definition,comparison")
    parser.add_argument("--type-values", default="", help="Comma-separated benchmark types to run, for example: factual,multi-part")
    parser.add_argument("--metrics", nargs="*", default=None, help="Metrics to use. Example: semantic keyword nli")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def normalize_base_config(cfg: Dict[str, Any], metric_names: List[str] | None) -> Dict[str, Any]:
    normalized = dict(cfg)
    normalized["model_path"] = normalized.get("model_path", normalized.get("gen_model"))
    normalized["retrieval_method"] = normalized.get("retrieval_method", normalized.get("ensemble_method", "rrf"))
    normalized["metrics"] = metric_names if metric_names else normalized.get("metrics", ["all"])
    normalized["output_mode"] = "terminal"
    normalized["threshold_override"] = normalized.get("threshold_override")
    normalized["disable_chunks"] = normalized.get("disable_chunks", False)
    normalized["use_golden_chunks"] = normalized.get("use_golden_chunks", False)
    normalized["index_prefix"] = normalized.get("index_prefix", "textbook_index")
    return normalized


def filter_benchmarks(
    benchmarks: List[Dict[str, Any]],
    selected_ids: str,
    only_with_intent: bool,
    intent_values: str,
    type_values: str,
) -> List[Dict[str, Any]]:
    filtered = benchmarks

    if selected_ids.strip():
        wanted = {item.strip() for item in selected_ids.split(",") if item.strip()}
        filtered = [benchmark for benchmark in filtered if benchmark.get("id") in wanted]

    if only_with_intent:
        filtered = [benchmark for benchmark in filtered if benchmark.get("intent")]

    if intent_values.strip():
        wanted_intents = {item.strip() for item in intent_values.split(",") if item.strip()}
        filtered = [benchmark for benchmark in filtered if benchmark.get("intent") in wanted_intents]

    if type_values.strip():
        wanted_types = {item.strip() for item in type_values.split(",") if item.strip()}
        filtered = [benchmark for benchmark in filtered if benchmark.get("type") in wanted_types]

    return filtered


def count_explicit_intents(benchmarks: List[Dict[str, Any]]) -> int:
    return sum(1 for benchmark in benchmarks if benchmark.get("intent"))


def mean_or_zero(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def derive_primary_intent(question: str) -> str:
    from src.main import detect_query_intent

    scores = detect_query_intent(question or "")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    if top_score <= 0:
        return "other"
    return top_name.replace("_intent", "")


def render_bar(value: float, max_value: float, width: int = 220, color: str = "#3b82f6") -> str:
    safe_value = max(0.0, value)
    safe_max = max(max_value, 1e-9)
    bar_width = int((safe_value / safe_max) * width)
    label = f"{safe_value:.3f}" if safe_value <= 1.0 else f"{safe_value:.1f}"
    return (
        f'<div class="bar-wrap"><div class="bar" style="width:{bar_width}px;background:{color};"></div>'
        f'<span class="bar-label">{label}</span></div>'
    )


def score_color(index: int) -> str:
    palette = ["#2563eb", "#059669", "#d97706"]
    return palette[index % len(palette)]


def run_single_benchmark(
    benchmark: Dict[str, Any],
    baseline: Dict[str, Any],
    base_config: Dict[str, Any],
    scorer: Any,
) -> Dict[str, Any]:
    from tests.test_benchmarks import get_tokensmith_answer

    effective_config = dict(base_config)
    effective_config.update(baseline["overrides"])

    question = benchmark["question"]
    expected_answer = benchmark["expected_answer"]
    keywords = benchmark.get("keywords", [])
    threshold = effective_config.get("threshold_override") or benchmark.get("similarity_threshold") or 0.6
    benchmark_type = benchmark.get("type", "unspecified")
    benchmark_intent = benchmark.get("intent") or derive_primary_intent(question)

    answer, chunks_info, hyde_query = get_tokensmith_answer(
        question=question,
        config=effective_config,
        golden_chunks=benchmark.get("golden_chunks") if effective_config.get("use_golden_chunks") else None,
    )

    scores = scorer.calculate_scores(
        answer,
        expected_answer,
        keywords,
        question=question,
        ideal_retrieved_chunks=benchmark.get("ideal_retrieved_chunks"),
        actual_retrieved_chunks=chunks_info,
    )
    final_score = scores.get("final_score", 0.0)
    passed = final_score >= threshold

    return {
        "benchmark_id": benchmark.get("id", "unknown"),
        "type": benchmark_type,
        "intent": benchmark_intent,
        "question": question,
        "expected_answer": expected_answer,
        "retrieved_answer": answer,
        "threshold": threshold,
        "scores": scores,
        "passed": passed,
        "chunks_info": chunks_info or [],
        "hyde_query": hyde_query,
        "baseline": baseline["name"],
        "baseline_label": baseline["label"],
        "config_overrides": baseline["overrides"],
    }


def summarize_baseline(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [item["scores"]["final_score"] for item in results]
    passed = sum(1 for item in results if item["passed"])
    summary = {
        "count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": (passed / len(results)) if results else 0.0,
        "avg_final_score": mean_or_zero(scores),
        "median_final_score": statistics.median(scores) if scores else 0.0,
        "min_final_score": min(scores) if scores else 0.0,
        "max_final_score": max(scores) if scores else 0.0,
    }

    metric_values: Dict[str, List[float]] = defaultdict(list)
    for item in results:
        for key, value in item["scores"].items():
            if key.endswith("_similarity"):
                metric_values[key].append(float(value))
    summary["metric_averages"] = {key: mean_or_zero(values) for key, values in sorted(metric_values.items())}
    return summary


def summarize_by_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item["type"]].append(item)
    return {group: summarize_baseline(items) for group, items in sorted(grouped.items())}


def summarize_by_intent(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in results:
        grouped[item["intent"]].append(item)
    return {group: summarize_baseline(items) for group, items in sorted(grouped.items())}


def compute_benchmark_winners(results_by_baseline: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    aligned: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for baseline_name, entries in results_by_baseline.items():
        for entry in entries:
            aligned[entry["benchmark_id"]][baseline_name] = entry

    winners = []
    for benchmark_id, mapping in sorted(aligned.items()):
        ranked = sorted(
            mapping.items(),
            key=lambda item: item[1]["scores"]["final_score"],
            reverse=True,
        )
        top_name, top_entry = ranked[0]
        runner_up_score = ranked[1][1]["scores"]["final_score"] if len(ranked) > 1 else top_entry["scores"]["final_score"]
        winners.append(
            {
                "benchmark_id": benchmark_id,
                "type": top_entry["type"],
                "intent": top_entry["intent"],
                "winner": top_name,
                "winner_label": top_entry["baseline_label"],
                "winner_score": top_entry["scores"]["final_score"],
                "margin": top_entry["scores"]["final_score"] - runner_up_score,
                "scores": {name: entry["scores"]["final_score"] for name, entry in mapping.items()},
            }
        )
    return winners


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_discussion(
    baseline_summaries: Dict[str, Dict[str, Any]],
    type_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    intent_summaries: Dict[str, Dict[str, Dict[str, Any]]],
) -> str:
    ordered = [
        (baseline_name, baseline_summaries[baseline_name])
        for baseline_name in [baseline["name"] for baseline in BASELINES]
    ]
    best_overall_name, best_overall = max(ordered, key=lambda item: item[1]["avg_final_score"])
    recursive = baseline_summaries["recursive_sections"]
    semantic = baseline_summaries["semantic_sections"]
    semantic_meta = baseline_summaries["semantic_metadata"]

    delta_semantic = semantic["avg_final_score"] - recursive["avg_final_score"]
    delta_meta = semantic_meta["avg_final_score"] - semantic["avg_final_score"]
    delta_total = semantic_meta["avg_final_score"] - recursive["avg_final_score"]

    lines = [
        "## Experimental Results Discussion",
        "",
        f"The strongest overall configuration was **{best_overall_name}**, with an average final score of **{best_overall['avg_final_score']:.3f}** and a pass rate of **{best_overall['pass_rate'] * 100:.1f}%**.",
        f"Moving from recursive chunking to semantic chunking changed the average score by **{delta_semantic:+.3f}** and the pass rate by **{(semantic['pass_rate'] - recursive['pass_rate']) * 100:+.1f}** percentage points.",
        f"Adding metadata-aware query-intent reranking on top of semantic chunking changed the average score by **{delta_meta:+.3f}** and the pass rate by **{(semantic_meta['pass_rate'] - semantic['pass_rate']) * 100:+.1f}** percentage points.",
        f"Relative to the original recursive baseline, the full semantic-plus-metadata configuration changed the average score by **{delta_total:+.3f}** and the pass rate by **{(semantic_meta['pass_rate'] - recursive['pass_rate']) * 100:+.1f}** percentage points.",
        "",
        "Question-type breakdown:",
    ]

    all_types = sorted({question_type for summary in type_summaries.values() for question_type in summary.keys()})
    for question_type in all_types:
        best_for_type = None
        best_score = float("-inf")
        for baseline_name, summaries in type_summaries.items():
            current = summaries.get(question_type)
            if current and current["avg_final_score"] > best_score:
                best_score = current["avg_final_score"]
                best_for_type = baseline_name
        if best_for_type is None:
            continue
        lines.append(
            f"- `{question_type}` questions were best handled by **{best_for_type}** with average score **{best_score:.3f}**."
        )

    lines.append("")
    lines.append("These comparisons isolate the impact of chunking and metadata-aware reranking while keeping the rest of the pipeline fixed to the current benchmark configuration.")
    lines.append("")
    lines.append("Intent breakdown:")

    all_intents = sorted({intent for summary in intent_summaries.values() for intent in summary.keys()})
    for intent in all_intents:
        recursive_summary = intent_summaries["recursive_sections"].get(intent)
        semantic_summary = intent_summaries["semantic_sections"].get(intent)
        metadata_summary = intent_summaries["semantic_metadata"].get(intent)
        if not metadata_summary:
            continue
        semantic_delta = (
            metadata_summary["avg_final_score"] - semantic_summary["avg_final_score"]
            if semantic_summary else 0.0
        )
        recursive_delta = (
            metadata_summary["avg_final_score"] - recursive_summary["avg_final_score"]
            if recursive_summary else 0.0
        )
        lines.append(
            f"- `{intent}`: semantic+metadata vs semantic-only **{semantic_delta:+.3f}**, vs recursive **{recursive_delta:+.3f}**."
        )
    return "\n".join(lines)


def generate_markdown_report(
    output_dir: Path,
    baseline_summaries: Dict[str, Dict[str, Any]],
    type_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    intent_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    winners: List[Dict[str, Any]],
) -> None:
    lines = [
        "# TokenSmith Baseline Comparison",
        "",
        build_discussion(baseline_summaries, type_summaries, intent_summaries),
        "",
        "## Overall Metrics",
        "",
        "| Baseline | Avg Score | Median | Pass Rate | Passed | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for baseline in BASELINES:
        summary = baseline_summaries[baseline["name"]]
        lines.append(
            f"| {baseline['label']} | {summary['avg_final_score']:.3f} | {summary['median_final_score']:.3f} | {summary['pass_rate'] * 100:.1f}% | {summary['passed']} | {summary['failed']} |"
        )

    lines.extend(["", "## Per-Type Metrics", ""])
    all_types = sorted({question_type for summary in type_summaries.values() for question_type in summary.keys()})
    for question_type in all_types:
        lines.append(f"### {question_type}")
        lines.append("")
        lines.append("| Baseline | Avg Score | Pass Rate | Count |")
        lines.append("|---|---:|---:|---:|")
        for baseline in BASELINES:
            summary = type_summaries[baseline["name"]].get(question_type)
            if not summary:
                continue
            lines.append(
                f"| {baseline['label']} | {summary['avg_final_score']:.3f} | {summary['pass_rate'] * 100:.1f}% | {summary['count']} |"
            )
        lines.append("")

    lines.extend(["## Per-Intent Metrics", ""])
    all_intents = sorted({intent for summary in intent_summaries.values() for intent in summary.keys()})
    for intent in all_intents:
        lines.append(f"### {intent}")
        lines.append("")
        lines.append("| Baseline | Avg Score | Pass Rate | Count |")
        lines.append("|---|---:|---:|---:|")
        for baseline in BASELINES:
            summary = intent_summaries[baseline["name"]].get(intent)
            if not summary:
                continue
            lines.append(
                f"| {baseline['label']} | {summary['avg_final_score']:.3f} | {summary['pass_rate'] * 100:.1f}% | {summary['count']} |"
            )
        lines.append("")

    lines.extend(["## Per-Benchmark Winners", "", "| Benchmark | Type | Intent | Winner | Margin | Scores |", "|---|---|---|---|---:|---|"])
    for row in winners:
        score_str = ", ".join(f"{name}={score:.3f}" for name, score in sorted(row["scores"].items()))
        lines.append(
            f"| {row['benchmark_id']} | {row['type']} | {row['intent']} | {row['winner_label']} | {row['margin']:.3f} | {score_str} |"
        )

    (output_dir / "baseline_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def generate_html_report(
    output_dir: Path,
    baseline_summaries: Dict[str, Dict[str, Any]],
    type_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    intent_summaries: Dict[str, Dict[str, Dict[str, Any]]],
    winners: List[Dict[str, Any]],
) -> None:
    max_avg = max(summary["avg_final_score"] for summary in baseline_summaries.values()) or 1.0
    discussion_html = build_discussion(baseline_summaries, type_summaries, intent_summaries).replace("\n", "<br>\n")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>TokenSmith Baseline Comparison</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#111827;}table{border-collapse:collapse;width:100%;margin:18px 0;}th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left;}th{background:#f3f4f6;}h1,h2,h3{margin-top:28px;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;} .card{border:1px solid #d1d5db;border-radius:8px;padding:16px;background:#fff;} .bar-wrap{display:flex;align-items:center;gap:10px;margin:6px 0;} .bar{height:14px;border-radius:999px;} .bar-label{font-variant-numeric:tabular-nums;} .small{color:#4b5563;font-size:0.92rem;} </style></head><body>",
        "<h1>TokenSmith Baseline Comparison</h1>",
        f"<p class='small'>Generated on {datetime.now().isoformat(timespec='seconds')}</p>",
        f"<div class='card'>{discussion_html}</div>",
        "<h2>Overall Metrics</h2><div class='grid'>",
    ]

    for index, baseline in enumerate(BASELINES):
        summary = baseline_summaries[baseline["name"]]
        color = score_color(index)
        parts.append(
            "<div class='card'>"
            f"<h3>{baseline['label']}</h3>"
            f"<div>Average final score {render_bar(summary['avg_final_score'], max_avg, color=color)}</div>"
            f"<div>Pass rate {render_bar(summary['pass_rate'], 1.0, color=color)}</div>"
            f"<p>Median score: {summary['median_final_score']:.3f}<br>Passed: {summary['passed']} / {summary['count']}</p>"
            "</div>"
        )
    parts.append("</div>")

    parts.append("<h2>Per-Type Metrics</h2>")
    all_types = sorted({question_type for summary in type_summaries.values() for question_type in summary.keys()})
    for question_type in all_types:
        parts.append(f"<h3>{question_type}</h3><table><tr><th>Baseline</th><th>Avg Score</th><th>Pass Rate</th><th>Count</th></tr>")
        for baseline in BASELINES:
            summary = type_summaries[baseline["name"]].get(question_type)
            if not summary:
                continue
            parts.append(
                f"<tr><td>{baseline['label']}</td><td>{summary['avg_final_score']:.3f}</td><td>{summary['pass_rate'] * 100:.1f}%</td><td>{summary['count']}</td></tr>"
            )
        parts.append("</table>")

    parts.append("<h2>Per-Intent Metrics</h2>")
    all_intents = sorted({intent for summary in intent_summaries.values() for intent in summary.keys()})
    for intent in all_intents:
        parts.append(f"<h3>{intent}</h3><table><tr><th>Baseline</th><th>Avg Score</th><th>Pass Rate</th><th>Count</th></tr>")
        for baseline in BASELINES:
            summary = intent_summaries[baseline["name"]].get(intent)
            if not summary:
                continue
            parts.append(
                f"<tr><td>{baseline['label']}</td><td>{summary['avg_final_score']:.3f}</td><td>{summary['pass_rate'] * 100:.1f}%</td><td>{summary['count']}</td></tr>"
            )
        parts.append("</table>")

    parts.append("<h2>Per-Benchmark Winners</h2><table><tr><th>Benchmark</th><th>Type</th><th>Intent</th><th>Winner</th><th>Margin</th><th>Scores</th></tr>")
    for row in winners:
        score_str = ", ".join(f"{name}={score:.3f}" for name, score in sorted(row["scores"].items()))
        parts.append(
            f"<tr><td>{row['benchmark_id']}</td><td>{row['type']}</td><td>{row['intent']}</td><td>{row['winner_label']}</td><td>{row['margin']:.3f}</td><td>{score_str}</td></tr>"
        )
    parts.append("</table></body></html>")
    (output_dir / "baseline_comparison_report.html").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    from tests.metrics import SimilarityScorer

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = normalize_base_config(load_yaml(args.config), args.metrics)
    benchmark_payload = load_yaml(args.benchmarks)
    all_benchmarks = benchmark_payload["benchmarks"]
    benchmarks = filter_benchmarks(
        all_benchmarks,
        args.benchmark_ids,
        args.only_with_intent,
        args.intent_values,
        args.type_values,
    )
    explicit_intent_count = count_explicit_intents(all_benchmarks)
    if not benchmarks:
        print("No benchmarks matched the requested filters.")
        print(f"Total benchmarks loaded: {len(all_benchmarks)}")
        print(f"Benchmarks with explicit intent: {explicit_intent_count}")
        if args.only_with_intent:
            print("`--only-with-intent` requires benchmarks to define an `intent:` field in tests/benchmarks.yaml.")
        if args.intent_values.strip():
            print(f"No benchmarks matched intent values: {args.intent_values}")
        if args.type_values.strip():
            print(f"No benchmarks matched type values: {args.type_values}")
        print("Example benchmark entry:")
        print("  - id: \"acid_properties\"")
        print("    type: \"factual\"")
        print("    intent: \"definition\"")
        print("    question: \"What are the ACID properties of a transaction?\"")
        return
    scorer = SimilarityScorer(enabled_metrics=base_config["metrics"])

    results_by_baseline: Dict[str, List[Dict[str, Any]]] = {}
    for baseline in BASELINES:
        print(f"\n=== Running baseline: {baseline['label']} ===")
        baseline_results: List[Dict[str, Any]] = []
        for benchmark in benchmarks:
            print(f"- {benchmark.get('id', 'unknown')}: {benchmark['question']}")
            try:
                baseline_results.append(run_single_benchmark(benchmark, baseline, base_config, scorer))
            except Exception as exc:
                baseline_results.append(
                    {
                        "benchmark_id": benchmark.get("id", "unknown"),
                        "type": benchmark.get("type", "unspecified"),
                        "question": benchmark["question"],
                        "threshold": benchmark.get("similarity_threshold", 0.0),
                        "scores": {"final_score": 0.0},
                        "passed": False,
                        "baseline": baseline["name"],
                        "baseline_label": baseline["label"],
                        "error": str(exc),
                    }
                )
                print(f"  ERROR: {exc}")
        results_by_baseline[baseline["name"]] = baseline_results

    baseline_summaries = {
        baseline["name"]: summarize_baseline(results_by_baseline[baseline["name"]])
        for baseline in BASELINES
    }
    type_summaries = {
        baseline["name"]: summarize_by_type(results_by_baseline[baseline["name"]])
        for baseline in BASELINES
    }
    intent_summaries = {
        baseline["name"]: summarize_by_intent(results_by_baseline[baseline["name"]])
        for baseline in BASELINES
    }
    winners = compute_benchmark_winners(results_by_baseline)

    raw_payload = {
        "generated_at": datetime.now().isoformat(),
        "base_config": base_config,
        "baselines": BASELINES,
        "baseline_summaries": baseline_summaries,
        "type_summaries": type_summaries,
        "intent_summaries": intent_summaries,
        "results_by_baseline": results_by_baseline,
        "benchmark_winners": winners,
    }
    (output_dir / "baseline_comparison_results.json").write_text(
        json.dumps(raw_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_rows = []
    for baseline in BASELINES:
        summary = baseline_summaries[baseline["name"]]
        summary_rows.append(
            {
                "baseline": baseline["name"],
                "label": baseline["label"],
                "avg_final_score": f"{summary['avg_final_score']:.6f}",
                "median_final_score": f"{summary['median_final_score']:.6f}",
                "pass_rate": f"{summary['pass_rate']:.6f}",
                "passed": summary["passed"],
                "failed": summary["failed"],
            }
        )
    write_csv_rows(
        output_dir / "baseline_summary.csv",
        summary_rows,
        ["baseline", "label", "avg_final_score", "median_final_score", "pass_rate", "passed", "failed"],
    )

    detailed_rows = []
    for baseline_name, entries in results_by_baseline.items():
        for entry in entries:
            detailed_rows.append(
                {
                    "baseline": baseline_name,
                    "benchmark_id": entry["benchmark_id"],
                    "type": entry["type"],
                    "intent": entry.get("intent", "other"),
                    "final_score": f"{entry['scores'].get('final_score', 0.0):.6f}",
                    "passed": entry["passed"],
                    "threshold": entry.get("threshold", 0.0),
                }
            )
    write_csv_rows(
        output_dir / "baseline_per_benchmark.csv",
        detailed_rows,
        ["baseline", "benchmark_id", "type", "intent", "final_score", "passed", "threshold"],
    )

    generate_markdown_report(output_dir, baseline_summaries, type_summaries, intent_summaries, winners)
    generate_html_report(output_dir, baseline_summaries, type_summaries, intent_summaries, winners)

    print(f"\nSaved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
