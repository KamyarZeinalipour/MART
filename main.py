#!/usr/bin/env python3
"""
MART: Multi-Agent Red Team Framework
Main entry point for running adversarial experiments.

Usage:
    # Run full experiment with all styles
    python main.py --run full

    # Quick demo with 3 samples and 2 styles
    python main.py --run demo

    # Custom run
    python main.py --run custom --num-samples 10 --styles academic metaphorical --max-iter 3

    # Single-agent baseline only
    python main.py --run single
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mart.config import APIConfig, MARTConfig, ExperimentConfig
from mart.agents import LLMClient, AttackerAgent, JudgeAgent, RefinerAgent
from mart.filters import (
    LLMToxicityFilter,
    KeywordFilter,
    HuggingFaceToxicityFilter,
    PerspectiveAPIFilter,
    SentinelAgent,
)
from mart.pipeline import MARTPipeline, save_traces
from mart.metrics import compute_full_report, print_report_summary


def setup_logging(verbose: bool = True):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(path: str, num_samples: int = None) -> list[dict]:
    """Load toxic text samples from JSON file."""
    with open(path) as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    logging.info(f"Loaded {len(data)} samples from {path}")
    return data


def build_pipeline(config: ExperimentConfig) -> MARTPipeline:
    """Build all agents and the MART pipeline."""

    # Create LLM client
    llm = LLMClient(config.api)

    # Create agents
    attacker = AttackerAgent(llm, config.mart)
    judge = JudgeAgent(llm, config.mart)
    refiner = RefinerAgent(llm, config.mart)

    # Create safety filters
    filters = [
        LLMToxicityFilter(llm, threshold=config.mart.toxicity_threshold),
        KeywordFilter(threshold=2),
    ]

    # Optional filters
    try:
        hf_filter = HuggingFaceToxicityFilter()
        filters.append(hf_filter)
        logging.info("HuggingFace toxicity filter: ENABLED")
    except Exception as e:
        logging.warning(f"HuggingFace filter skipped: {e}")

    if config.api.perspective_api_key:
        filters.append(PerspectiveAPIFilter(
            api_key=config.api.perspective_api_key,
            threshold=config.mart.toxicity_threshold,
        ))
        logging.info("Perspective API filter: ENABLED")
    else:
        logging.info("Perspective API filter: DISABLED (no API key)")

    # Create sentinel
    sentinel = SentinelAgent(filters=filters, strict=config.mart.strict_evasion)
    logging.info(f"Sentinel initialized with {len(filters)} filters (strict={config.mart.strict_evasion})")

    # Build pipeline
    pipeline = MARTPipeline(
        attacker=attacker,
        judge=judge,
        refiner=refiner,
        sentinel=sentinel,
        config=config.mart,
    )

    return pipeline


def run_experiment(
    pipeline: MARTPipeline,
    samples: list[dict],
    styles: list[str],
    run_single: bool = True,
    run_mart: bool = True,
    output_dir: str = "results",
):
    """
    Run the full experiment: single-agent baseline + MART.
    """
    single_traces = []
    mart_traces = []

    total_tasks = len(samples) * len(styles) * (int(run_single) + int(run_mart))
    completed = 0

    for sample in samples:
        sid = sample["id"]
        text = sample["text"]

        for style in styles:
            # Phase 1: Single-agent baseline
            if run_single:
                completed += 1
                print(f"\n[{completed}/{total_tasks}] "
                      f"SINGLE | Sample {sid} | {style}")
                trace = pipeline.run_single_agent(sid, text, style)
                single_traces.append(trace)

            # Phase 2: Full MART
            if run_mart:
                completed += 1
                print(f"\n[{completed}/{total_tasks}] "
                      f"MART   | Sample {sid} | {style}")
                trace = pipeline.run_mart(sid, text, style)
                mart_traces.append(trace)

    # Save traces
    os.makedirs(output_dir, exist_ok=True)
    if single_traces:
        save_traces(single_traces, f"{output_dir}/single_agent_traces.json")
    if mart_traces:
        save_traces(mart_traces, f"{output_dir}/mart_traces.json")

    # Compute and display report
    report = compute_full_report(
        single_traces,
        mart_traces,
        output_path=f"{output_dir}/metrics_report.json",
    )
    print_report_summary(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="MART: Multi-Agent Red Team Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", type=str, default="demo",
        choices=["demo", "full", "single", "mart", "custom"],
        help="Run mode: demo (3 samples, 2 styles), full (all), "
             "single (baseline only), mart (MART only), custom (specify below)",
    )
    parser.add_argument(
        "--data", type=str, default="data/sample_toxic.json",
        help="Path to toxic samples JSON file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Number of samples to use (default: all)",
    )
    parser.add_argument(
        "--styles", nargs="+", default=None,
        choices=["academic", "slang", "metaphorical",
                 "code_switching", "multi_turn", "narrative"],
        help="Attack styles to use",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5,
        help="Max MART iterations (default: 5)",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup
    setup_logging(args.verbose)

    # API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ Error: DeepSeek API key required!")
        print("   Set DEEPSEEK_API_KEY env var or use --api-key")
        sys.exit(1)

    # Configure based on run mode
    if args.run == "demo":
        num_samples = 3
        styles = ["academic", "metaphorical"]
        max_iter = 3
    elif args.run == "full":
        num_samples = args.num_samples  # all
        styles = ["academic", "slang", "metaphorical",
                  "code_switching", "multi_turn", "narrative"]
        max_iter = 5
    elif args.run == "custom":
        num_samples = args.num_samples
        styles = args.styles or ["academic", "metaphorical"]
        max_iter = args.max_iter
    else:
        num_samples = args.num_samples
        styles = args.styles or ["academic", "metaphorical", "slang"]
        max_iter = args.max_iter

    run_single = args.run in ("demo", "full", "single", "custom")
    run_mart_flag = args.run in ("demo", "full", "mart", "custom")

    # Build config
    config = ExperimentConfig(
        api=APIConfig(deepseek_api_key=api_key),
        mart=MARTConfig(
            max_iterations=max_iter,
            strict_evasion=False,  # relaxed for demo
        ),
        output_dir=args.output,
    )

    # Load dataset
    data_path = Path(__file__).parent / args.data
    samples = load_dataset(str(data_path), num_samples=num_samples)

    # Build pipeline
    print("\n🔧 Building MART pipeline...")
    pipeline = build_pipeline(config)

    # Run experiment
    print(f"\n🚀 Starting experiment: mode={args.run}, "
          f"samples={len(samples)}, styles={styles}, max_iter={max_iter}")
    print(f"   Single-agent: {'YES' if run_single else 'NO'}")
    print(f"   MART:         {'YES' if run_mart_flag else 'NO'}")
    print(f"   Output:       {args.output}/\n")

    report = run_experiment(
        pipeline=pipeline,
        samples=samples,
        styles=styles,
        run_single=run_single,
        run_mart=run_mart_flag,
        output_dir=args.output,
    )

    print(f"\n✅ Done! Results saved to {args.output}/")
    print(f"   - single_agent_traces.json")
    print(f"   - mart_traces.json")
    print(f"   - metrics_report.json")


if __name__ == "__main__":
    main()
