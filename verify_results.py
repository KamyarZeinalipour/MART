#!/usr/bin/env python3
"""
Verify whether adversarial rewrites genuinely preserve toxic intent
or if the LLM secretly detoxified the text.

Usage:
    # Verify results from a previous MART run
    python verify_results.py --traces results_with_perspective/mart_traces.json

    # Verify only successful evasions
    python verify_results.py --traces results/mart_traces.json --only-successful

    # Quick demo: verify 2 samples
    python verify_results.py --traces results/mart_traces.json --max-samples 2
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mart.config import APIConfig
from mart.agents import LLMClient
from mart.verifier import (
    ToxicityPreservationVerifier,
    VerificationResult,
    print_verification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Verify toxicity preservation in adversarial rewrites"
    )
    parser.add_argument(
        "--traces", required=True,
        help="Path to traces JSON file (e.g., results/mart_traces.json)",
    )
    parser.add_argument(
        "--only-successful", action="store_true",
        help="Only verify rewrites that successfully evaded filters",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to verify",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Combined score threshold for 'genuinely toxic' (default: 0.5)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for verification results JSON",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)",
    )

    args = parser.parse_args()

    # API key
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ Error: DeepSeek API key required!")
        sys.exit(1)

    # Load traces
    with open(args.traces) as f:
        traces = json.load(f)

    print(f"📂 Loaded {len(traces)} traces from {args.traces}")

    # Filter traces
    candidates = []
    for t in traces:
        if t.get("refused", False):
            continue
        if args.only_successful and not t.get("success", False):
            continue
        if t.get("final_rewrite"):
            candidates.append(t)

    if args.max_samples:
        candidates = candidates[:args.max_samples]

    print(f"🔍 Verifying {len(candidates)} rewrites...")
    if args.only_successful:
        print(f"   (filtered to successful evasions only)")

    # Build verifier
    llm = LLMClient(APIConfig(deepseek_api_key=api_key))
    verifier = ToxicityPreservationVerifier(llm, threshold=args.threshold)

    # Run verification
    results = []
    for i, t in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] "
              f"Sample {t['sample_id']} | {t['style']} | "
              f"{'✅ evaded' if t.get('success') else '❌ caught'}")

        result = verifier.verify(
            original=t["original_text"],
            rewrite=t["final_rewrite"],
            style=t["style"],
            sample_id=t["sample_id"],
        )
        results.append(result)

    # Print report
    print_verification_report(results)

    # Save results
    output_path = args.output or str(
        Path(args.traces).parent / "verification_report.json"
    )
    out_data = [asdict(r) for r in results]
    with open(output_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\n💾 Saved verification results to {output_path}")


if __name__ == "__main__":
    main()
