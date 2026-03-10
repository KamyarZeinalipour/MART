#!/usr/bin/env python3
"""
Advanced Attack Runner

Run the 4 advanced attack strategies:
  1. Genetic Algorithm Attack
  2. Multi-Turn Conversation Poisoning
  3. Adversarial Token Manipulation
  4. Contrastive Self-Play

Usage:
    # Run all advanced attacks on sample data
    python run_advanced.py --attacks all

    # Run specific attacks
    python run_advanced.py --attacks genetic multiturn

    # Quick demo
    python run_advanced.py --attacks all --max-samples 2
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mart.config import APIConfig, MARTConfig, ExperimentConfig
from mart.agents import LLMClient, AttackerAgent, JudgeAgent, RefinerAgent
from mart.filters import (
    LLMToxicityFilter, KeywordFilter, HuggingFaceToxicityFilter,
    PerspectiveAPIFilter, SentinelAgent,
)
from mart.genetic_attack import GeneticAttacker
from mart.multiturn_attack import MultiTurnPoisoner
from mart.token_attack import TokenManipulator
from mart.self_play import ContrastiveSelfPlay, print_selfplay_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def load_dataset(path: str, num_samples: int = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if num_samples:
        data = data[:num_samples]
    return data


def build_components(config: ExperimentConfig):
    """Build shared components used by all attacks."""
    llm = LLMClient(config.api)
    attacker = AttackerAgent(llm, config.mart)
    judge = JudgeAgent(llm, config.mart)
    refiner = RefinerAgent(llm, config.mart)

    filters = [
        LLMToxicityFilter(llm, threshold=config.mart.toxicity_threshold),
        KeywordFilter(threshold=2),
    ]
    try:
        filters.append(HuggingFaceToxicityFilter())
    except Exception:
        pass
    if config.api.perspective_api_key:
        filters.append(PerspectiveAPIFilter(
            api_key=config.api.perspective_api_key,
            threshold=config.mart.toxicity_threshold,
        ))

    sentinel = SentinelAgent(filters=filters, strict=config.mart.strict_evasion)

    return llm, attacker, judge, refiner, sentinel


def run_genetic(llm, attacker, judge, sentinel, config, samples, output_dir):
    """Run Genetic Algorithm Attack."""
    print("\n" + "=" * 70)
    print("  🧬 GENETIC ALGORITHM ATTACK")
    print("=" * 70)

    ga = GeneticAttacker(
        llm=llm, attacker=attacker, judge=judge, sentinel=sentinel,
        config=config.mart, population_size=6, num_generations=4, top_k=3,
    )

    results = []
    for sample in samples:
        result = ga.run(sample["id"], sample["text"])
        results.append(asdict(result))

    # Summary
    successes = sum(1 for r in results if r["success"])
    print(f"\n  📊 Genetic Attack: {successes}/{len(results)} successful")
    if results:
        avg_fitness = sum(r["best_fitness"] for r in results) / len(results)
        print(f"  Average best fitness: {avg_fitness:.3f}")

    # Save
    out = Path(output_dir) / "genetic_attack_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")

    return results


def run_multiturn(llm, judge, sentinel, config, samples, output_dir):
    """Run Multi-Turn Conversation Poisoning."""
    print("\n" + "=" * 70)
    print("  💬 MULTI-TURN CONVERSATION POISONING")
    print("=" * 70)

    poisoner = MultiTurnPoisoner(
        llm=llm, judge=judge, sentinel=sentinel,
        config=config.mart, num_turns=6,
    )

    results = []
    strategies = ["gradual", "split", "context"]

    for sample in samples:
        for strategy in strategies:
            result = poisoner.attack(sample["id"], sample["text"], strategy)
            results.append(asdict(result))

    # Summary
    successes = sum(1 for r in results if r["success"])
    by_strategy = {}
    for r in results:
        # Extract strategy from the results
        for s in strategies:
            if s in str(r.get("turns", [])):
                by_strategy.setdefault(s, []).append(r)

    print(f"\n  📊 Multi-Turn: {successes}/{len(results)} successful")
    no_single_flagged = sum(1 for r in results if not r["any_turn_flagged"])
    print(f"  No individual turn flagged: {no_single_flagged}/{len(results)}")

    # Save
    out = Path(output_dir) / "multiturn_attack_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")

    return results


def run_token(sentinel, samples, output_dir):
    """Run Token Manipulation Attack."""
    print("\n" + "=" * 70)
    print("  🔤 ADVERSARIAL TOKEN MANIPULATION")
    print("=" * 70)

    manipulator = TokenManipulator(sentinel=sentinel)

    all_results = []
    strategy_stats = {}

    for sample in samples:
        results = manipulator.attack_all_strategies(sample["id"], sample["text"])
        for r in results:
            rd = asdict(r)
            all_results.append(rd)

            # Track per-strategy stats
            s = rd["strategy"]
            if s not in strategy_stats:
                strategy_stats[s] = {"total": 0, "success": 0}
            strategy_stats[s]["total"] += 1
            if rd["success"]:
                strategy_stats[s]["success"] += 1

    # Summary
    total_success = sum(1 for r in all_results if r["success"])
    print(f"\n  📊 Token Attack: {total_success}/{len(all_results)} successful")
    print(f"\n  Per-strategy:")
    for s, stats in sorted(strategy_stats.items()):
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"    {s:25s}: {stats['success']}/{stats['total']} ({rate:.0%})")

    # Save
    out = Path(output_dir) / "token_attack_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")

    return all_results


def run_selfplay(llm, attacker, judge, refiner, config, samples, output_dir):
    """Run Contrastive Self-Play."""
    print("\n" + "=" * 70)
    print("  ⚔️  CONTRASTIVE SELF-PLAY")
    print("=" * 70)

    sp = ContrastiveSelfPlay(
        llm=llm, attacker=attacker, judge=judge, refiner=refiner,
        config=config.mart, num_rounds=4, attacks_per_round=4,
    )

    results = []
    for sample in samples:
        result = sp.run(sample["id"], sample["text"])
        results.append(result)

    # Print report
    print_selfplay_report(results)

    # Save
    out = Path(output_dir) / "selfplay_results.json"
    serialized = [asdict(r) for r in results]
    with open(out, "w") as f:
        json.dump(serialized, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Advanced MART Attack Runner")
    parser.add_argument(
        "--attacks", nargs="+", default=["all"],
        choices=["all", "genetic", "multiturn", "token", "selfplay"],
        help="Which attacks to run",
    )
    parser.add_argument("--data", default="data/sample_toxic.json")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output", default="results_advanced")
    parser.add_argument("--api-key", default=None)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY required")
        sys.exit(1)

    config = ExperimentConfig(
        api=APIConfig(deepseek_api_key=api_key),
        mart=MARTConfig(max_iterations=3, strict_evasion=False),
        output_dir=args.output,
    )

    data_path = Path(__file__).parent / args.data
    samples = load_dataset(str(data_path), args.max_samples)
    print(f"📂 Loaded {len(samples)} samples")

    llm, attacker, judge, refiner, sentinel = build_components(config)

    run_all = "all" in args.attacks

    if run_all or "genetic" in args.attacks:
        run_genetic(llm, attacker, judge, sentinel, config, samples, args.output)

    if run_all or "multiturn" in args.attacks:
        run_multiturn(llm, judge, sentinel, config, samples, args.output)

    if run_all or "token" in args.attacks:
        run_token(sentinel, samples, args.output)

    if run_all or "selfplay" in args.attacks:
        run_selfplay(llm, attacker, judge, refiner, config, samples, args.output)

    print(f"\n✅ All done! Results in {args.output}/")


if __name__ == "__main__":
    main()
