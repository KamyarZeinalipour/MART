#!/usr/bin/env python3
"""
New Advanced Attack Runner (Wave 2)

Run the 4 NEW attack strategies:
  1. Cross-Lingual Code-Switching
  2. Persona/Roleplay Attack
  3. Socratic Questioning
  4. Semantic Trojan

Usage:
    python run_new_attacks.py --attacks all --max-samples 2
    python run_new_attacks.py --attacks persona socratic
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
from mart.agents import LLMClient, JudgeAgent
from mart.filters import (
    LLMToxicityFilter, KeywordFilter, HuggingFaceToxicityFilter,
    PerspectiveAPIFilter, SentinelAgent,
)
from mart.crosslingual_attack import CrossLingualAttacker
from mart.persona_attack import PersonaAttacker
from mart.socratic_attack import SocraticAttacker
from mart.trojan_attack import SemanticTrojanAttacker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def load_dataset(path: str, n: int = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:n] if n else data


def build_components(config: ExperimentConfig):
    llm = LLMClient(config.api)
    judge = JudgeAgent(llm, config.mart)

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
    return llm, judge, sentinel


def run_crosslingual(llm, judge, sentinel, config, samples, output_dir):
    print("\n" + "=" * 70)
    print("  🌍 CROSS-LINGUAL CODE-SWITCHING ATTACK")
    print("=" * 70)

    attacker = CrossLingualAttacker(llm, judge, sentinel, config.mart)
    results = []

    strategies = ["codeswitch", "keyword_sub", "academic_foreign", "dialogue"]
    for sample in samples:
        for strategy in strategies:
            r = attacker.attack(sample["id"], sample["text"], strategy)
            results.append(asdict(r))

    successes = sum(1 for r in results if r["success"])
    print(f"\n  📊 Cross-Lingual: {successes}/{len(results)} successful")

    by_strategy = {}
    for r in results:
        s = r["strategy"]
        by_strategy.setdefault(s, {"t": 0, "s": 0})
        by_strategy[s]["t"] += 1
        if r["success"]:
            by_strategy[s]["s"] += 1

    for s, st in sorted(by_strategy.items()):
        rate = st["s"] / st["t"] if st["t"] else 0
        print(f"    {s:22s}: {st['s']}/{st['t']} ({rate:.0%})")

    out = Path(output_dir) / "crosslingual_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")
    return results


def run_persona(llm, judge, sentinel, config, samples, output_dir):
    print("\n" + "=" * 70)
    print("  🎭 PERSONA / ROLEPLAY ATTACK")
    print("=" * 70)

    attacker = PersonaAttacker(llm, judge, sentinel, config.mart)
    results = []

    for sample in samples:
        batch = attacker.attack_all_personas(sample["id"], sample["text"])
        results.extend([asdict(r) for r in batch])

    successes = sum(1 for r in results if r["success"])
    print(f"\n  📊 Persona: {successes}/{len(results)} successful")

    by_persona = {}
    for r in results:
        p = r["persona"]
        by_persona.setdefault(p, {"t": 0, "s": 0})
        by_persona[p]["t"] += 1
        if r["success"]:
            by_persona[p]["s"] += 1

    for p, st in sorted(by_persona.items()):
        rate = st["s"] / st["t"] if st["t"] else 0
        print(f"    {p:22s}: {st['s']}/{st['t']} ({rate:.0%})")

    out = Path(output_dir) / "persona_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")
    return results


def run_socratic(llm, judge, sentinel, config, samples, output_dir):
    print("\n" + "=" * 70)
    print("  ❓ SOCRATIC QUESTIONING ATTACK")
    print("=" * 70)

    attacker = SocraticAttacker(llm, judge, sentinel, config.mart)
    results = []

    for sample in samples:
        batch = attacker.attack_all_strategies(sample["id"], sample["text"])
        results.extend([asdict(r) for r in batch])

    successes = sum(1 for r in results if r["success"])
    print(f"\n  📊 Socratic: {successes}/{len(results)} successful")

    by_strategy = {}
    for r in results:
        s = r["strategy"]
        by_strategy.setdefault(s, {"t": 0, "s": 0})
        by_strategy[s]["t"] += 1
        if r["success"]:
            by_strategy[s]["s"] += 1

    for s, st in sorted(by_strategy.items()):
        rate = st["s"] / st["t"] if st["t"] else 0
        print(f"    {s:22s}: {st['s']}/{st['t']} ({rate:.0%})")

    out = Path(output_dir) / "socratic_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")
    return results


def run_trojan(llm, judge, sentinel, config, samples, output_dir):
    print("\n" + "=" * 70)
    print("  🎯 SEMANTIC TROJAN ATTACK")
    print("=" * 70)

    attacker = SemanticTrojanAttacker(llm, judge, sentinel, config.mart)
    results = []

    for sample in samples:
        batch = attacker.attack_all_strategies(sample["id"], sample["text"])
        results.extend([asdict(r) for r in batch])

    successes = sum(1 for r in results if r["success"])
    print(f"\n  📊 Trojan: {successes}/{len(results)} successful")

    by_strategy = {}
    for r in results:
        s = r["strategy"]
        by_strategy.setdefault(s, {"t": 0, "s": 0})
        by_strategy[s]["t"] += 1
        if r["success"]:
            by_strategy[s]["s"] += 1

    for s, st in sorted(by_strategy.items()):
        rate = st["s"] / st["t"] if st["t"] else 0
        print(f"    {s:22s}: {st['s']}/{st['t']} ({rate:.0%})")

    out = Path(output_dir) / "trojan_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Saved to {out}")
    return results


def main():
    parser = argparse.ArgumentParser(description="New Attack Runner (Wave 2)")
    parser.add_argument(
        "--attacks", nargs="+", default=["all"],
        choices=["all", "crosslingual", "persona", "socratic", "trojan"],
    )
    parser.add_argument("--data", default="data/sample_toxic.json")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output", default="results_new_attacks")
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

    samples = load_dataset(str(Path(__file__).parent / args.data), args.max_samples)
    print(f"📂 Loaded {len(samples)} samples")

    llm, judge, sentinel = build_components(config)
    run_all = "all" in args.attacks

    if run_all or "crosslingual" in args.attacks:
        run_crosslingual(llm, judge, sentinel, config, samples, args.output)

    if run_all or "persona" in args.attacks:
        run_persona(llm, judge, sentinel, config, samples, args.output)

    if run_all or "socratic" in args.attacks:
        run_socratic(llm, judge, sentinel, config, samples, args.output)

    if run_all or "trojan" in args.attacks:
        run_trojan(llm, judge, sentinel, config, samples, args.output)

    print(f"\n✅ All done! Results in {args.output}/")


if __name__ == "__main__":
    main()
