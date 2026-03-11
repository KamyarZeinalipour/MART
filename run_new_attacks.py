#!/usr/bin/env python3
"""
New Advanced Attack Runner (Wave 2)

Run the 4 NEW attack strategies:
  1. Cross-Lingual Code-Switching
  2. Persona/Roleplay Attack
  3. Socratic Questioning
  4. Semantic Trojan

Supports the multilingual benchmark (15 languages, 100 native samples each).

Usage:
    # Run on default sample data
    python run_new_attacks.py --attacks all --max-samples 2

    # Run on multilingual benchmark (all 15 languages)
    python run_new_attacks.py --attacks all --dataset multilingual --max-samples 5

    # Run on specific languages only
    python run_new_attacks.py --attacks socratic --dataset multilingual --language en de fr

    # Run on a single language
    python run_new_attacks.py --attacks all --dataset multilingual --language ar --max-samples 10
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


AVAILABLE_LANGUAGES = [
    'en', 'de', 'es', 'it', 'fr', 'ar', 'hi', 'zh',
    'ru', 'ja', 'uk', 'he', 'hin', 'am', 'tt',
]

LANG_NAMES = {
    'en': 'English', 'de': 'German', 'es': 'Spanish', 'it': 'Italian',
    'fr': 'French', 'ar': 'Arabic', 'hi': 'Hindi', 'zh': 'Chinese',
    'ru': 'Russian', 'ja': 'Japanese', 'uk': 'Ukrainian', 'he': 'Hebrew',
    'hin': 'Hinglish', 'am': 'Amharic', 'tt': 'Tatar',
}


def load_dataset(path: str, n: int = None, languages: list = None) -> list[dict]:
    """Load dataset with optional language filtering."""
    with open(path) as f:
        data = json.load(f)

    # Filter by language if specified
    if languages and languages != ['all']:
        data = [s for s in data if s.get('language', 'en') in languages]

    # Normalize: ensure 'text' field exists (multilingual uses 'toxic_sentence')
    for s in data:
        if 'text' not in s and 'toxic_sentence' in s:
            s['text'] = s['toxic_sentence']
        if 'id' not in s:
            s['id'] = data.index(s) + 1

    if n:
        data = data[:n]

    # Print language distribution
    by_lang = {}
    for s in data:
        l = s.get('language', 'en')
        by_lang[l] = by_lang.get(l, 0) + 1
    lang_str = ', '.join(f'{LANG_NAMES.get(l,l)}({c})' for l, c in sorted(by_lang.items()))
    print(f'📂 Loaded {len(data)} samples: {lang_str}')

    return data


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
    parser = argparse.ArgumentParser(
        description="MART Attack Runner (Wave 2) — Supports Multilingual Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--attacks", nargs="+", default=["all"],
        choices=["all", "crosslingual", "persona", "socratic", "trojan"],
        help="Attack types to run",
    )
    parser.add_argument(
        "--dataset", default="sample",
        choices=["sample", "multilingual"],
        help="Dataset to use: 'sample' (10 English) or 'multilingual' (1500, 15 languages)",
    )
    parser.add_argument(
        "--language", nargs="+", default=["all"],
        help=f"Languages to test (default: all). Options: all, {', '.join(AVAILABLE_LANGUAGES)}",
    )
    parser.add_argument("--data", default=None, help="Custom data file path (overrides --dataset)")
    parser.add_argument("--max-samples", type=int, default=2, help="Max samples per language")
    parser.add_argument("--output", default="results_new_attacks", help="Output directory")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key")

    args = parser.parse_args()
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY required")
        sys.exit(1)

    # Resolve dataset path
    if args.data:
        data_path = args.data
    elif args.dataset == "multilingual":
        data_path = str(Path(__file__).parent / "data" / "benchmark_multilingual.json")
    else:
        data_path = str(Path(__file__).parent / "data" / "sample_toxic.json")

    config = ExperimentConfig(
        api=APIConfig(deepseek_api_key=api_key),
        mart=MARTConfig(max_iterations=3, strict_evasion=False),
        output_dir=args.output,
    )

    # Load and filter by language
    samples = load_dataset(data_path, args.max_samples, args.language)

    if not samples:
        print(f"❌ No samples found for language(s): {args.language}")
        print(f"   Available: {', '.join(AVAILABLE_LANGUAGES)}")
        sys.exit(1)

    # Build output dir with language suffix
    if args.dataset == "multilingual" and args.language != ["all"]:
        output_dir = f"{args.output}/{'_'.join(args.language)}"
    else:
        output_dir = args.output

    llm, judge, sentinel = build_components(config)
    run_all = "all" in args.attacks

    if run_all or "crosslingual" in args.attacks:
        run_crosslingual(llm, judge, sentinel, config, samples, output_dir)

    if run_all or "persona" in args.attacks:
        run_persona(llm, judge, sentinel, config, samples, output_dir)

    if run_all or "socratic" in args.attacks:
        run_socratic(llm, judge, sentinel, config, samples, output_dir)

    if run_all or "trojan" in args.attacks:
        run_trojan(llm, judge, sentinel, config, samples, output_dir)

    print(f"\n✅ All done! Results in {output_dir}/")


if __name__ == "__main__":
    main()
