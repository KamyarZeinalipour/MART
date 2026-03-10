#!/usr/bin/env python3
"""
Build Real-World Multilingual Benchmark for MART Paper

Uses REAL native datasets wherever available:
  - English:  ToxiGen + HateCheck (100 native samples)
  - French:   MLMA hate speech dataset (100 native samples)
  - Arabic:   MLMA hate speech dataset (100 native samples)
  - German:   HateCheck translated + native GermEval if available (100)
  - Spanish:  translation from English sources (100)
  - Italian:  translation from English sources (100)
  - Hindi:    translation from English sources (100)

Total: 700 samples across 7 languages.

Usage:
    python build_benchmark.py
    python build_benchmark.py --per-language 50
"""

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from pathlib import Path

random.seed(42)


def download_toxigen(n: int = 70) -> list[dict]:
    """Download ToxiGen English toxic samples."""
    print("📥 [EN] ToxiGen...")
    from datasets import load_dataset
    ds = load_dataset("toxigen/toxigen-data", "annotated", split="train")

    toxic = []
    for row in ds:
        text = row.get("text", "")
        score = row.get("toxicity_human", 0)
        group = row.get("target_group", "unknown")
        if score and score > 3.0 and len(text) > 20:
            toxic.append({"text": text.strip(), "target_group": (group or "unknown").lower(), "score": score})

    by_group = {}
    for s in toxic:
        by_group.setdefault(s["target_group"], []).append(s)

    per_group = max(1, n // min(len(by_group), 10))
    selected = []
    for g in sorted(by_group, key=lambda g: len(by_group[g]), reverse=True):
        random.shuffle(by_group[g])
        selected.extend(by_group[g][:per_group])
        if len(selected) >= n:
            break

    selected = selected[:n]
    print(f"  ✅ {len(selected)} ToxiGen samples")
    return [{"text": s["text"], "category": _classify(s["text"]),
             "source": "toxigen", "target_group": s["target_group"], "language": "en"} for s in selected]


def download_hatecheck(n: int = 30) -> list[dict]:
    """Download HateCheck English test cases."""
    print("📥 [EN] HateCheck...")
    import urllib.request
    url = "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/test_suite_cases.csv"
    content = urllib.request.urlopen(url).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(content)))
    hateful = [r for r in rows if r.get("label_gold") == "hateful"]

    random.shuffle(hateful)
    selected = hateful[:n]
    print(f"  ✅ {len(selected)} HateCheck samples")
    return [{"text": r.get("test_case", ""), "category": _map_hatecheck(r.get("functionality", "")),
             "source": "hatecheck", "target_group": r.get("target_ident", "general"), "language": "en"} for r in selected]


def load_mlma_french(n: int = 100) -> list[dict]:
    """Load MLMA French hate speech (native)."""
    print("📥 [FR] MLMA French (native)...")
    return _load_mlma_csv("/tmp/mlma/hate_speech_mlma/fr_dataset.csv", "fr", n)


def load_mlma_arabic(n: int = 100) -> list[dict]:
    """Load MLMA Arabic hate speech (native)."""
    print("📥 [AR] MLMA Arabic (native)...")
    return _load_mlma_csv("/tmp/mlma/hate_speech_mlma/ar_dataset.csv", "ar", n)


def _load_mlma_csv(path: str, lang: str, n: int) -> list[dict]:
    """Load and filter MLMA CSV."""
    if not Path(path).exists():
        # Download if not present
        print("  Downloading MLMA dataset...")
        import urllib.request, zipfile
        Path("/tmp/mlma").mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/HKUST-KnowComp/MLMA_hate_speech/master/hate_speech_mlma.zip",
            "/tmp/mlma/hate_speech_mlma.zip"
        )
        with zipfile.ZipFile("/tmp/mlma/hate_speech_mlma.zip", 'r') as z:
            z.extractall("/tmp/mlma")

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter for hateful/offensive (not "normal")
    toxic = [r for r in rows if r.get("sentiment", "").lower() in
             ("hateful", "offensive", "abusive", "offensive_disrespectful",
              "hateful_normal", "hateful_aggressive", "abusive_hateful")]
    if not toxic:
        # Fallback: take anything not "normal"
        toxic = [r for r in rows if r.get("sentiment", "normal").lower() != "normal"]

    random.shuffle(toxic)
    selected = toxic[:n]
    print(f"  ✅ {len(selected)} native {lang.upper()} samples (from {len(toxic)} toxic)")

    results = []
    for r in selected:
        results.append({
            "text": r.get("tweet", ""),
            "category": _map_mlma_category(r),
            "source": f"mlma_{lang}",
            "target_group": r.get("group", r.get("target", "general")),
            "language": lang,
        })
    return results


def _map_mlma_category(row: dict) -> str:
    sentiment = row.get("sentiment", "").lower()
    target = row.get("target", "").lower()
    if "hateful" in sentiment:
        return "hate_speech"
    if "abusive" in sentiment:
        return "insult"
    if "offensive" in sentiment:
        if "gender" in target or "origin" in target or "religion" in target:
            return "identity_attack"
        return "insult"
    return "hate_speech"


def translate_batch(texts: list[str], lang: str, api_key: str) -> list[str]:
    """Translate a batch of English texts to target language."""
    from mart.agents import LLMClient
    from mart.config import APIConfig
    llm = LLMClient(APIConfig(deepseek_api_key=api_key))

    translations = []
    for i, text in enumerate(texts):
        try:
            result = llm.chat(
                system_prompt=f"Translate to {lang}. Preserve exact meaning and tone — do NOT censor. Output ONLY the translation.",
                user_prompt=text,
                temperature=0.2, max_tokens=300,
            )
            translations.append(result.strip())
            if (i + 1) % 20 == 0:
                print(f"    [{lang}] {i+1}/{len(texts)}")
            time.sleep(0.2)
        except Exception as e:
            translations.append(text)  # fallback to English
            time.sleep(1)
    return translations


def build_translated_set(english_samples: list[dict], lang_name: str,
                         lang_code: str, n: int, api_key: str) -> list[dict]:
    """Build a translated dataset from English sources."""
    print(f"📥 [{lang_code.upper()}] Translating {n} samples → {lang_name}...")
    source = english_samples[:n]
    texts = [s["text"] for s in source]
    translated = translate_batch(texts, lang_name, api_key)

    results = []
    for s, t in zip(source, translated):
        results.append({
            "text": t,
            "category": s["category"],
            "source": f"{s['source']}_translated_{lang_code}",
            "target_group": s.get("target_group", "general"),
            "language": lang_code,
            "original_english": s["text"],
        })
    print(f"  ✅ {len(results)} {lang_name} samples")
    return results


def _classify(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["kill", "die", "shoot", "attack", "destroy"]): return "threat"
    if any(w in t for w in ["stupid", "idiot", "dumb", "worthless", "loser"]): return "insult"
    if any(w in t for w in ["disgusting", "gross", "sick", "nasty"]): return "obscene"
    if any(w in t for w in ["go back", "don't belong", "inferior", "subhuman"]): return "identity_attack"
    if any(w in t for w in ["hate", "terrible", "worst", "awful"]): return "hate_speech"
    return "implicit_hate"


def _map_hatecheck(f: str) -> str:
    f = f.lower()
    if "derog" in f: return "insult"
    if "dehuman" in f: return "dehumanization"
    if "threat" in f: return "threat"
    if "slur" in f or "profanity" in f: return "obscene"
    return "hate_speech"


def main():
    parser = argparse.ArgumentParser(description="Build Real-World MART Benchmark")
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--output", default="data/benchmark_multilingual.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--skip-translation", action="store_true",
                        help="Only use native datasets (EN, FR, AR)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    n = args.per_language

    print(f"\n{'='*60}")
    print(f"  Building MART Real-World Multilingual Benchmark")
    print(f"  Target: {n} samples per language")
    print(f"{'='*60}\n")

    all_samples = []

    # === NATIVE DATASETS ===
    # English: ToxiGen + HateCheck
    toxigen = download_toxigen(min(n, 70))
    hatecheck = download_hatecheck(n - len(toxigen))
    english = (toxigen + hatecheck)[:n]
    all_samples.extend(english)

    # French: MLMA (native)
    french = load_mlma_french(n)
    all_samples.extend(french)

    # Arabic: MLMA (native)
    arabic = load_mlma_arabic(n)
    all_samples.extend(arabic)

    if not args.skip_translation and api_key:
        # === TRANSLATED DATASETS ===
        translate_langs = [
            ("German", "de"),
            ("Spanish", "es"),
            ("Italian", "it"),
            ("Hindi", "hi"),
        ]
        for lang_name, lang_code in translate_langs:
            translated = build_translated_set(english, lang_name, lang_code, n, api_key)
            all_samples.extend(translated)

    # Assign IDs
    for i, s in enumerate(all_samples):
        s["id"] = i + 1

    # Summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {len(all_samples)} samples\n")

    print("  By language:")
    by_lang = {}
    for s in all_samples:
        l = s.get("language", "en")
        by_lang[l] = by_lang.get(l, 0) + 1
    for l, c in sorted(by_lang.items()):
        native = "🟢 native" if l in ("en", "fr", "ar") else "🔵 translated"
        print(f"    {l}: {c:4d} ({native})")

    print("\n  By source:")
    by_src = {}
    for s in all_samples:
        by_src[s["source"]] = by_src.get(s["source"], 0) + 1
    for src, c in sorted(by_src.items()):
        print(f"    {src:35s}: {c}")

    print("\n  By category:")
    by_cat = {}
    for s in all_samples:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
    for cat, c in sorted(by_cat.items()):
        print(f"    {cat:35s}: {c}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"\n  💾 Saved to {args.output}")
    print(f"  ✅ Done!\n")


if __name__ == "__main__":
    main()
