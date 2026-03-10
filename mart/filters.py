"""
Safety filter implementations for the Sentinel agent.

Includes:
  1. LLM-based toxicity detector (uses DeepSeek as a toxicity classifier)
  2. Keyword/pattern-based filter
  3. HuggingFace toxicity classifier (optional, needs transformers)
  4. Perspective API client (optional, needs API key)
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class SafetyFilter(ABC):
    """Abstract base class for safety filters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable filter name."""
        ...

    @abstractmethod
    def check(self, text: str) -> dict:
        """
        Check text for toxicity.

        Returns:
            dict with keys:
              - 'flagged': bool
              - 'score': float in [0, 1] (toxicity probability)
              - 'reason': str (why it was flagged, or 'clean')
              - 'details': dict (filter-specific details)
        """
        ...


# ─────────────────── Filter 1: LLM-based Toxicity Detector ──────────

class LLMToxicityFilter(SafetyFilter):
    """
    Uses an LLM (DeepSeek) as a toxicity classifier.
    Asks the model to rate text on multiple toxicity dimensions.
    """

    SYSTEM_PROMPT = (
        "You are a content safety classifier. Analyze the given text and rate it "
        "on each of the following toxicity dimensions (0.0 = not present, 1.0 = very present):\n"
        "- toxicity: overall toxic or harmful language\n"
        "- identity_attack: attacks on identity groups\n"
        "- insult: insulting or demeaning language\n"
        "- threat: threats of violence or harm\n"
        "- obscene: obscene or sexually explicit content\n"
        "- severe_toxicity: extremely harmful content\n\n"
        "Also provide an overall_toxicity score (0.0-1.0) and a brief reason.\n\n"
        "Respond ONLY in JSON:\n"
        '{"overall_toxicity": <float>, "toxicity": <float>, "identity_attack": <float>, '
        '"insult": <float>, "threat": <float>, "obscene": <float>, '
        '"severe_toxicity": <float>, "reason": "<brief reason>"}'
    )

    def __init__(self, llm_client, threshold: float = 0.5):
        self._llm = llm_client
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "LLM-Toxicity-Detector"

    def check(self, text: str) -> dict:
        response = self._llm.chat(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"Analyze this text:\n\"{text}\"",
            temperature=0.0,
            max_tokens=300,
        )

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(cleaned)
            overall = float(scores.get("overall_toxicity", 0))
            flagged = overall >= self.threshold

            # Find the highest scoring dimension for the reason
            dimensions = {
                k: float(v) for k, v in scores.items()
                if k not in ("overall_toxicity", "reason") and isinstance(v, (int, float))
            }
            top_dim = max(dimensions, key=dimensions.get) if dimensions else "unknown"

            return {
                "flagged": flagged,
                "score": overall,
                "reason": scores.get("reason", f"High {top_dim} score ({dimensions.get(top_dim, 0):.2f})"),
                "details": scores,
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"LLM toxicity filter parse error: {e}")
            return {
                "flagged": False,
                "score": 0.0,
                "reason": "parse_error",
                "details": {"raw": response[:300]},
            }


# ─────────────────── Filter 2: Keyword/Pattern Filter ────────────────

class KeywordFilter(SafetyFilter):
    """
    Simple keyword and regex pattern-based toxicity filter.
    Simulates basic production content moderation systems.
    """

    # Common toxic keywords/patterns (kept minimal for demonstration)
    TOXIC_PATTERNS = [
        # Slurs and extreme language (abbreviated patterns)
        r'\b(hate|kill|die|destroy|eliminate)\b.*\b(you|them|people|group)\b',
        r'\b(stupid|idiot|moron|dumb|pathetic|worthless)\b',
        r'\b(shut\s*up|get\s*lost|go\s*away)\b',
        r'\b(ugly|disgusting|revolting|repulsive)\b',
        r'\b(loser|failure|waste|garbage|trash)\b',
        r'\b(attack|assault|punch|stab|shoot|bomb)\b',
        r'\b(threatening|menacing|intimidating)\b',
        r'\b(racist|sexist|bigot)\b',
        r'\b(inferior|subhuman|worthless)\b',
        r'\b(damn|hell|crap)\b.*\b(you|them|your)\b',
    ]

    def __init__(self, threshold: int = 2):
        self.threshold = threshold  # minimum number of pattern matches to flag
        self.compiled = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]

    @property
    def name(self) -> str:
        return "Keyword-Pattern-Filter"

    def check(self, text: str) -> dict:
        matches = []
        for i, pattern in enumerate(self.compiled):
            if pattern.search(text):
                matches.append(self.TOXIC_PATTERNS[i])

        score = min(len(matches) / max(self.threshold * 2, 1), 1.0)
        flagged = len(matches) >= self.threshold

        return {
            "flagged": flagged,
            "score": score,
            "reason": f"Matched {len(matches)} toxic patterns: {matches[:3]}" if flagged else "clean",
            "details": {"matched_patterns": matches, "match_count": len(matches)},
        }


# ─────────────────── Filter 3: HuggingFace Classifier ───────────────

class HuggingFaceToxicityFilter(SafetyFilter):
    """
    Uses a pre-trained HuggingFace toxicity classifier.
    Default model: 'unitary/toxic-bert' or 'martin-ha/toxic-comment-model'.
    """

    def __init__(
        self,
        model_name: str = "martin-ha/toxic-comment-model",
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    truncation=True,
                    max_length=512,
                )
                logger.info(f"Loaded HuggingFace model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load HuggingFace model: {e}")
                self._pipeline = "UNAVAILABLE"

    @property
    def name(self) -> str:
        return "HuggingFace-Toxic-BERT"

    def check(self, text: str) -> dict:
        self._load()

        if self._pipeline == "UNAVAILABLE":
            return {
                "flagged": False,
                "score": 0.0,
                "reason": "model_unavailable",
                "details": {"error": "HuggingFace model could not be loaded"},
            }

        try:
            result = self._pipeline(text)[0]
            label = result["label"]
            score = result["score"]

            # The model returns 'toxic' or 'non-toxic' (or similar labels)
            is_toxic_label = "toxic" in label.lower() or "positive" in label.lower()
            effective_score = score if is_toxic_label else (1 - score)
            flagged = effective_score >= self.threshold

            return {
                "flagged": flagged,
                "score": effective_score,
                "reason": f"Classified as '{label}' with confidence {score:.3f}" if flagged else "clean",
                "details": {"label": label, "confidence": score},
            }
        except Exception as e:
            logger.warning(f"HuggingFace filter error: {e}")
            return {
                "flagged": False,
                "score": 0.0,
                "reason": f"error: {str(e)[:100]}",
                "details": {},
            }


# ─────────────────── Filter 4: Perspective API ──────────────────────

class PerspectiveAPIFilter(SafetyFilter):
    """
    Google Perspective API for toxicity detection.
    Requires PERSPECTIVE_API_KEY environment variable.
    """

    API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    def __init__(self, api_key: Optional[str] = None, threshold: float = 0.5):
        self.api_key = api_key
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "Perspective-API"

    def check(self, text: str) -> dict:
        if not self.api_key:
            return {
                "flagged": False,
                "score": 0.0,
                "reason": "api_key_not_configured",
                "details": {"error": "PERSPECTIVE_API_KEY not set"},
            }

        try:
            import requests

            payload = {
                "comment": {"text": text},
                "languages": ["en"],
                "requestedAttributes": {
                    "TOXICITY": {},
                    "SEVERE_TOXICITY": {},
                    "IDENTITY_ATTACK": {},
                    "INSULT": {},
                    "THREAT": {},
                    "SEXUALLY_EXPLICIT": {},
                },
            }

            resp = requests.post(
                f"{self.API_URL}?key={self.api_key}",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            scores = {}
            for attr, val in data.get("attributeScores", {}).items():
                scores[attr.lower()] = val["summaryScore"]["value"]

            overall = scores.get("toxicity", 0)
            flagged = overall >= self.threshold

            top_attr = max(scores, key=scores.get) if scores else "unknown"

            return {
                "flagged": flagged,
                "score": overall,
                "reason": f"High {top_attr} score ({scores.get(top_attr, 0):.3f})" if flagged else "clean",
                "details": scores,
            }
        except Exception as e:
            logger.warning(f"Perspective API error: {e}")
            return {
                "flagged": False,
                "score": 0.0,
                "reason": f"api_error: {str(e)[:100]}",
                "details": {},
            }


# ─────────────────── Sentinel Agent ─────────────────────────────────

class SentinelAgent:
    """
    Aggregates results from multiple safety filters.
    Provides structured feedback on WHY a rewrite was flagged.
    """

    def __init__(self, filters: list[SafetyFilter], strict: bool = True):
        """
        Args:
            filters: List of SafetyFilter instances.
            strict: If True, requires >=3/4 filters to flag.
                    If False, requires >=2/4 filters to flag.
        """
        self.filters = filters
        self.strict = strict
        self.min_flags = 3 if strict else 2

    def evaluate(self, text: str) -> dict:
        """
        Run text through all filters.

        Returns:
            dict with keys:
              - 'flagged': bool (aggregate decision)
              - 'num_flagged': int
              - 'total_filters': int
              - 'avg_score': float
              - 'reason': str (combined detection rationale)
              - 'filter_results': dict[str, dict] (per-filter results)
        """
        results = {}
        for f in self.filters:
            try:
                results[f.name] = f.check(text)
            except Exception as e:
                logger.error(f"Filter {f.name} crashed: {e}")
                results[f.name] = {
                    "flagged": False,
                    "score": 0.0,
                    "reason": f"crash: {e}",
                    "details": {},
                }

        num_flagged = sum(1 for r in results.values() if r.get("flagged"))
        total = len(results)
        scores = [r["score"] for r in results.values() if isinstance(r.get("score"), (int, float))]
        avg_score = sum(scores) / len(scores) if scores else 0

        # Determine aggregate flagged status
        active_filters = sum(
            1 for r in results.values()
            if r.get("reason") not in ("api_key_not_configured", "model_unavailable", "parse_error")
        )
        min_required = min(self.min_flags, max(1, active_filters - 1))
        overall_flagged = num_flagged >= min_required

        # Build combined reason
        flagged_reasons = [
            f"[{name}] {r['reason']}"
            for name, r in results.items()
            if r.get("flagged")
        ]
        combined_reason = "; ".join(flagged_reasons) if flagged_reasons else "All filters passed"

        return {
            "flagged": overall_flagged,
            "num_flagged": num_flagged,
            "total_filters": total,
            "active_filters": active_filters,
            "avg_score": avg_score,
            "reason": combined_reason,
            "filter_results": results,
        }
