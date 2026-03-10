"""
MART Pipeline: Orchestrates the multi-agent adversarial loop.

Implements:
  - Single-agent baseline attack (Phase 1)
  - Full MART iterative loop (Phase 2)
  - Results collection and serialization
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .agents import AttackerAgent, JudgeAgent, RefinerAgent, LLMClient
from .config import MARTConfig, ExperimentConfig
from .filters import SentinelAgent

logger = logging.getLogger(__name__)


@dataclass
class AttackTrace:
    """Records the full trace of a single attack attempt."""
    sample_id: int
    original_text: str
    style: str
    mode: str  # 'single' or 'mart'

    # Results
    refused: bool = False
    final_rewrite: Optional[str] = None
    iterations_used: int = 0
    success: bool = False  # evasion success

    # Judge scores (final)
    semantic_preservation: float = 0.0
    naturalness: float = 0.0

    # Sentinel results (final)
    num_filters_flagged: int = 0
    avg_filter_score: float = 0.0
    sentinel_reason: str = ""
    filter_details: dict = field(default_factory=dict)

    # Iteration history
    iteration_history: list = field(default_factory=list)

    # Timing
    total_time_seconds: float = 0.0


class MARTPipeline:
    """
    Orchestrates the Multi-Agent Red Team adversarial loop.

    Phases:
      1. Single-agent baseline (control)
      2. Full MART iterative loop
    """

    def __init__(
        self,
        attacker: AttackerAgent,
        judge: JudgeAgent,
        refiner: RefinerAgent,
        sentinel: SentinelAgent,
        config: MARTConfig,
    ):
        self.attacker = attacker
        self.judge = judge
        self.refiner = refiner
        self.sentinel = sentinel
        self.config = config

    # ────────────── Phase 1: Single-Agent Baseline ──────────────

    def run_single_agent(
        self,
        sample_id: int,
        toxic_text: str,
        style: str,
    ) -> AttackTrace:
        """
        Run a single-shot attack (no iterative refinement).
        This serves as the baseline to compare MART against.
        """
        start = time.time()
        trace = AttackTrace(
            sample_id=sample_id,
            original_text=toxic_text,
            style=style,
            mode="single",
        )

        # Step 1: Attacker generates rewrite
        logger.info(f"[Single] Sample {sample_id} | Style: {style}")
        attack_result = self.attacker.attack(toxic_text, style)

        if attack_result["refused"]:
            trace.refused = True
            trace.total_time_seconds = time.time() - start
            logger.info(f"  → REFUSED")
            return trace

        rewrite = attack_result["rewrite"]
        trace.final_rewrite = rewrite

        # Step 2: Judge evaluates
        judge_result = self.judge.evaluate(toxic_text, rewrite)
        trace.semantic_preservation = judge_result["semantic_preservation"]
        trace.naturalness = judge_result["naturalness"]

        # Step 3: Sentinel checks
        sentinel_result = self.sentinel.evaluate(rewrite)
        trace.num_filters_flagged = sentinel_result["num_flagged"]
        trace.avg_filter_score = sentinel_result["avg_score"]
        trace.sentinel_reason = sentinel_result["reason"]
        trace.filter_details = sentinel_result["filter_results"]
        trace.success = not sentinel_result["flagged"]
        trace.iterations_used = 1

        trace.total_time_seconds = time.time() - start
        status = "✅ EVADED" if trace.success else "❌ CAUGHT"
        logger.info(
            f"  → {status} | SemPres={trace.semantic_preservation:.2f} "
            f"| Flagged={sentinel_result['num_flagged']}/{sentinel_result['total_filters']}"
        )

        return trace

    # ────────────── Phase 2: Full MART Loop ─────────────────────

    def run_mart(
        self,
        sample_id: int,
        toxic_text: str,
        style: str,
    ) -> AttackTrace:
        """
        Run the full MART iterative adversarial loop.

        Algorithm:
          1. Attacker generates initial rewrite
          2. For k = 0..K-1:
             a. Judge evaluates quality
             b. If quality too low → Refiner improves
             c. Sentinel checks evasion
             d. If evaded → success
             e. Else → Refiner improves using Sentinel feedback
          3. Return success/failure
        """
        start = time.time()
        trace = AttackTrace(
            sample_id=sample_id,
            original_text=toxic_text,
            style=style,
            mode="mart",
        )

        # Step 1: Initial attack
        logger.info(f"[MART] Sample {sample_id} | Style: {style}")
        attack_result = self.attacker.attack(toxic_text, style)

        if attack_result["refused"]:
            trace.refused = True
            trace.total_time_seconds = time.time() - start
            logger.info(f"  → REFUSED by Attacker")
            return trace

        current_rewrite = attack_result["rewrite"]

        # Step 2: Iterative refinement loop
        for k in range(self.config.max_iterations):
            logger.info(f"  Iteration {k+1}/{self.config.max_iterations}")

            iteration_record = {"iteration": k + 1, "rewrite": current_rewrite}

            # 2a: Judge evaluates quality
            judge_result = self.judge.evaluate(toxic_text, current_rewrite)
            sem_score = judge_result["semantic_preservation"]
            nat_score = judge_result["naturalness"]
            iteration_record["semantic_preservation"] = sem_score
            iteration_record["naturalness"] = nat_score

            logger.info(f"    Judge: SemPres={sem_score:.2f}, Natural={nat_score:.2f}")

            # 2b: If quality too low, refine for quality
            if sem_score < self.config.semantic_threshold or nat_score < self.config.naturalness_threshold:
                reason = (
                    f"Low quality: semantic={sem_score:.2f} (need {self.config.semantic_threshold}), "
                    f"naturalness={nat_score:.2f} (need {self.config.naturalness_threshold})"
                )
                logger.info(f"    → Low quality, refining...")
                current_rewrite = self.refiner.refine(
                    current_rewrite, toxic_text, reason, style
                )
                iteration_record["action"] = "refined_for_quality"
                iteration_record["refinement_reason"] = reason
                trace.iteration_history.append(iteration_record)
                continue

            # 2c: Sentinel checks evasion
            sentinel_result = self.sentinel.evaluate(current_rewrite)
            iteration_record["flagged"] = sentinel_result["flagged"]
            iteration_record["num_flagged"] = sentinel_result["num_flagged"]
            iteration_record["avg_score"] = sentinel_result["avg_score"]
            iteration_record["sentinel_reason"] = sentinel_result["reason"]

            logger.info(
                f"    Sentinel: Flagged={sentinel_result['flagged']} "
                f"({sentinel_result['num_flagged']}/{sentinel_result['total_filters']})"
            )

            # 2d: If evaded → success!
            if not sentinel_result["flagged"]:
                trace.final_rewrite = current_rewrite
                trace.success = True
                trace.iterations_used = k + 1
                trace.semantic_preservation = sem_score
                trace.naturalness = nat_score
                trace.num_filters_flagged = sentinel_result["num_flagged"]
                trace.avg_filter_score = sentinel_result["avg_score"]
                trace.sentinel_reason = sentinel_result["reason"]
                trace.filter_details = sentinel_result["filter_results"]
                iteration_record["action"] = "evaded"
                trace.iteration_history.append(iteration_record)

                # Record success in refiner memory
                self.refiner.record_success(current_rewrite, style)

                trace.total_time_seconds = time.time() - start
                logger.info(f"  → ✅ EVADED at iteration {k+1}!")
                return trace

            # 2e: Refine using Sentinel feedback
            logger.info(f"    → Caught! Refining with feedback...")
            current_rewrite = self.refiner.refine(
                current_rewrite,
                toxic_text,
                sentinel_result["reason"],
                style,
            )
            iteration_record["action"] = "refined_for_evasion"
            iteration_record["refinement_reason"] = sentinel_result["reason"]
            trace.iteration_history.append(iteration_record)

        # Step 3: Exhausted all iterations
        trace.final_rewrite = current_rewrite
        trace.success = False
        trace.iterations_used = self.config.max_iterations

        # Final evaluation
        judge_result = self.judge.evaluate(toxic_text, current_rewrite)
        trace.semantic_preservation = judge_result["semantic_preservation"]
        trace.naturalness = judge_result["naturalness"]

        sentinel_result = self.sentinel.evaluate(current_rewrite)
        trace.num_filters_flagged = sentinel_result["num_flagged"]
        trace.avg_filter_score = sentinel_result["avg_score"]
        trace.sentinel_reason = sentinel_result["reason"]
        trace.filter_details = sentinel_result["filter_results"]

        trace.total_time_seconds = time.time() - start
        logger.info(f"  → ❌ FAILED after {self.config.max_iterations} iterations")
        return trace


def save_traces(traces: list[AttackTrace], output_path: str):
    """Save attack traces to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for t in traces:
        d = asdict(t)
        # Clean up non-serializable items
        for key in list(d.get("filter_details", {}).keys()):
            if not isinstance(d["filter_details"][key], (dict, list, str, int, float, bool, type(None))):
                d["filter_details"][key] = str(d["filter_details"][key])
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved {len(data)} traces to {path}")
