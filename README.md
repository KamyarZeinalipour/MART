# 🔴 MART: Multi-Agent Red Team Framework

**A modular adversarial testing framework for evaluating the robustness of LLM content moderation systems.**

MART orchestrates multiple specialized AI agents that collaboratively generate, evaluate, refine, and verify adversarial rewrites of toxic text — exposing blind spots in safety filters that single-pass attacks miss.

---

## 🏗️ Architecture

```
                    ┌────────────────┐
                    │   Toxic Text   │
                    └───────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │  Attacker Agent  │  ← 6 attack styles
                   └────────┬────────┘
                            ▼
              ┌─────────────────────────┐
              │      Judge Agent        │  ← semantic & naturalness scoring
              └─────┬──────────┬────────┘
                    │          │
            pass ◄──┘          └──► fail → Refiner Agent (with memory)
                    │                         │
                    ▼                         │
           ┌────────────────┐                 │
           │ Sentinel Agent │ ◄───────────────┘
           │  (4 filters)   │
           └───────┬────────┘
                   │
          ┌────────┴────────┐
       evaded            caught → loop back to Refiner
          │
          ▼
   ┌──────────────┐
   │   Verifier   │  ← confirms toxic intent preserved
   └──────────────┘
```

## 🧩 Components

### Core Agents
| Agent | Role |
|-------|------|
| **Attacker** | Generates adversarial rewrites using 6 styles (academic, metaphorical, slang, code-switching, multi-turn, narrative) |
| **Judge** | Evaluates semantic preservation and naturalness of rewrites |
| **Refiner** | Iteratively improves rewrites using feedback + memory of past successes/failures |
| **Sentinel** | Aggregates 4 independent safety filters to detect toxicity |
| **Verifier** | 3-strategy verification that rewrites genuinely preserve toxic intent |

### Safety Filters Under Test
| Filter | Type | Source |
|--------|------|--------|
| LLM Toxicity Detector | LLM-based | DeepSeek as classifier |
| Keyword Pattern Filter | Rule-based | Regex pattern matching |
| HuggingFace Toxic-BERT | ML classifier | `martin-ha/toxic-comment-model` |
| Perspective API | External API | Google Jigsaw |

### Advanced Attack Modules
| Module | Strategy |
|--------|----------|
| 🧬 **Genetic Algorithm** | Evolutionary population-based optimization with mutation & crossover |
| 💬 **Multi-Turn Poisoning** | Distributes toxic meaning across conversation turns |
| 🔤 **Token Manipulation** | Character-level perturbations (homoglyphs, zero-width, leetspeak) |
| ⚔️ **Contrastive Self-Play** | GAN-like co-evolution of attacker and adaptive defender |

## 📊 Key Results

| Attack Method | ASR | Key Insight |
|---------------|-----|-------------|
| Single-Agent Baseline | 66.7% | — |
| **MART Loop** (3 filters) | **100%** | 1.5× amplification over single-agent |
| **MART Loop** (4 filters) | 50% | Perspective API blocks academic style |
| **Genetic Algorithm** | **100%** | Population diversity beats sequential refinement |
| **Multi-Turn Poisoning** | **67%** | 0.90–0.95 semantic preservation, per-turn undetectable |
| Token Manipulation | 0% | LLM filters immune to character tricks |
| Contrastive Self-Play | 6.25% | Adaptive defender learns from 1 example |

See [REPORT.md](REPORT.md) for full experiment details, cross-filter transferability matrices, and analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- DeepSeek API key ([get one here](https://platform.deepseek.com/))
- (Optional) Google Perspective API key

### Installation

```bash
git clone https://github.com/KamyarZeinalipour/MART.git
cd MART
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Set API Keys

```bash
export DEEPSEEK_API_KEY="your-key-here"
export PERSPECTIVE_API_KEY="your-key-here"  # optional
```

### Run Experiments

```bash
# Quick demo (3 samples, 2 styles, 3 iterations)
python main.py --run demo

# Full experiment (all 10 samples, all 6 styles)
python main.py --run full --max-iter 5

# Advanced attacks
python run_advanced.py --attacks all --max-samples 3

# Verify toxicity is preserved (not secretly detoxified)
python verify_results.py --traces results/mart_traces.json
```

### CLI Options

```bash
python main.py --help
python run_advanced.py --help

# Run specific advanced attacks
python run_advanced.py --attacks genetic multiturn
python run_advanced.py --attacks selfplay token
```

## 📁 Project Structure

```
MART/
├── main.py                    # CLI for core MART experiments
├── run_advanced.py            # Runner for advanced attacks
├── verify_results.py          # Toxicity preservation verifier
├── requirements.txt
├── REPORT.md                  # Comprehensive experiment report
├── mart/                      # Framework package
│   ├── config.py              # Configuration dataclasses
│   ├── agents.py              # Attacker, Judge, Refiner
│   ├── filters.py             # 4 safety filters + Sentinel
│   ├── pipeline.py            # MART orchestration loop
│   ├── metrics.py             # Evaluation metrics (ASR, MAAF, SI, CFT...)
│   ├── verifier.py            # 3-strategy toxicity verification
│   ├── genetic_attack.py      # Genetic algorithm attack
│   ├── multiturn_attack.py    # Multi-turn conversation poisoning
│   ├── token_attack.py        # Adversarial token manipulation
│   └── self_play.py           # Contrastive self-play
└── data/
    └── sample_toxic.json      # Sample dataset (10 examples)
```

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| **ASR** | Attack Success Rate — % of rewrites evading all filters |
| **MAAF** | Multi-Agent Amplification Factor — MART vs single-agent improvement |
| **SI** | Stealth Index — combined quality × evasion score |
| **FDR** | Filter Disagreement Rate — how often filters disagree |
| **CFT** | Cross-Filter Transferability — evasion transfer between filters |
| **SPS** | Semantic Preservation Score — meaning retention |

## ⚠️ Ethical Statement

This framework is designed for **defensive research purposes** — identifying vulnerabilities in content moderation systems so they can be strengthened. The toxic examples used are synthetic or sourced from established research datasets. We do not condone the use of these techniques to bypass safety systems in production.

## 📝 Citation

```bibtex
@article{mart2026,
  title={MART: Multi-Agent Red Teaming for Auditing LLM Safety Filters},
  author={Zeinalipour, Kamyar},
  year={2026}
}
```

## 📄 License

MIT License
