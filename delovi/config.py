"""Shared settings: paths, models, prices, request limits and run defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"  # questions/<split>.json and memories/<dataset>/<name>/
RUNS_DIR = ROOT / "runs"  # one folder per evaluation
CACHE_DIR = ROOT / "cache"  # LLM completions and built memory trees

# Decision model (Jev, through the TypeSafe SDK)
JEV_MODEL = "jev-1.13.0"
JEV_PRICE_PER_MTOK = 0.042  # USD per 1M input tokens; output is free
JEV_RPM, JEV_TPS = 1200, 250_000  # account rate limits; requests are paced at 90% / 80% of them
JEV_CHOICE_TOKENS = 29_000  # estimated tokens per ranking (Choice) request; Jev accepts 32k
JEV_MAX_CHOICES = 255  # candidates per ranking request
JEV_SCORE_MAX_QUESTIONS = 64  # score questions per request
JEV_SCORE_TOKENS = 48_000  # estimated tokens per score request

# LLMs (OpenAI Responses API)
QA_MODEL = "gpt-6-luna"  # answers every question; fixed for all policies
QA_REASONING = "medium"  # the model's default effort
JUDGE_MODEL = "gpt-5"  # grades open-ended answers
PRICES = {  # USD per 1M tokens: (input, cached input, output incl. reasoning)
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-6.1-sol": (2.0, 0.10, 10.0),
}

# Evaluation defaults (run speed only; results do not depend on them)
PARALLEL = 20  # questions answered at once
CONCURRENCY = 32  # Jev requests and QA calls in flight
BUILD_CONCURRENCY = 96  # memory-building LLM calls in flight
