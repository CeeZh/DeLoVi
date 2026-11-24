"""Hyper-parameters of the policy evolution (the setting that produced DeLoVi-E)."""

from delovi.config import ROOT

# Coding agent: Claude Code in non-interactive mode, the prompt on stdin
AGENT_MODEL = "claude-opus-5-5"
AGENT_EFFORT = "high"
AGENT_CMD = ("CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 claude -p --model {model} --effort {effort} "
             "--permission-mode bypassPermissions")

# Loop
ITERATIONS = 30
INIT_POLICY = ROOT / "hierarchical_search"  # the first parent
SELECT_SPLIT = "dev"  # every policy is evaluated on it; the agents read its logs
CONFIRM_SPLIT = "val"  # policies on the dev frontier are also evaluated on it; never shown to the agents

# Objective: accuracy, and input tokens per question = Jev tokens + CHECK_QA_WEIGHT x tokens of intermediate
# QA calls + FINAL_QA_WEIGHT x tokens of the final QA call
CHECK_QA_WEIGHT = 2.0
FINAL_QA_WEIGHT = 4.0
INIT_FLOOR_MARGIN = 0.03  # a policy more than this below the initial policy's dev accuracy is discarded

# Budgets (USD, nominal: cached calls count as if paid)
BUILD_BUDGET_USD = 20.0  # memory building of one evaluation
SEARCH_BUDGET_USD = 20.0  # Jev requests of one evaluation
PROBE_BUDGET_USD = 3.0  # the agent's live checks in one iteration (real spend)

# Evaluation
PARALLEL = 32  # questions answered at once
