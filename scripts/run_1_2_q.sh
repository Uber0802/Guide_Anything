#!/usr/bin/env bash
# Checklist 1.2, Q-learner arm: five seeds of DecisionQ over the scripted motor, then the press threshold.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=8
PY=${PY:-python}
for s in ${SEEDS:-0 1 2 3 4}; do
  $PY scripts/train_decision_q.py --headless --seed "$s" > "outputs/train_q_s$s.log" 2>&1
  $PY scripts/eval_decision.py --headless --decision_q "logs/decision_q/decision_q_s$s.pt" > "outputs/eval_q_s$s.log" 2>&1
  echo "seed $s: $(grep -E '^1.2 ' "outputs/eval_q_s$s.log") | $(grep -E '^  return' "outputs/eval_q_s$s.log")"
done
