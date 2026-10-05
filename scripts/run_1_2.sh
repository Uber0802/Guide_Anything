#!/usr/bin/env bash
# Checklist 1.2: five seeds of warm-started, motor-frozen PPO, then the press threshold of each.
# Pass: at least 4 of 5 seeds within 0.05 of the analytic threshold.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES OMP_NUM_THREADS=8
PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2 3 4"}
TAG=${TAG:-_frozen}
for s in $SEEDS; do
  $PY scripts/train_ppo.py --headless --seed "$s" --num_envs 1024 --iterations 300 --warm_start_rounds 6 \
    --gamma 0.998 --lam 0.99 --entropy_coef ${ENTROPY:-0.003} --lr 1e-4 --schedule fixed --freeze_motor \
    --warm_std 0.005 0.005 0.02 0.3 --tag "$TAG" > "outputs/train${TAG}_s$s.log" 2>&1 &
done
wait
for s in $SEEDS; do
  run=$(ls -d logs/ppo/*_calibrated${TAG}_s"$s"/ | tail -1)
  ckpt=$(ls "$run"model_*.pt | sort -V | tail -1)
  for mode in sampled mean; do
    flag=$([ "$mode" = sampled ] && echo --stochastic)
    log="outputs/eval${TAG}_${mode}_s$s.log"
    $PY scripts/eval_decision.py --headless $flag --checkpoint "$ckpt" > "$log" 2>&1
    echo "seed $s $mode: $(grep -E '^1.2 ' "$log") | $(grep -E '^  return' "$log")"
  done
done
