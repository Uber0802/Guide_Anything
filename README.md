# GuideAnything

Guidance as belief for contact-rich manipulation. The first environment is
V-alias (checklist item 1.4 of the design discussion): square peg-in-hole
with a virtual latch whose two worlds give identical wrist F/T readings until
the part would break.

## Setup

```bash
conda create -p /weka/robots-default/poyiw/miniconda3/envs/guideanything --override-channels -c conda-forge python=3.11
conda activate guideanything
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install "setuptools<81" && pip install --no-build-isolation flatdict==4.0.1  # isaaclab dep needs pkg_resources
pip install "isaaclab[isaacsim,all]==2.3.2" --extra-index-url https://pypi.nvidia.com
pip install pytest matplotlib ruff
pip install -e .
```

Isaac Sim asks for the Omniverse EULA on first launch; set `OMNI_KIT_ACCEPT_EULA=YES` once you have accepted it.
On this 243-core machine set `OMP_NUM_THREADS=8`; torch's default thread count makes small CPU jobs about 10x slower.

## Layout

```
guide_anything/
  control.py                 task-space impedance with a dynamically consistent null space
  scripted.py                oracle and blind-press controllers (measuring instruments)
  certificate.py             held-out logistic / MLP classifiers for the certificate
  decision.py                press threshold and Bayes return for the press-or-stop decision
  app.py                     starts Isaac Sim
  rl/
    rsl.py                   rsl_rl VecEnv adapter and PPO config
    warm_start.py            DAgger warm start from a scripted teacher
    split.py                 freeze the motor outputs and normalizer, train only the commit output
    decision_q.py            Q-learner for the one-shot decision
  tasks/peg_insert/
    latch.py                 virtual latch, pure torch
    env_cfg.py               geometry, control, reset, reward, latch parameters
    env.py                   PegInsertEnv (DirectRLEnv), gym id GuideAnything-PegInsert-v0
scripts/
  solve_home_pose.py         start joint angles for the current geometry
  check_contact.py           force-vs-depth in both worlds -> outputs/contact_curve.{csv,png}
  scripted_insert.py         oracle success per world and throughput
  aliasing_certificate.py    can a classifier tell the worlds apart before the part breaks?
  train_ppo.py               belief-conditioned PPO, optional warm start and motor freeze
  eval_decision.py           press threshold, return, per-belief table for a policy or scripted reference
  run_1_2.sh                 checklist 1.2, PPO arm: five seeds, train then evaluate
  train_decision_q.py        Q-learner for the one-shot press-or-stop decision
  run_1_2_q.sh               checklist 1.2, Q-learner arm
tests/                       latch, classifier and decision unit tests
EXPERIMENTS.md               experiment log (Chinese), in order, failures included
```

## Environment

- Franka Panda, square 8 mm peg rigidly attached to the hand, socket from four
  kinematic walls, 1 mm clearance, 25 mm deep. The ground is the hole bottom.
- Wrist F/T is the incoming joint wrench of the hand link, rotated to world,
  sign flipped to "force the environment exerts on the tool".
- Action (4): 3-D offset of the tip target from the current tip, scaled by
  5 cm, plus a commit. Without commit the downward offset is clamped so the
  commanded force stays under 8 N. The commit is read once, when the wrist
  first reads contact; if set, the arm then presses with the full
  `400 N/m * 0.05 m` = 20 N.
  Orientation is held pointing down.
- Observation (14): tip minus observed hole, tip velocity, wrist force / 10,
  previous tip-offset action, belief p = P(LATCH), episode time. A separate
  `critic` group carries the world, damaged and released flags for an
  asymmetric critic; the policy never sees it.
- Belief modes: `calibrated` (p ~ U(0, 1), world drawn from p), `oracle`, `prior`.
- Latch 12 mm above the bottom. LATCH releases at 15 N and the peg slides to
  the bottom. FLOOR is a hard stop that breaks above 10 N.
- Success: tip within 2.5 mm of the hole axis and within 2 mm of the goal
  depth (bottom for LATCH, the stop for FLOOR), not broken.
- Reward: every episode runs the full 10 s; the outcome (+1 success, -3
  damage) is paid on the last step, plus potential-based shaping toward the
  hole entrance. Optimal decision: press iff p > 0.8.
- 120 Hz physics, 15 Hz policy.

## Status (2026-10-05, L40S)

| Check | Result |
|---|---|
| Unit tests (latch, classifiers, decision) | 17/17 pass |
| Wrist F/T vs known external force (10 N z, 5 N x, 5 N y) | 9.97, 4.92, 5.08 N |
| Wrist F/T vs latch force while engaged | max error 0.027 N |
| Both worlds, same blind press, before damage | max difference 0.05 N |
| FLOOR breaks / LATCH releases | 10.4 N / 15.0 N |
| Oracle success, z revealed (2048 episodes) | LATCH 1.000, FLOOR 1.000, no damage |
| Throughput, oracle controller | 1.7k / 6.6k / 22.8k env steps/s at 256 / 1024 / 4096 envs |
| Aliasing certificate, blind press held at 8 N, 2500 episodes, held-out 1250 | logistic 0.503, MLP 0.494 (chance s.e. 0.014): PASS |
| Positive control, same pipeline pressed to 18 N | logistic 1.000, MLP 1.000 |

The certificate only covers the histories it saw: blind press, ramp 0.5 N per
step, hold at 8 N. Below 15 N the two worlds share one force law, so other
safe probes should not separate them either, but that is by construction, not
measured.

## Checklist 1.2: does the learner find the press threshold?

Optimal: press iff p > 0.8 (Bayes return 0.60 with a perfect motor; scripted
reference 0.606). Five seeds each, pass = at least 4 within 0.05.

| Learner | Thresholds | Returns | Pass |
|---|---|---|---|
| PPO, warm-started motor frozen, commit latched at first contact (`run_1_2.sh`) | 0.812, 0.788, 0.941, 0.810, 0.773 | 0.543 to 0.597 | 4/5 |
| Q-learner over the one-shot decision (`run_1_2_q.sh`) | 0.819, 0.802, 0.788, 0.795, 0.772 | 0.597 to 0.606 | 5/5 |

The Q-learner uses about a tenth of the samples. PPO from scratch failed in
every setting tried; see `EXPERIMENTS.md` for the full sequence of failures
and the fixes they led to.

Not done yet: 1.3 with a probe action, 2.0 identification curves, 2.1 oracle
gap, V-force, observation noise on the wrist force.
