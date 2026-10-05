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
  app.py                     starts Isaac Sim
  tasks/peg_insert/
    latch.py                 virtual latch, pure torch
    env_cfg.py               geometry, control, reset, reward, latch parameters
    env.py                   PegInsertEnv (DirectRLEnv), gym id GuideAnything-PegInsert-v0
scripts/
  solve_home_pose.py         start joint angles for the current geometry
  check_contact.py           force-vs-depth in both worlds -> outputs/contact_curve.{csv,png}
  scripted_insert.py         oracle success per world and throughput
  aliasing_certificate.py    can a classifier tell the worlds apart before the part breaks?
tests/                       latch and classifier unit tests
```

## Environment

- Franka Panda, square 8 mm peg rigidly attached to the hand, socket from four
  kinematic walls, 0.5 mm clearance, 25 mm deep. The ground is the hole bottom.
- Wrist F/T is the incoming joint wrench of the hand link, rotated to world,
  sign flipped to "force the environment exerts on the tool".
- Action: 3-D offset of the tip target from the current tip, scaled by 5 cm.
  In contact, action z of `a` presses with `400 N/m * 0.05 m * a` = up to 20 N.
  Orientation is held pointing down.
- Observation (12, +2 with `reveal_world`): tip minus observed hole, tip
  velocity, wrist force / 10, previous action, [one-hot world].
- Latch 12 mm above the bottom. LATCH releases at 15 N and the peg slides to
  the bottom. FLOOR is a hard stop that breaks above 10 N and ends the episode.
- Success: tip within 2.5 mm of the hole axis and within 2 mm of the goal
  depth (bottom for LATCH, the stop for FLOOR), not broken.
- 120 Hz physics, 15 Hz policy, 10 s episodes.

## Status (2026-10-05, L40S)

| Check | Result |
|---|---|
| Unit tests (latch, classifiers) | 12/12 pass |
| Wrist F/T vs known external force (10 N z, 5 N x, 5 N y) | 9.97, 4.92, 5.08 N |
| Wrist F/T vs latch force while engaged | max error 0.027 N |
| Both worlds, same blind press, before damage | max difference 0.05 N |
| FLOOR breaks / LATCH releases | 10.4 N / 15.0 N |
| Oracle success, z revealed (2048 episodes) | LATCH 1.000, FLOOR 1.000, no damage |
| Throughput, oracle controller | 1.7k / 6.6k / 22.8k env steps/s at 256 / 1024 / 4096 envs |
| Aliasing certificate, blind press held at 8 N, 2500 episodes, held-out 1250 | logistic 0.516, MLP 0.502 (chance s.e. 0.014): PASS |
| Same, 5 train/test splits | logistic mean 0.497 (0.475 to 0.517), MLP mean 0.497 (0.483 to 0.518) |
| Positive control, same pipeline pressed to 18 N | logistic 1.000, MLP 1.000 |

The certificate only covers the histories it saw: blind press, ramp 0.5 N per
step, hold at 8 N. Below 15 N the two worlds share one force law, so other
safe probes should not separate them either, but that is by construction, not
measured.

Not done yet: an RL baseline and observation noise on the wrist force.
