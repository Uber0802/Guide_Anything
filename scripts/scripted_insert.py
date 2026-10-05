"""Motor check: success of the oracle controller in each world, and simulation throughput.

Checklist item 1.4 needs both worlds above 0.9 when the world is revealed.

    python scripts/scripted_insert.py --headless --num_envs 256 --episodes 1000
"""

import argparse
import time

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--episodes", type=int, default=1000)
args, app = launch(parser)

import torch

from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg
from guide_anything.tasks.peg_insert.latch import FLOOR, LATCH

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = args.num_envs
env = PegInsertEnv(cfg)
env.reset()
ctrl = ScriptedController(env)

outcomes = {LATCH: [], FLOOR: []}  # (success, damaged) per finished episode
steps, start = 0, time.perf_counter()
while sum(len(v) for v in outcomes.values()) < args.episodes:
    _, _, terminated, truncated, _ = env.step(ctrl.oracle())
    steps += 1
    for i in (terminated | truncated).nonzero().flatten().tolist():
        outcomes[env.final_world[i].item()].append((env.final_success[i].item(), env.final_damaged[i].item()))
elapsed = time.perf_counter() - start

for world, name in [(LATCH, "LATCH"), (FLOOR, "FLOOR")]:
    o = torch.tensor(outcomes[world], dtype=torch.float)
    n = len(o)
    succ, dmg = o.mean(0).tolist() if n else (float("nan"), float("nan"))
    se = (succ * (1 - succ) / max(n, 1)) ** 0.5
    print(f"{name}: episodes {n}  success {succ:.3f} +- {se:.3f}  damaged {dmg:.3f}")
fps = steps * args.num_envs / elapsed
print(f"throughput: {fps:.0f} env steps/s ({fps * cfg.decimation:.0f} physics steps/s), {elapsed:.0f} s wall")

env.close()
app.close()
