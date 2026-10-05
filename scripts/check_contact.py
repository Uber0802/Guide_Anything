"""Press into the latch in both worlds and record force against depth.

Checks that
  1. the wrist force reads the latch force (sign, frame, magnitude);
  2. the two worlds give identical curves until the FLOOR part breaks;
  3. LATCH releases near its release force and FLOOR breaks near its damage force.

Writes outputs/contact_curve.csv and outputs/contact_curve.png.

    python scripts/check_contact.py --headless
"""

import argparse
from pathlib import Path

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--press_force", type=float, default=20.0)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--out", type=Path, default=Path("outputs"))
args, app = launch(parser)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg
from guide_anything.tasks.peg_insert.latch import FLOOR, LATCH

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = 2
cfg.reset.tip_xy_noise = 0.0
cfg.episode_length_s = 1e3  # never time out inside this script
cfg.terminate_on_damage = False  # keep pressing the FLOOR part after it breaks
env = PegInsertEnv(cfg)
env.reset()
ids = torch.arange(2, device=env.device)
env.latch.reset(ids, torch.tensor([LATCH, FLOOR], device=env.device))
ctrl = ScriptedController(env)

rows = []
for step in range(args.steps):
    env.step(ctrl.blind_press(stop_force=args.press_force))
    pen = (env.latch_z - env.tip_pos_w[:, 2]) * 1e3
    for i, name in enumerate(["latch", "floor"]):
        rows.append(
            (
                step,
                name,
                pen[i].item(),
                env.wrist_force_w[i, 2].item(),
                env.latch.force[i].item(),
                bool(env.latch.released[i]),
                bool(env.latch.damaged[i]),
            )
        )

args.out.mkdir(parents=True, exist_ok=True)
header = "step,world,penetration_mm,wrist_fz,latch_force,released,damaged"
with open(args.out / "contact_curve.csv", "w") as f:
    f.write(header + "\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n")

latch = [r for r in rows if r[1] == "latch"]
floor = [r for r in rows if r[1] == "floor"]
engaged = [r for r in rows if r[4] > 0.5]
gap = max(abs(r[3] - r[4]) for r in engaged) if engaged else float("nan")
alias = [abs(a[3] - b[3]) for a, b in zip(latch, floor) if not b[6]]
print(f"steps run: {len(latch)}")
print(f"max |wrist_fz - latch_force| while engaged: {gap:.3f} N")
print(f"max |wrist_fz(latch) - wrist_fz(floor)| before damage: {max(alias):.4f} N")
rel = next((r for r in latch if r[5]), None)
dmg = next((r for r in floor if r[6]), None)
print(f"LATCH released: {rel is not None}" + (f" at step {rel[0]}" if rel else ""))
print(f"FLOOR damaged: {dmg is not None}" + (f" at step {dmg[0]}" if dmg else ""))
bottom_mm = cfg.geometry.latch_height * 1e3
print(f"LATCH final penetration {latch[-1][2]:.1f} mm (hole bottom is {bottom_mm:.0f} mm below the latch)")

fig, ax = plt.subplots(figsize=(6, 4))
for name, data in [("LATCH", latch), ("FLOOR", floor)]:
    d = np.array([(r[2], r[3]) for r in data])
    ax.plot(d[:, 0], d[:, 1], label=name)
ax.axhline(cfg.latch.damage_force, ls="--", c="gray", lw=0.8)
ax.axhline(cfg.latch.release_force, ls=":", c="gray", lw=0.8)
ax.set_xlabel("tip past latch plane (mm)")
ax.set_ylabel("wrist force z (N)")
ax.legend()
fig.tight_layout()
fig.savefig(args.out / "contact_curve.png", dpi=150)

env.close()
app.close()
