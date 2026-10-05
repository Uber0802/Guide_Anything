"""Aliasing certificate for V-alias (checklist item 1.4).

Half the episodes are LATCH, half FLOOR. A blind controller that does not know
the world presses on contact, ramps the force, and holds at ``--stop_force``.
Held-out classifiers then try to recover the world from the full observation
history the policy would see. The certificate passes when every classifier
stays within ``--tol`` of chance.

A positive control runs the same pipeline with a stop force past the latch
release, where the worlds do differ, to show the classifiers can find a leak.

    python scripts/aliasing_certificate.py --headless
"""

import argparse
from pathlib import Path

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=2500, help="split evenly between the worlds")
parser.add_argument("--steps", type=int, default=110, help="policy steps recorded per episode")
parser.add_argument("--stop_force", type=float, default=8.0, help="N, below the damage force")
parser.add_argument("--control_force", type=float, default=18.0, help="N, past the latch release")
parser.add_argument("--tol", type=float, default=0.02)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("outputs"))
args, app = launch(parser)

import torch

from guide_anything.certificate import holdout_accuracy
from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg
from guide_anything.tasks.peg_insert.latch import FLOOR, LATCH

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = args.episodes
cfg.seed = args.seed
cfg.episode_length_s = 1e3  # one episode per env, never time out
cfg.belief.mode = "prior"  # the policy observation must carry nothing about the world
env = PegInsertEnv(cfg)
ctrl = ScriptedController(env)
worlds = (torch.arange(args.episodes, device=env.device) >= args.episodes // 2).long()  # 0 = LATCH, 1 = FLOOR
assert LATCH == 0 and FLOOR == 1


def record(stop_force: float) -> dict:
    """Run one episode in every env and return the observation histories."""
    obs, _ = env.reset()
    env.latch.reset(torch.arange(args.episodes, device=env.device), worlds)
    ctrl.reset()
    history = []
    for _ in range(args.steps):
        obs, _, _, _, _ = env.step(ctrl.blind_press(stop_force=stop_force))
        history.append(obs["policy"].clone())
    history = torch.stack(history, dim=1)  # (episodes, steps, obs_dim)
    force_z = history[..., 8] * 10.0  # observation layout: tip rel. hole, tip vel, wrist force / 10, action
    return {
        "obs": history,
        "world": worlds.clone(),
        "damaged": env.latch.damaged.clone(),
        "released": env.latch.released.clone(),
        "max_force": force_z.amax(1),
        "contact": (force_z > ctrl.contact_force).any(1),
    }


def evaluate(run: dict, name: str) -> list:
    x = run["obs"].flatten(1)
    print(f"\n[{name}] {len(x)} episodes, {run['obs'].shape[1]} steps x {run['obs'].shape[2]} features")
    print(
        f"  contact reached {run['contact'].float().mean():.3f}"
        f" | peak force {run['max_force'].min():.2f}..{run['max_force'].max():.2f} N"
        f" | damaged {run['damaged'].float().mean():.3f} | released {run['released'].float().mean():.3f}"
    )
    results = []
    for kind in ["logistic", "mlp"]:
        r = holdout_accuracy(x, run["world"], kind, seed=args.seed)
        print(f"  {kind:8s} held-out accuracy {r.accuracy:.4f}  (n_test {r.n_test}, chance s.e. {r.stderr:.4f})")
        results.append(r)
    return results


certificate = record(args.stop_force)
control = record(args.control_force)
args.out.mkdir(parents=True, exist_ok=True)
torch.save({"certificate": certificate, "control": control, "args": vars(args)}, args.out / "aliasing_certificate.pt")

cert_results = evaluate(certificate, f"certificate, stop at {args.stop_force} N")
control_results = evaluate(control, f"positive control, stop at {args.control_force} N")

safe = not certificate["damaged"].any() and not certificate["released"].any() and certificate["contact"].all()
aliased = all(r.within(args.tol) for r in cert_results)
detects = all(r.accuracy > 0.9 for r in control_results)
print("\nchecks")
print(f"  every certificate episode touched the latch, none broke or released: {safe}")
print(f"  classifiers at chance (|acc - 0.5| <= {args.tol}): {aliased}")
print(f"  classifiers find the positive-control leak (acc > 0.9): {detects}")
print(f"CERTIFICATE {'PASS' if safe and aliased and detects else 'FAIL'}")

env.close()
app.close()
