"""Press-or-stop decision against the belief, for a trained policy or the scripted reference.

One episode per env with p ~ U(0, 1) and the world drawn from p. Reports the
fitted press threshold (pass for checklist 1.2: within 0.05 of the analytic
value), the outcome return, and its gap to the Bayes decision with a perfect motor.

    python scripts/eval_decision.py --headless --checkpoint logs/ppo/<run>/model_299.pt [--stochastic]
    python scripts/eval_decision.py --headless --scripted 0.8
"""

import argparse
from pathlib import Path

from guide_anything.app import launch

parser = argparse.ArgumentParser()
source = parser.add_mutually_exclusive_group(required=True)
source.add_argument("--checkpoint", type=Path)
source.add_argument("--scripted", type=float, nargs="+", help="belief thresholds for the scripted controller")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--tol", type=float, default=0.05)
parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the policy mean")
args, app = launch(parser)

import torch

from guide_anything.decision import binned, fit_threshold, optimal_return, optimal_threshold
from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.belief.mode = "calibrated"
env = PegInsertEnv(cfg)
rew = cfg.reward
analytic = optimal_threshold(rew.success, rew.damage)


def run_episodes(act) -> dict:
    obs, _ = env.reset()
    for _ in range(int(env.max_episode_length)):
        obs, _, terminated, _, _ = env.step(act(obs))
        if terminated.any():
            break
    assert terminated.all(), "every episode should end on the same step"
    return {
        "belief": env.final_belief.clone(),
        "pressed": env.final_pressed.clone(),
        "success": env.final_success.clone(),
        "damaged": env.final_damaged.clone(),
        "tip_error": env.final_tip_error.clone(),
    }


def report(name: str, ep: dict):
    outcome = rew.success * ep["success"].float() - rew.damage * ep["damaged"].float()
    bayes = optimal_return(ep["belief"], rew.success, rew.damage)
    t, err = fit_threshold(ep["belief"], ep["pressed"])
    print(f"\n[{name}] {len(outcome)} episodes")
    print(f"  press threshold {t:.3f} (analytic {analytic:.3f}), episodes off the step {err:.3f}")
    print(
        f"  return {outcome.mean():.3f} | Bayes with perfect motor {bayes.mean():.3f}"
        f" | always stop {(1 - ep['belief']).mean():.3f}"
    )
    print(
        f"  success {ep['success'].float().mean():.3f} | damaged {ep['damaged'].float().mean():.3f}"
        f" | pressed {ep['pressed'].float().mean():.3f}"
    )
    for label, mask in [("pressed", ep["pressed"]), ("stopped", ~ep["pressed"])]:
        err = ep["tip_error"][mask] * 1e3
        if len(err):
            xy = err[:, :2].norm(dim=-1)
            print(
                f"  {label}: final tip minus goal, z median {err[:, 2].median():.1f} mm"
                f" (90% {err[:, 2].quantile(0.9):.1f}), xy median {xy.median():.1f} mm (90% {xy.quantile(0.9):.1f})"
            )
    print("  p bin   pressed  success  damaged  return  bayes   n")
    rows = zip(
        binned(ep["belief"], ep["pressed"]),
        binned(ep["belief"], ep["success"]),
        binned(ep["belief"], ep["damaged"]),
        binned(ep["belief"], outcome),
        binned(ep["belief"], bayes),
    )
    for (c, pr, n), (_, su, _), (_, da, _), (_, ret, _), (_, ba, _) in rows:
        print(f"  {c:.2f}    {pr:.3f}    {su:.3f}    {da:.3f}   {ret:6.3f}  {ba:6.3f}  {n}")
    return t


if args.checkpoint:
    from rsl_rl.runners import OnPolicyRunner

    from guide_anything.rl.rsl import RslRlEnv, ppo_config
    from guide_anything.rl.split import freeze_motor

    wrapped = RslRlEnv(env)
    runner = OnPolicyRunner(wrapped, ppo_config(0, gamma=rew.gamma), log_dir=None, device=env.device)
    saved = torch.load(args.checkpoint, map_location=env.device, weights_only=False)["model_state_dict"]
    if any(k.startswith("actor.motor.") for k in saved):
        freeze_motor(runner, decision_dims=[3])
    runner.load(str(args.checkpoint))
    model = runner.alg.policy
    model.eval()
    act = model.act if args.stochastic else model.act_inference
    mode = "sampled" if args.stochastic else "mean"
    with torch.inference_mode():
        t = report(f"PPO ({mode} actions) {args.checkpoint}", run_episodes(lambda obs: act(wrapped._wrap(obs))))
    print(f"\n1.2 {'PASS' if abs(t - analytic) <= args.tol else 'FAIL'}: |{t:.3f} - {analytic:.3f}| <= {args.tol}")
else:
    ctrl = ScriptedController(env)
    for threshold in args.scripted:
        report(f"scripted, press iff p > {threshold}", run_episodes(lambda obs, th=threshold: ctrl.by_belief(th)))

env.close()
app.close()
