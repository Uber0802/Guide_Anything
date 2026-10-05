"""Train a belief-conditioned PPO policy on the peg-insertion task (checklist item 1.2).

python scripts/train_ppo.py --headless --seed 0 --iterations 300
"""

import argparse
import time
from pathlib import Path

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=300)
parser.add_argument("--belief", default="calibrated", choices=["calibrated", "oracle", "prior"])
parser.add_argument("--init_noise_std", type=float, default=1.0, help="in action units; 1.0 is 20 N in contact")
parser.add_argument(
    "--warm_std",
    type=float,
    nargs=4,
    default=[0.005, 0.005, 0.02, 0.3],
    help="action std after warm start: x, y, z, commit. 0.005 in x or y is 0.25 mm; 0.02 in z is 0.4 N",
)
parser.add_argument("--gamma", type=float, default=0.99, help="also the shaping discount")
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--entropy_coef", type=float, default=0.005)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--desired_kl", type=float, default=0.01)
parser.add_argument("--schedule", default="adaptive", choices=["adaptive", "fixed"])
parser.add_argument("--warm_start_rounds", type=int, default=0, help="DAgger rounds imitating a never-press teacher")
parser.add_argument(
    "--warm_press_fraction", type=float, default=0.5, help="warm-start episodes executed with commit forced on"
)
parser.add_argument("--freeze_motor", action="store_true", help="after warm start, train only the commit output")
parser.add_argument("--tag", default="")
parser.add_argument("--log_root", type=Path, default=Path("logs/ppo"))
args, app = launch(parser)

from rsl_rl.runners import OnPolicyRunner

from guide_anything.rl.rsl import RslRlEnv, ppo_config
from guide_anything.rl.split import freeze_motor
from guide_anything.rl.warm_start import warm_start
from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.belief.mode = args.belief
cfg.reward.gamma = args.gamma
env = RslRlEnv(PegInsertEnv(cfg))

log_dir = args.log_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{args.belief}{args.tag}_s{args.seed}"
train_cfg = ppo_config(
    args.seed,
    gamma=args.gamma,
    lam=args.lam,
    init_noise_std=args.init_noise_std,
    entropy_coef=args.entropy_coef,
    learning_rate=args.lr,
    desired_kl=args.desired_kl,
    schedule=args.schedule,
)
runner = OnPolicyRunner(env, train_cfg, str(log_dir), device=env.device)
if args.warm_start_rounds:
    # The teacher never presses: it gives motor skill (approach, align, rest on the latch) and no decision.
    teacher = ScriptedController(env.env)
    warm_start(
        env,
        runner.alg.policy,
        lambda: teacher.by_belief(threshold=float("inf")),
        gamma=args.gamma,
        rounds=args.warm_start_rounds,
        action_std=args.warm_std,
        commit_dim=3,
        press_fraction=args.warm_press_fraction,
    )
    env.reset()
if args.freeze_motor:
    freeze_motor(runner, decision_dims=[3])
# Every episode must run the full horizon for the outcome reward to be decision-faithful.
runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
print(f"checkpoints in {log_dir}")

env.env.close()
app.close()
