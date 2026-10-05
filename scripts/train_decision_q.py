"""Learn the press-or-stop decision with a Q-learner over a fixed motor (checklist 1.2, non-PPO arm).

Each round runs one episode per env. At first contact the decision is random
with probability epsilon, otherwise argmax Q; Q is refit on all data so far.
The motor is the scripted teacher, which the PPO pipeline's warm-started actor
imitates (both branches verified at 1.000 success).

    python scripts/train_decision_q.py --headless --seed 0
"""

import argparse
from pathlib import Path

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2, 0.1, 0.1])
parser.add_argument("--out", type=Path, default=Path("logs/decision_q"))
args, app = launch(parser)

import torch

from guide_anything.rl.decision_q import PRESS, DecisionQ
from guide_anything.scripted import ScriptedController
from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.belief.mode = "calibrated"
env = PegInsertEnv(cfg)
ctrl = ScriptedController(env)
torch.manual_seed(args.seed)
q = DecisionQ(cfg.observation_space).to(env.device)
rew = cfg.reward


def run_round(epsilon: float):
    """One episode per env. Returns the observation at first contact, the action, and the outcome."""
    obs, _ = env.reset()
    n = env.num_envs
    decided = torch.zeros(n, dtype=torch.bool, device=env.device)
    contact_obs = torch.zeros(n, cfg.observation_space, device=env.device)
    press = torch.zeros(n, dtype=torch.bool, device=env.device)
    while True:
        o = obs["policy"]
        # Mirror the env: it latches the commit on the first step the wrist reads contact.
        first = (env.wrist_force_w[:, 2] > cfg.control.commit_contact_force) & ~decided
        if first.any():
            with torch.no_grad():
                greedy = q.press(o[first])
            explore = torch.rand(int(first.sum()), device=env.device) < epsilon
            coin = torch.rand(int(first.sum()), device=env.device) < 0.5
            press[first] = torch.where(explore, coin, greedy)
            contact_obs[first] = o[first]
            decided |= first
        action = ctrl.by_belief(threshold=float("inf"))  # motor only; the commit comes from the decision
        action[:, 3] = torch.where(press, 1.0, -1.0)
        obs, _, terminated, _, _ = env.step(action)
        if terminated.any():
            break
    outcome = rew.success * env.final_success.float() - rew.damage * env.final_damaged.float()
    return contact_obs[decided], press[decided].long() * PRESS, outcome[decided], env.final_belief[decided]


data_o, data_a, data_r = [], [], []
for r, eps in enumerate(args.epsilons):
    o, a, out, belief = run_round(eps)
    data_o.append(o)
    data_a.append(a)
    data_r.append(out)
    q.fit(torch.cat(data_o), torch.cat(data_a), torch.cat(data_r))
    print(
        f"[round {r}] epsilon {eps:.2f} | decided {len(o)} | pressed {a.float().mean():.3f}"
        f" | return {out.mean():.3f} | samples {sum(len(x) for x in data_o)}"
    )

args.out.mkdir(parents=True, exist_ok=True)
path = args.out / f"decision_q_s{args.seed}.pt"
torch.save(q.state_dict(), path)
print(f"saved {path}")

env.close()
app.close()
