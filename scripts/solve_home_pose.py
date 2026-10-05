"""Drive the tip to its start pose with the env's own controller and print the joint angles.

Paste the printed values into PegInsertEnvCfg.robot.init_state.joint_pos.

    python scripts/solve_home_pose.py --headless
"""

import argparse

from guide_anything.app import launch

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
args, app = launch(parser)


from guide_anything.tasks.peg_insert.env import PegInsertEnv
from guide_anything.tasks.peg_insert.env_cfg import PegInsertEnvCfg

cfg = PegInsertEnvCfg()
cfg.scene.num_envs = 1
cfg.reset.tip_xy_noise = 0.0
cfg.episode_length_s = 1e3
env = PegInsertEnv(cfg)
env.reset()

target = env.hole_pos_w.clone()
target[:, 2] += cfg.geometry.hole_depth + cfg.reset.tip_height
for _ in range(args.steps):
    action = ((target - env.tip_pos_w) / cfg.control.action_scale).clamp(-1, 1)
    env.step(action)

err_mm = (env.tip_pos_w - target).norm().item() * 1e3
q = env._robot.data.joint_pos[0]
names = env._robot.joint_names
print(f"tip error {err_mm:.2f} mm, hand quat {env.hand_pose_w[1][0].tolist()}")
print("joint_pos={")
for name, value in zip(names, q.tolist()):
    print(f'    "{name}": {value:.4f},')
print("}")

env.close()
app.close()
