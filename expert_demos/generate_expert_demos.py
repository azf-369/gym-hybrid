import os
import sys

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import gymnasium.logger as gym_logger

if not hasattr(gym_logger, "set_level"):
    if hasattr(gym_logger, "setLevel"):
        gym_logger.set_level = gym_logger.setLevel
    else:
        gym_logger.set_level = lambda *_args, **_kwargs: None
if not hasattr(gym.logger, "set_level"):
    gym.logger.set_level = gym_logger.set_level

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import gym_hybrid


ENV_ID = "Moving-v0"
OUTPUT = os.path.join(os.path.dirname(__file__), "expert_demo.npz")
EPISODES = 1
MAX_STEPS = 200

TARGET_X, TARGET_Y = 0.5, 0.0
START_X, START_Y = -0.5, 0.0
START_THETA = 0.0


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def make_action(action_id: int, param_value: float, action_space):
    action = {"id": action_id}
    for key, space in action_space.spaces.items():
        if key == "id":
            continue
        if space.shape[0] == 0:
            action[key] = np.zeros((0,), dtype=space.dtype)
        else:
            action[key] = np.zeros(space.shape, dtype=space.dtype)
    param_key = f"params{action_id}"
    if param_key in action and action[param_key].shape[0] > 0:
        action[param_key][0] = param_value
    return action


def concat_params(action, action_space):
    params = []
    for key, space in action_space.spaces.items():
        if key == "id":
            continue
        params.append(action[key].astype(np.float32))
    return np.concatenate(params) if params else np.zeros((0,), dtype=np.float32)


def expert_policy(obs, target_radius):
    agent_x, agent_y = obs[0], obs[1]
    speed = obs[2]
    cos_theta, sin_theta = obs[3], obs[4]
    target_x, target_y = obs[5], obs[6]
    distance = obs[7]
    in_target = obs[8] > 0.5

    heading = np.arctan2(sin_theta, cos_theta)
    desired = np.arctan2(target_y - agent_y, target_x - agent_x)
    delta = normalize_angle(desired - heading)

    # If within the target region, always issue BREAK to come to rest and terminate
    if in_target or distance <= target_radius * 1.2:
        return 2, 0.0
    if abs(delta) > 0.15:
        return 1, float(np.clip(delta / (np.pi / 2), -1.0, 1.0))
    if distance > target_radius * 2.0:
        return 0, 1.0
    return 0, float(np.clip(distance * 2.0, 0.0, 0.3))


def run_episode(env):
    # random reset (env chooses start and target)
    try:
        env.reset()
    except TypeError:
        env.reset()

    # capture start and target for saving
    try:
        start = (float(env.unwrapped.agent.x), float(env.unwrapped.agent.y), float(env.unwrapped.agent.theta))
    except Exception:
        start = None
    try:
        target = (float(env.unwrapped.target.x), float(env.unwrapped.target.y))
    except Exception:
        target = None

    obs = env.unwrapped.get_state()
    obs_traj, ids, p0s, p1s, p2s, pcat = [], [], [], [], [], []
    done = False
    steps = 0
    success = False

    while not done and steps < MAX_STEPS:
        action_id, param = expert_policy(obs, env.unwrapped.target_radius)
        action = make_action(action_id, param, env.action_space)
        obs_traj.append(obs)
        ids.append(action_id)
        p0s.append(action["params0"].astype(np.float32))
        p1s.append(action["params1"].astype(np.float32))
        p2s.append(action["params2"].astype(np.float32))
        pcat.append(concat_params(action, env.action_space))

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if terminated and info.get("reward", 0) > 0:
            success = True
        steps += 1

    return success, obs_traj, ids, p0s, p1s, p2s, pcat, start, target


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", "-n", type=int, default=1, help="number of successful demos to collect")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(__file__), help="where to save demo files")
    parser.add_argument("--max-attempts", type=int, default=10, help="max attempts per demo")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    env = gym.make(ENV_ID)

    success = 0
    attempts = 0

    while success < args.episodes and attempts < args.episodes * args.max_attempts:
        ok, obs, ids, p0s, p1s, p2s, pcat, start, target = run_episode(env)
        attempts += 1
        if not ok:
            continue

        out_name = os.path.join(args.output_dir, f"expert_demo_{success}.npz")
        np.savez(
            out_name,
            obs=np.asarray(obs, dtype=np.float32),
            action_id=np.asarray(ids, dtype=np.int64),
            params0=np.asarray(p0s, dtype=np.float32),
            params1=np.asarray(p1s, dtype=np.float32),
            params2=np.asarray(p2s, dtype=np.float32),
            action_params=np.asarray(pcat, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32) if target is not None else np.asarray([], dtype=np.float32),
            start=np.asarray(start, dtype=np.float32) if start is not None else np.asarray([], dtype=np.float32),
        )
        success += 1

    env.close()

    if success < args.episodes:
        raise RuntimeError(f"Failed to collect {args.episodes} successful demo(s) after {attempts} attempts")


if __name__ == "__main__":
    main()
