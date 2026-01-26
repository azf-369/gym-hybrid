import os
import sys
import subprocess
import numpy as np

import gymnasium as gym
import gymnasium.logger as gym_logger

if not hasattr(gym_logger, "set_level"):
    if hasattr(gym_logger, "setLevel"):
        gym_logger.set_level = gym_logger.setLevel
    else:
        gym_logger.set_level = lambda *_args, **_kwargs: None
if not hasattr(gym.logger, "set_level"):
    gym.logger.set_level = gym_logger.set_level

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
import gym_hybrid  # registers Moving-v0

DEMOS = os.path.join(ROOT, "expert_demos")
GEN = os.path.join(DEMOS, "generate_expert_demos.py")


def generate_demos(num=2):
    for f in os.listdir(DEMOS):
        if f.startswith("expert_demo_") and f.endswith(".npz"):
            os.remove(os.path.join(DEMOS, f))
    subprocess.check_call([sys.executable, GEN, "--episodes", str(num)])
    files = sorted(
        [os.path.join(DEMOS, f) for f in os.listdir(DEMOS) if f.startswith("expert_demo_") and f.endswith(".npz")]
    )
    assert len(files) >= num
    return files


def test_demo_format_matches_action_space():
    files = generate_demos(num=2)
    env = gym.make("Moving-v0")

    # compute expected param concat size from action_space
    expected_param_dim = 0
    for key, space in env.action_space.spaces.items():
        if key == "id":
            continue
        expected_param_dim += int(space.shape[0])

    for path in files:
        data = np.load(path)
        obs = data["obs"]
        action_id = data["action_id"]
        action_params = data["action_params"]
        params0 = data["params0"]
        params1 = data["params1"]
        params2 = data["params2"]
        start = data["start"]
        target = data["target"]

        assert obs.ndim == 2
        assert action_id.ndim == 1
        assert action_params.ndim == 2
        assert obs.shape[0] == action_id.shape[0] == action_params.shape[0]
        assert action_params.shape[1] == expected_param_dim
        assert params0.shape[0] == action_id.shape[0]
        assert params1.shape[0] == action_id.shape[0]
        assert params2.shape[0] == action_id.shape[0]
        assert start.shape[0] == 3 and target.shape[0] == 2

        # validate a reconstructed action against action_space
        idx = 0
        action = {
            "id": int(action_id[idx]),
            "params0": np.asarray(params0[idx], dtype=np.float32),
            "params1": np.asarray(params1[idx], dtype=np.float32),
            "params2": np.asarray(params2[idx], dtype=np.float32),
        }
        assert env.action_space.contains(action)

    env.close()


if __name__ == "__main__":
    test_demo_format_matches_action_space()
