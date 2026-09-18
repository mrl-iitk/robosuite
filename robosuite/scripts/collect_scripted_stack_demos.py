"""
Collects demonstrations of the scripted Stack policy (see scripted_stack_policy.py) into
the standard robosuite hdf5 dataset format -- the same "data/demo_N -> {states, actions,
model_file}" layout produced by collect_human_demonstrations.py, and consumed by
robosuite/scripts/playback_demonstrations_from_hdf5.py or (outside this repo) robomimic's
dataset_states_to_obs.py, which replays states through the sim to add observations
(including camera images) on top.

Only successful episodes (cubeA ends up stacked on cubeB) are kept, exactly as in
collect_human_demonstrations.py's gather_demonstrations_as_hdf5. See
_robomimic_hdf5_utils.py for why `finalize_for_robomimic` is called on the result.

Usage:
    python robosuite/scripts/collect_scripted_stack_demos.py --num-episodes 20 --directory stack_datasets
"""

import argparse
import json
import os
import tempfile

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts._robomimic_hdf5_utils import finalize_for_robomimic
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.scripts.scripted_stack_policy import run_episode
from robosuite.wrappers import DataCollectionWrapper


def main(args):
    controller_config = load_composite_controller_config(controller=None, robot=args.robot)
    config = {
        "env_name": "Stack",
        "robots": args.robot,
        "controller_configs": controller_config,
    }

    env = suite.make(
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
        horizon=args.horizon,
    )
    env_info = json.dumps(config)

    tmp_directory = tempfile.mkdtemp()
    env = DataCollectionWrapper(env, tmp_directory)

    os.makedirs(args.directory, exist_ok=True)

    num_successful = 0
    for ep in range(args.num_episodes):
        env.reset()
        success = run_episode(env, args, verbose=args.verbose)
        num_successful += int(success)
        print(f"Episode {ep + 1}/{args.num_episodes}: {'success' if success else 'incomplete'}")

    env.close()

    print(f"{num_successful}/{args.num_episodes} episodes were successful and will be saved.")
    gather_demonstrations_as_hdf5(tmp_directory, args.directory, env_info)
    finalize_for_robomimic(os.path.join(args.directory, "demo.hdf5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--directory", type=str, default="stack_datasets", help="Where to write the combined demo.hdf5.")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering while collecting.")
    parser.add_argument("--verbose", action="store_true", help="Print per-cube grasp/success progress for each episode.")
    parser.add_argument("--horizon", type=int, default=1200)
    parser.add_argument("--speed-scale", type=float, default=0.6, help="Fraction of max controller translation output used per step.")
    parser.add_argument("--rot-speed-scale", type=float, default=0.6, help="Fraction of max controller rotation output used per step.")
    parser.add_argument("--pos-tolerance", type=float, default=0.005, help="Position tolerance (m) to consider a waypoint reached.")
    parser.add_argument("--yaw-tolerance", type=float, default=np.deg2rad(3.0), help="Yaw tolerance (rad) to consider a waypoint reached.")
    parser.add_argument("--hold-steps", type=int, default=5, help="Consecutive in-tolerance steps required before advancing to the next waypoint.")
    parser.add_argument("--max-phase-steps", type=int, default=150, help="Max steps allowed per movement phase before giving up and moving on.")
    parser.add_argument("--grasp-steps", type=int, default=20, help="Steps spent closing the gripper before checking/using the grasp.")
    parser.add_argument("--release-steps", type=int, default=10, help="Steps spent opening the gripper to release a cube.")
    args = parser.parse_args()

    main(args)
