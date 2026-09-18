"""
Collects demonstrations of the scripted Wipe policy (see scripted_wipe_policy.py) into the
standard robosuite hdf5 dataset format -- the same "data/demo_N -> {states, actions, model_file}"
layout produced by collect_human_demonstrations.py, and consumed by
robosuite/scripts/playback_demonstrations_from_hdf5.py or (outside this repo) robomimic's
dataset_states_to_obs.py, which replays states through the sim to add observations (including
camera images) on top.

Only successful episodes (env._check_success() at some point during the rollout) are kept,
exactly as in collect_human_demonstrations.py's gather_demonstrations_as_hdf5.

gather_demonstrations_as_hdf5 only writes the raw robosuite attribute `env_info` on the `data`
group. robomimic's own tooling (get_dataset_info.py, dataset_states_to_obs.py,
playback_dataset.py, ...) instead reads an `env_args` attribute (plus `total`/`num_samples`),
which is normally added by a separate manual pass through robomimic's
scripts/conversion/convert_robosuite.py. `finalize_for_robomimic` (see
_robomimic_hdf5_utils.py) replicates that conversion in-place so the hdf5 this script
writes is directly usable by robomimic without that extra step.

Usage:
    python robosuite/scripts/collect_scripted_wipe_demos.py --num-episodes 20 --directory wipe_datasets
"""

import argparse
import json
import os
import tempfile

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts._robomimic_hdf5_utils import finalize_for_robomimic
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.scripts.scripted_wipe_policy import run_episode
from robosuite.wrappers import DataCollectionWrapper


def main(args):
    controller_config = load_composite_controller_config(controller=None, robot=args.robot)
    config = {
        "env_name": "Wipe",
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
    parser.add_argument("--directory", type=str, default="wipe_datasets", help="Where to write the combined demo.hdf5.")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering while collecting.")
    parser.add_argument("--verbose", action="store_true", help="Print per-step policy progress for each episode.")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--z-bias", type=float, default=0.012, help="Downward offset (m) below marker height to press into the table."
    )
    parser.add_argument("--speed-scale", type=float, default=0.6, help="Fraction of max controller translation output used per step.")
    parser.add_argument("--rot-speed-scale", type=float, default=0.6, help="Fraction of max controller rotation output used per step.")
    parser.add_argument(
        "--stall-limit", type=int, default=100, help="Steps without switching target before giving up on a marker."
    )
    parser.add_argument("--return-steps", type=int, default=200, help="Max steps allowed to return to the initial position after wiping.")
    parser.add_argument("--return-tolerance", type=float, default=0.005, help="Position tolerance (m) for the return-to-start phase.")
    args = parser.parse_args()

    main(args)
