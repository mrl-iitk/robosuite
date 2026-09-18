"""
Collects demonstrations of the scripted PyramidStack policy (see scripted_pyramid_policy.py)
into the standard robosuite hdf5 dataset format -- the same "data/demo_N -> {states, actions,
model_file}" layout produced by collect_human_demonstrations.py / collect_scripted_stack_demos.py,
and consumed by robosuite/scripts/playback_demonstrations_from_hdf5.py or (outside this repo)
robomimic's dataset_states_to_obs.py, which replays states through the sim to add observations
(including camera images) on top.

Relevant data to collect for this task: since states+actions are enough to replay/re-render an
episode (robomimic's tooling regenerates observations from state, per the module docstring
above), the only real design decision is *what to expose as observables* so a downstream
`dataset_states_to_obs.py` pass has something useful to attach at each timestep. PyramidStack's
`_setup_observables` already provides, per cube: its world pose (`cube_i_pos`/`cube_i_quat`),
its vector to its fixed pyramid goal slot (`cube_i_to_target` -- directly useful as a per-cube
goal-conditioning signal, since unlike Stack's 2 cubes, the goal here isn't "wherever the other
cube ended up" but a fixed slot known in advance), and gripper-to-cube distance per arm (grasp
affordance signal). It also adds one scalar not present in Stack/CubeRow, `num_cubes_placed`
(how many of the 10 cubes are currently correctly placed and resting) -- useful downstream as a
cheap progress/staging signal (e.g. for curriculum learning or as an auxiliary prediction target)
given how much longer and more failure-prone a 10-cube build is than a 2-cube stack.

Only successful episodes (all 10 cubes end up built into the pyramid) are kept, exactly as in
collect_human_demonstrations.py's gather_demonstrations_as_hdf5. See _robomimic_hdf5_utils.py for
why `finalize_for_robomimic` is called on the result.

Speed note: a 10-cube pyramid episode takes ~35-45s of *wall-clock* time regardless (that's the
simulated motion itself, deliberately unhurried -- see scripted_pyramid_policy.py's --speed-scale)
but episodes are independent of each other, and MuJoCo simulates a single env on a single CPU
core. --num-workers > 1 runs that many episodes at once in separate processes (each with its own
env instance), which is a wall-clock-only speedup for collecting a *batch* -- it changes nothing
about any individual episode's simulated motion, duration, or precision. Each worker writes to
its own temp directory (DataCollectionWrapper timestamps episode folders, so concurrent writers
into one shared directory could theoretically collide); after all workers finish, their episode
folders are merged into one directory before the single `gather_demonstrations_as_hdf5` call, so
the output is one combined demo.hdf5 either way, identical in format to the --num-workers 1 case.

Usage:
    python robosuite/scripts/collect_scripted_pyramid_demos.py --num-episodes 20 --directory pyramid_datasets --num-workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
import time

import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.scripts._robomimic_hdf5_utils import finalize_for_robomimic
from robosuite.scripts.collect_human_demonstrations import gather_demonstrations_as_hdf5
from robosuite.scripts.scripted_pyramid_policy import run_episode
from robosuite.wrappers import DataCollectionWrapper


def _format_duration(seconds):
    """Formats a duration in seconds as e.g. "45s" or "3m 12s", for ETA printing."""
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    if minutes == 0:
        return f"{seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours == 0:
        return f"{minutes}m {seconds}s"
    return f"{hours}h {minutes}m"


def _collect_in_worker(
    args, config, num_episodes, worker_idx, num_workers, episode_num_offset, total_episodes,
    shared_progress, progress_lock, batch_start_time,
):
    """Runs `num_episodes` episodes in a fresh env + DataCollectionWrapper, entirely within this
    process (must build its own env rather than receiving one from the parent -- environments
    aren't picklable/fork-shareable across a process boundary in general). Returns
    (episode_folder_paths, num_successful) rather than writing straight to the shared output dir,
    so the parent can merge every worker's episodes into one directory before the single
    gather_demonstrations_as_hdf5 call that produces the final combined demo.hdf5.

    Prints progress throughout (not gated by --verbose): a line per cube as it's placed (from
    `run_episode`'s `progress_prefix`), plus a line when each episode finishes that also reports
    the running success count/rate across *all* workers (via `shared_progress`, a
    multiprocessing.Manager dict, updated under `progress_lock` so concurrent workers don't race
    each other) and an ETA for the whole batch, extrapolated from the batch's actual throughput
    so far (`completed_episodes / elapsed_wall_time`) -- this naturally accounts for
    --num-workers > 1 running episodes concurrently, not just per-worker episode time. With
    --num-workers > 1 these interleave across processes -- each line is tagged with which worker
    and which (global) episode number it's from, so the interleaving stays legible rather than
    looking like garbled output.
    """
    np.random.seed(worker_idx if num_workers > 1 else 0)
    env = suite.make(
        **config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        reward_shaping=False,  # the scripted policy never reads the reward; shaping it is wasted
        control_freq=20,       # per-step computation, not a shortcut on the actual motion
        horizon=args.horizon,
    )
    tmp_directory = tempfile.mkdtemp()
    env = DataCollectionWrapper(env, tmp_directory)

    worker_tag = f"[worker {worker_idx + 1}/{num_workers}] " if num_workers > 1 else ""
    num_successful = 0
    for i in range(num_episodes):
        global_ep_num = episode_num_offset + i + 1
        prefix = f"{worker_tag}episode {global_ep_num}/{total_episodes}"
        env.reset()
        t0 = time.time()
        success = run_episode(env, args, verbose=args.verbose, progress_prefix=prefix)
        num_successful += int(success)

        with progress_lock:
            shared_progress["completed"] += 1
            shared_progress["successful"] += int(success)
            completed = shared_progress["completed"]
            successful = shared_progress["successful"]

        elapsed = time.time() - batch_start_time
        remaining = total_episodes - completed
        eta = _format_duration(elapsed / completed * remaining) if completed and remaining else "0s"
        print(
            f"{prefix}: {'success' if success else 'incomplete'} ({time.time() - t0:.1f}s) | "
            f"overall: {successful}/{completed} successful so far "
            f"({100.0 * successful / completed:.0f}%), {completed}/{total_episodes} episodes done, "
            f"ETA {eta}",
            flush=True,
        )
    env.close()

    episode_dirs = [os.path.join(tmp_directory, d) for d in os.listdir(tmp_directory)]
    return episode_dirs, num_successful


def main(args):
    controller_config = load_composite_controller_config(controller=None, robot=args.robot)
    config = {
        "env_name": "PyramidStack",
        "robots": args.robot,
        "controller_configs": controller_config,
    }
    env_info = json.dumps(config)
    os.makedirs(args.directory, exist_ok=True)

    if args.render and args.num_workers > 1:
        print("--render forces --num-workers 1 (can't render multiple parallel envs to one viewer).")
    num_workers = 1 if args.render else max(1, min(args.num_workers, args.num_episodes))

    # shared, cross-process running tally (successful/completed episode counts) and a fixed batch
    # start time, so every worker's per-episode print line can report the *overall* batch's
    # progress and ETA, not just its own -- a multiprocessing.Manager is used even in the
    # num_workers==1 case so this code path doesn't need to branch (overhead is negligible next
    # to a single ~40s episode).
    manager = mp.Manager()
    shared_progress = manager.dict(completed=0, successful=0)
    progress_lock = manager.Lock()
    batch_start_time = time.time()

    if num_workers == 1:
        # single-process path -- identical to running _collect_in_worker directly, no pool
        # overhead, and easiest to reach for with --render (workers can't render to a shared
        # on-screen viewer anyway).
        episode_dirs, num_successful = _collect_in_worker(
            args, config, args.num_episodes, worker_idx=0, num_workers=1,
            episode_num_offset=0, total_episodes=args.num_episodes,
            shared_progress=shared_progress, progress_lock=progress_lock, batch_start_time=batch_start_time,
        )
        merged_directory = os.path.dirname(episode_dirs[0]) if episode_dirs else tempfile.mkdtemp()
    else:
        # split episodes as evenly as possible across workers (e.g. 20 episodes / 8 workers ->
        # sizes [3,3,3,3,2,2,2,2]), each running in its own process/env/DataCollectionWrapper.
        base, extra = divmod(args.num_episodes, num_workers)
        per_worker = [base + (1 if i < extra else 0) for i in range(num_workers)]
        per_worker = [n for n in per_worker if n > 0]
        offsets = np.cumsum([0] + per_worker[:-1])
        print(f"Collecting {args.num_episodes} episodes across {len(per_worker)} parallel workers: {per_worker}", flush=True)

        with mp.get_context("spawn").Pool(len(per_worker)) as pool:
            results = pool.starmap(
                _collect_in_worker,
                [
                    (
                        args, config, n, worker_idx, len(per_worker), int(offsets[worker_idx]), args.num_episodes,
                        shared_progress, progress_lock, batch_start_time,
                    )
                    for worker_idx, n in enumerate(per_worker)
                ],
            )

        num_successful = sum(n for _, n in results)
        # merge every worker's episode folders into one directory before the single
        # gather_demonstrations_as_hdf5 pass -- each folder name is already unique (per-worker
        # temp directories, timestamp-based names), so a plain move is enough.
        merged_directory = tempfile.mkdtemp()
        for episode_dirs, _ in results:
            for ep_dir in episode_dirs:
                shutil.move(ep_dir, os.path.join(merged_directory, os.path.basename(ep_dir)))

    success_rate = 100.0 * num_successful / args.num_episodes if args.num_episodes else 0.0
    print(f"{num_successful}/{args.num_episodes} episodes were successful ({success_rate:.1f}%) and will be saved.")
    gather_demonstrations_as_hdf5(merged_directory, args.directory, env_info)
    finalize_for_robomimic(os.path.join(args.directory, "demo.hdf5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--directory", type=str, default="stack_datasets", help="Where to write the combined demo.hdf5.")
    parser.add_argument("--num-workers", type=int, default=1, help="Collect this many episodes in parallel (separate processes/envs). Only speeds up collecting a *batch* -- each individual episode's simulated motion/duration is unchanged. Ignored (forced to 1) if --render is set.")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering while collecting (forces --num-workers to 1 -- can't render multiple parallel envs to one viewer).")
    parser.add_argument("--verbose", action="store_true", help="Print per-cube grasp/placement progress for each episode.")
    parser.add_argument("--horizon", type=int, default=8000)
    parser.add_argument("--speed-scale", type=float, default=0.4, help="Fraction of max controller translation output used per step (lower = slower, more precise, less overshoot -- worth it for a 10-cube build).")
    parser.add_argument("--rot-speed-scale", type=float, default=0.4, help="Fraction of max controller rotation output used per step (see --speed-scale).")
    parser.add_argument("--pos-tolerance", type=float, default=0.005, help="Position tolerance (m) to consider a waypoint reached.")
    parser.add_argument("--yaw-tolerance", type=float, default=np.deg2rad(3.0), help="Yaw tolerance (rad) to consider a waypoint reached.")
    parser.add_argument("--hold-steps", type=int, default=5, help="Consecutive in-tolerance steps required before advancing to the next waypoint.")
    parser.add_argument("--max-phase-steps", type=int, default=300, help="Max steps allowed per movement phase before giving up and moving on (higher than Stack's 150: --speed-scale 0.4 plus a wide spawn region means farther waypoints take longer to converge).")
    parser.add_argument("--align-phase-steps", type=int, default=500, help="Max steps allowed to laterally align above a cube/goal slot before descending -- generous on purpose, since an unconverged xy alignment turns the following vertical descent into a diagonal sweep that can clip the cube being grasped or an already-placed neighbor.")
    parser.add_argument("--grasp-steps", type=int, default=20, help="Steps spent closing the gripper before checking/using the grasp.")
    parser.add_argument("--release-steps", type=int, default=10, help="Steps spent opening the gripper to release a cube.")
    args = parser.parse_args()

    main(args)
