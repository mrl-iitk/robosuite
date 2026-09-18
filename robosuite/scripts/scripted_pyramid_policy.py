"""
Hand-coded (non-RL) policy that solves robosuite's PyramidStack task.

Generalizes `scripted_stack_policy.py`'s single `pick_and_place` routine (already
cube-agnostic -- it takes the cube's obs key, body id, geoms, and target as plain arguments)
from stacking 2 cubes to building the full 10-cube, 4-3-2-1 pyramid: cubes are picked up and
placed strictly in index order (0..9), which `PyramidStack` guarantees is always a valid
bottom-up build order (a cube's supports, if any, always have a lower index than the cube
itself). Reuses `move_to` / `hold` / `get_rot_action` / `get_reference_down_axis` /
`nearest_face_yaw` / `get_body_yaw` unmodified.

The one thing this script cannot reuse as-is is `scripted_stack_policy.pick_and_place`'s fixed
hover height (table height + 0.15m): that's comfortably above a 2-cube stack, but not above a
4-layer pyramid (apex is at ~table height + 0.16m for the default cube size), so transiting over
an already-built section of the pyramid at that height would clip it. This script instead hovers
at a height that clears the *tallest* layer the pyramid ever reaches, computed once from the env
(`table_offset[2] + num_layers * cube_size + margin`).

Usage:
    python robosuite/scripts/scripted_pyramid_policy.py --render
"""

import argparse
import time

import numpy as np

import robosuite as suite
from robosuite.scripts.scripted_stack_policy import get_reference_down_axis, hold, move_to

GRASP_HEIGHT_OFFSET = -0.005  # eef target z relative to cube center while grasping/placing
HOVER_MARGIN_ABOVE_PYRAMID = 0.07  # clearance (m) above the tallest layer while transiting
OPEN, CLOSE = -1.0, 1.0


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def nearest_face_yaw(yaw):
    """Folds an arbitrary yaw into the smallest rotation in [-pi/4, pi/4) that reaches an
    equivalent orientation for a box symmetric under 90-degree rotations about its vertical axis."""
    return ((yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4


def get_yaw_from_xmat(xmat_flat):
    R = xmat_flat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def get_body_yaw(env, body_id):
    return get_yaw_from_xmat(env.sim.data.body_xmat[body_id])


def pyramid_hover_height(env):
    """Height (world z) comfortably above every layer of the pyramid, including the apex, used
    as the transit altitude between cubes so the arm never drags a carried cube through an
    already-built section."""
    return env.table_offset[2] + len(env.layer_sizes) * env.cube_size + HOVER_MARGIN_ABOVE_PYRAMID


def _align_xy(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_xy, z, target_yaw, gripper_action, tol):
    """Moves to (target_xy, z), with a generous budget dedicated to actually nailing the xy
    component (not just "close enough within the shared step budget"). Always called *before* a
    vertical approach near an object -- see `_descend_vertically` for why."""
    return move_to(
        env, args, arm, pose_idx, grip_idx, obs, local_down_axis, np.array([*target_xy, z]), target_yaw,
        gripper_action, args.align_phase_steps, tol,
    )


def _descend_vertically(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, z, target_yaw, gripper_action, tol):
    """Drops (or rises) straight to `z`, holding the eef's *current* xy fixed as the target so
    move_to's computed xy action is ~0 and the motion is a pure vertical line -- never a diagonal
    sweep that could clip the cube being approached (while grasping) or an already-placed pyramid
    neighbor (while placing). Must only be called once xy is already aligned (via `_align_xy`)."""
    xy = obs["robot0_eef_pos"][:2].copy()
    return move_to(
        env, args, arm, pose_idx, grip_idx, obs, local_down_axis, np.array([*xy, z]), target_yaw,
        gripper_action, args.max_phase_steps, tol,
    )


def pick_and_place_cube(
    env, args, arm, pose_idx, grip_idx, obs, local_down_axis, cube_idx, hover_z, verbose
):
    """Picks up `env.cubes[cube_idx]` from wherever it currently is and places it at its fixed
    pyramid goal slot (`env.target_positions[cube_idx]`), level. Returns (obs, grasped).

    Every vertical approach near an object (descending onto the cube to grasp it, descending onto
    its goal slot next to already-placed neighbors) is xy-aligned first, then dropped straight
    down -- see `_align_xy`/`_descend_vertically` -- so the gripper never sweeps sideways into
    something while still descending.
    """
    cube = env.cubes[cube_idx]
    cube_pos_key = f"cube_{cube_idx}_pos"
    cube_body_id = env.cube_body_ids[cube_idx]
    target = env.table_offset + env.target_positions[cube_idx]
    target_xy, place_z = target[:2], target[2]
    tol = args.pos_tolerance

    # aligns the gripper jaws with the cube's current face; spawn rotation is fixed axis-aligned,
    # so this is normally ~0, but computing it from the live pose keeps this robust regardless
    target_yaw = nearest_face_yaw(get_body_yaw(env, cube_body_id))

    cube_xy = obs[cube_pos_key][:2].copy()
    obs, _ = _align_xy(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, cube_xy, hover_z, target_yaw, OPEN, tol)

    # re-read the cube's xy now that we're precisely hovering above where it actually is (xy is
    # already locked from the align step, so this just fixes the z target for the vertical drop)
    grasp_z = obs[cube_pos_key][2] + GRASP_HEIGHT_OFFSET
    obs, _ = _descend_vertically(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, grasp_z, target_yaw, OPEN, tol)

    obs = hold(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_yaw, CLOSE, args.grasp_steps)
    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=cube)
    if verbose:
        print(f"  [cube_{cube_idx}] grasped={grasped}")

    obs, _ = _descend_vertically(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, hover_z, target_yaw, CLOSE, tol)

    obs, _ = _align_xy(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_xy, hover_z, target_yaw, CLOSE, tol)

    place_z = place_z + GRASP_HEIGHT_OFFSET
    obs, _ = _descend_vertically(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, place_z, target_yaw, CLOSE, tol)

    obs = hold(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_yaw, OPEN, args.release_steps)

    obs, _ = _descend_vertically(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, hover_z, target_yaw, OPEN, tol)

    return obs, grasped


def run_episode(env, args, verbose=True, progress_prefix=None):
    """
    Runs one scripted pyramid-building episode on an already-reset `env`: picks up and places
    all 10 cubes in index order (guaranteed bottom-up by `PyramidStack`).

    Args:
        progress_prefix (str or None): if given, prints one line per cube as it's placed
            (`"{progress_prefix} cube 3/10 placed (t=14.2s)"`), independent of `verbose` (which
            controls the more detailed per-cube grasp-outcome prints). A single 10-cube episode
            takes ~30-50s, so per-cube updates -- rather than only a start/end message -- are what
            actually lets someone watching tell the episode is progressing vs. stuck.

    Returns:
        bool: True if the full pyramid is built (`env._check_success()` semantics).
    """
    arm = env.robots[0].arms[0]
    pose_idx = env.robots[0].composite_controller._action_split_indexes[arm]
    grip_idx = env.robots[0].composite_controller._action_split_indexes[f"{arm}_gripper"]

    obs = env._get_observations(force_update=True)
    # local eef-site axis that currently points down -- captured once, right after reset,
    # while the arm is in its known-level default pose
    local_down_axis = get_reference_down_axis(env, arm)
    hover_z = pyramid_hover_height(env)

    start_time = time.time()
    num_grasped = 0
    for cube_idx in range(env.num_cubes):
        obs, grasped = pick_and_place_cube(
            env, args, arm, pose_idx, grip_idx, obs, local_down_axis, cube_idx, hover_z, verbose
        )
        num_grasped += int(grasped)
        if progress_prefix is not None:
            placed = env._cube_placed(env.cubes[cube_idx])
            status = "placed" if placed else ("grasped, misplaced" if grasped else "MISSED")
            print(
                f"{progress_prefix} cube {cube_idx + 1}/{env.num_cubes} {status} "
                f"(t={time.time() - start_time:.1f}s)",
                flush=True,
            )

    success = env._check_success()
    if verbose:
        num_placed = sum(env._cube_placed(c) for c in env.cubes)
        print(
            f"{'Success' if success else 'Failed'}: {num_placed}/{env.num_cubes} cubes placed, "
            f"{num_grasped}/{env.num_cubes} grasped."
        )
    return success


def main(args):
    env = suite.make(
        env_name="PyramidStack",
        robots=args.robot,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        use_camera_obs=False,
        control_freq=20,
        horizon=args.horizon,
        ignore_done=True,
    )
    env.reset()
    run_episode(env, args, progress_prefix="episode 1/1")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering.")
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
