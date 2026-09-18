"""
Hand-coded (non-RL) policy that solves robosuite's Wipe task.

The wiping tool has zero gripper actuators, so the action vector is just the
arm controller's translation/rotation deltas (OSC_POSE, 6-dim). Each control
step the policy:
  1. Corrects orientation so the wiping pad's flat face points straight down
     at the table (Panda's default ready pose does NOT start the pad flat --
     it starts nearly vertical, since WipingGripper mounts differently than a
     standard parallel gripper).
  2. Drives the end-effector toward the nearest still-dirty marker in XY,
     pressing slightly below the marker's height in Z to generate the real
     mujoco contact that the env's marker-detection check requires.
  3. If a marker can't actually be reached (arm near a workspace/reach limit)
     it gets skipped after a stall timeout so the policy doesn't deadlock.
  4. Once wiping is done (or the wipe horizon runs out), the arm returns to
     its initial (reset) end-effector position before the episode ends.

Usage:
    python robosuite/scripts/scripted_wipe_policy.py --render
"""

import argparse

import numpy as np

import robosuite as suite

TARGET_NORMAL = np.array([0.0, 0.0, -1.0])  # pad should face straight down


def get_pad_normal(env, corner_ids):
    """Returns the unit normal of the wiping pad's plane, from its 4 corner geoms."""
    c1, c2, c3, c4 = [env.sim.data.geom_xpos[i] for i in corner_ids]
    normal = np.cross(c2 - c1, c3 - c1)
    return normal / np.linalg.norm(normal)


def get_orientation_action(normal, output_max_rot, rot_speed_scale):
    """Proportional control driving `normal` toward TARGET_NORMAL, as a clipped axis-angle delta."""
    axis = np.cross(normal, TARGET_NORMAL)
    axis_norm = np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(normal, TARGET_NORMAL), -1.0, 1.0))
    if axis_norm < 1e-6:
        if angle < 1e-3:
            return np.zeros(3)  # already aligned, no rotation axis needed
        # normal is antiparallel to the target (angle == pi) -- the cross product is degenerate
        # here, so pick an arbitrary axis perpendicular to normal to rotate around instead.
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(normal, arbitrary)
    axis = axis / np.linalg.norm(axis)
    rotvec = axis * angle
    return np.clip(rotvec / output_max_rot, -1, 1) * rot_speed_scale


def get_nearest_unwiped_marker(env, eef_pos, skip_indices):
    """Returns (marker_index, marker_world_pos) of the closest unwiped, non-skipped marker, or None."""
    best_idx, best_pos, best_dist = None, None, np.inf
    for i, marker in enumerate(env.model.mujoco_arena.markers):
        if marker in env.wiped_markers or i in skip_indices:
            continue
        marker_pos = np.array(env.sim.data.body_xpos[env.sim.model.body_name2id(marker.root_body)])
        dist = np.linalg.norm(marker_pos - eef_pos)
        if dist < best_dist:
            best_idx, best_pos, best_dist = i, marker_pos, dist
    return None if best_idx is None else (best_idx, best_pos)


def run_episode(env, args, verbose=True):
    """
    Runs one scripted wipe episode on an already-reset `env`: wipes every marker it can
    reach, then returns the arm to its position at the start of the episode.

    Returns:
        bool: True if every marker was wiped (env._check_success() semantics).
    """
    arm = env.robots[0].arms[0]
    start_idx, end_idx = env.robots[0].composite_controller._action_split_indexes[arm]
    output_max = env.robots[0].part_controllers[arm].output_max
    corner_ids = [env.sim.model.geom_name2id(n) for n in env.robots[0].gripper[arm].important_geoms["corners"]]
    initial_eef_pos = env._get_eef_xpos(arm)

    skip_indices = set()
    current_idx = None
    stall_steps = 0

    for t in range(args.horizon):
        eef_pos = env._get_eef_xpos(arm)
        target = get_nearest_unwiped_marker(env, eef_pos, skip_indices)

        action = np.zeros(env.action_dim)
        normal = get_pad_normal(env, corner_ids)
        delta = np.zeros(6)
        delta[3:6] = get_orientation_action(normal, output_max[3:6], args.rot_speed_scale)

        if target is not None:
            idx, marker_pos = target
            if idx != current_idx:
                current_idx, stall_steps = idx, 0
            else:
                stall_steps += 1
            if stall_steps > args.stall_limit:
                skip_indices.add(idx)

            press_target = marker_pos.copy()
            press_target[2] -= args.z_bias
            delta[:3] = np.clip((press_target - eef_pos) / output_max[:3], -1, 1) * args.speed_scale

        action[start_idx:end_idx] = delta
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()

        if len(env.wiped_markers) == env.num_markers:
            if verbose:
                print(f"Success: wiped all {env.num_markers} markers at step {t} (skipped {len(skip_indices)}).")
            break
    else:
        if verbose:
            print(
                f"Did not finish: wiped {len(env.wiped_markers)}/{env.num_markers} markers "
                f"in {args.horizon} steps (skipped {len(skip_indices)})."
            )

    if verbose:
        print(f"proportion_wiped: {len(env.wiped_markers) / env.num_markers:.2f}")

    for t in range(args.return_steps):
        eef_pos = env._get_eef_xpos(arm)
        dist = np.linalg.norm(initial_eef_pos - eef_pos)
        if dist < args.return_tolerance:
            if verbose:
                print(f"Returned to initial position after {t} steps (dist={dist:.4f}m).")
            break

        action = np.zeros(env.action_dim)
        normal = get_pad_normal(env, corner_ids)
        delta = np.zeros(6)
        delta[3:6] = get_orientation_action(normal, output_max[3:6], args.rot_speed_scale)
        delta[:3] = np.clip((initial_eef_pos - eef_pos) / output_max[:3], -1, 1) * args.speed_scale
        action[start_idx:end_idx] = delta

        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
    else:
        if verbose:
            print(f"Did not fully return to initial position within {args.return_steps} steps (dist={dist:.4f}m).")

    return len(env.wiped_markers) == env.num_markers


def main(args):
    env = suite.make(
        env_name="Wipe",
        robots=args.robot,
        has_renderer=args.render,
        has_offscreen_renderer=not args.render,
        use_camera_obs=False,
        control_freq=20,
        horizon=args.horizon,
        ignore_done=True,
    )
    env.reset()
    run_episode(env, args)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering.")
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
