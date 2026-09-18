"""
Hand-coded (non-RL) policy that solves robosuite's Stack task.

The action vector is the arm controller's translation/rotation deltas (OSC_POSE, 6-dim)
plus a 1-dim parallel-jaw gripper actuation. Panda's reset pose already points the
gripper straight down at the table for this task (unlike the Wipe task's WipingGripper),
so this policy leaves roll/pitch deltas at zero throughout and only actively drives
position plus yaw.

Cubes spawn with a fully random yaw (robosuite's UniformRandomSampler samples rotation
uniformly in [0, 2*pi) by default). Approaching with a fixed world-frame gripper yaw
means the jaws sometimes have to close across a cube's diagonal (~1.4x wider than its
face) instead of a flat face, which is a real source of dropped/failed grasps. Since a
box is symmetric under 90-degree rotations about its vertical axis, `nearest_face_yaw`
folds the cube's sampled yaw into the smallest correction (+/-45 degrees) that aligns
the gripper jaws with the nearest pair of opposite faces, and that target yaw is held
for the whole pick-and-place of that cube (computed once, before the cube is touched).

The policy always moves cubeB (the larger, green base cube) to a fixed target location
on the table first, then picks up cubeA (the smaller, red cube) and places it directly
on top of cubeB there -- matching the Stack task's success condition (cubeA resting on,
and in contact with, cubeB once released).

Each cube is moved by the same `pick_and_place` routine:
  1. hover above the cube (gripper open)
  2. descend to grasp height
  3. close the gripper and hold briefly to get a firm grasp
  4. lift straight up
  5. translate laterally to the target (x, y)
  6. descend to the target height
  7. open the gripper to release
  8. retreat back up to the hover height

Yaw is the only rotation this policy *targets* somewhere other than level -- roll/pitch
are always actively driven back to level (gripper pointing straight down) at every step,
using the same normal-alignment approach as scripted_wipe_policy.py's `get_orientation_action`
(reused here rather than duplicated). This keeps a grasped cube horizontal throughout the
carry, so it is placed down flat instead of however it happened to tip during the grasp.

Usage:
    python robosuite/scripts/scripted_stack_policy.py --render
"""

import argparse

import numpy as np

import robosuite as suite
from robosuite.scripts.scripted_wipe_policy import TARGET_NORMAL, get_orientation_action

HOVER_HEIGHT_ABOVE_TABLE = 0.15  # eef height (m) above the table while transporting cubes
GRASP_HEIGHT_OFFSET = -0.005  # eef target z relative to cube center while grasping/placing
OPEN, CLOSE = -1.0, 1.0


def wrap_angle(angle):
    """Wraps an angle (rad) to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def nearest_face_yaw(yaw, reference=0.0):
    """Folds `yaw` to the equivalent orientation (mod 90 degrees, since a box is symmetric under
    90-degree rotations about its vertical axis) nearest to `reference`. With the default
    reference=0.0, this picks the smallest-magnitude representative in [-pi/4, pi/4).

    Folding relative to the *current eef yaw* (rather than always relative to 0) matters between
    successive cubes: the eef's yaw carries over from the previous pick-and-place, and if the new
    target is folded to the smallest-magnitude representative regardless of where the eef already
    is, the two can differ by up to ~90 degrees even though an equivalent grasp orientation only
    ~0 degrees away exists. That unnecessary swing can carry the wrist through a joint limit on
    some arms (observed on Nero7, whose wrist has less range than Panda's), stalling the OSC
    controller for the rest of the episode -- so callers should pass the eef's current yaw here.
    """
    diff = wrap_angle(yaw - reference)
    return reference + ((diff + np.pi / 4) % (np.pi / 2)) - np.pi / 4


def get_yaw_from_xmat(xmat_flat):
    """Extracts the world-frame yaw (rotation about z) from a flattened 3x3 rotation matrix."""
    R = xmat_flat.reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def get_eef_yaw(env, arm):
    return get_yaw_from_xmat(env.sim.data.site_xmat[env.robots[0].eef_site_id[arm]])


def get_body_yaw(env, body_id):
    return get_yaw_from_xmat(env.sim.data.body_xmat[body_id])


def get_reference_down_axis(env, arm):
    """Picks whichever local eef-site axis currently points closest to world -z, so the
    leveling correction below works regardless of how a given gripper's site is mounted."""
    R = env.sim.data.site_xmat[env.robots[0].eef_site_id[arm]].reshape(3, 3)
    return int(np.argmax(R.T @ TARGET_NORMAL))


def get_eef_normal(env, arm, local_down_axis):
    R = env.sim.data.site_xmat[env.robots[0].eef_site_id[arm]].reshape(3, 3)
    return R[:, local_down_axis]


def get_rot_action(env, arm, local_down_axis, target_yaw, output_max_rot, rot_speed_scale):
    """Combines a leveling correction (roll/pitch, driving the eef back to pointing straight
    down) with an explicit yaw correction (driving toward `target_yaw`) into one 3-vector."""
    normal = get_eef_normal(env, arm, local_down_axis)
    level_action = get_orientation_action(normal, output_max_rot, rot_speed_scale)
    yaw_error = wrap_angle(target_yaw - get_eef_yaw(env, arm))
    rot_action = level_action.copy()
    rot_action[2] = np.clip(yaw_error / output_max_rot[2], -1, 1) * rot_speed_scale
    return rot_action, yaw_error


def move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_pos, target_yaw, gripper_action, max_steps, tol):
    """Drives the eef toward `target_pos`/`target_yaw` (while leveling roll/pitch) until both
    position and yaw stay within tolerance for `args.hold_steps` consecutive steps, or
    `max_steps` is exhausted. Returns (obs, success)."""
    output_max = env.robots[0].part_controllers[arm].output_max
    stable_steps = 0
    for _ in range(max_steps):
        eef_pos = obs["robot0_eef_pos"]
        rot_action, yaw_error = get_rot_action(env, arm, local_down_axis, target_yaw, output_max[3:6], args.rot_speed_scale)
        action = np.zeros(env.action_dim)
        action[pose_idx[0] : pose_idx[0] + 3] = (
            np.clip((target_pos - eef_pos) / output_max[:3], -1, 1) * args.speed_scale
        )
        action[pose_idx[0] + 3 : pose_idx[0] + 6] = rot_action
        action[grip_idx[0] : grip_idx[1]] = gripper_action
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
        if np.linalg.norm(obs["robot0_eef_pos"] - target_pos) < tol and abs(yaw_error) < args.yaw_tolerance:
            stable_steps += 1
            if stable_steps >= args.hold_steps:
                return obs, True
        else:
            stable_steps = 0
    return obs, False


def hold(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_yaw, gripper_action, num_steps):
    """Holds the current position (zero position delta) while leveling roll/pitch, correcting
    yaw, and actuating the gripper for `num_steps`."""
    output_max = env.robots[0].part_controllers[arm].output_max
    for _ in range(num_steps):
        rot_action, _ = get_rot_action(env, arm, local_down_axis, target_yaw, output_max[3:6], args.rot_speed_scale)
        action = np.zeros(env.action_dim)
        action[pose_idx[0] + 3 : pose_idx[0] + 6] = rot_action
        action[grip_idx[0] : grip_idx[1]] = gripper_action
        obs, reward, done, info = env.step(action)
        if args.render:
            env.render()
    return obs


def pick_and_place(
    env, args, arm, pose_idx, grip_idx, obs, local_down_axis, cube_pos_key, cube_body_id, cube_geoms, target_xy, place_z, verbose, label
):
    """Picks up the cube at `cube_pos_key` and places it so its center ends up at
    (target_xy, place_z), level. Returns (obs, grasped)."""
    hover_z = env.table_offset[2] + HOVER_HEIGHT_ABOVE_TABLE
    tol = args.pos_tolerance
    # aligns the gripper jaws with the cube's nearest face, folded to whichever of the 4
    # equivalent (90-degree-symmetric) orientations is closest to the eef's current yaw -- this
    # minimizes the rotation the arm has to execute, rather than always folding toward 0; held for
    # the whole pick-and-place
    target_yaw = nearest_face_yaw(get_body_yaw(env, cube_body_id), reference=get_eef_yaw(env, arm))

    cube_xy = obs[cube_pos_key][:2].copy()
    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, np.array([*cube_xy, hover_z]), target_yaw, OPEN, args.max_phase_steps, tol)

    grasp_target = obs[cube_pos_key].copy()
    grasp_target[2] += GRASP_HEIGHT_OFFSET
    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, grasp_target, target_yaw, OPEN, args.max_phase_steps, tol)

    obs = hold(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_yaw, CLOSE, args.grasp_steps)
    grasped = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=cube_geoms)
    if verbose:
        print(f"  [{label}] grasped={grasped}")

    lift_target = obs["robot0_eef_pos"].copy()
    lift_target[2] = hover_z
    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, lift_target, target_yaw, CLOSE, args.max_phase_steps, tol)

    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, np.array([*target_xy, hover_z]), target_yaw, CLOSE, args.max_phase_steps, tol)

    place_target = np.array([*target_xy, place_z + GRASP_HEIGHT_OFFSET])
    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, place_target, target_yaw, CLOSE, args.max_phase_steps, tol)

    obs = hold(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, target_yaw, OPEN, args.release_steps)

    retreat_target = obs["robot0_eef_pos"].copy()
    retreat_target[2] = hover_z
    obs, _ = move_to(env, args, arm, pose_idx, grip_idx, obs, local_down_axis, retreat_target, target_yaw, OPEN, args.max_phase_steps, tol)

    return obs, grasped


def run_episode(env, args, verbose=True):
    """
    Runs one scripted stacking episode on an already-reset `env`: moves cubeB to a fixed
    target location, then stacks cubeA on top of it.

    Returns:
        bool: True if cubeA ends up stacked on cubeB (env._check_success() semantics).
    """
    arm = env.robots[0].arms[0]
    pose_idx = env.robots[0].composite_controller._action_split_indexes[arm]
    grip_idx = env.robots[0].composite_controller._action_split_indexes[f"{arm}_gripper"]

    obs = env._get_observations(force_update=True)
    # local eef-site axis that currently points down -- captured once, right after reset,
    # while the arm is in its known-level default pose
    local_down_axis = get_reference_down_axis(env, arm)

    target_xy = env.stack_target_pos.copy()
    cubeB_half_height = env.cubeB.size[2]
    cubeA_half_height = env.cubeA.size[2]
    cubeB_place_z = env.table_offset[2] + cubeB_half_height
    cubeA_place_z = cubeB_place_z + cubeB_half_height + cubeA_half_height

    obs, _ = pick_and_place(
        env, args, arm, pose_idx, grip_idx, obs, local_down_axis, "cubeB_pos", env.cubeB_body_id, env.cubeB, target_xy, cubeB_place_z, verbose, "cubeB"
    )
    # re-measure where cubeB actually settled before stacking cubeA on it
    obs = env._get_observations(force_update=True)
    cubeA_place_z = obs["cubeB_pos"][2] + cubeB_half_height + cubeA_half_height

    obs, _ = pick_and_place(
        env, args, arm, pose_idx, grip_idx, obs, local_down_axis, "cubeA_pos", env.cubeA_body_id, env.cubeA, target_xy, cubeA_place_z, verbose, "cubeA"
    )

    success = env._check_success()
    if verbose:
        print(f"{'Success' if success else 'Failed'}: cubeA {'is' if success else 'is not'} stacked on cubeB.")
    return success


def main(args):
    env = suite.make(
        env_name="Stack",
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
