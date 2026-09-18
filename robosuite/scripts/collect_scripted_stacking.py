import time
import numpy as np

import robosuite as suite
from robosuite.controllers import load_composite_controller_config


# ============================================================
# CONFIG
# ============================================================

NUM_EPISODES = 2
CONTROL_FREQ = 20
HORIZON = 600

# OSC translation:
# action +/-1 ~= +/-5 cm
DELTA_SCALE = 0.05

# OSC rotation:
# action +/-1 ~= +/-0.5 rad
ROT_SCALE = 0.5

NORMAL_TOLERANCE = 0.012
GRASP_TOLERANCE = 0.004
YAW_TOLERANCE = np.deg2rad(2.0)

APPROACH_HEIGHT = 0.15
GRASP_HEIGHT = 0.005
LIFT_HEIGHT = 0.20

# Height above desired XY while carrying
TRANSFER_HEIGHT = 0.20

# Cube B placement:
# CHANGE THESE TO YOUR DESIRED LOCATION
TARGET_X = -0.15
TARGET_Y = 0.35

# Cube dimensions:
# cube B center is around z=0.835
# table top is around z=0.805
# cube height ~= 0.05
CUBE_B_PLACE_Z = 0.86

# Cube A sits on top of Cube B
CUBE_A_PLACE_Z = 0.91


# ============================================================
# ENVIRONMENT
# ============================================================

def create_environment():

    print("\nCreating Stack environment...")

    controller_config = load_composite_controller_config(
        robot="Panda"
    )

    env = suite.make(
        env_name="Stack",
        robots="Panda",
        controller_configs=controller_config,

        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,

        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
    )

    return env


# ============================================================
# EEF POSITION
# ============================================================

def get_eef(env):

    site_id = env.robots[0].eef_site_id["right"]

    return np.array(
        env.sim.data.site_xpos[site_id]
    )


# ============================================================
# EEF YAW
# ============================================================

def get_eef_yaw(env):

    site_id = env.robots[0].eef_site_id["right"]

    # MuJoCo gives site orientation as a 3x3 rotation matrix
    R = env.sim.data.site_xmat[site_id].reshape(3, 3)

    # Rotation around world Z
    yaw = np.arctan2(
        R[1, 0],
        R[0, 0],
    )

    return yaw


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return np.arctan2(
        np.sin(angle),
        np.cos(angle),
    )


# ============================================================
# ACTION
# ============================================================

def make_action(
    eef,
    target,
    gripper,
    desired_yaw=0.0,
):

    # --------------------------------------------------------
    # POSITION
    # --------------------------------------------------------

    position_error = target - eef

    position_delta = np.clip(
        position_error / DELTA_SCALE,
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # YAW
    # --------------------------------------------------------

    current_yaw = get_current_yaw_global

    # --------------------------------------------------------
    # We calculate yaw outside this function below.
    # --------------------------------------------------------

    return np.array([
        position_delta[0],
        position_delta[1],
        position_delta[2],

        0.0,       # roll
        0.0,       # pitch
        0.0,       # yaw -- filled by caller

        gripper,
    ])


# ============================================================
# MOVE TO TARGET + ALIGN YAW
# ============================================================

def move_to(
    env,
    target,
    gripper,
    name,
    desired_yaw=0.0,
    tolerance=NORMAL_TOLERANCE,
    max_steps=180,
):

    print(f"\n  -> {name}")

    for step in range(max_steps):

        eef = get_eef(env)

        # ----------------------------------------------------
        # Position error
        # ----------------------------------------------------

        position_error = target - eef

        distance = np.linalg.norm(
            position_error
        )

        # ----------------------------------------------------
        # Yaw error
        # ----------------------------------------------------

        current_yaw = get_eef_yaw(env)

        yaw_error = wrap_angle(
            desired_yaw - current_yaw
        )

        # ----------------------------------------------------
        # SUCCESS CONDITION
        #
        # Both position AND yaw must be correct.
        # ----------------------------------------------------

        if (
            distance < tolerance
            and abs(yaw_error) < YAW_TOLERANCE
        ):

            print(
                f"     reached: "
                f"pos={distance:.4f} m, "
                f"yaw={np.degrees(yaw_error):.2f}°"
            )

            return True

        # ----------------------------------------------------
        # Position command
        # ----------------------------------------------------

        position_delta = np.clip(
            position_error / DELTA_SCALE,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Yaw command
        # ----------------------------------------------------

        yaw_delta = np.clip(
            yaw_error / ROT_SCALE,
            -1.0,
            1.0,
        )

        # ----------------------------------------------------
        # Final action
        # ----------------------------------------------------

        action = np.array([
            position_delta[0],
            position_delta[1],
            position_delta[2],

            0.0,          # roll
            0.0,          # pitch
            yaw_delta,    # yaw

            gripper,
        ])

        env.step(action)

        if step % 30 == 0:

            print(
                f"     step={step:3d} "
                f"pos_error={distance:.3f} "
                f"yaw={np.degrees(current_yaw):.1f}°"
            )

        time.sleep(
            1.0 / CONTROL_FREQ
        )

    print(
        f"     WARNING: target not reached "
        f"pos={distance:.4f} m "
        f"yaw={np.degrees(yaw_error):.2f}°"
    )

    return False


# ============================================================
# HOLD POSITION + GRIPPER
# ============================================================

def hold_gripper(
    env,
    target,
    gripper,
    desired_yaw=0.0,
    steps=30,
):

    for _ in range(steps):

        eef = get_eef(env)

        position_error = target - eef

        position_delta = np.clip(
            position_error / DELTA_SCALE,
            -1.0,
            1.0,
        )

        current_yaw = get_eef_yaw(env)

        yaw_error = wrap_angle(
            desired_yaw - current_yaw
        )

        yaw_delta = np.clip(
            yaw_error / ROT_SCALE,
            -1.0,
            1.0,
        )

        action = np.array([
            position_delta[0],
            position_delta[1],
            position_delta[2],

            0.0,
            0.0,
            yaw_delta,

            gripper,
        ])

        env.step(action)

        time.sleep(
            1.0 / CONTROL_FREQ
        )


# ============================================================
# PICK CUBE
# ============================================================

def pick_cube(
    env,
    cube_pos,
    name,
):

    print(f"\n  PICK {name}")

    # --------------------------------------------------------
    # 1. Approach
    # --------------------------------------------------------

    target = cube_pos.copy()

    target[2] += APPROACH_HEIGHT

    move_to(
        env,
        target,
        gripper=-1.0,
        desired_yaw=0.0,
        name=f"APPROACH {name}",
    )

    # --------------------------------------------------------
    # 2. Descend
    # --------------------------------------------------------

    target = cube_pos.copy()

    target[2] += GRASP_HEIGHT

    move_to(
        env,
        target,
        gripper=-1.0,
        desired_yaw=0.0,
        tolerance=GRASP_TOLERANCE,
        name=f"DESCEND TO {name}",
    )

    # --------------------------------------------------------
    # 3. Close gripper
    # --------------------------------------------------------

    print("     Closing gripper...")

    hold_gripper(
        env,
        target,
        gripper=1.0,
        desired_yaw=0.0,
        steps=35,
    )

    # --------------------------------------------------------
    # 4. Lift
    # --------------------------------------------------------

    target = cube_pos.copy()

    target[2] += LIFT_HEIGHT

    move_to(
        env,
        target,
        gripper=1.0,
        desired_yaw=0.0,
        name=f"LIFT {name}",
    )

    return target


# ============================================================
# PLACE CUBE
# ============================================================

def place_cube(
    env,
    target,
    name,
    place_z,
):

    # --------------------------------------------------------
    # Move above destination
    # --------------------------------------------------------

    above = np.array([
        target[0],
        target[1],
        place_z + TRANSFER_HEIGHT,
    ])

    move_to(
        env,
        above,
        gripper=1.0,
        desired_yaw=0.0,
        name=f"MOVE {name} TO TARGET",
    )

    # --------------------------------------------------------
    # Descend
    # --------------------------------------------------------

    place = np.array([
        target[0],
        target[1],
        place_z,
    ])

    move_to(
        env,
        place,
        gripper=1.0,
        desired_yaw=0.0,
        name=f"PLACE {name}",
    )

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    print("     Opening gripper...")

    hold_gripper(
        env,
        place,
        gripper=-1.0,
        desired_yaw=0.0,
        steps=35,
    )


# ============================================================
# ONE EPISODE
# ============================================================

def run_episode(
    env,
    episode,
):

    print("\n" + "=" * 55)
    print(f"EPISODE {episode}")
    print("=" * 55)

    obs = env.reset()

    cubeA = np.array(
        obs["cubeA_pos"]
    )

    cubeB = np.array(
        obs["cubeB_pos"]
    )

    print("Cube A:", cubeA)
    print("Cube B:", cubeB)

    # ========================================================
    # DESIRED STACK LOCATION
    # ========================================================

    stack_xy = np.array([
        TARGET_X,
        TARGET_Y,
    ])

    print(
        "Target XY:",
        stack_xy
    )

    # ========================================================
    # 1. PICK CUBE B
    # ========================================================

    pick_cube(
        env,
        cubeB,
        "CUBE B",
    )

    # ========================================================
    # 2. PLACE CUBE B
    # ========================================================

    print("\n  PLACE CUBE B")

    place_cube(
        env,
        stack_xy,
        "CUBE B",
        CUBE_B_PLACE_Z,
    )

    # ========================================================
    # 3. PICK CUBE A
    # ========================================================

    pick_cube(
        env,
        cubeA,
        "CUBE A",
    )

    # ========================================================
    # 4. PLACE CUBE A ON B
    # ========================================================

    print("\n  PLACE CUBE A ON CUBE B")

    place_cube(
        env,
        stack_xy,
        "CUBE A",
        CUBE_A_PLACE_Z,
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n  STACK COMPLETE")

    time.sleep(2)


# ============================================================
# MAIN
# ============================================================

def main():

    env = None

    try:

        env = create_environment()

        print("Environment ready.")

        for episode in range(
            1,
            NUM_EPISODES + 1,
        ):

            run_episode(
                env,
                episode,
            )

        print("\nAll episodes finished.")

        time.sleep(2)

    except KeyboardInterrupt:

        print("\nInterrupted by user.")

    finally:

        print("Closing environment...")

        if env is not None:

            try:
                env.close()
            except Exception:
                pass


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()