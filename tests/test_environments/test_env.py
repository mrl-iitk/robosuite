import time
import traceback

import numpy as np

import robosuite
from robosuite.controllers import load_composite_controller_config


def print_sep(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    print_sep("Creating Environment")

    env = robosuite.make(
        "ServeBread",
        robots="Nero7",
        controller_configs=load_composite_controller_config(controller="BASIC"),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        control_freq=20,
    )

    print_sep("Reset")

    env.reset()

    print_sep("Body Names")
    print(env.sim.model.body_names)

    print_sep("Geom Names")
    print(env.sim.model.geom_names)

    print_sep("Joint Names")
    print(env.sim.model.joint_names)

    print_sep("Site Names")
    print(env.sim.model.site_names)

    print_sep("Bread")

    print("Root body :", env.bread.root_body)
    print("Contact   :", env.bread.contact_geoms)
    print("Visual    :", env.bread.visual_geoms)

    print_sep("Plate")

    print("Root body :", env.plate.root_body)
    print("Contact   :", env.plate.contact_geoms)
    print("Visual    :", env.plate.visual_geoms)

    print_sep("Body IDs")

    print("Bread ID :", env.bread_body_id)
    print("Plate ID :", env.plate_body_id)

    print_sep("Body State")

    print("Bread Position :", env.sim.data.body_xpos[env.bread_body_id])
    print("Bread Quaternion :", env.sim.data.body_xquat[env.bread_body_id])

    print("Plate Position :", env.sim.data.body_xpos[env.plate_body_id])
    print("Plate Quaternion :", env.sim.data.body_xquat[env.plate_body_id])

    print_sep("Mass")

    print("Bread mass :", env.sim.model.body_mass[env.bread_body_id])
    print("Plate mass :", env.sim.model.body_mass[env.plate_body_id])

    print_sep("Inertia")

    print("Bread inertia :", env.sim.model.body_inertia[env.bread_body_id])
    print("Plate inertia :", env.sim.model.body_inertia[env.plate_body_id])

    print_sep("Joints")

    print("Bread joints :", env.bread.joints)
    print("Plate joints :", env.plate.joints)

    print_sep("Joint qpos")

    for j in env.bread.joints:
        print(j, env.sim.data.get_joint_qpos(j))

    for j in env.plate.joints:
        print(j, env.sim.data.get_joint_qpos(j))

    print_sep("Gravity")

    print(env.sim.model.opt.gravity)

    print_sep("Camera")

    try:
        print("lookat :", env.viewer.viewer.cam.lookat)
        print("distance :", env.viewer.viewer.cam.distance)
        print("azimuth :", env.viewer.viewer.cam.azimuth)
        print("elevation :", env.viewer.viewer.cam.elevation)
    except Exception:
        print("Camera info unavailable")

    print_sep("Simulation")

    action = np.zeros(env.action_dim)

    step = 0

    while True:

        try:

            env.step(action)

        except Exception:

            print("\nSTEP FAILED")
            traceback.print_exc()
            break
        

        bread = env.sim.data.body_xpos[env.bread_body_id]
        plate = env.sim.data.body_xpos[env.plate_body_id]

        print(
            f"\nSTEP {step}"
        )

        print("Bread Pos :", bread)
        print("Plate Pos :", plate)

        print("Bread Vel :", env.sim.data.cvel[env.bread_body_id])
        print("Plate Vel :", env.sim.data.cvel[env.plate_body_id])

        print("Bread qvel :", env.sim.data.qvel[:])
        print("Bread qacc :", env.sim.data.qacc[:])

        print("Bread mass:", env.sim.model.body_mass[env.bread_body_id])
        print("Plate mass:", env.sim.model.body_mass[env.plate_body_id])

        print("Bread inertia:", env.sim.model.body_inertia[env.bread_body_id])
        print("Plate inertia:", env.sim.model.body_inertia[env.plate_body_id])

        if np.isnan(env.sim.data.qpos).any():
            print("NaN in qpos")
            break

        if np.isnan(env.sim.data.qvel).any():
            print("NaN in qvel")
            break

        if np.isnan(env.sim.data.qacc).any():
            print("NaN in qacc")
            break

        env.render()

        step += 1

        time.sleep(0.02)
        input("Press Enter to continue...")
    env.close()


if __name__ == "__main__":
    main()