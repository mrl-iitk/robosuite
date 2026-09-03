import argparse
import datetime
import json
import os
import sys
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import time
import glob
import h5py
import numpy as np
# Ensure we use the local robosuite repository
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper
from robosuite.utils.placement_samplers import UniformRandomSampler

def gather_demonstrations_as_hdf5(directory, out_dir, env_info):
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    if os.path.exists(hdf5_path):
        try:
            os.remove(hdf5_path)
        except Exception:
            pass
    f = h5py.File(hdf5_path, "w")

    grp = f.create_group("data")

    num_eps = 0
    env_name = None

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        success = False

        for state_file in sorted(glob.glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            success = success or dic["successful"]

        if len(states) == 0:
            continue

        if not success:
            print(f"Demonstration in {ep_directory} is unsuccessful. Discarding.")
            continue
            
        print(f"Demonstration in {ep_directory} is successful and has been saved")
            
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps - 1))

        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    env_info_dict = json.loads(env_info)
    env_args = {
        "env_name": env_info_dict["env_name"],
        "type": 1,
        "env_kwargs": env_info_dict
    }
    grp.attrs["env_args"] = json.dumps(env_args)

    f.close()


def collect_lift(env, max_steps, max_fr=20, absolute_actions=False):
    """
    Executes a simple scripted state machine to solve the Lift task.
    """
    obs = env.reset()
    
    # State machine phases:
    # 0: Hover above cube
    # 1: Lower to cube
    # 2: Grasp
    # 3: Lift up
    phase = 0
    step_count = 0
    
    # Assuming single arm "right" for now
    active_arm = env.robots[0].arms[0] if len(env.robots[0].arms) > 0 else "right"
    
    while step_count < max_steps:
        # Get observations
        cube_pos = obs.get("cube_pos")
        if cube_pos is None:
            # If not Lift env, fallback to generic object position
            cube_pos = obs.get("object-state", np.zeros(3))[:3]
            
        eef_pos = obs.get(f"robot0_eef_pos")
        
        # Calculate delta position
        delta_pos = np.zeros(3)
        gripper = -1.0 # open
        
        if phase == 0: # Hover above cube
            target_pos = cube_pos + np.array([0, 0, 0.35])
            dist = np.linalg.norm(target_pos - eef_pos)
            if dist < 0.05:
                phase = 1
            else:
                delta_pos = (target_pos - eef_pos) * 10.0
                
        elif phase == 1: # Lower down
            target_pos = cube_pos + np.array([0, 0, -0.03]) # slightly below center to grasp
            dist = np.linalg.norm(target_pos - eef_pos)
            if dist < 0.02 or eef_pos[2] < (cube_pos[2] + 0.01):
                phase = 2
            else:
                delta_pos = (target_pos - eef_pos) * 10.0
                
        elif phase == 2: # Grasp
            gripper = 1.0 # close
            # Wait a bit for gripper to close (20 steps ~ 1 sec)
            if 'grasp_start_step' not in locals():
                grasp_start_step = step_count
            if step_count - grasp_start_step > 20:
                phase = 3
            delta_pos = np.zeros(3)
            
        elif phase == 3: # Lift
            gripper = 1.0
            delta_pos = np.array([0, 0, 1.0]) # Move straight up
            
            if env._check_success():
                print("Task successful!")
                return True
                
        # Delta orientation is 0 (keep starting orientation)
        delta_ori = np.zeros(3)
        
        # Clip delta pos for safety
        delta_pos = np.clip(delta_pos, -1, 1)
        
        # Action array: [dx, dy, dz, ax, ay, az, gripper]
        action = np.concatenate([delta_pos, delta_ori, [gripper]])
        
        if absolute_actions:
            from scipy.spatial.transform import Rotation as R
            abs_pos = eef_pos + np.clip(delta_pos, -1.0, 1.0) * 0.05
            eef_quat_xyzw = obs.get(f"robot0_eef_quat", np.array([0, 0, 0, 1.0]))
            abs_ori = (R.from_rotvec(np.clip(delta_ori, -1.0, 1.0) * 0.5) * R.from_quat(eef_quat_xyzw)).as_rotvec()
            action = np.concatenate([abs_pos, abs_ori, [gripper]])
        
        start_time = time.time()
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        env.render()
        
        elapsed = time.time() - start_time
        if elapsed < 1.0 / max_fr:
            time.sleep(1.0 / max_fr - elapsed)

    env.close()
    return False

def collect_nut_assembly(env, max_steps, device=None, max_fr=20, absolute_actions=False):
    """
    Executes a simple scripted state machine to solve the NutAssembly task.
    """
    obs = None
    for _ in range(5):
        try:
            obs = env.reset()
            break
        except Exception as e:
            print(f"env.reset() encountered error: {e}. Retrying reset...")
    if obs is None:
        return False
    
    # State machine phases:
    # 0: Hover above nut
    # 1: Lower to nut
    # 2: Grasp
    # 3: Lift up
    # 4: Hover above peg
    # 5: Lower onto peg
    # 6: Release and move away
    phase = 0
    step_count = 0
    
    active_arm = env.robots[0].arms[0] if len(env.robots[0].arms) > 0 else "right"
    
    # Keep track of grasp start    phase = 0
    grasp_start_step = 0
    release_start_step = 0
    lower_start_step = 0
    
    chosen_yaw_offset = None
    while step_count < max_steps:
        # Get observations
        eef_pos = obs.get(f"robot0_eef_pos")
        eef_quat = obs.get(f"robot0_eef_quat")
        # Target the handle of the nut instead of its center
        nut_pos_body = obs.get("SquareNut0_pos", obs.get("RoundNut0_pos"))
        if nut_pos_body is None:
            nut_pos_body = obs.get("SquareNut_pos", obs.get("RoundNut_pos"))
            
        if nut_pos_body is None:
            nut_pos_body = obs.get("object-state", np.zeros(3))[:3]
            
        nut_quat = obs.get("SquareNut0_quat", obs.get("RoundNut0_quat"))
        if nut_quat is None:
            nut_quat = obs.get("SquareNut_quat", obs.get("RoundNut_quat", np.array([0, 0, 0, 1])))
            
        # Calculate yaw from nut quaternion
        nut_mat = T.quat2mat(nut_quat)
        nut_yaw = np.arctan2(nut_mat[1, 0], nut_mat[0, 0])
        
        # Target position along handle extension (0.065) further away from nut center
        local_x_axis = np.array([np.cos(nut_yaw), np.sin(nut_yaw), 0])
        nut_pos = nut_pos_body + local_x_axis * 0.065
            
        # Peg coordinates from sim data directly (since they aren't standard obs)
        peg1_pos = np.array(env.sim.data.body_xpos[env.peg1_body_id])
        
        # Calculate delta position
        delta_pos = np.zeros(3)
        gripper = -1.0 # open
        
        # Calculate orientation error to point straight down but aligned with the nut's yaw!
        # During placement (phase >= 4), align the nut to the nearest 90 degrees to fit the square peg
        if phase >= 4:
            current_target_yaw = round(nut_yaw / (np.pi / 2)) * (np.pi / 2)
        else:
            current_target_yaw = nut_yaw
            
        # Reverse yaw because euler2mat applies X-rotation (pi) AFTER Z-rotation, flipping the Y axis!
        base_target_yaw = -current_target_yaw
        
        # To avoid joint limits, we test both the forward and backward 180-deg symmetric grasps
        # We only compute this once in Phase 0 to prevent the gripper from randomly flipping!
        if chosen_yaw_offset is None:
            quat1 = T.mat2quat(T.euler2mat([np.pi, 0, base_target_yaw]))
            quat2 = T.mat2quat(T.euler2mat([np.pi, 0, base_target_yaw + np.pi]))
            err1 = np.linalg.norm(T.get_orientation_error(quat1, eef_quat))
            err2 = np.linalg.norm(T.get_orientation_error(quat2, eef_quat))
            chosen_yaw_offset = 0 if err1 < err2 else np.pi
            
        target_quat = T.mat2quat(T.euler2mat([np.pi, 0, base_target_yaw + chosen_yaw_offset]))
        ori_error = T.get_orientation_error(target_quat, eef_quat)
        delta_ori = ori_error * 10.0
        
        if phase == 0: # Hover above nut
            target_pos = nut_pos + np.array([0, 0, 0.30])
            dist = np.linalg.norm(target_pos - eef_pos)
            xy_dist = np.linalg.norm(target_pos[:2] - eef_pos[:2])
                
            if (dist < 0.05 and xy_dist < 0.01 and np.linalg.norm(ori_error) < 0.1) or step_count > 60:
                print(f"Phase 0 -> 1. Reached hover nut. eef: {eef_pos}, nut: {nut_pos}")
                phase = 1
            else:
                delta_pos = (target_pos - eef_pos) * 10.0 # Fast hover motion
                
        elif phase == 1: # Lower down to nut
            target_pos = nut_pos + np.array([0, 0, -0.025]) # Lower onto nut handle
            dist = np.linalg.norm(target_pos - eef_pos)
            if eef_pos[2] < (nut_pos[2] - 0.005): # Triggers when EEF reaches nut handle contact (~0.824)
                if lower_start_step == 0:
                    lower_start_step = step_count
                elif step_count - lower_start_step > 10: # Fast depth check
                    print(f"Phase 1 -> 2. Lowered to nut. eef_z={eef_pos[2]:.3f}, nut_z={nut_pos[2]:.3f}")
                    phase = 2
                    grasp_start_step = step_count
            else:
                delta_pos = (target_pos - eef_pos) * 8.0 # Responsive lowering
                
        elif phase == 2: # Grasp
            gripper = 1.0 # close
            if step_count - grasp_start_step > 50: # Allow full closure of physical gripper fingers before lifting
                print(f"Phase 2 -> 3. Grasped nut.")
                phase = 3
            # Maintain slight downward pressure while fingers close around nut
            delta_pos = np.array([0, 0, -0.05])
            # Keep delta_ori active to maintain downward orientation
            
        elif phase == 3: # Lift
            gripper = 1.0
            target_pos = np.array([eef_pos[0], eef_pos[1], 1.05]) # Lift straight up to a fixed height
            dist = np.linalg.norm(target_pos - eef_pos)
            if eef_pos[2] > 1.00: # Reached clearance height
                # Verify that the nut was physically lifted off table (table z=0.80, nut rest z=0.83)
                if nut_pos_body[2] > 0.88:
                    print(f"Phase 3 -> 4. Lifted nut successfully (nut_z={nut_pos_body[2]:.3f}).")
                    phase = 4
                else:
                    print(f"Grasp failed: Nut stayed on table (nut_z={nut_pos_body[2]:.3f}). Discarding episode...")
                    return False
            else:
                delta_pos = (target_pos - eef_pos) * 8.0 # Fast lift speed
                
        elif phase == 4: # Hover above peg
            gripper = 1.0
            # Fail fast if nut drops off during transport
            if nut_pos_body[2] < 0.85:
                print(f"Grasp lost: Nut dropped mid-air (nut_z={nut_pos_body[2]:.3f})! Discarding episode...")
                return False
            # Small forward offset to align nut opening with peg rod
            place_offset = np.array([0.012, 0, 0]) # 12mm forward offset for peg alignment
            peg_target = peg1_pos + place_offset
            
            # Object-centric tracking: offset eef_pos by the exact difference between eef and the nut!
            target_pos = peg_target + np.array([0, 0, 0.15]) + (eef_pos - nut_pos_body)
            dist = np.linalg.norm(nut_pos_body[:2] - peg_target[:2])
            xy_dist = dist
            
            # Gentle orientation correction in Phase 4 to avoid violently snapping the nut out of grasp!
            delta_ori = ori_error * 2.0
            
            if step_count % 50 == 0:
                print(f"Phase 4 tracking. xy_dist={xy_dist:.3f}, nut_z={nut_pos_body[2]:.3f}, peg_z={peg1_pos[2]:.3f}")
            if xy_dist < 0.04 and abs(nut_pos_body[2] - (peg1_pos[2] + 0.15)) < 0.04:
                print(f"Phase 4 -> 5. Hovering above peg.")
                phase = 5
            else:
                delta_pos = (target_pos - eef_pos) * 4.0 # Steady move to peg
                
        elif phase == 5: # Lower onto peg
            gripper = 1.0
            place_offset = np.array([0.012, 0, 0])
            peg_target = peg1_pos + place_offset
            # Use object-centric tracking for lowering as well
            target_pos = peg_target + np.array([0, 0, 0.00]) + (eef_pos - nut_pos_body)
            dist = np.linalg.norm(nut_pos_body[:2] - peg1_pos[:2])
            if nut_pos_body[2] < (peg1_pos[2] + 0.04):
                print(f"Phase 5 -> 6. Lowered onto peg. dist={dist:.3f}")
                phase = 6
                release_start_step = step_count
            else:
                delta_pos = (target_pos - eef_pos) * 8.0 # Fast lowering
                
        elif phase == 6: # Release
            gripper = -1.0
            place_offset = np.array([0.012, 0, 0])
            peg_target = peg1_pos + place_offset
            # Keep the offset fixed to what it was at release to avoid jerking
            target_pos = peg_target + np.array([0, 0, 0.25]) + (eef_pos - nut_pos_body)
            if step_count - release_start_step > 10:
                delta_pos = (target_pos - eef_pos) * 8.0
            else:
                delta_pos = np.zeros(3)
                
            if env._check_success():
                print("Task successful!")
                return True

        # Clip delta pos for safety while allowing brisk movement
        delta_pos = np.clip(delta_pos, -0.6, 0.6)
        
        active_arm = env.robots[0].arms[0] if len(env.robots[0].arms) > 0 else "right"
        if device is not None:
            input_ac_dict = device.input2action(goal_update_mode="target")
            if input_ac_dict is None:
                print("User triggered reset via device!")
                return False
                
            dpos = input_ac_dict[f"{active_arm}_delta"][:3]
            dquat = input_ac_dict[f"{active_arm}_delta"][3:6]
            dgrasp = input_ac_dict[f"{active_arm}_gripper"][0]
            
            # Exclusionary override: If user provides input, take full control.
            # Once user stops, the script takes control back instantly.
            if np.any(np.abs(dpos) > 0.05) or np.any(np.abs(dquat) > 0.05):
                delta_pos = dpos * 10.0
                delta_ori = dquat * 10.0
                
            # Gripper override
            if device.__class__.__name__.lower() == "spacemouse":
                if dgrasp == 1.0:
                    gripper = -gripper # Invert script's gripper while holding button
        
        # Action array: [dx, dy, dz, ax, ay, az, gripper]
        action = np.concatenate([delta_pos, delta_ori, [gripper]])
        
        if absolute_actions:
            from scipy.spatial.transform import Rotation as R
            abs_pos = eef_pos + np.clip(delta_pos, -1.0, 1.0) * 0.05
            abs_ori = (R.from_rotvec(np.clip(delta_ori, -1.0, 1.0) * 0.5) * R.from_quat(eef_quat)).as_rotvec()
            action = np.concatenate([abs_pos, abs_ori, [gripper]])
        
        start_time = time.time()
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        env.render()
        
        elapsed = time.time() - start_time
        if elapsed < 1.0 / max_fr:
            time.sleep(1.0 / max_fr - elapsed)

    env.close()
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default=os.path.join(suite.models.assets_root, "demonstrations_private"))
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Nero")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--device", type=str, default="none", help="Choice of device: none, keyboard, spacemouse")
    parser.add_argument("--max_fr", type=int, default=10, help="Max frame rate (fps) to sleep to; 20 is real-time, 10 is half-speed.")
    parser.add_argument("--absolute_actions", action="store_true", help="Record absolute actions instead of delta actions.")
    args = parser.parse_args()

    controller_config = load_composite_controller_config(
        controller=None,
        robot=args.robots[0],
    )
    
    # We ensure OSC_POSE uses delta inputs (default)
    for arm in ["right", "left"]:
        if arm in controller_config["body_parts"]:
            controller_config["body_parts"][arm]["type"] = "OSC_POSE"
            controller_config["body_parts"][arm]["input_type"] = "absolute" if args.absolute_actions else "delta"
            controller_config["body_parts"][arm]["input_ref_frame"] = "world"
            controller_config["body_parts"][arm]["use_action_scaling"] = True

    # Configure aggressive domain randomization for object placement
    placement_initializer = None
    if args.environment == "Lift":
        placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=None, # Will be set internally by the env
            x_range=[-0.15, 0.15], # Expanded 30cm spawn width
            y_range=[-0.15, 0.15],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=(0, 0, 0.8),
            z_offset=0.01,
        )
    elif args.environment in ["NutAssembly", "NutAssemblySquare", "NutAssemblyRound"]:
        from robosuite.utils.placement_samplers import SequentialCompositeSampler
        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="SquareNutSampler",
                x_range=[0.05, 0.1], # Spawn further away from the robot
                y_range=[-0.1, 0.1], # Centered
                rotation=[3*np.pi/4, 5*np.pi/4], # Full 360-degree rotation
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.82),
                z_offset=0.02,
            )
        )
        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name="RoundNutSampler",
                x_range=[-0.115, -0.11],
                y_range=[-0.225, -0.11],
                rotation=None,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.82),
                z_offset=0.02,
            )
        )

    env = suite.make(
        args.environment,
        robots=args.robots,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="frontview",
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        controller_configs=controller_config,
        placement_initializer=placement_initializer,
    )

    new_dir = args.directory
    os.makedirs(new_dir, exist_ok=True)
    tmp_directory = os.path.join(args.directory, ".tmp_collection")
    os.makedirs(tmp_directory, exist_ok=True)

    env = DataCollectionWrapper(env, tmp_directory)

    env_info = json.dumps({
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
        "has_renderer": True,
        "has_offscreen_renderer": True,
        "render_camera": "frontview",
        "ignore_done": True,
        "use_camera_obs": True,
        "reward_shaping": True,
        "control_freq": 20,
    })

    # initialize device
    device = None
    if args.device == "keyboard":
        from robosuite.devices import Keyboard
        device = Keyboard(env=env)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse
        device = SpaceMouse(env=env)
        
    if device is not None:
        device.start_control()

    # Check for existing successful episodes in the temp directory to support resumption
    import glob
    import shutil
    existing_eps = glob.glob(os.path.join(tmp_directory, "ep_*"))
    successful_episodes = 0
    for ep_dir in existing_eps:
        state_paths = glob.glob(os.path.join(ep_dir, "state_*.npz"))
        is_success = False
        for state_file in state_paths:
            try:
                dic = np.load(state_file, allow_pickle=True)
                if dic["successful"]:
                    is_success = True
                    break
            except Exception:
                pass
        
        if is_success:
            successful_episodes += 1
        else:
            shutil.rmtree(ep_dir, ignore_errors=True)
            
    total_attempts = successful_episodes
    
    if successful_episodes > 0:
        print(f"\nResuming collection: Found {successful_episodes} successful episodes in temporary directory.")
    
    while successful_episodes < args.num_episodes:
        total_attempts += 1
        print(f"Collecting episode {successful_episodes+1}/{args.num_episodes} (Attempt {total_attempts})...")
        if args.environment == "Lift":
            success = collect_lift(env, args.max_steps, args.max_fr, args.absolute_actions)
        elif args.environment in ["NutAssembly", "NutAssemblySquare", "NutAssemblyRound"]:
            success = collect_nut_assembly(env, args.max_steps, device, args.max_fr, args.absolute_actions)
        else:
            print(f"No scripted policy available for {args.environment}")
            break
            
        if success is None:
            print("\nUser aborted via device. Exiting.")
            sys.exit(0)
            
        # We check the return value because DataCollectionWrapper wipes env.successful on flush!
        if success or getattr(env, "successful", False):
            env.successful = True  # Force wrapper's flag to True so _flush saves it as successful
            successful_episodes += 1
            print(f"Episode successful! Total collected: {successful_episodes}")
        else:
            print(f"Episode failed. Discarding and retrying...")
            import shutil
            if getattr(env, "ep_directory", None) is not None and os.path.exists(env.ep_directory):
                shutil.rmtree(env.ep_directory, ignore_errors=True)
                env.has_interaction = False # Prevent wrapper from flushing to deleted directory on next reset!
                if hasattr(env, "states"):
                    env.states = []
                if hasattr(env, "action_infos"):
                    env.action_infos = []
            if device is not None:
                device.start_control()

    print("\nCollection finished. Gathering demonstrations...")
    env.close()
    gather_demonstrations_as_hdf5(tmp_directory, new_dir, env_info)
    
    dataset_path = os.path.join(new_dir, "demo.hdf5")
    output_name = os.path.join(new_dir, "demo_image.hdf5")
    
    # Dynamically locate robomimic script without hardcoded user paths
    robomimic_script = None
    robomimic_root = None

    try:
        import robomimic
        script_candidate = os.path.join(os.path.dirname(robomimic.__file__), "scripts", "dataset_states_to_obs.py")
        if os.path.exists(script_candidate):
            robomimic_script = script_candidate
            robomimic_root = os.path.dirname(os.path.dirname(robomimic.__file__))
    except ImportError:
        pass

    if robomimic_script is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        candidates = [
            os.path.join(repo_root, "dependencies", "robomimic", "robomimic", "scripts", "dataset_states_to_obs.py"),
            os.path.join(repo_root, "..", "dependencies", "robomimic", "robomimic", "scripts", "dataset_states_to_obs.py"),
            os.path.join(os.path.dirname(repo_root), "Beyond-Action-Residuals-ZPRL-", "dependencies", "robomimic", "robomimic", "scripts", "dataset_states_to_obs.py"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                robomimic_script = candidate
                robomimic_root = os.path.abspath(os.path.join(os.path.dirname(candidate), "../.."))
                break

    if robomimic_script and os.path.exists(robomimic_script):
        cmd = [
            "python", robomimic_script,
            "--dataset", dataset_path,
            "--output_name", "demo_image.hdf5",
            "--camera_names", "agentview", "robot0_eye_in_hand",
            "--camera_height", "84",
            "--camera_width", "84",
            "--done_mode", "2"
        ]
        print(f"Running command: {' '.join(cmd)}")
        my_env = os.environ.copy()
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        pythonpath_entries = [repo_root]
        if robomimic_root:
            pythonpath_entries.append(robomimic_root)
        if "PYTHONPATH" in my_env:
            pythonpath_entries.append(my_env["PYTHONPATH"])
        my_env["PYTHONPATH"] = ":".join(pythonpath_entries)

        import subprocess
        subprocess.run(cmd, env=my_env)
        print(f"Extraction complete! Saved to {output_name}")
        
        # Post-process the extracted dataset to exactly match the multihuman HDF5 structure
        print("Post-processing dataset structure to match multihuman dataset...")
        try:
            import h5py
            import numpy as np
            with h5py.File(output_name, "r+") as f:
                demos = list(f["data"].keys())
                demos = sorted(demos, key=lambda x: int(x.split("_")[1]))
                
                # Fix structure
                for demo in demos:
                    for obs_key in ["obs", "next_obs"]:
                        grp = f["data"][demo][obs_key]
                        if "robot0_eef_quat_site" in grp:
                            del grp["robot0_eef_quat_site"]
                        if "robot0_joint_acc" in grp:
                            del grp["robot0_joint_acc"]
                            
                        n_samples = grp["robot0_eef_pos"].shape[0]
                        if "robot0_eef_vel_ang" not in grp:
                            grp.create_dataset("robot0_eef_vel_ang", data=np.zeros((n_samples, 3), dtype=np.float64))
                        if "robot0_eef_vel_lin" not in grp:
                            grp.create_dataset("robot0_eef_vel_lin", data=np.zeros((n_samples, 3), dtype=np.float64))
                            
                # Create mask groups
                if "mask" in f:
                    del f["mask"]
                mask_grp = f.create_group("mask")
                
                mid = len(demos) // 2
                d1 = [d.encode("utf-8") for d in demos[:mid]]
                d2 = [d.encode("utf-8") for d in demos[mid:]]
                
                mask_grp.create_dataset("50_percent", data=np.array(d1))
                mask_grp.create_dataset("50_percent_in_origin", data=np.array(d2))
            print(f"Post-processing complete! Dataset is now fully aligned with multihuman structure.")
        except Exception as e:
            print(f"Failed to post-process dataset: {e}")

