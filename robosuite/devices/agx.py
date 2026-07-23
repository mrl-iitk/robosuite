"""
Driver class for PyAgxArm teleoperation controller.
"""

import numpy as np
import atexit

from robosuite.devices import Device

class AgxArm(Device):
    """
    A teleoperation controller using a physical AgxArm (Nero).
    Maps leader arm joint angles directly to the simulated arm.
    """

    def __init__(self, env):
        super().__init__(env)
        
        self._reset_state = 0
        self._enabled = False
        
        self.robot = None
        self.end_effector = None
        self.gripper_state = -1 # start open (-1)
        self.last_joint_angles = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0])
        
        atexit.register(self.shutdown)
        
        try:
            from pynput.keyboard import Key, Listener
            self.listener = Listener(on_press=self.on_press, on_release=self.on_release)
            self.listener.start()
        except ImportError:
            print("[AgxArm] pynput not installed. Keyboard gripper control disabled.")
        
        try:
            from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
            
            # Use socketcan with can0 for now, but gracefully handle missing hardware
            print("[AgxArm] Initializing PyAgxArm for Nero...")
            cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT, channel="can0", auto_connect=True)
            self.robot = AgxArmFactory.create_arm(cfg)
            
        except ImportError:
            print("[AgxArm] ERROR: pyAgxArm not installed! Teleop will not work.")
        except Exception as e:
            print(f"[AgxArm] WARNING: Failed to initialize pyAgxArm ({e})")

    def shutdown(self):
        if self.robot is not None:
            try:
                print("[AgxArm] Restoring to follower mode before disconnecting...")
                self.robot.set_follower_mode()
                self.robot.disconnect()
            except Exception:
                pass

    def _reset_internal_state(self):
        super()._reset_internal_state()
        self.grasp_states[self.active_robot][self.active_arm_index] = False
        self.gripper_state = -1

    def on_press(self, key):
        try:
            from pynput.keyboard import Key
            if key == Key.space:
                current_grasp = self.grasp_states[self.active_robot][self.active_arm_index]
                self.grasp_states[self.active_robot][self.active_arm_index] = not current_grasp
                self.gripper_state = 1 if not current_grasp else -1
        except Exception:
            pass

    def on_release(self, key):
        pass

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        
        if self.robot is not None:
            try:
                if self.end_effector is None:
                    self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
                self.robot.connect()
                self.robot.set_leader_mode()
                print("[AgxArm] Connected to physical arm and set to leader mode with AgxGripper.")
            except Exception as e:
                print(f"[AgxArm] WARNING: Could not connect to physical arm ({e})")
                self.robot = None
                self.end_effector = None

    def get_controller_state(self):
        """
        Not used directly by input2action since we override input2action for joint control,
        but required to implement the abstract method.
        """
        return dict(
            dpos=np.zeros(3),
            rotation=np.eye(3),
            raw_drotation=np.zeros(3),
            grasp=int(self.grasp),
            reset=self._reset_state,
            base_mode=int(self.base_mode),
        )
        
    def get_leader_joints(self):
        if self.robot is not None:
            try:
                ja = self.robot.get_leader_joint_angles() or self.robot.get_joint_angles()
                if ja is not None:
                    self.last_joint_angles = np.array(ja.msg)
            except Exception:
                pass
        
        return self.last_joint_angles

    def get_leader_gripper(self):
        """Reads the physical AgxGripper (if available) or uses keyboard state."""
        if self.end_effector is not None:
            try:
                gs = self.end_effector.get_gripper_status()
                if gs is not None:
                    # Mode width: value is in meters. Mode angle: value is in degrees.
                    if gs.msg.value < 0.02: # Adjust threshold as needed
                        self.gripper_state = 1
                    else:
                        self.gripper_state = -1
            except Exception:
                pass
        return self.gripper_state

    def input2action(self, mirror_actions=False, goal_update_mode="target"):
        """
        Overrides the default Device.input2action to provide absolute joint position control.
        """
        if self._reset_state:
            return None
            
        active_arm = self.active_arm
        robot = self.env.robots[self.active_robot]
        
        
        # Get absolute joint angles from the leader arm
        target_joint_angles = self.get_leader_joints()
        grasp = self.get_leader_gripper()
        
        ac_dict = {}
        for arm in robot.arms:
            ac_dict[f"{arm}_abs"] = np.zeros(7)
            ac_dict[f"{arm}_delta"] = np.zeros(7)
            ac_dict[f"{arm}_gripper"] = np.zeros(robot.gripper[arm].dof)
            
        # Overwrite active arm
        ac_dict[f"{active_arm}_abs"] = target_joint_angles
        
        gripper = robot.gripper[active_arm]
        gripper_dof = gripper.dof
        
        if hasattr(gripper, "grasp_qpos"):
            ac_dict[f"{active_arm}_gripper"] = getattr(gripper, "grasp_qpos")[grasp]
        else:
            ac_dict[f"{active_arm}_gripper"] = np.array([grasp] * gripper_dof)
            
        return ac_dict
