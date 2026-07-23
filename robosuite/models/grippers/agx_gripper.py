"""
AGX Gripper model for the Nero robotic arm.
"""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class AgxGripper(GripperModel):
    """
    AGX Gripper
    """

    def __init__(self, idn=0):
        """
        Args:
            idn (int or str): Number or a string name to append to this model. To
                prevent naming clashes when instantiating multiple of the same
                model, this idn is used to append to each model element name.
        """
        super().__init__(xml_path_completion("grippers/agx_gripper.xml"), idn=idn)

    def format_action(self, action):
        """
        Maps continuous action into gripper joint commands

        Args:
            action (np.array): continuous action (-1 to 1)

        Returns:
            np.array: gripper joint commands
        """
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0, 1.0]) * self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def init_qpos(self):
        """
        Returns:
            np.array: default initial qpos for the gripper (open state)
        """
        # Open state: joint1 at 0.05, joint2 at -0.05
        return np.array([0.05, -0.05])

    @property
    def _important_geoms(self):
        """
        Returns:
            dict: Important geom names associated with this gripper
        """
        return {
            "left_finger": ["gripper_link1"],
            "right_finger": ["gripper_link2"],
            "left_fingerpad": ["gripper_link1"],
            "right_fingerpad": ["gripper_link2"],
        }
        
    @property
    def dof(self):
        return 1

    @property
    def speed(self):
        return 0.2
