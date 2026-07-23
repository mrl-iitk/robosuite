import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Nero(ManipulatorModel):
    """
    Nero is a 7-DoF robotic arm designed by AgileX Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/nero/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)))

    @property
    def default_base(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return {"right": "AgxGripper"} # Fallback gripper, since Nero doesn't have one specified in robosuite

    @property
    def default_controller_config(self):
        return {"right": "default_panda"} # Use panda config as baseline

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]) # Default safe position

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.25, 0, 0.8), # Moved base much closer so peg is well within 0.5m reach
        }

    @property
    def default_base(self):
        return "NullMount"

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
