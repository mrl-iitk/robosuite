import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Nero7(ManipulatorModel):
    """
    Panda is a sensitive single-arm robot designed by Franka.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/nero/robot.xml"), idn=idn)

        # Set joint damping
        self.set_joint_attribute(attrib="damping", values=np.array((0.4, 0.4, 0.2, 0.2 , 0.1 , 0.1 , 0.1))/10)

    @property
    def default_base(self):
        return "NullMount"

    @property
    def default_gripper(self):
        return {"right": "PiperGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_nero7"}

    @property
    def init_qpos(self):
        # All-zero qpos extends the arm straight up (eef ends up ~0.9m above the table),
        # far outside the workspace the Stack task expects. This pose instead points the
        # gripper straight down (local z-axis aligned with world -z, matching the
        # convention scripted policies assume) with the eef centered over the table at
        # the task's hover height (table height + 0.15m). This particular IK solution was
        # chosen (out of the arm's 1-DOF-redundant solution family for that pose) because
        # it keeps every joint well clear of its limits, leaving headroom to move in any
        # direction -- other solutions reach the same pose but pin 2+ joints near their
        # limits, which stalls the OSC controller as soon as it tries to move off-center.
        return np.array([0.3312, 0.4951, -0.4556, 1.7133, 0.0425, -0.3086, 0.9589])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
