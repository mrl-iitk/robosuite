from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat


class Stack(ManipulationEnv):
    """
    Three-cube stacking task for a single robot arm.

    Task:

        Cube A
           ↓
        Cube B
           ↓
        Cube C

    The robot sequentially stacks:
        1. Cube A on Cube B
        2. Cube C on the existing A-B stack

    This environment is designed to provide a simple
    3-cube manipulation benchmark for sim-to-sim transfer.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):

        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    # ============================================================
    # REWARD
    # ============================================================

    def reward(self, action=None):
        """
        Three-stage stacking reward.

        Stage 1:
            Cube A stacked on Cube B.

        Stage 2:
            Cube C stacked on top of Cube A.

        Sparse reward:
            1.0 -> complete 3-cube stack
            0.0 -> otherwise

        Dense reward:
            reaching / grasping / lifting / aligning / stacking
            for the currently active cube.
        """

        r_stage1, r_stage2 = self.staged_rewards()

        if self.reward_shaping:
            reward = max(r_stage1, r_stage2)
        else:
            reward = 2.0 if r_stage2 > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Returns:

            r_stage1:
                Cube A successfully stacked on Cube B.

            r_stage2:
                Cube C successfully stacked on top of Cube A.
        """

        cubeA_pos = self.sim.data.body_xpos[self.cubeA_body_id]
        cubeB_pos = self.sim.data.body_xpos[self.cubeB_body_id]
        cubeC_pos = self.sim.data.body_xpos[self.cubeC_body_id]

        # --------------------------------------------------------
        # Stage 1: A on B
        # --------------------------------------------------------

        dist_A = min(
            [
                np.linalg.norm(
                    self.sim.data.site_xpos[
                        self.robots[0].eef_site_id[arm]
                    ]
                    - cubeA_pos
                )
                for arm in self.robots[0].arms
            ]
        )

        r_reach_A = (1 - np.tanh(10.0 * dist_A)) * 0.25

        grasping_A = self._check_grasp(
            gripper=self.robots[0].gripper,
            object_geoms=self.cubeA,
        )

        if grasping_A:
            r_reach_A += 0.25

        cubeA_height = cubeA_pos[2]
        table_height = self.table_offset[2]

        cubeA_lifted = cubeA_height > table_height + 0.04

        r_stage1 = 0.0

        if cubeA_lifted:

            r_lift_A = 1.0

            horiz_dist_AB = np.linalg.norm(
                np.array(cubeA_pos[:2])
                - np.array(cubeB_pos[:2])
            )

            r_align_AB = 0.5 * (
                1 - np.tanh(horiz_dist_AB)
            )

            r_stage1 = r_lift_A + r_align_AB

        # A must be released and touching B
        cubeA_touching_B = self.check_contact(
            self.cubeA,
            self.cubeB,
        )

        if (
            not grasping_A
            and cubeA_lifted
            and cubeA_touching_B
        ):
            stage1_success = True
        else:
            stage1_success = False

        # --------------------------------------------------------
        # Stage 2: C on top of A
        # --------------------------------------------------------

        # Only evaluate C stacking once A-B exists
        if stage1_success:

            dist_C = min(
                [
                    np.linalg.norm(
                        self.sim.data.site_xpos[
                            self.robots[0].eef_site_id[arm]
                        ]
                        - cubeC_pos
                    )
                    for arm in self.robots[0].arms
                ]
            )

            r_reach_C = (
                1 - np.tanh(10.0 * dist_C)
            ) * 0.25

            grasping_C = self._check_grasp(
                gripper=self.robots[0].gripper,
                object_geoms=self.cubeC,
            )

            if grasping_C:
                r_reach_C += 0.25

            # C must be lifted above the existing stack
            cubeC_lifted = (
                cubeC_pos[2] > table_height + 0.08
            )

            if cubeC_lifted:

                horiz_dist_CA = np.linalg.norm(
                    np.array(cubeC_pos[:2])
                    - np.array(cubeA_pos[:2])
                )

                r_align_CA = 0.5 * (
                    1 - np.tanh(horiz_dist_CA)
                )

                r_stage2 = 1.0 + r_align_CA

            else:
                r_stage2 = r_reach_C

            cubeC_touching_A = self.check_contact(
                self.cubeC,
                self.cubeA,
            )

            if (
                not grasping_C
                and cubeC_lifted
                and cubeC_touching_A
            ):
                r_stage2 = 2.0

        else:
            r_stage2 = 0.0

        return r_stage1, r_stage2

    # ============================================================
    # LOAD MODEL
    # ============================================================

    def _load_model(self):

        super()._load_model()

        # Robot placement
        xpos = self.robots[0].robot_model.base_xpos_offset[
            "table"
        ](self.table_full_size[0])

        self.robots[0].robot_model.set_base_xpos(xpos)

        # Table
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        mujoco_arena.set_origin([0, 0, 0])

        # ========================================================
        # Materials
        # ========================================================

        tex_attrib = {
            "type": "cube",
        }

        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="greenwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="bluewood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        # ========================================================
        # Three cubes
        # ========================================================

        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.02, 0.02, 0.02],
            size_max=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=greenwood,
        )

        self.cubeC = BoxObject(
            name="cubeC",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 0, 1, 1],
            material=bluewood,
        )

        cubes = [
            self.cubeA,
            self.cubeB,
            self.cubeC,
        ]

        # ========================================================
        # Placement sampler
        # ========================================================

        if self.placement_initializer is not None:

            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)

        else:

            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=cubes,

                x_range=[-0.08, 0.08],
                y_range=[-0.08, 0.08],

                rotation=None,

                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,

                reference_pos=self.table_offset,

                z_offset=0.01,
            )

        # ========================================================
        # Create task
        # ========================================================

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[
                robot.robot_model
                for robot in self.robots
            ],
            mujoco_objects=cubes,
        )

    # ============================================================
    # REFERENCES
    # ============================================================

    def _setup_references(self):

        super()._setup_references()

        self.cubeA_body_id = (
            self.sim.model.body_name2id(
                self.cubeA.root_body
            )
        )

        self.cubeB_body_id = (
            self.sim.model.body_name2id(
                self.cubeB.root_body
            )
        )

        self.cubeC_body_id = (
            self.sim.model.body_name2id(
                self.cubeC.root_body
            )
        )

    # ============================================================
    # RESET
    # ============================================================

    def _reset_internal(self):

        super()._reset_internal()

        if not self.deterministic_reset:

            object_placements = (
                self.placement_initializer.sample()
            )

            for obj_pos, obj_quat, obj in (
                object_placements.values()
            ):

                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate(
                        [
                            np.array(obj_pos),
                            np.array(obj_quat),
                        ]
                    ),
                )

    # ============================================================
    # OBSERVATIONS
    # ============================================================

    def _setup_observables(self):

        observables = super()._setup_observables()

        if self.use_object_obs:

            modality = "object"

            # ----------------------------------------------------
            # Cube A
            # ----------------------------------------------------

            @sensor(modality=modality)
            def cubeA_pos(obs_cache):
                return np.array(
                    self.sim.data.body_xpos[
                        self.cubeA_body_id
                    ]
                )

            @sensor(modality=modality)
            def cubeA_quat(obs_cache):
                return convert_quat(
                    np.array(
                        self.sim.data.body_xquat[
                            self.cubeA_body_id
                        ]
                    ),
                    to="xyzw",
                )

            # ----------------------------------------------------
            # Cube B
            # ----------------------------------------------------

            @sensor(modality=modality)
            def cubeB_pos(obs_cache):
                return np.array(
                    self.sim.data.body_xpos[
                        self.cubeB_body_id
                    ]
                )

            @sensor(modality=modality)
            def cubeB_quat(obs_cache):
                return convert_quat(
                    np.array(
                        self.sim.data.body_xquat[
                            self.cubeB_body_id
                        ]
                    ),
                    to="xyzw",
                )

            # ----------------------------------------------------
            # Cube C
            # ----------------------------------------------------

            @sensor(modality=modality)
            def cubeC_pos(obs_cache):
                return np.array(
                    self.sim.data.body_xpos[
                        self.cubeC_body_id
                    ]
                )

            @sensor(modality=modality)
            def cubeC_quat(obs_cache):
                return convert_quat(
                    np.array(
                        self.sim.data.body_xquat[
                            self.cubeC_body_id
                        ]
                    ),
                    to="xyzw",
                )

            # ----------------------------------------------------
            # Relative object positions
            # ----------------------------------------------------

            @sensor(modality=modality)
            def cubeA_to_cubeB(obs_cache):

                return (
                    obs_cache["cubeB_pos"]
                    - obs_cache["cubeA_pos"]
                    if (
                        "cubeA_pos" in obs_cache
                        and "cubeB_pos" in obs_cache
                    )
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cubeC_to_cubeA(obs_cache):

                return (
                    obs_cache["cubeA_pos"]
                    - obs_cache["cubeC_pos"]
                    if (
                        "cubeA_pos" in obs_cache
                        and "cubeC_pos" in obs_cache
                    )
                    else np.zeros(3)
                )

            # ----------------------------------------------------
            # EEF → objects
            # ----------------------------------------------------

            arm_prefixes = self._get_arm_prefixes(
                self.robots[0],
                include_robot_name=False,
            )

            full_prefixes = self._get_arm_prefixes(
                self.robots[0]
            )

            sensors = [
                cubeA_pos,
                cubeA_quat,
                cubeB_pos,
                cubeB_quat,
                cubeC_pos,
                cubeC_quat,
                cubeA_to_cubeB,
                cubeC_to_cubeA,
            ]

            sensors += [
                self._get_obj_eef_sensor(
                    full_pf,
                    f"{cube}_pos",
                    f"{arm_pf}gripper_to_{cube}",
                    modality,
                )
                for arm_pf, full_pf in zip(
                    arm_prefixes,
                    full_prefixes,
                )
                for cube in [
                    "cubeA",
                    "cubeB",
                    "cubeC",
                ]
            ]

            names = [
                s.__name__
                for s in sensors
            ]

            for name, s in zip(names, sensors):

                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    # ============================================================
    # SUCCESS
    # ============================================================

    def _check_success(self):

        r_stage1, r_stage2 = self.staged_rewards()

        return r_stage2 > 0

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize(self, vis_settings):

        super().visualize(
            vis_settings=vis_settings
        )

        if vis_settings["grippers"]:

            self._visualize_gripper_to_target(
                gripper=self.robots[0].gripper,
                target=self.cubeA,
            )