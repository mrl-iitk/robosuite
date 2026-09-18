from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

NUM_CUBES = 4  # this task is fixed to exactly 4 cubes


class CubeRow(ManipulationEnv):
    """
    This class corresponds to a cube-arranging task for a single robot arm: pick up exactly
    4 plain (uncolored / unmarked) cubes and place them in a straight row with a fixed gap
    between adjacent cubes. Structured the same way as robosuite's built-in `Lift` task --
    same lifecycle hooks (`_load_model`, `_setup_references`, `_setup_observables`, `reward`,
    `_check_success`, `visualize`) -- just generalized from one cube to four, with an explicit
    goal (a row layout) instead of "lift off the table".

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for the table.

        cube_size (float): full side length of each cube, in meters (default 0.045 = 4.5cm).

        gap (float): spacing between adjacent cubes in the goal row, in meters (default 0.005 = 0.5cm).

        spawn_location (2-tuple): explicit (x, y) anchor, in table-local meters relative to the
            table center, for the cube_0 (row=0, col=0) spawn cell. The remaining 3 cubes are laid
            out on a 2x2 grid extending from this anchor by `spawn_spacing` in the +x and +y
            directions, so all 4 start with generous, known separation instead of scattered randomly
            across the whole table. E.g. (-0.25, -0.25) anchors the grid in the corner nearest the
            robot's -x/-y side.

        spawn_spacing (float): center-to-center spacing (m) between the 4 spawn grid cells.

        placement_tolerance (float): per-cube xy distance (m) to its goal slot in the row required
            to count as "placed".

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will be used to place the 4
            cubes on every reset (must already contain all 4 `self.cubes` via `add_objects`), else
            a per-cube `UniformRandomSampler` grid anchored at `spawn_location` is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        lite_physics (bool): Whether to optimize for mujoco forward and step calls to reduce total simulation overhead.
            Set to False to preserve backward compatibility with datasets collected in robosuite <= 1.4.1.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
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
        cube_size=0.045,
        gap=0.005,
        spawn_location=(-0.25, -0.25),
        spawn_spacing=0.12,
        placement_tolerance=0.01,
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
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))

        # cube / layout settings
        self.num_cubes = NUM_CUBES
        self.cube_size = cube_size
        self.gap = gap
        self.spawn_location = np.array(spawn_location, dtype=float)
        self.spawn_spacing = spawn_spacing
        self.placement_tolerance = placement_tolerance
        # fixed (x, y) goal slots for the row, spaced by cube_size + gap, table-local coords
        self.target_positions = self._compute_row_layout(self.num_cubes, cube_size, gap)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer -- kept separately from self.placement_initializer because
        # _load_model() runs on every hard reset and rebuilds a *new* SequentialCompositeSampler each
        # time (composite samplers don't support add_objects(), unlike a flat UniformRandomSampler),
        # so the "did the user pass one in" check must not be against self.placement_initializer itself
        self._external_placement_initializer = placement_initializer
        self.placement_initializer = placement_initializer

        # populated in _load_model / _setup_references
        self.cubes = []
        self.cube_body_ids = []

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

    def _compute_row_layout(self, n, cube_size, gap):
        """Fixed table-local (x, y) goal slots for an n-cube row, spaced by cube_size + gap."""
        pitch = cube_size + gap
        start = -(n - 1) * pitch / 2.0
        return [np.array([0.15, start + i * pitch]) for i in range(n)]

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if all 4 cubes are arranged in the row

        Un-normalized summed components if using reward shaping, averaged per-cube:

            - Reaching: in [0, 1], to encourage the arm to reach an unplaced cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping that cube
            - Placing: in [0, 1], non-zero (and only counted) once the cube is grasped, encourages
              carrying it to its row slot

        The sparse reward only consists of the row-complete component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:
            gripper = self.robots[0].gripper
            per_cube_reward = 0.0
            for cube in self.cubes:
                dist_to_target = self._cube_xy_dist_to_target(cube)
                placed = dist_to_target < self.placement_tolerance and self._cube_resting(cube)
                if placed:
                    per_cube_reward += 1.0
                    continue

                # reaching reward
                reach_dist = self._gripper_to_target(
                    gripper=gripper, target=cube.root_body, target_type="body", return_distance=True
                )
                reaching_reward = 1 - np.tanh(10.0 * reach_dist)
                per_cube_reward += reaching_reward

                # grasping reward
                grasping = self._check_grasp(gripper=gripper, object_geoms=cube)
                if grasping:
                    per_cube_reward += 0.25
                    # placing reward, only once grasped (encourages carrying toward the slot)
                    per_cube_reward += 1 - np.tanh(10.0 * dist_to_target)

            reward = per_cube_reward / self.num_cubes

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize the 4 cubes -- plain, uniform material, no color-coded goal markers
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        plain_material = CustomMaterial(
            texture="WoodLight",
            tex_name="cube_tex",
            mat_name="cube_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        cube_half = self.cube_size / 2.0
        self.cubes = [
            BoxObject(
                name=f"cube_{i}",
                size_min=[cube_half, cube_half, cube_half],
                size_max=[cube_half, cube_half, cube_half],
                material=plain_material,
                density=300,
            )
            for i in range(self.num_cubes)
        ]

        # Create placement initializer.
        # NOTE: unlike Lift's single flat UniformRandomSampler, our default sampler here is a
        # SequentialCompositeSampler (one sub-sampler per cube, for the spawn grid). Composite
        # samplers don't support add_objects() -- they're rebuilt from scratch every _load_model()
        # call instead, using freshly-created self.cubes each time. Only an *externally supplied*
        # sampler (passed in at construction) gets the reset()+add_objects() treatment.
        if self._external_placement_initializer is not None:
            self._external_placement_initializer.reset()
            self._external_placement_initializer.add_objects(self.cubes)
            self.placement_initializer = self._external_placement_initializer
        else:
            # 2x2 grid of per-cube samplers anchored at spawn_location, so the 4 cubes start
            # with generous, known separation instead of anywhere on the table.
            self.placement_initializer = SequentialCompositeSampler(name="SpawnGridSampler")
            ncols = 2
            jitter = self.spawn_spacing * 0.15  # small jitter, cubes stay well separated
            for i, cube in enumerate(self.cubes):
                row, col = divmod(i, ncols)
                cell_center = self.spawn_location + np.array([row * self.spawn_spacing, col * self.spawn_spacing])
                self.placement_initializer.append_sampler(
                    UniformRandomSampler(
                        name=f"CubeSampler{i}",
                        mujoco_objects=cube,
                        x_range=[cell_center[0] - jitter, cell_center[0] + jitter],
                        y_range=[cell_center[1] - jitter, cell_center[1] + jitter],
                        rotation=(0, 0),  # keep axis-aligned so the final row packs cleanly
                        rotation_axis="z",
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=True,
                        reference_pos=self.table_offset,
                        z_offset=0.01,
                    )
                )

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cubes,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube_body_ids = [self.sim.model.body_name2id(cube.root_body) for cube in self.cubes]

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
            full_prefixes = self._get_arm_prefixes(self.robots[0])

            sensors = []
            for i, cube in enumerate(self.cubes):
                target_xy = self.table_offset[:2] + self.target_positions[i]

                @sensor(modality=modality)
                def cube_pos(obs_cache, body_id=self.cube_body_ids[i] if self.cube_body_ids else None, cube=cube):
                    bid = body_id if body_id is not None else self.sim.model.body_name2id(cube.root_body)
                    return np.array(self.sim.data.body_xpos[bid])

                @sensor(modality=modality)
                def cube_quat(obs_cache, body_id=self.cube_body_ids[i] if self.cube_body_ids else None, cube=cube):
                    bid = body_id if body_id is not None else self.sim.model.body_name2id(cube.root_body)
                    return convert_quat(np.array(self.sim.data.body_xquat[bid]), to="xyzw")

                @sensor(modality=modality)
                def cube_to_target(obs_cache, key=f"cube_{i}_pos", tgt=target_xy):
                    if key in obs_cache:
                        xy = obs_cache[key][:2] - tgt
                        return np.array([xy[0], xy[1], 0.0])
                    return np.zeros(3)

                cube_pos.__name__ = f"cube_{i}_pos"
                cube_quat.__name__ = f"cube_{i}_quat"
                cube_to_target.__name__ = f"cube_{i}_to_target"
                sensors += [cube_pos, cube_quat, cube_to_target]

                # gripper to cube position sensor; one for each arm
                sensors += [
                    self._get_obj_eef_sensor(full_pf, f"cube_{i}_pos", f"{arm_pf}gripper_to_cube_{i}_pos", modality)
                    for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
                ]

            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the
        nearest not-yet-placed cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the closest unplaced cube
        if vis_settings["grippers"]:
            unplaced = [
                cube
                for cube in self.cubes
                if not (self._cube_xy_dist_to_target(cube) < self.placement_tolerance and self._cube_resting(cube))
            ]
            target_cube = unplaced[0] if unplaced else self.cubes[-1]
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=target_cube)

    def _cube_body_id(self, cube):
        return self.sim.model.body_name2id(cube.root_body)

    def _cube_xy_dist_to_target(self, cube):
        i = self.cubes.index(cube)
        cube_pos = self.sim.data.body_xpos[self._cube_body_id(cube)]
        target_xy = self.table_offset[:2] + self.target_positions[i]
        return np.linalg.norm(cube_pos[:2] - target_xy)

    def _cube_resting(self, cube):
        cube_z = self.sim.data.body_xpos[self._cube_body_id(cube)][2]
        table_z = self.model.mujoco_arena.table_offset[2]
        return abs(cube_z - table_z) < 0.03

    def _check_success(self):
        """
        Check if all 4 cubes have been arranged into their row slots.

        Returns:
            bool: True if every cube is within `placement_tolerance` of its goal slot and resting
        """
        return all(
            self._cube_xy_dist_to_target(cube) < self.placement_tolerance and self._cube_resting(cube)
            for cube in self.cubes
        )


if __name__ == "__main__":
    import robosuite

    env = robosuite.make(
        "CubeRow",
        robots="Panda",
        spawn_location=(-0.25, -0.25),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()
    low, high = env.action_spec
    for _ in range(500):
        action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)
        env.render()
    env.close()