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

# Cubes per layer, bottom to top -- a triangular (4+3+2+1) brick pyramid, built in a single
# row/height cross-section (like CubeRow generalized with a height axis), not a full square
# pyramid. Fixed, like CubeRow's NUM_CUBES.
LAYER_SIZES = [4, 3, 2, 1]
NUM_CUBES = sum(LAYER_SIZES)


class PyramidStack(ManipulationEnv):
    """
    Pyramid-stacking task for a single robot arm: pick up 10 plain cubes and stack them into a
    4-3-2-1 triangular pyramid (four cubes in a row on the table, three cubes bridging each
    adjacent pair on top of those, two bridging those, and a single cube at the apex).
    Structured the same way as `CubeRow` (itself modeled on `Lift`/`Stack`) -- same lifecycle
    hooks, just generalized from one row to a stack of shrinking rows.

    Each cube has a fixed identity: cubes 0-3 are the base row, 4-6 sit on top of adjacent base
    pairs, 7-8 sit on top of those, and cube 9 is the apex, so the build order is always
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9 -- pick cube i only after every cube supporting it already has
    a nonzero layer index less than i's.

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

        gap (float): horizontal clearance left between adjacent cubes within the same layer of
            the goal pyramid, in meters (default 0.010 = 1cm). Small enough that the cube one
            layer up still gets a wide bearing area on both cubes it bridges (17.5mm overlap per
            side at the default cube_size), but wide enough to give the gripper real clearance
            when placing a cube next to an already-placed same-layer neighbor.

        pyramid_anchor (2-tuple): (x, y) location, in table-local meters relative to the table
            center, of the pyramid's base-row center / footprint center.

        spawn_x_range (2-tuple): (min, max) x bound, in table-local meters relative to the table
            center, of the single shared region all 10 cubes are randomly scattered within at
            spawn. Kept entirely on the opposite side of the table from `pyramid_anchor` (whose
            footprint starts at x=0.15 by default) so a spawned cube can never land on a slot the
            pyramid needs later -- see `_cube_supported`/`_check_success` and the module-level
            discussion in this class's docstring for why that matters. Also kept inside the region
            a reference arm (Panda) can reliably reach *at the pyramid-clearing transit height*
            (see `scripted_pyramid_policy.pyramid_hover_height`), not just at table height -- a
            wider region looked fine for a single static reach check at grasp height, but reaching
            its far corners while also up at transit height turned out to leave several
            centimeters of steady-state error (not something a bigger step budget fixes), which
            was enough to cause missed grasps.

        spawn_y_range (2-tuple): (min, max) y bound of that same shared spawn region.

        placement_tolerance (float): per-cube xy distance (m) to its goal slot, and z distance
            (m) to its goal layer height, required to count as "placed" (default 0.008 = 8mm,
            ~18% of the default cube_size -- tight enough that a "success" looks like a genuinely
            square pyramid rather than a sloppy one, while still comfortably above the ~5-8mm
            residual the scripted policy typically achieves per cube).

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will be used to place the 10
            cubes on every reset (must already contain all 10 `self.cubes` via `add_objects`),
            else a per-cube `UniformRandomSampler` grid anchored at `spawn_location` is used by
            default.

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
        gap=0.010,
        pyramid_anchor=(0.15, 0.0),
        spawn_x_range=(-0.30, -0.03),
        spawn_y_range=(-0.25, 0.25),
        placement_tolerance=0.008,
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
        self.layer_sizes = LAYER_SIZES
        self.cube_size = cube_size
        self.gap = gap
        self.pyramid_anchor = np.array(pyramid_anchor, dtype=float)
        self.spawn_x_range = spawn_x_range
        self.spawn_y_range = spawn_y_range
        self.placement_tolerance = placement_tolerance

        # fixed (x, y, z) goal slot, per cube, table-local coords (xy) + absolute height offset
        # from the table (z); and, per cube, the indices of the (0-2) cubes directly below it
        # that it must end up resting on.
        self.target_positions, self.cube_layer, self.cube_supports = self._compute_pyramid_layout(
            self.layer_sizes, cube_size, gap, self.pyramid_anchor
        )

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer -- kept separately from self.placement_initializer because
        # _load_model() runs on every hard reset and rebuilds a *new* default sampler each time,
        # bound to the freshly-created self.cubes for that reset, so the "did the user pass one
        # in" check must not be against self.placement_initializer itself
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

    @staticmethod
    def _compute_pyramid_layout(layer_sizes, cube_size, gap, anchor):
        """
        Fixed table-local (x, y, z) goal slots for a triangular pyramid with `layer_sizes` cubes
        per layer (bottom to top), plus each cube's supporting cube indices in the layer below.

        Layer k's row is centered on `anchor` just like layer 0's, because the center-to-center
        pitch between adjacent cubes is the same (cube_size + gap) at every layer: a cube one
        layer up sits at the midpoint of the two cubes below it, and consecutive midpoints of a
        constant-pitch row are themselves a constant-pitch row with one fewer point, still
        centered on the same anchor.

        Returns:
            3-tuple:
                - list of np.array([dx, dy, dz]) goal offsets (dz measured from the table surface,
                  dx/dy from `anchor`), one per cube, ordered layer-major (bottom row first)
                - list of int, the layer index (0 = base) of each cube, same ordering
                - list of list of int, the indices (into the same flat cube list) of the 0-2
                  cubes each cube rests on, same ordering (empty for the base layer)
        """
        pitch = cube_size + gap
        cube_half = cube_size / 2.0

        targets = []
        layer_of = []
        supports = []
        # global index of each layer's cubes, so layer k+1 can reference layer k's indices
        layer_index_offsets = []
        idx = 0
        for k, n in enumerate(layer_sizes):
            layer_index_offsets.append(idx)
            start = -(n - 1) * pitch / 2.0
            z = cube_half + k * cube_size
            for i in range(n):
                x = anchor[0]
                y = anchor[1] + start + i * pitch
                targets.append(np.array([x, y, z]))
                layer_of.append(k)
                if k == 0:
                    supports.append([])
                else:
                    below_offset = layer_index_offsets[k - 1]
                    supports.append([below_offset + i, below_offset + i + 1])
                idx += 1

        return targets, layer_of, supports

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided once the full 10-cube pyramid is built

        Un-normalized summed components if using reward shaping, averaged per-cube:

            - Reaching: in [0, 1], to encourage the arm to reach the next not-yet-placed cube
              whose supports (if any) are already placed
            - Grasping: in {0, 0.25}, non-zero if the arm is grasping that cube
            - Placing: in [0, 1], non-zero (and only counted) once the cube is grasped, encourages
              carrying it to its goal slot

        Only one cube (the next "buildable" one -- lowest index whose supports, if any, are
        already placed) is shaped at a time, mirroring how the pyramid must physically be built
        bottom-up.

        The sparse reward only consists of the pyramid-complete component.

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
            target_cube = self._next_buildable_cube()
            if target_cube is not None:
                dist_to_target = self._cube_dist_to_target(target_cube)

                reach_dist = self._gripper_to_target(
                    gripper=gripper, target=target_cube.root_body, target_type="body", return_distance=True
                )
                reaching_reward = 1 - np.tanh(10.0 * reach_dist)

                grasping = self._check_grasp(gripper=gripper, object_geoms=target_cube)
                grasp_reward = 0.25 if grasping else 0.0
                place_reward = (1 - np.tanh(10.0 * dist_to_target)) if grasping else 0.0

                reward = reaching_reward + grasp_reward + place_reward

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

        # initialize the 10 cubes -- plain, uniform material, no color-coded goal markers
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
        # NOTE: unlike CubeRow's per-cube grid, the default sampler here is a single
        # UniformRandomSampler covering all 10 cubes at once (the same pattern Stack uses for its
        # 2 cubes) -- ensure_valid_placement=True already checks each new placement against every
        # previously-placed object in this same sample() call (see
        # robosuite/utils/placement_samplers.py), so scattering all 10 cubes across one wide
        # region still guarantees no two cubes spawn overlapping, without needing a rigid grid.
        # The region itself (spawn_x_range/spawn_y_range) is confined to the opposite side of the
        # table from the pyramid's goal footprint (pyramid_anchor), so a spawned cube can never
        # land on a slot the pyramid needs later -- see this class's docstring.
        if self._external_placement_initializer is not None:
            self._external_placement_initializer.reset()
            self._external_placement_initializer.add_objects(self.cubes)
            self.placement_initializer = self._external_placement_initializer
        else:
            self.placement_initializer = UniformRandomSampler(
                name="SpawnSampler",
                mujoco_objects=self.cubes,
                x_range=self.spawn_x_range,
                y_range=self.spawn_y_range,
                rotation=(0, 0),  # keep axis-aligned so the pyramid packs cleanly
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
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
                target_xyz = self.table_offset + self.target_positions[i]

                @sensor(modality=modality)
                def cube_pos(obs_cache, body_id=self.cube_body_ids[i] if self.cube_body_ids else None, cube=cube):
                    bid = body_id if body_id is not None else self.sim.model.body_name2id(cube.root_body)
                    return np.array(self.sim.data.body_xpos[bid])

                @sensor(modality=modality)
                def cube_quat(obs_cache, body_id=self.cube_body_ids[i] if self.cube_body_ids else None, cube=cube):
                    bid = body_id if body_id is not None else self.sim.model.body_name2id(cube.root_body)
                    return convert_quat(np.array(self.sim.data.body_xquat[bid]), to="xyzw")

                @sensor(modality=modality)
                def cube_to_target(obs_cache, key=f"cube_{i}_pos", tgt=target_xyz):
                    if key in obs_cache:
                        return np.array(obs_cache[key]) - tgt
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

            # scalar progress observable: how many cubes are correctly placed right now
            @sensor(modality=modality)
            def num_cubes_placed(obs_cache):
                return np.array([sum(self._cube_placed(cube) for cube in self.cubes)])

            num_cubes_placed.__name__ = "num_cubes_placed"
            sensors.append(num_cubes_placed)

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
        next buildable (supports already placed) cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the next buildable cube
        if vis_settings["grippers"]:
            target_cube = self._next_buildable_cube()
            if target_cube is not None:
                self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=target_cube)

    def _cube_body_id(self, cube):
        return self.sim.model.body_name2id(cube.root_body)

    def _cube_index(self, cube):
        return self.cubes.index(cube)

    def _cube_dist_to_target(self, cube):
        """Full 3D distance from `cube`'s current position to its goal slot."""
        i = self._cube_index(cube)
        cube_pos = self.sim.data.body_xpos[self._cube_body_id(cube)]
        target = self.table_offset + self.target_positions[i]
        return np.linalg.norm(cube_pos - target)

    def _cube_xy_dist_to_target(self, cube):
        i = self._cube_index(cube)
        cube_pos = self.sim.data.body_xpos[self._cube_body_id(cube)]
        target_xy = (self.table_offset + self.target_positions[i])[:2]
        return np.linalg.norm(cube_pos[:2] - target_xy)

    def _cube_z_dist_to_target(self, cube):
        i = self._cube_index(cube)
        cube_z = self.sim.data.body_xpos[self._cube_body_id(cube)][2]
        target_z = (self.table_offset + self.target_positions[i])[2]
        return abs(cube_z - target_z)

    def _cube_supported(self, cube):
        """Whether `cube` is in contact with every cube its goal slot says it should rest on
        (always True for the base layer, which rests on the table instead)."""
        i = self._cube_index(cube)
        supports = self.cube_supports[i]
        if not supports:
            return True
        return all(self.check_contact(cube, self.cubes[j]) for j in supports)

    def _cube_placed(self, cube):
        """A cube counts as placed once it is at its goal (x, y, z) within tolerance AND resting
        on the cube(s) (or table, for the base layer) its slot says it should rest on."""
        return (
            self._cube_xy_dist_to_target(cube) < self.placement_tolerance
            and self._cube_z_dist_to_target(cube) < self.placement_tolerance
            and self._cube_supported(cube)
        )

    def _next_buildable_cube(self):
        """Lowest-index not-yet-placed cube whose supports (if any) are already placed, i.e. the
        cube the build order says to pick up next. Returns None once the pyramid is complete."""
        for i, cube in enumerate(self.cubes):
            if self._cube_placed(cube):
                continue
            supports = self.cube_supports[i]
            if all(self._cube_placed(self.cubes[j]) for j in supports):
                return cube
        return None

    def _check_success(self):
        """
        Check if all 10 cubes have been built into the pyramid.

        Returns:
            bool: True if every cube is at its goal slot (within `placement_tolerance`) and
                resting on the cube(s)/table its slot says it should rest on
        """
        return all(self._cube_placed(cube) for cube in self.cubes)


if __name__ == "__main__":
    import robosuite

    env = robosuite.make(
        "PyramidStack",
        robots="Panda",
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
