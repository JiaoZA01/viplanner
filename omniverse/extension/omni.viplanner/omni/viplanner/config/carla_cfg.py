# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import DCMotorCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.viplanner.utils import UnRealImporterCfg

# isort: off
from .base_cfg import ViPlannerBaseCfg
from ..viplanner import DATA_DIR


LIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/recklessdriver/projects/Models/limo0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            "front_left_wheel": 0.0,
            "front_right_wheel": 0.0,
            "rear_left_wheel": 0.0,
            "rear_right_wheel": 0.0,
        },
    ),
    actuators={
        "wheels": DCMotorCfg(
            joint_names_expr=[
                "front_left_wheel",
                "front_right_wheel",
                "rear_left_wheel",
                "rear_right_wheel",
            ],
            effort_limit=400.0,
            saturation_effort=400.0,
            velocity_limit=100.0,
            stiffness={".*": 0.0},
            damping={".*": 10.0},
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the Carla scene with a Limo differential-drive robot."""

    terrain = UnRealImporterCfg(
        prim_path="/World/Carla",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path="/home/recklessdriver/projects/viplanner/models/viplanner/carla_export/new_carla_export/carla.usd",
        groundplane=True,
        cw_config_file=os.path.join(DATA_DIR, "town01", "cw_multiply_cfg.yml"),
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "town01", "keyword_mapping.yml"),
        people_config_file=os.path.join(DATA_DIR, "town01", "people_cfg.yml"),
        vehicle_config_file=os.path.join(DATA_DIR, "town01", "vehicle_cfg.yml"),
        axis_up="Z",
    )

    robot = LIMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/limo/chassis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/GroundPlane"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/limo/.*",
        history_length=3,
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
        ),
    )

    depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/limo/chassis_link/depth_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=3.8,
        ),
        width=848,
        height=480,
        data_types=["distance_to_image_plane"],
    )

    semantic_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/limo/chassis_link/semantic_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.15),
            rot=(-0.5, 0.5, -0.5, 0.5),
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=3.8,
        ),
        width=1280,
        height=720,
        data_types=["semantic_segmentation", "rgb"],
        colorize_semantic_segmentation=False,
    )


@configclass
class ViPlannerCarlaCfg(ViPlannerBaseCfg):
    """Configuration for the Carla navigation environment."""

    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()

        self.viewer.eye = (133, 127.5, 8.5)
        self.viewer.lookat = (125.5, 120, 1.0)

        self.scene.robot.init_state.pos = (125.5, 330.5, 0.001)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)
