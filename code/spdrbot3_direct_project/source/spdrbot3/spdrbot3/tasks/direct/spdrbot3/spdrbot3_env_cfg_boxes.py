# spdrbot3_env_cfg_boxes.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Box / cube terrain environment config for SpdrBot3.
# Terrain is composed of discrete rectangular blocks at varying heights,
# similar to the environments used by Boston Dynamics Spot training.
# Observation space kept at 48 (same as flat & rough) for weight compatibility.

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from isaaclab_assets.robots.spdrbot import SPDRBOT_CFG  # isort: skip


# ---------------------------------------------------------------------------
# Box / cube terrain generator configuration
# Primarily discrete blocks and stairs – the "cubed" look.
# curriculum=True organises rows by difficulty (easy→hard from row 0→9).
# ---------------------------------------------------------------------------
SPDR_BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=50,                # 10 x 50 = 500 patches (must be >= num_envs!)
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # 60% – small grid boxes (the primary terrain)
        "random_grid_small": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.60,
            grid_width=0.45,                      # 45 cm wide cubes
            grid_height_range=(0.005, 0.02),      # 0.5–2 cm tall (robot is only 10 cm!)
            platform_width=2.5,
        ),
        # 40% – larger grid boxes for variety
        "random_grid_large": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.40,
            grid_width=0.75,                      # 75 cm wide blocks
            grid_height_range=(0.005, 0.015),     # 0.5–1.5 cm tall
            platform_width=2.0,
        ),
    },
)


@configclass
class BoxEventCfg:
    """Domain randomization for box / cube terrain training.

    Wider friction range and periodic push perturbation for robustness.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class Spdrbot3BoxEnvCfg(DirectRLEnvCfg):
    """SpdrBot3 box / cube terrain environment configuration.

    Same observation (48) and action space (12) as flat and rough configs
    so that a pretrained model can be loaded directly for transfer learning.
    """

    # env – identical dims to flat & rough
    episode_length_s = 40.0
    decimation = 4
    action_scale = 1.0       # same as plane – training from scratch
    action_space = 12
    observation_space = 48   # MUST match for transfer learning
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=6.0,
            dynamic_friction=5.0,
            restitution=0.0,
        ),
    )

    # ── BOX / CUBE TERRAIN ─────────────────────────────────────────────────
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SPDR_BOX_TERRAINS_CFG,
        max_init_terrain_level=None,    # no restriction – all 500 patches used so robots don’t overlap
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=6.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # scene – replicate_physics=False for per-env terrain patches
    # num_envs must be <= num_rows * num_cols in the terrain generator!
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=500,
        env_spacing=0.0,
        replicate_physics=False,
    )

    # events (domain randomization)
    events: BoxEventCfg = BoxEventCfg()

    # robot
    robot: ArticulationCfg = SPDRBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── REWARD SCALES (identical to plane config for training from scratch) ──
    lin_vel_reward_scale = 8.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -1e-7
    action_rate_reward_scale = -0.005
    flat_orientation_reward_scale = -3.0
    max_tilt_angle_deg = 45.0             # same as flat config

    # ── BODY HEIGHT REWARD (encourage robot to lift its base off the ground) ─
    base_height_reward_scale = 5.0        # strong incentive to keep body high
    base_height_target = 0.10             # target base height (matches spawn z=0.1)
