# spdrbot3_env_cfg_rough.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Rough terrain environment config for SpdrBot3.
# Uses transfer learning from the flat-terrain pretrained model.
# Observation space kept at 48 (same as flat) for weight compatibility.

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
# Rough terrain generator configuration
# Focused on rocky / uneven surfaces with some gentle slopes for variety.
# curriculum=True organises rows by difficulty (easy→hard from row 0→9).
# ---------------------------------------------------------------------------
SPDR_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # 40% – mild random bumps (lower curriculum rows)
        "random_rough_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        # 30% – heavier random bumps (higher curriculum rows)
        "random_rough_high": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.04, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
        # 15% – gentle pyramid slopes
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        # 15% – inverted pyramid slopes
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


@configclass
class RoughEventCfg:
    """Domain randomization for rough-terrain training.

    Wider friction range than flat-terrain to improve robustness.
    Interval push perturbation forces the policy to recover from disturbances.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),   # wider than flat (was 0.8, 0.8)
            "dynamic_friction_range": (0.4, 0.8),   # wider than flat (was 0.6, 0.6)
            "restitution_range": (0.0, 0.1),
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
    # Periodically push the robot to train recovery behaviour
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Spdrbot3RoughEnvCfg(DirectRLEnvCfg):
    """SpdrBot3 rough-terrain environment configuration.

    Same observation space (48) and action space (12) as the flat-terrain
    config so that a pretrained flat-surface model can be loaded directly
    for transfer learning / fine-tuning.
    """

    # env – identical to flat
    episode_length_s = 40.0
    decimation = 4
    action_scale = 1
    action_space = 12
    observation_space = 48   # MUST match flat config for transfer learning
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

    # ── ROUGH TERRAIN ──────────────────────────────────────────────────────
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SPDR_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,       # start on easier half of the grid
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

    # scene – replicate_physics=False because each env sits on a different
    # terrain patch; env_spacing=0.0 lets the terrain generator handle placement.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200,
        env_spacing=0.0,
        replicate_physics=False,
    )

    # events (domain randomization)
    events: RoughEventCfg = RoughEventCfg()

    # robot
    robot: ArticulationCfg = SPDRBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ── REWARD SCALES (adjusted for rough terrain) ─────────────────────────
    lin_vel_reward_scale = 8.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -1.0           # relaxed – vertical motion expected on bumpy ground
    ang_vel_reward_scale = -0.02        # relaxed – uneven terrain causes angular velocity
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -1e-7
    action_rate_reward_scale = -0.005
    flat_orientation_reward_scale = -1.5  # relaxed – terrain isn't flat
    max_tilt_angle_deg = 55.0             # more tolerance before termination
    # body height reward (disabled for rough terrain)
    base_height_reward_scale = 0.0
    base_height_target = 0.10