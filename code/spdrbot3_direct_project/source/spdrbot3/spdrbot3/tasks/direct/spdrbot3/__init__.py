# source/spdrbot3/spdrbot3/tasks/direct/spdrbot3/__init__.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers[](https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Spdrbot3-Direct-v0",
    entry_point=f"{__name__}.spdrbot3_env:Spdrbot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spdrbot3_env_cfg:Spdrbot3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Rough terrain variant – same env class, different terrain & reward config.
# Use --resume --checkpoint <absolute_path_to_flat_model> for transfer learning.
gym.register(
    id="Template-Spdrbot3-Rough-Direct-v0",
    entry_point=f"{__name__}.spdrbot3_env:Spdrbot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spdrbot3_env_cfg_rough:Spdrbot3RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_rough:PPORoughRunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Box / cube terrain variant – discrete blocks at varying heights.
# Use --resume --checkpoint <path_to_rough_model> for transfer learning.
gym.register(
    id="Template-Spdrbot3-Boxes-Direct-v0",
    entry_point=f"{__name__}.spdrbot3_env:Spdrbot3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spdrbot3_env_cfg_boxes:Spdrbot3BoxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg_boxes:PPOBoxRunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)