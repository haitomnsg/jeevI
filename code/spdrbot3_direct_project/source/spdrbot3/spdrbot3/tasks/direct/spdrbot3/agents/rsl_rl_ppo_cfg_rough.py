# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# PPO runner config for rough-terrain training.
#
# Network architecture is IDENTICAL to the flat-terrain config ([64, 64])
# so that pretrained weights can be loaded directly for transfer learning.
# Learning rate is lower (5e-5 vs 1e-4) to fine-tune without destroying
# the flat-terrain knowledge.

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORoughRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 100               # much longer than flat (was 500)
    save_interval = 50
    experiment_name = "spdr3_rough"      # separate log directory
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[64, 64],      # SAME as flat – required for weight transfer
        critic_hidden_dims=[64, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,              # slightly more exploration (was 0.002)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,            # lower LR for fine-tuning (was 1e-4)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
