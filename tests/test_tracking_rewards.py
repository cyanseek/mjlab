"""Tests for motion tracking rewards."""

import math
from unittest.mock import Mock

import pytest
import torch

from mjlab.tasks.tracking.mdp import (
  motion_relative_body_angular_velocity_error_exp,
  motion_relative_body_linear_velocity_error_exp,
)
from mjlab.utils.lab_api.math import quat_apply, quat_mul

NUM_ENVS, NUM_BODIES = 4, 5


def _offset_reference_env(offset: torch.Tensor) -> Mock:
  """An env stub whose robot moves like the reference rotated by the
  world-frame quaternion ``offset``."""
  gen = torch.Generator().manual_seed(0)
  shape = (NUM_ENVS, NUM_BODIES)
  cmd = Mock()
  cmd.cfg.body_names = tuple(f"body_{i}" for i in range(NUM_BODIES))
  quat = torch.randn(NUM_ENVS, 4, generator=gen)
  cmd.anchor_quat_w = quat / quat.norm(dim=-1, keepdim=True)
  cmd.body_lin_vel_w = torch.randn(*shape, 3, generator=gen)
  cmd.body_ang_vel_w = torch.randn(*shape, 3, generator=gen)
  cmd.robot_anchor_quat_w = quat_mul(offset.expand(NUM_ENVS, 4), cmd.anchor_quat_w)
  cmd.robot_body_lin_vel_w = quat_apply(offset.expand(*shape, 4), cmd.body_lin_vel_w)
  cmd.robot_body_ang_vel_w = quat_apply(offset.expand(*shape, 4), cmd.body_ang_vel_w)
  env = Mock()
  env.command_manager.get_term.return_value = cmd
  return env


@pytest.mark.parametrize("yaw", [-3.0, 0.4, math.pi])
def test_body_velocity_rewards_ignore_heading_offset(yaw: float):
  """A robot that follows the reference at a different heading gets the full
  body velocity rewards."""
  env = _offset_reference_env(
    torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
  )
  for func, std in (
    (motion_relative_body_linear_velocity_error_exp, 1.0),
    (motion_relative_body_angular_velocity_error_exp, 3.14),
  ):
    torch.testing.assert_close(func(env, "motion", std), torch.ones(NUM_ENVS))


def test_body_velocity_rewards_keep_tilt_offset():
  """A robot tilted away from the reference loses body velocity reward as if
  the velocities were compared in the world frame."""
  env = _offset_reference_env(torch.tensor([math.cos(0.2), math.sin(0.2), 0.0, 0.0]))
  cmd = env.command_manager.get_term("motion")
  for func, ref_vel, robot_vel, std in (
    (
      motion_relative_body_linear_velocity_error_exp,
      cmd.body_lin_vel_w,
      cmd.robot_body_lin_vel_w,
      1.0,
    ),
    (
      motion_relative_body_angular_velocity_error_exp,
      cmd.body_ang_vel_w,
      cmd.robot_body_ang_vel_w,
      3.14,
    ),
  ):
    error = torch.sum(torch.square(ref_vel - robot_vel), dim=-1).mean(-1)
    torch.testing.assert_close(func(env, "motion", std), torch.exp(-error / std**2))
