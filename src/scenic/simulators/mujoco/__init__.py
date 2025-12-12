"""MuJoCo simulator package for Scenic.

This file exposes the primary simulator classes so tests and users can import
them as ``from scenic.simulators.mujoco import MujocoSimulator``.
"""

from .simulator import MujocoSimulator, MujocoSimulation

__all__ = ["MujocoSimulator", "MujocoSimulation"]
