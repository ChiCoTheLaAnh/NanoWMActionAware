"""
DataSource package for world model datasets.
"""

from .base import DataSource, TrajectoryData
from .factory import create_data_source


def __getattr__(name):
    """Load optional dataset backends only when their classes are requested."""
    if name in {
        "DinoWorldModelDataSource",
        "PushTDataSource",
        "DeformableEnvDataSource",
    }:
        from . import dino_wm

        return getattr(dino_wm, name)
    if name == "LeRobotDataSource":
        from .lerobot import LeRobotDataSource

        return LeRobotDataSource
    if name in {"CSGODataSource", "VizDoomDataSource"}:
        from . import game

        return getattr(game, name)
    raise AttributeError(name)

__all__ = [
    "DataSource",
    "TrajectoryData",
    "DinoWorldModelDataSource",
    "PushTDataSource",
    "DeformableEnvDataSource",
    "LeRobotDataSource",
    "CSGODataSource",
    "VizDoomDataSource",
    "create_data_source",
]
