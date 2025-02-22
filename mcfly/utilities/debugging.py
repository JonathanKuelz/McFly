from abc import ABC, abstractmethod

from isaacsim.core.utils.stage import get_current_stage
from omni.isaac.core import World
from pxr import Usd


class DebugInterfaceBaseClass(ABC):
    """A helper class that defines the interfaces necessary to attach custom code to the debug extension."""

    _world: World
    _stage: Usd.Stage

    @abstractmethod
    def execute():
        pass

    @abstractmethod
    def setup_scene():
        pass

    @property
    def scene(self):
        return self._world.scene

    @property
    def world(self) -> World:
        if self._world is None:
            self._world = World()
        return self._world

    @property
    def stage(self):
        return get_current_stage()
