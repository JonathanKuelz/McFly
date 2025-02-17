import asyncio
import logging
from pathlib import Path
from typing import Optional

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.gui.components.ui_utils import btn_builder, dropdown_builder, get_style, state_btn_builder, str_builder
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import open_stage, open_stage_async
from omni.isaac.manipulators import SingleManipulator
import omni.ui as ui
import omni.usd
from pxr import Usd

from mcfly.control.rmpflow import FrankaRMPFlowController
from mcfly.utilities import usd_util
from mcfly.extensions.templates.extension_ui_templates import ExtensionUiTemplate


class AssemblyUI(ExtensionUiTemplate):

    def __init__(self,
                 window_title: str,
                 menu_path: Optional[str] = None):
        self._buttons = dict()
        self._current_stage_prims = []
        self._dropdowns = dict()
        self._robot = None
        self._usd_file = ""
        self._world = None
        self._world_settings = {"physics_dt": 1.0 / 60.0, "stage_units_in_meters": 1.0, "rendering_dt": 1.0 / 60.0}

        self._controller = None
        self._articulation_controller = None

        super().__init__(window_title, menu_path)

    @property
    def scene(self) -> Scene:
        return self._world.scene

    def build_window(self):
        """
        Build the main window that holds the UI elements
        """
        if not self._window:
            self._window = ui.Window(title=self._window_title, visible=False, width=340, height=300)

    def build_ui(self):
        # Create a section in the window where control inputs can be entered
        self._controls_stack = ui.VStack(spacing=5, height=0)
        with self._controls_stack:
            self._meta_frame = ui.CollapsableFrame(
                title="General Information",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            )
            self._controls_frame = ui.CollapsableFrame(
                title="World Controls",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
            )
            self.extra_stacks = ui.VStack(margin=5, spacing=5, height=0)

        with self._meta_frame:
            self._usd_field = str_builder("USD file:",
                                          on_clicked_fn=self._on_enter_usd,
                                          tooltip="Enter a path to a USD file you want to observe here.",
                                          use_folder_picker=True)

        with self._controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                dict = {
                    "label": "Load USD",
                    "type": "button",
                    "text": "Load",
                    "tooltip": "Load USD and Task",
                    "on_clicked_fn": self._on_load_usd,
                }
                self._buttons["Load USD"] = btn_builder(**dict)
                self._buttons["Load USD"].enabled = False
                self._goal_prim_field = str_builder("Goal Prim:",
                                                    on_clicked_fn=self._on_set_goal_prim,
                                                    tooltip="Enter the prim path of the goal to be reached")
                self._robot_prim_field = str_builder("Robot Prim:",
                                                     on_clicked_fn=self._on_enter_robot_prim,
                                                     tooltip="Enter the prim path of the robot to be controlled")
                dict = {
                    "label": "Load Robot",
                    "type": "button",
                    "text": "Load bot",
                    "tooltip": "Load Robot object",
                    "on_clicked_fn": self._on_load_robot,
                }
                self._buttons["Load Robot"] = btn_builder(**dict)
                self._buttons["Load Robot"].enabled = True
                dict = {
                    "label": "Move",
                    "type": "button",
                    "a_text": "START",
                    "b_text": "STOP",
                    "tooltip": "Move to goal prim",
                    "on_clicked_fn": self._on_move,
                }
                self._buttons["Move"] = state_btn_builder(**dict)
                self._buttons["Move"].enabled = False
                dict = {
                    "label": "Reset",
                    "type": "button",
                    "text": "Reset",
                    "tooltip": "Reset robot and environment",
                    "on_clicked_fn": self._on_reset,
                }
                self._buttons["Reset"] = btn_builder(**dict)
                self._buttons["Reset"].enabled = True

            self.setup_post_load()

    def setup_scene(self):
        print('setup scene')

    def setup_post_load(self, *args, **kwargs):
        pass

    async def _init_world(self):
        """See https://docs.omniverse.nvidia.com/isaacsim/latest/how_to_guides/environment_setup.html"""
        await self._world.reset_async()

    def _get_stage(self) -> Usd.Stage:
        return omni.usd.get_context().get_stage()

    def _on_enter_usd(self, usd: ui.SimpleStringModel):
        """The callback when someone changes the value of the USD path field."""
        self._usd_file = usd.get_value_as_string()
        self._buttons["Load USD"].enabled = self._usd_file != ""

    def _on_enter_robot_prim(self, robot_prim: ui.SimpleStringModel):
        # TODO: Remove bot prim from scene
        self._bot_prim = robot_prim.get_value_as_string().strip()

    def _on_load_robot(self):
        prim_path = self._bot_prim
        self._robot = Robot(prim_path, name='franka')
        logging.info("Robot loaded")

        self._controller = FrankaRMPFlowController(name="target_follower_controller", robot_articulation=self._robot)
        self._articulation_controller = self._robot.get_articulation_controller()
        logging.info("Controller loaded")

        if self.scene.object_exists('Manipulator'):
            self.scene.remove_object('Manipulator', registry_only=True)

        self._robot.initialize()
        self.scene.add(SingleManipulator(self._bot_prim, end_effector_prim_name='tcp', name='Manipulator'))
        if self._goal_prim not in ("", "/World"):
            if self.scene.object_exists("Goal"):
                self.scene.remove_object("Goal", registry_only=True)
            self.scene.add(SingleRigidPrim(self._goal_prim, name="Goal"))

        self._buttons["Move"].enabled = True

    def _on_set_goal_prim(self, goal_prim: ui.SimpleStringModel):
        prim_path = goal_prim.get_value_as_string().strip()
        if prim_path not in ("", "/World"):
            logging.info(f"Goal set to prim {prim_path}")
            self._buttons["Move"].enabled = True
        else:
            self._buttons["Move"].enabled = False
        self._goal_prim = prim_path

    def _on_load_usd(self, *args, **kwargs):
        async def _on_load_usd_async():
            await self._on_load_usd_async()
            await omni.kit.app.get_app().next_update_async()

        asyncio.ensure_future(_on_load_usd_async())

    async def _on_load_usd_async(self):
        """The user pressed the LOAD button."""
        pth = Path(self._usd_file)
        if self._usd_file.strip() != '':
            if not pth.exists():
                logging.error(f"File {pth} does not exist.")
                return
            if pth.suffix not in (".usd", ".usda"):
                logging.error(f"File {pth} is not a USD file.")
                return

        if self._usd_file.strip() != '':
            logging.info("Loading USD file")
            await open_stage_async(usd_path=self._usd_file)
            await self._on_reset_async()

    def _on_move(self, val):
        asyncio.ensure_future(self._on_follow_target_event_async(val))

    def _on_reset(self):
        asyncio.ensure_future(self._on_reset_async())

    async def _on_reset_async(self):
        if self._world is not None:
            self._world.stop()
            self._world.clear_all_callbacks()

        self._world = World(**self._world_settings)
        await self._world.initialize_simulation_context_async()

        await self._world.play_async()
        await update_stage_async()

        self.scene.clear(registry_only=True)
        self.setup_scene()

        await self._init_world()
        await self._world.pause_async()
        self.setup_post_load()

    async def _on_follow_target_event_async(self, val):
        if val:
            await self._world.play_async()
            self._world.add_physics_callback("sim_step", self._on_follow_target_simulation_step)
        else:
            self._world.remove_physics_callback("sim_step")

    def _on_follow_target_simulation_step(self, step_size):
        try:
            goal_state = self.scene.get_object('Goal').get_current_dynamic_state()
            actions = self._controller.forward(
                target_end_effector_position=goal_state.position,
                target_end_effector_orientation=goal_state.orientation,
            )
            self._articulation_controller.apply_action(actions)
        except Exception as e:
            print(e)
