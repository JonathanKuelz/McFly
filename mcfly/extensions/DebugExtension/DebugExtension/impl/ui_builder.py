import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from isaacsim.core.utils.stage import open_stage_async, update_stage_async
from isaacsim.gui.components.ui_utils import btn_builder, get_style, str_builder

from omni.isaac.core import World
import omni.kit.app
import omni.ui as ui
import omni.usd

from mcfly.extensions.templates.extension_ui_templates import ExtensionUiTemplate
from mcfly.utilities.debugging import DebugInterfaceBaseClass


class DebuggerUiBuilder(ExtensionUiTemplate):
    """Manage the debugger Extension"""

    def __init__(self, window_title, menu_path: Optional[str] = None):
        super().__init__(window_title, menu_path)
        self._text_fields = dict()
        self._buttons = dict()
        self._usd_path: str = ''

        self._stage = None
        self._world = None
        self._on_load_world()

    @property
    def debug_interface(self) -> DebugInterfaceBaseClass:
        return self._debug_interface

    @debug_interface.setter
    def debug_interface(self, val):
        self._debug_interface = val
        if val is not None:
            self._buttons['Execute'].enabled = True

    @property
    def scene(self):
        return self._world.scene

    @property
    def world(self) -> World:
        return self._world

    @property
    def stage(self):
        return self._stage

    def build_window(self):
        """Build the main window that holds the UI elements"""
        if not self._window:
            self._window = ui.Window(title=self._window_title, visible=False, width=340, height=300)

    def build_ui(self):
        """Build the Graphical User Interface (GUI) in the underlying windowing system"""
        self._ui_stack = ui.VStack(style=get_style(), spacing=5, height=0)
        with self._ui_stack:
            self._frame = ui.CollapsableFrame(
                title="Debugger",
                width=ui.Fraction(1),
                height=0,
                collapsed=False,
                style=get_style(),
                horizontal_scrollbar=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                vertical_scrollbar=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            )
        with self._frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Stage setup
                self._text_fields['USD stage'] = str_builder(
                    "USD file:", on_clicked_fn=self._on_enter_usd, tooltip="Path to a USD stage that you want to load.",
                    use_folder_picker=True)
                self._buttons['Load USD'] = btn_builder(
                    label="LOAD", text="LOAD", on_clicked_fn=self._on_load_usd, tooltip="Load the USD stage.")
                self._buttons['Load USD'].enabled = False

                # User Code
                self._text_fields['User Code'] = str_builder(
                    "Sourcecode:", on_clicked_fn=self._on_enter_code,
                    tooltip="Enter the path to a source file that exposes a DebuggerInterface class.",
                    use_folder_picker=True,
                    default_val='/home/chicken/Code/McFly/mcfly/extensions/DebugExtension/DebugExtension/grasp_demo.py'  # TODO: remove after debugging
                )
                self._code_path = '/home/chicken/Code/McFly/mcfly/extensions/DebugExtension/DebugExtension/grasp_demo.py'  # TODO: remove after debugging

                self._buttons['Load Code'] = btn_builder(label="LOAD", text="LOAD", on_clicked_fn=self._on_load_code,
                                                         tooltip="Load the code provided and sets up the stage.")
                self._buttons['Load Code'].enabled = Path(self._code_path).exists()

                self._buttons['Execute'] = btn_builder(label="EXECUTE", text="EXECUTE",
                                                       on_clicked_fn=self._on_execute_code,
                                                       tooltip="Execute the code provided by the DebuggerInterface.")
                self._buttons['Execute'].enabled = False

                # Reset
                self._buttons['Reset'] = btn_builder(
                    label="RESET", text="RESET", on_clicked_fn=self._on_reset, tooltip="Reset the scene.")

    def setup(self):
        """Can this be left empty?"""
        pass

    # --------------- UI Callbacks ---------------

    def _on_enter_usd(self, pth: ui.SimpleStringModel):
        """Callback for entering the USD file path"""
        pth = pth.get_value_as_string().strip()
        logging.debug(f"Entered USD file path: {pth}")
        self._usd_path = pth
        if Path(self._usd_path).exists():
            self._buttons['Load USD'].enabled = True

    def _on_load_usd(self, *args, **kwargs):
        async def _on_load_usd_async():
            await self._on_load_usd_async()
            await omni.kit.app.get_app().next_update_async()

        asyncio.ensure_future(_on_load_usd_async())

    def _on_enter_code(self, pth: ui.SimpleStringModel):
        """Callback for entering the source code path"""
        pth = pth.get_value_as_string().strip()
        logging.debug(f"Entered source code path: {pth}")
        self._code_path = pth
        if Path(self._code_path).exists():
            self._buttons['Load Code'].enabled = True

    def _on_execute_code(self):
        """Executes the code of the present DebugInterface"""
        self.debug_interface.execute()

    def _on_load_code(self, *args, **kwargs):
        async def _on_load_code_async():
            await self._on_load_code_async()
            await omni.kit.app.get_app().next_update_async()
        asyncio.ensure_future(_on_load_code_async())
        logging.info("Loaded code to debug.")

    # ---------------  Backend ---------------

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

    async def _on_load_code_async(self):
        """The user pressed the LOAD button."""
        pth = Path(self._code_path)
        if pth.suffix != ".py":
            logging.error(f"File {pth} is not a Python file.")
            return

        logging.info("Executing source code")
        await self.instantiate_debugging_class_async(code_path=pth)

    async def instantiate_debugging_class_async(self, code_path: Path):
        """Creates a debugging class instance from the source code"""
        module_name = code_path.stem
        try:
            # Load the module from the given filename
            spec = importlib.util.spec_from_file_location(module_name, str(code_path))
            if spec is None:
                raise ImportError(f"Could not create a module spec for '{code_path}'.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logging.error(f"Error executing code: {e}")
            return
        try:
            self.debug_interface = getattr(module, 'DebugInterface')(self.world)
        except AttributeError as e:
            logging.error(f"Error executing code: {e}")
            return
        self.debug_interface.setup_scene()

    # ---------------  Reset ---------------

    def _on_load_world(self):
        asyncio.ensure_future(self._on_load_world_async())

    async def _on_load_world_async(self):
        """Function called at startup"""
        self._world = World()
        await self._world.initialize_simulation_context_async()
        await self._world.reset_async()
        await self._world.pause_async()
        await omni.kit.app.get_app().next_update_async()

    def _on_reset(self):
        asyncio.ensure_future(self._on_reset_async())

    async def _on_reset_async(self):
        self._stage = omni.usd.get_context().get_stage()
        self._world.clear_all_callbacks()
        self.scene.clear(registry_only=True)
        await self._world.play_async()
        await update_stage_async()
        await self._world.reset_async()
        await self._world.pause_async()
        await omni.kit.app.get_app().next_update_async()

        self.setup()

    def _world_cleanup(self):
        self._world.stop()
        self._world.clear_all_callbacks()
        self._on_reset()
