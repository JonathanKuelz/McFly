import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Optional

from isaacsim.core.utils.stage import open_stage_async, update_stage_async
from isaacsim.gui.components.ui_utils import btn_builder, get_style, str_builder

from omni.isaac.core import World
from omni.isaac.dynamic_control import _dynamic_control
import omni.kit.app
import omni.ui as ui
import omni.usd

from mcfly.extensions.templates.extension_ui_templates import ExtensionUiTemplate


class DebuggerUiBuilder(ExtensionUiTemplate):
    """Manage the debugger Extension"""

    def __init__(self, window_title, menu_path: Optional[str] = None):
        super().__init__(window_title, menu_path)
        self._text_fields = dict()
        self._buttons = dict()
        self._usd_path: str = ''

        self._stage = None
        self._world = None

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
                self._text_fields['USD stage'] = str_builder(
                    "USD file:", on_clicked_fn=self._on_enter_usd, tooltip="Path to a USD stage that you want to load.",
                    use_folder_picker=True)
                self._buttons['Load USD'] = btn_builder(
                    label="LOAD", on_clicked_fn=self._on_load_usd, tooltip="Load the USD stage.")
                self._buttons['Load USD'].enabled = False

                self._text_fields['User Code'] = str_builder(
                    "Sourcecode:", on_clicked_fn=self._on_enter_usd,
                    tooltip="Enter the path to a source file that exposes a debug_extension() method.",
                    use_folder_picker=True)
                self._buttons['Execute'] = btn_builder(
                    label="EXECUTE", on_clicked_fn=self._on_load_usd, tooltip="Execute the debug_extension() method.")
                self._buttons['Execute'].enabled = False

                self._buttons['Reset'] = btn_builder(
                    label="RESET", on_clicked_fn=self._on_reset, tooltip="Reset the scene.")

    def setup(self):
        """Can this be left empty?"""
        pass

    # --------------- UI Callbacks ---------------

    def _on_enter_usd(self, pth: ui.SimpleStringModel):
        """Callback for entering the USD file path"""
        pth = pth.get_value_as_string()
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
        pth = pth.get_value_as_string()
        logging.debug(f"Entered source code path: {pth}")
        self._code_path = pth
        if Path(self._code_path).exists():
            self._buttons['Execute'].enabled = True

    def _on_execute_code(self, *args, **kwargs):
        async def _on_execute_code_async():
            await self._on_execute_code_async()
            await omni.kit.app.get_app().next_update_async()
        _on_execute_code_async()

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

    async def _on_execute_code_async(self):
        """The user pressed the EXECUTE button."""
        pth = Path(self._code_path)
        if pth.suffix != ".py":
            logging.error(f"File {pth} is not a Python file.")
            return

        logging.info("Executing source code")
        await self.execute_code_async(code_path=pth)
        await self._on_reset_async()

    async def execute_code_async(self, code_path: Path):
        """Execute the debug_extension() method from the source code"""
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
            getattr(module, 'debug_extension')()
        except AttributeError as e:
            logging.error(f"Error executing code: {e}")

    # ---------------  Reset ---------------

    def _on_reset(self):
        asyncio.ensure_future(self._on_reset_async())

    async def _on_reset_async(self):
        await update_stage_async()

        self._stage = omni.usd.get_context().get_stage()
        dc = _dynamic_control.acquire_dynamic_control_interface()
        self._world = dc.get_world()
        self.scene.clear(registry_only=True)
        self.setup_scene()

        self.setup()
