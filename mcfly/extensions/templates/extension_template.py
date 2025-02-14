import carb
import omni.ext
import omni.kit.app

from mcfly.extensions.templates.extension_ui_template import ExtensionUiTemplate


class ExtensionTemplate(omni.ext.IExt):
    """A minimal extension with a UI window"""

    ui_builder_ref = ExtensionUiTemplate

    def __init__(self):
        """Set up the extension

        Args:
            ui_builder: A UI builder that should handle the UI for this extension.
        """
        super().__init__()
        self.ui_builder = None

    @property
    def name(self) -> str:
        """Name of the extension"""
        return str(self.__class__.__name__)

    def on_startup(self, ext_id):
        """Method called when the extension is loaded/enabled"""
        carb.log_info(f"on_startup {ext_id}")
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)  # noqa
        self.ui_builder = self.ui_builder_ref(window_title=self.name, menu_path=f"Window/{self.name}")

    def on_shutdown(self):
        """Method called when the extension is disabled"""
        carb.log_info("on_shutdown")

        # clean up UI
        self.ui_builder.cleanup()
