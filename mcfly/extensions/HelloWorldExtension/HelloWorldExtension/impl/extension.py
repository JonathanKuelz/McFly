import carb
import omni.ext
import omni.kit.app

from .ui import HelloWorldUI


class HelloWorldExtension(omni.ext.IExt):
    """A minimal extension with a UI window"""

    def on_startup(self, ext_id):
        """Method called when the extension is loaded/enabled"""
        carb.log_info(f"on_startup {ext_id}")
        ext_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)  # noqa

        # UI handler
        self.ui_builder = HelloWorldUI(window_title="Wow, fantastic!", menu_path="Window/HelloUIExtension")

    def on_shutdown(self):
        """Method called when the extension is disabled"""
        carb.log_info("on_shutdown")

        # clean up UI
        self.ui_builder.cleanup()
