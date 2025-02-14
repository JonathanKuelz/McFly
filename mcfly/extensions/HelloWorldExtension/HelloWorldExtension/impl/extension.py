from mcfly.extensions.templates.extension_template import ExtensionTemplate
from .ui import HelloWorldUI


class HelloWorldExtension(ExtensionTemplate):
    """A minimal extension with a UI window"""

    ui_builder_ref = HelloWorldUI

    @property
    def name(self) -> str:
        """Name of the extension"""
        return "HelloWorldExtension"
