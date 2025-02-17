from mcfly.extensions.templates.extension_templates import ExtensionTemplate
from .ui import AssemblyUI


class AssemblyExtension(ExtensionTemplate):
    """A minimal extension with a UI window"""

    ui_builder_ref = AssemblyUI

    def __init__(self):
        super().__init__()

    @property
    def name(self) -> str:
        """Name of the extension"""
        return "Assembly Extension"
