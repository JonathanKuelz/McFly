from .ui_builder import DebuggerUiBuilder

from mcfly.extensions.templates.extension_templates import ExtensionTemplate


class DebugExtension(ExtensionTemplate):
    """The Extension class"""

    ui_builder_ref = DebuggerUiBuilder
