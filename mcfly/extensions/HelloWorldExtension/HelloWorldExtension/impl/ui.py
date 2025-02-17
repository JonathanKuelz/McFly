import omni.ui as ui

from mcfly.extensions.templates.extension_ui_templates import ExtensionUiTemplate


class HelloWorldUI(ExtensionUiTemplate):

    def build_window(self):
        """
        Build a small window with a button that does nothing.
        """
        if not self._window:
            self._window = ui.Window(title=self._window_title, visible=False, width=300, height=300)
            with self._window.frame:
                self._button = ui.Button("Click me", clicked_fn=lambda: print("Button clicked"))
