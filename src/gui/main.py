from pathlib import Path
from typing import Any, Self, override

import wx
from wx.lib import scrolledpanel

from helper.constants import APP_NAME


class ExpandableItem(wx.Panel):
    @override
    def __init__(
        self: Self,
        parent: wx.Panel,
        title: str,
        sub_items: list[str],
    ) -> None:
        super().__init__(parent)
        self.sub_items = sub_items
        self.expanded = False

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header with toggle button (arrow)
        self.toggle_button = wx.ToggleButton(
            self, label="▶ " + title, style=wx.BU_EXACTFIT,
        )
        self.toggle_button.Bind(wx.EVT_TOGGLEBUTTON, self.on_toggle)
        self.main_sizer.Add(self.toggle_button, flag=wx.EXPAND | wx.ALL, border=5)

        # Panel to contain sub-items (initially hidden)
        self.sub_panel = wx.Panel(self)
        self.sub_sizer = wx.BoxSizer(wx.VERTICAL)

        for sub in sub_items:
            label = wx.StaticText(self.sub_panel, label=f"• {sub}")
            self.sub_sizer.Add(label, flag=wx.LEFT | wx.TOP, border=10)

        self.sub_panel.SetSizer(self.sub_sizer)
        self.sub_panel.Hide()  # Hidden by default
        self.main_sizer.Add(self.sub_panel, flag=wx.EXPAND)

        self.SetSizer(self.main_sizer)

    def on_toggle(self: Self, _event: Any) -> None:
        self.expanded = not self.expanded
        if self.expanded:
            self.toggle_button.SetLabel("▼ " + self.toggle_button.GetLabel()[2:])
            self.sub_panel.Show()
        else:
            self.toggle_button.SetLabel("▶ " + self.toggle_button.GetLabel()[2:])
            self.sub_panel.Hide()

        self.GetParent().Layout()
        self.GetParent().FitInside()  # Important: updates scrolling region


class ListScrollPanel(scrolledpanel.ScrolledPanel):
    @override
    def __init__(self: Self, parent: wx.Panel) -> None:
        super().__init__(parent, style=wx.VSCROLL)

        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        data = [
            ("Fruits", ["Apple", "Banana", "Cherry"]),
            ("Vegetables", ["Carrot", "Lettuce", "Potato"]),
            ("Dairy", ["Milk", "Cheese"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
            ("Grains", ["Bread", "Rice", "Pasta"]),
        ]

        for title, sub_items in data:
            row = ExpandableItem(self, title, sub_items)
            scroll_sizer.Add(row, flag=wx.EXPAND | wx.ALL, border=5)

        self.SetSizer(scroll_sizer)
        self.SetupScrolling(scroll_x=False, scroll_y=True)


class ListPanel(wx.Panel):
    __scroll_panel: ListScrollPanel

    @override
    def __init__(self: Self, parent: wx.Panel) -> None:
        super().__init__(parent)

        self.SetBackgroundColour(wx.Colour(200, 220, 240))

        self.__scroll_panel = ListScrollPanel(self)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.__scroll_panel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(main_sizer)


class DetailViewPanel(wx.Panel):
    @override
    def __init__(self: Self, parent: wx.Panel) -> None:
        super().__init__(parent)

        self.SetBackgroundColour(wx.Colour(240, 220, 200))


class MainPanel(wx.Panel):
    __left_panel: ListPanel
    __right_panel: DetailViewPanel

    @override
    def __init__(self: Self, parent: wx.Window) -> None:
        super().__init__(parent)

        self.__left_panel = ListPanel(self)
        self.__right_panel = DetailViewPanel(self)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        main_sizer.Add(
            self.__left_panel, proportion=60, flag=wx.EXPAND | wx.ALL, border=5,
        )
        main_sizer.Add(
            self.__right_panel, proportion=40, flag=wx.EXPAND | wx.ALL, border=5,
        )

        self.SetSizer(main_sizer)


class MainFrame(wx.Frame):
    __panel: MainPanel

    @override
    def __init__(self: Self) -> None:
        super().__init__(parent=None, title=APP_NAME)
        self.__panel = MainPanel(self)
        self.__panel.SetBackgroundColour(wx.Colour(240, 240, 240))

        self.Maximize(maximize=True)
        self.Show(show=True)


class MainApp(wx.App):
    __frame: MainFrame

    def __on_focus_event(self: Self, event: Any) -> None:
        print(f"EVENT: {event.GetEventType()} from {event.GetEventObject()}")
        event.Skip()

    @override
    def OnInit(self: Self) -> bool:
        # wx.Log.SetActiveTarget(wx.LogStderr())
        # wx.Log.SetVerbose(True)
        self.Bind(wx.EVT_SET_FOCUS, self.__on_focus_event)

        self.__frame = MainFrame()
        self.__frame.Show()
        self.SetTopWindow(frame=self.__frame)
        return True


def launch_gui(_config: Path) -> None:
    app = MainApp()

    app.MainLoop()
