from pathlib import Path
from typing import Self

import wx

from helper.constants import APP_NAME


class ListPanel(wx.Panel):
    def __init__(self, parent) -> None:
        super().__init__(parent)

        self.SetBackgroundColour(wx.Colour(200, 220, 240))


class DetailViewPanel(wx.Panel):
    def __init__(self, parent) -> None:
        super().__init__(parent)

        self.SetBackgroundColour(wx.Colour(240, 220, 200))


class MainPanel(wx.Panel):
    __left_panel: ListPanel
    __left_panel: DetailViewPanel

    def __init__(self, parent) -> None:
        super().__init__(parent)

        self.__left_panel = ListPanel(self)
        self.__right_panel = DetailViewPanel(self)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        main_sizer.Add(
            self.__left_panel, proportion=60, flag=wx.EXPAND | wx.ALL, border=5
        )
        main_sizer.Add(
            self.__right_panel, proportion=40, flag=wx.EXPAND | wx.ALL, border=5
        )

        self.SetSizer(main_sizer)


class MainFrame(wx.Frame):
    __panel: MainPanel

    def __init__(self: Self) -> None:
        super().__init__(parent=None, title=APP_NAME)
        self.__panel = MainPanel(self)
        self.__panel.SetBackgroundColour(wx.Colour(240, 240, 240))

        self.Maximize(maximize=True)
        self.Show(show=True)


def launch_gui(_config: Path) -> None:
    app = wx.App()
    frame = MainFrame()
    app.SetTopWindow(frame)
    app.MainLoop()
