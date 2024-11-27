from pathlib import Path

import wx

from helper.constants import APP_NAME


def launch(_config: Path) -> None:
    app = wx.App()
    window = wx.Frame(None, title=APP_NAME, size=(300, 200))
    panel = wx.Panel(window)
    wx.StaticText(panel, label="WIP", pos=(100, 50))
    window.Show(show=True)
    app.MainLoop()
