from pathlib import Path

import wx


## TODO: this is just an exampel gui application atm
def launch(_settings_folder: Path) -> None:
    app = wx.App()
    window = wx.Frame(None, title="wxPython Frame", size=(300, 200))
    panel = wx.Panel(window)
    wx.StaticText(panel, label="Hello World", pos=(100, 50))
    window.Show(show=True)
    app.MainLoop()
