import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import customtkinter as ctk

from utilities import Logger
from modules import State
from initializer import Initializer

ctk.set_appearance_mode("dark")

class Application(ctk.CTk):
    def __init__(self):        
        super().__init__()

        self.title("CalciferNet")
        self.attributes('-fullscreen', True)        
        self._bind_events()

        self.after(0, self.state, 'zoomed')
        
        from views import Home
        Home(self)

    def _bind_events(self):
        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", self._exit_fullscreen)

    def _toggle_fullscreen(self, event=None):
        self.attributes("-fullscreen", not self.attributes("-fullscreen"))

    def _exit_fullscreen(self, event=None):
        self.attributes("-fullscreen", False)

if __name__ == "__main__":
    Logger()

    initializer = Initializer()
    initializer.mainloop()
    
    if State().state == 1:
        app = Application()
        app.mainloop()