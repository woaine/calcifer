import customtkinter as ctk

from components import Menu, Display, Log, SideFeed

class Home(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=67)
        self.columnconfigure(1, weight=33)
        self.rowconfigure(0, weight=1)

        Main(self)
        Side(self)
        
        self.pack(fill="both", expand=True)

class Main(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, corner_radius=0, **kwargs)
        self.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=100)
        self.columnconfigure(0, weight=1)

        Menu(self)
        Display(self)

class Side(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, corner_radius=0, **kwargs)
        self.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=100)
        self.columnconfigure(0, weight=1)
        
        SideFeed(self)
        Log(self)