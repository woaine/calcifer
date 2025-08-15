import customtkinter as ctk
import os

from modules import Logger

class Log(ctk.CTkTextbox):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, font=("Calibri", 18), state="disabled", **kwargs)
        self.grid(row=2, column=0, sticky="nsew", padx=25, pady=(0, 25))
        
        Logger().current_log.add_listener(self.log)
        self._load_log()

    def log(self, *args):
        self.after(100, self._set_text(Logger().current_log.value + "\n"))
    
    def _load_log(self):
        log_file_path = Logger().get_logs()
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as log_file:
                self._set_text(log_file.read())

    def _set_text(self, text):
        self.configure(state="normal")
        self.insert("end", text)
        self.configure(state="disabled")
        self.see("end")
