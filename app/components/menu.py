import customtkinter as ctk

from views import Settings
from modules import State, Model
from utilities import Logger
class Menu(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, corner_radius=0, fg_color="darkgray", **kwargs)
        self.grid(row=0, column=0, sticky="nsew")

        self.is_predicting = ctk.BooleanVar(value=0)
        self.is_saving = ctk.BooleanVar(value=0)
        self.is_recording = ctk.BooleanVar(value=0)

        self._create_widgets()
        self._bind_events()
        self._trace_variables()

    def _create_widgets(self):
        self._config_button = ctk.CTkButton(self, text="Configure", fg_color="#1D1E1E", command=self._open_config_dialog)
        self._config_button.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        self._predict_switch = ctk.CTkSwitch(self, text="Predict Temperature", variable=self.is_predicting, state="enabled", onvalue=1, offvalue=0)
        self._predict_switch.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        self._save_switch = ctk.CTkSwitch(self, text="Save Data", variable=self.is_saving, state="disabled", onvalue=1, offvalue=0)
        self._save_switch.grid(row=0, column=2, sticky="ew", padx=5, pady=5)

        self._record_switch = ctk.CTkSwitch(self, text="Record Stream", variable=self.is_recording, state="enabled", onvalue=1, offvalue=0)
        self._record_switch.grid(row=0, column=3, sticky="ew", padx=5, pady=5)

    def _bind_events(self):
        self._config_button.bind("<Enter>", lambda e: self._config_button.configure(cursor="hand2"))
        self._config_button.bind("<Leave>", lambda e: self._config_button.configure(cursor=""))

    def _trace_variables(self):
        State().config_updated.add_listener(self._update_menu_state)
        self.is_predicting.trace_add("write", self._update_predict_value)
        self.is_saving.trace_add("write", self._update_save_value)
        self.is_recording.trace_add("write", self._update_recording_value)

    def _update_predict_value(self, *args):
        State().predicting = self.is_predicting.get()

        if State().predicting:
            self._save_switch.configure(state="normal")
        else:
            self._save_switch.deselect()
            self._save_switch.configure(state="disabled")

    def _update_save_value(self, *args):
        State().saving = self.is_saving.get()

    def _update_recording_value(self, *args):
        State().recording.value = self.is_recording.get()

    def _update_menu_state(self, *args):
        if State().config_updated.value:
            self._record_switch.deselect()
            self._predict_switch.deselect()

            self._predict_switch.configure(state="normal" if Model().model.name != "None" else "disabled")

            State().config_updated.value = 0

    def _open_config_dialog(self):
        Settings(self.master.master.master)