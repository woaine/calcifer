import customtkinter as ctk

from modules import State, Model, ModelType, Logger

class Settings(ctk.CTkToplevel):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self._prediction_model = ctk.StringVar(value=Model().model.name)

        self._setup_window(master)
        self._create_widgets()
        self._trace_variables()

    def _setup_window(self, master):
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.title("Configure")
        self.transient(master)
        self.grab_set()
        
        self.update_idletasks()

    def _create_widgets(self):
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(padx=5, pady=5, expand=True)

        label = ctk.CTkLabel(self.frame, text="Prediction Model:")
        label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        options = [model.name for model in ModelType]

        option_menu = ctk.CTkOptionMenu(self.frame, values=options, variable=self._prediction_model, width=150)
        option_menu.grid(row=1, column=1, padx=10, pady=10, sticky="w")

        self.confirm_button = ctk.CTkButton(self.frame, text="Confirm", command=self._save_config)
        self.confirm_button.grid(row=3, column=0, columnspan=3, padx=10, pady=20)
        self.confirm_button.configure(state="disabled")

    def _trace_variables(self):
        self._prediction_model.trace_add("write", self._check_changes)

    def _check_changes(self, *args):
        current_prediction_model = Model().model.name

        if self._prediction_model.get() != current_prediction_model:
            self.confirm_button.configure(state="normal")
        else:
            self.confirm_button.configure(state="disabled")
        
    def _save_config(self):
        new_value = self._prediction_model.get()
        Model().load_model(new_value, ModelType[new_value].value)

        State().config_updated.value = 1
        Logger().log(f"Updated prediction model to {new_value} successfully.")
        self.destroy()