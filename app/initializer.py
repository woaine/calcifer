import customtkinter as ctk

from utilities import Logger
from modules import State

class Initializer(ctk.CTk):
    def __init__(self):
        super().__init__(fg_color="#1D1E1E")
        self.title("Initializer")

        width = self.winfo_screenwidth() // 2
        height = self.winfo_screenheight() // 3
        x = (self.winfo_screenwidth() // 2) - (self.winfo_width() // 2)
        y = (self.winfo_screenheight() // 2) - (self.winfo_height() // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.resizable(False, False)
        
        self._create_widgets()

        self.after(100, self._initialize_modules)

    def _create_widgets(self):
        self.init_label_text = ctk.StringVar(value="Initializing...")
        self.init_label = self._create_label(self.init_label_text, ("Arial", 20), pady=20, anchor="n", side="top")

        self.progress_bar = ctk.CTkProgressBar(self)
        self.progress_bar.pack(pady=20, padx=20, fill="x", anchor="s", side="bottom")
        self.progress_bar.set(0)

        self.progress_info_text = ctk.StringVar(value="Preparing modules")
        self.progress_info = self._create_label(self.progress_info_text, ("Arial", 10), pady=5, anchor="s", expand=True, fill='both', side="bottom")

        self.error_info_text = ctk.StringVar(value="")
        self.error_info = self._create_label(self.error_info_text, ("Arial", 10), expand=True, fill='both',)
        self.error_info.pack_forget()

        self.restart_label = ctk.CTkLabel(self, text="Please restart the app.", font=("Arial", 10))
        self.restart_label.pack(expand=True, fill='both')
        self.restart_label.pack_forget()

    def _create_label(self, text_var: ctk.StringVar, font: tuple, **pack_options):
        label = ctk.CTkLabel(self, textvariable=text_var, font=font)
        label.pack(**pack_options)

        return label

    def _initialize_modules(self):
        from modules import Database, Sensor, ThermalCamera, Model
        from src.superresolution import SuperResolution
        from src.detection import YOLO11Person, YOLO11Face

        errors = []
        modules = [
            ("Database", Database),
            ("Model", Model),
            ("Sensor", Sensor),
            ("Thermal Camera", ThermalCamera),
            ("Superresolution", SuperResolution),
            ("Person Detection", YOLO11Person),
            ("Face Detection", YOLO11Face)
        ]

        total_modules = len(modules)
        for i, (name, module_class) in enumerate(modules):
            self.progress_info_text.set(f"Initializing {name}...")
            Logger().log(f"Initializing {name}...")
            self.update_idletasks()

            try:
                module_class()
            except Exception as e:
                errors.append(f"{name} failed to initialize: {str(e)}")

            self._update_progress(name, i, total_modules)

        if not errors:
            self.progress_info_text.set("Starting the app...")
            Logger().log("Starting the app...")
            self.update_idletasks()

            State().state = 1
            self.destroy()
        else:
            self._show_errors(errors)

    def _update_progress(self, name: str, index: int, total: int):
        self.progress_bar.set((index + 1) / total)
        self.update_idletasks()
        
    def _show_errors(self, errors):
        self.init_label_text.set("Initialization failed")
        Logger().log("Initialization failed", "error")
        self.init_label.configure(text_color="red")

        self.progress_info.pack_forget()
        self.error_info_text.set("\n".join(errors))
        Logger().log("\n".join(errors), "error")
        self.error_info.pack(pady=5, padx=20)

        self.progress_bar.pack_forget()
        self.restart_label.pack(pady=10, padx=10, anchor="s", side="bottom")

        self.update_idletasks()