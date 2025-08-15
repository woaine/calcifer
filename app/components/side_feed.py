import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from modules import Database

class SideFeed(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.grid(row=0, rowspan=2, column=0, sticky="nsew", padx=25, pady=25)
        
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        Database().get_data(self._initialize_data)
        Database().add_listener(self._on_new_data)

    def _initialize_data(self, data):
        """Safely initialize the dataframe from the database."""
        try:
            if data is None or len(data) == 0:
                self._data = pd.DataFrame(columns=["model", "Tg", "Tp"])
            else:
                self._data = pd.DataFrame(data, columns=["model", "Tg", "Tp"])
        except Exception as e:
            print(f"Error initializing data: {e}")
            self._data = pd.DataFrame(columns=["model", "Tg", "Tp"])
        
        self._update_plot()

    def _on_new_data(self, data):
        """Handle new data when notified by the Database."""
        model, Tg, Tp = data
        
        self._data.loc[len(self._data)] = [model, Tg, Tp]
        
        self._update_plot()

    def _update_plot(self):
        """Update the plot with new data."""
        self.ax.clear()
        
        if len(self._data) == 0:
            self.ax.text(0.5, 0.5, "No data available", 
                        ha='center', va='center', fontsize=14,
                        transform=self.ax.transAxes)
            self.ax.set_xlabel('Tg (°C)', fontsize=12)
            self.ax.set_ylabel('Tp (°C)', fontsize=12)
            self.ax.set_title('Temperature Analysis by Model', fontsize=14)
        else:
            # Plot data as before
            unique_models = self._data['model'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
            
            for i, model in enumerate(unique_models):
                mask = self._data['model'] == model
                self.ax.scatter(
                    self._data.loc[mask, 'Tg'], 
                    self._data.loc[mask, 'Tp'],
                    label=model,
                    color=colors[i],
                    alpha=0.7,
                    s=50
                )
            
            self.ax.set_xlabel('Tg (°C)', fontsize=12)
            self.ax.set_ylabel('Tp (°C)', fontsize=12)
            self.ax.set_title('Temperature Analysis by Model', fontsize=14)
            
            self.ax.grid(True, linestyle='--', alpha=0.6)
            self.ax.legend(title='Model', loc='best', frameon=True, fontsize=10)
        
        self.fig.tight_layout()
        
        self.canvas.draw_idle()