import tkinter as tk
from tkinter import ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # initialize data
        self.plot_type = ('Correlations', 'Correlation times', 'End-to-end distances', 'End-to-end times',
                          'Establishment times and silent patches', 'Final state', 'Fractions of "ON" cells', 'Heatmap',
                          'Monomer interactions', 'Monomer states', 'Monomer states (time-space plot)', 'Optimization',
                          'Optimization result', 'RMS', 'Successful recruited conversions', 'Time dynamics')

        self.parameters = ('model', 'n_processes', 'rms', 'cenH_size', 'cenH_init_idx', 'ATF1_idx', 'N', 't_total',
                           'noise', 'initial_state', 'dt', 'alpha_1', 'alpha_2', 'beta', 'seed', 'cell_division')

        self.entries = [ttk.Entry(self) for _ in range(len(self.parameters))]
        self.input_param_vals = {}
        self.plot_func = ''

        self.prefill_params = ('CMOL', 1, 2, 8, 16, None, 40, 10000, 0.5, 'A', 0.02, 0.07, 0.1, 0.004, 0, 0)

        # set up variable
        self.option_var = tk.StringVar(self)

        # create widget
        self.create_widget()

    def create_widget(self):
        self.title('')
        self.geometry("440x440")

        # padding for widgets using the grid layout
        paddings = {'padx': 5, 'pady': 5}

        # Option menu
        headline_font = ('Helvetica', 12, 'bold')
        option_label = ttk.Label(self,  text='Select plot type:', font = headline_font)
        option_label.grid(column=0, row=0, sticky=tk.W, **paddings)
        option_menu = ttk.OptionMenu(self, self.option_var, self.plot_type[0], *self.plot_type, command=None)
        option_menu.grid(column=1, row=0, sticky=tk.W, **paddings)

        # Entry fields
        ttk.Label(self, text='Parameters:', font=headline_font).grid(column=0, row=1)
        for i, param in enumerate(self.parameters):
            row_idx = i+2

            # Label
            ttk.Label(self, text=param).grid(row=row_idx)

            # Entry field
            self.entries[i].grid(row=row_idx, column=1)

            # Prefill
            self.entries[i].insert(tk.END, str(self.prefill_params[i]))

        # OK button
        ok_button = tk.Button(self, text='OK', state=tk.NORMAL, command=self.get_input)
        ok_button_row_idx = len(self.parameters) + 2
        ok_button.grid(column=1, row=ok_button_row_idx)

    def get_input(self):
        # Get all input data
        for i, entry in enumerate(self.entries):
            parameter = self.parameters[i]
            self.input_param_vals[parameter] = (entry.get())

        self.plot_func = self.option_var.get()

        self.destroy()



#
