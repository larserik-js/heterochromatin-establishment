from plot_class import Plots
import gui

if __name__ == '__main__':
    # Create widget
    widget_obj = gui.App()
    widget_obj.mainloop()

    # Get plot function and input parameter values
    plot_func, input_param_vals = widget_obj.plot_func, widget_obj.input_param_vals

    # Make plot object and plot
    plot_obj = Plots(plot_func, input_param_vals)
    plot_obj.plot()


