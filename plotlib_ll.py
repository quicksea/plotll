import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.cm as cm
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from datetime import datetime
import shutil  # Import shutil for deleting directories

def display_dataframe(df):
    """
    Display a pandas DataFrame in a Tkinter window.

    Parameters:
    df (pd.DataFrame): The DataFrame to display.

    Returns:
    None
    """

    # Create a new window
    window = tk.Tk()
    window.title("DataFrame Viewer")

    # Create a Treeview widget
    tree = ttk.Treeview(window, columns=list(df.columns), show="headings")
    tree.pack(expand=True, fill=tk.BOTH)

    # Set up the column headers
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor=tk.CENTER)

    # Insert the data into the Treeview
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

    # Add a scrollbar
    scrollbar = ttk.Scrollbar(window, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscroll=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Run the Tkinter event loop
    window.mainloop()


def colormap(groups, colorscheme=None):
    """
    Generate a colormap for the unique groups in the data.

    Parameters:
    groups (pd.Series): Series containing the group labels.
    colorscheme (str or list, optional): The name of the colormap to use or a custom list of colors.
        If None, a custom high-contrast palette will be used.

    Returns:
    dict: A dictionary mapping each unique group to a color.
    """
    unique_groups = groups.unique()

    # Define a custom high-contrast color palette
    custom_palette = ["red", "blue", "darkgreen", "orange", "purple", "brown", "pink", "cyan", "olive", "black"]

    # Use the custom palette if no colorscheme is provided
    if not colorscheme:
        colors = custom_palette
    elif isinstance(colorscheme, list):
        colors = colorscheme  # Use the provided custom list
    else:
        colormap = cm.get_cmap(colorscheme, len(unique_groups))
        colors = [colormap(i) for i in range(len(unique_groups))]

    # Map each unique group to a color
    return {group: colors[i % len(colors)] for i, group in enumerate(unique_groups)}

def nqplot(df, data_col, group_col, colors=None, symbol_col=None, color_col=None, x_range=None, y_range=None, 
           x_label=None, y_label=None, fit_line=False, connect_points=True, data_on_x=True, figsize=(800, 600), 
           title_fontsize=16, label_fontsize=14, tick_fontsize=12, legend_fontsize=12, minor_ticks=True, 
           xscale='linear', yscale='linear', display_stats=True, save_stats=True, hollow=False, 
           save_figures=False, silent_mode=False, output_folder=None):
    """
    Generate a normal quantile (Q-Q) plot grouped by a specified column.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing the data.
    data_col (str): Column name for the data to plot.
    group_col (str): Column name to group data by.
    colors (dict, optional): Mapping of groups to colors. Defaults to auto-generated colors.
    symbol_col (str, optional): Column name to determine marker symbols. Defaults to None.
    color_col (str, optional): Column name to determine marker colors. Defaults to group_col.
    x_range (tuple, optional): Range for the x-axis. Defaults to auto-determined range.
    y_range (tuple, optional): Range for the y-axis. Defaults to auto-determined range.
    x_label (str, optional): Label for the x-axis. Defaults to 'Theoretical Quantiles'.
    y_label (str, optional): Label for the y-axis. Defaults to 'Ordered Values'.
    fit_line (bool, optional): Whether to plot a fitted line. Defaults to False.
    connect_points (bool, optional): Whether to connect points within groups. Defaults to True.
    data_on_x (bool, optional): Whether to plot data on the x-axis. Defaults to True.
    figsize (tuple, optional): Figure size in pixels (width, height). Defaults to (800, 600).
    title_fontsize (int, optional): Font size for the plot title. Defaults to 16.
    label_fontsize (int, optional): Font size for axis labels. Defaults to 14.
    tick_fontsize (int, optional): Font size for tick labels. Defaults to 12.
    legend_fontsize (int, optional): Font size for legend labels. Defaults to 12.
    minor_ticks (bool, optional): Whether to show minor ticks and grid lines. Defaults to True.
    xscale (str, optional): Scale for the x-axis ('linear', 'log', etc.). Defaults to 'linear'.
    yscale (str, optional): Scale for the y-axis ('linear', 'log', etc.). Defaults to 'linear'.
    display_stats (bool, optional): Whether to display group statistics. Defaults to True.
    save_stats (bool, optional): Whether to save group statistics to a CSV file. Defaults to True.
    hollow (bool, optional): Whether to use hollow markers. Defaults to False.

    Returns:
    dict: A dictionary containing Q-Q plot data and fit parameters for each group.
    """
    if color_col is None:
        color_col = group_col
    if colors is None:
        colors = colormap(df[color_col])
    symbols = ["o", "s", "D", "^", "v", "<", ">"]  # List of symbols to rotate
    symbol_map = {}  # Map unique values in symbol_col to symbols
    if symbol_col:
        unique_symbols = df[symbol_col].unique()
        symbol_map = {value: symbols[i % len(symbols)] for i, value in enumerate(unique_symbols)}

    plt.figure(figsize=(figsize[0] / 100, figsize[1] / 100))
    qq_data = {}
    statslist = []
    # Loop over each group and create a Q-Q plot
    for group, group_df in df.groupby(group_col):
        # Generate theoretical quantiles and ordered sample values using probplot
        (osm, osr), (slope, intercept, r) = stats.probplot(group_df[data_col], dist="norm", fit=True)
        group_df = group_df.sort_values(data_col)
        group_df['nq'] = osm
        # Store the Q-Q plot data and fit parameters
        qq_data[group] = {
            'osm': osm,
            'osr': osr,
            'slope': slope,
            'intercept': intercept,
            'r': r
        }

        # Plot the sample points
        # Plot each point with its corresponding symbol
        for _, row in group_df.iterrows():
            symbol = symbol_map.get(row[symbol_col], "o") if symbol_col else "o"
            marker_style = symbol
            colorsel = colors[row[color_col]]
            facecolors='none' if hollow else colorsel
            if data_on_x:
                plt.scatter(row[data_col], row['nq'], alpha=0.6, color=colorsel, marker=marker_style, facecolors=facecolors)
            else:
                plt.scatter(row['nq'], row[data_col], alpha=0.6, color=colorsel, marker=marker_style, facecolors=facecolors)
        
        # Connect the data points piecewise if connect_points is True
        line_color = colors.get(group)
        if line_color is None:
            line_color = 'black'
        if connect_points:
            if data_on_x:
                plt.plot(osr, osm, color=line_color, alpha=0.6)
            else:
                plt.plot(osm, osr, color=line_color, alpha=0.6)

        # Plot the fitted line if fit_line is True
        if fit_line:
            x = np.linspace(np.min(osm), np.max(osm), 100)
            if data_on_x:
                plt.plot(slope * x + intercept, x, linestyle='--', label=f'{group} fit', color=line_color)
            else:
                plt.plot(x, slope * x + intercept, linestyle='--', label=f'{group} fit', color=line_color)
        groupstats = {
            'group': group,
            'mean': np.nanmean(group_df[data_col]),
            'stddev': np.nanstd(group_df[data_col]),
            'min': np.nanmin(group_df[data_col]),
            'max': np.nanmax(group_df[data_col]),
            'median': np.nanmedian(group_df[data_col]),
            'count_exna': np.count_nonzero(~np.isnan(group_df[data_col])),
            'count': len(group_df),
            'slope': slope,
            'intercept': intercept,
            'r': r,
        }
        for sig in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
            # pct = stats.norm.cdf(sig)
            # groupstats[f'NQ_{sig}'] = group_df[data_col].quantile(pct)
            if sig < np.nanmin(osm) or sig > np.nanmax(osm): # if sig is outside the range of osr
                groupstats[f'Q_{sig}'] = np.nan
            else:
                groupstats[f'Q_{sig}'] = np.interp(sig, osm, osr)
        statslist.append(groupstats)
    # Create a DataFrame from the stats list
    stats_df = pd.DataFrame(statslist)
    if save_stats:
        try: 
            stats_df.to_csv(os.path.join(output_folder, 'nqplot group_stats.csv'), index=False)
        except Exception as e:
            print(f"Error saving stats DataFrame: {e}")
    # Set x and y range if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    plt.xscale(xscale)
    plt.yscale(yscale)
    # Set x and y labels if specified, otherwise infer from column names
    plt.xlabel(x_label if x_label else (data_col if data_on_x else 'Quantiles'), fontsize=label_fontsize)
    plt.ylabel(y_label if y_label else ('Quantiles' if data_on_x else data_col), fontsize=label_fontsize)
    
    plt.title(f'Normal Quantile Plot of {data_col} by {group_col}', fontsize=title_fontsize)
    # plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(True)

    # Add minor ticks and grid lines if minor_ticks is True
    if minor_ticks:
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgrey')

    # Add legends for color and symbol outside the plot
    color_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', markersize=8, label=label)
                     for label, color in colors.items()]
    symbol_handles = None
    if symbol_col:
        symbol_handles = [plt.Line2D([0], [0], marker=symbol, color='black', linestyle='None', markersize=8, markerfacecolor='none' if hollow else 'black', label=label)
                          for label, symbol in symbol_map.items()]
    if symbol_handles is None:
        all_handles = color_handles
    else:
        all_handles = color_handles + symbol_handles
    all_labels = [handle.get_label() for handle in all_handles]
    legend_title = f"Colors: '{color_col}'\nSymbols: '{symbol_col}'" if symbol_col else f"Legend\nColors: '{color_col}'"
    plt.legend(all_handles, all_labels, title=legend_title, fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the layout to make space for legends
    plt.subplots_adjust(right=0.8)
    if save_figures:
        plt.draw()
        output_path = os.path.join(output_folder, f"nqplot_{data_col}-{group_col}.png")
        plt.savefig(output_path, bbox_inches="tight")
    if not silent_mode:
        plt.show(block=False)
    if display_stats and not silent_mode:
        display_dataframe(stats_df)

    return qq_data

def xyplot(df, x_col, y_col, group_col, symbol_col=None, color_col=None, colors=None, x_range=None, y_range=None, x_label=None, y_label=None, fit_line=True, connect_points=False, figsize=(800, 600), title_fontsize=16, label_fontsize=14, tick_fontsize=12, legend_fontsize=12, minor_ticks=True, xscale='linear', yscale='linear', hollow=False,
           save_figures=False, silent_mode=False, output_folder=None):
    """
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    x_col (str): The name of the x column to plot.
    y_col (str): The name of the y column to plot.
    group_col (str): The name of the group column to group by.
    colors (dict, optional): A dictionary mapping each group to a color. If None, colors will be assigned automatically.
    x_range (tuple, optional): The range for the x-axis. If None, the range will be determined automatically.
    y_range (tuple, optional): The range for the y-axis. If None, the range will be determined automatically.
    x_label (str, optional): The label for the x-axis. If None, the name of the x column will be used.
    y_label (str, optional): The label for the y-axis. If None, the name of the y column will be used.
    fit_line (bool, optional): Whether to plot the fitted line. Default is True.
    connect_points (bool, optional): Whether to connect the data points piecewise with lines of the same color as the symbols. Default is False.
    figsize (tuple, optional): The size of the figure in pixels. Default is (800, 600).
    title_fontsize (int, optional): The font size for the title. Default is 16.
    label_fontsize (int, optional): The font size for the x and y labels. Default is 14.
    tick_fontsize (int, optional): The font size for the tick labels. Default is 12.
    legend_fontsize (int, optional): The font size for the legend labels. Default is 12.
    minor_ticks (bool, optional): Whether to add minor ticks and grid lines. Default is True.
    hollow (bool, optional): Whether to use hollow symbols for the plot. Default is False.

    Returns:
    dict: A dictionary containing the plot data and fit parameters for each group.
    """
    if color_col is None:
        color_col = group_col
    if colors is None:
        colors = colormap(df[color_col])
    symbols = ["o", "s", "D", "^", "v"]  # List of symbols to rotate
    symbol_map = {}  # Map unique values in symbol_col to symbols
    if symbol_col:
        unique_symbols = df[symbol_col].unique()
        symbol_map = {value: symbols[i % len(symbols)] for i, value in enumerate(unique_symbols)}

    plt.figure(figsize=(figsize[0] / 100, figsize[1] / 100))
    plot_data = {}

    # Loop over each group and create an XY plot
    for group, group_df in df.groupby(group_col):
        x = group_df[x_col]
        y = group_df[y_col]
        
        # Store the plot data
        plot_data[group] = {
            'x': x,
            'y': y
        }
        
        # Plot each point with its corresponding symbol
        for _, row in group_df.iterrows():
            symbol = symbol_map.get(row[symbol_col], "o") if symbol_col else "o"
            marker_style = symbol
            colorsel = colors[row[color_col]]
            facecolors='none' if hollow else colorsel
            plt.scatter(row[x_col], row[y_col], alpha=0.6, color=colorsel, marker=marker_style, facecolors=facecolors)
        line_color = colors.get(group)
        if line_color is None:
            line_color = 'black'
        # Connect the data points piecewise if connect_points is True
        if connect_points:
            plt.plot(x, y, color=line_color, alpha=0.6)

        # Plot the fitted line if fit_line is True
        if fit_line:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            plt.plot(x, slope * x + intercept, linestyle='--', label=f'{group} fit', color=line_color)
    plt.xscale(xscale)
    plt.yscale(yscale)
    # Set x and y range if specified
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    # Set x and y labels if specified, otherwise infer from column names
    plt.xlabel(x_label if x_label else x_col, fontsize=label_fontsize)
    plt.ylabel(y_label if y_label else y_col, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(True)
    plt.title(f'XY Plot of {y_col} vs {x_col} by {group_col}', fontsize=title_fontsize)
    # Add minor ticks and grid lines if minor_ticks is True
    if minor_ticks:
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgrey')
    # Add legends for color and symbol
    color_handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='None', markersize=8, label=label)
                     for label, color in colors.items()]
    symbol_handles = None
    if symbol_col:
        symbol_handles = [plt.Line2D([0], [0], marker=symbol, color='black', linestyle='None', markersize=8, markerfacecolor='none' if hollow else 'black', label=label)
                          for label, symbol in symbol_map.items()]
    if symbol_handles is None:
        all_handles = color_handles
    else:
        all_handles = color_handles + symbol_handles
    all_labels = [handle.get_label() for handle in all_handles]
    legend_title = f"Colors: '{color_col}'\nSymbols: '{symbol_col}'" if symbol_col else f"Legend\nColors: '{color_col}'"
    plt.legend(all_handles, all_labels, title=legend_title, fontsize=legend_fontsize, loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the layout to make space for legends
    plt.subplots_adjust(right=0.8)
    if save_figures:
        plt.draw()
        output_path = os.path.join(output_folder, f"xyplot_{y_col}-{x_col}-{group_col}.png")
        plt.savefig(output_path, bbox_inches="tight")
    if not silent_mode:
        plt.show(block=False)
    return plot_data

def generate_sample_data(output_folder=None):
    np.random.seed(0)
    usa_heights = np.random.normal(175, 10, size=100)    # Mean 175 cm, SD 10 cm
    china_heights = np.random.normal(165, 8, size=100)   # Mean 165 cm, SD 8 cm
    india_heights = np.random.normal(170, 9, size=100)   # Mean 170 cm, SD 9 cm
    japan_heights = np.random.normal(160, 7, size=100)   # Mean 160 cm, SD 7 cm
    uk_heights = np.random.normal(172, 10, size=100)     # Mean 172 cm, SD 10 cm

    def generate_weights(heights, mean_weight, std_weight, slope=0.5, intercept=0):
        noise = np.random.normal(0, std_weight, size=heights.size)
        weights = slope * heights + intercept + noise
        return weights

    usa_weights = generate_weights(usa_heights, 70, 5)
    china_weights = generate_weights(china_heights, 60, 4)
    india_weights = generate_weights(india_heights, 65, 4.5)
    japan_weights = generate_weights(japan_heights, 55, 3.5)
    uk_weights = generate_weights(uk_heights, 68, 5)

    blood_pressure = np.random.normal(120, 15, size=500)
    iq = np.random.normal(100, 15, size=500)
    sex = np.random.choice(['Male', 'Female'], size=500)
    occupation = np.random.choice(['Engineer', 'Doctor', 'Artist', 'Scientist'], size=500)

    df = pd.DataFrame({
        'height': np.concatenate([usa_heights, china_heights, india_heights, japan_heights, uk_heights]),
        'weight': np.concatenate([usa_weights, china_weights, india_weights, japan_weights, uk_weights]),
        'blood_pressure': blood_pressure,
        'iq': iq,
        'country': ['USA'] * 100 + ['China'] * 100 + ['India'] * 100 + ['Japan'] * 100 + ['UK'] * 100,
        'sex': sex,
        'occupation': occupation,
    })
    if output_folder is None: # write to current directory
        output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, 'sample_data.csv'), index=False)
    return df

def create_gui():
    global df
    df = None

    def exit_program():
        """
        Exit the program by destroying the root window.
        """
        root.destroy()

    def delete_plot_folders():
        """
        Delete all subfolders in the current directory starting with 'plot_'.
        """
        current_dir = os.getcwd()
        for folder_name in os.listdir(current_dir):
            folder_path = os.path.join(current_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name.startswith("plot_"):
                try:
                    shutil.rmtree(folder_path)  # Delete the folder and its contents
                    print(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print(f"Error deleting folder {folder_path}: {e}")

    def generate_sample_data_gui():
        """
        Generate sample data and populate the GUI dropdowns.
        """
        global df
        df = generate_sample_data(output_folder=output_folder_var.get())
        columns = df.columns.tolist()
        x_col_var.set(columns[0])
        y_col_var.set(columns[1])
        group_col_var.set(columns[4])
        symbol_col_var.set("")  # Reset symbol column selection
        color_col_var.set("")  # Reset color column selection
        x_col_combobox['values'] = columns
        y_col_combobox['values'] = columns
        group_col_combobox['values'] = columns
        symbol_col_combobox['values'] = [""] + columns  # Allow empty selection
        color_col_combobox['values'] = [""] + columns  # Allow empty selection
        plot_button.config(state=tk.NORMAL)

    def load_file():
        """
        Load a file and populate the GUI dropdowns.
        """
        global df
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")])
        if file_path:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            columns = df.columns.tolist()
            x_col_var.set(columns[0])
            y_col_var.set(columns[1])
            group_col_var.set(columns[4])
            symbol_col_var.set("")  # Reset symbol column selection
            color_col_var.set("")  # Reset color column selection
            x_col_combobox['values'] = columns
            y_col_combobox['values'] = columns
            group_col_combobox['values'] = columns
            symbol_col_combobox['values'] = [""] + columns  # Allow empty selection
            color_col_combobox['values'] = [""] + columns  # Allow empty selection
            plot_button.config(state=tk.NORMAL)

    def plot():
        """
        Generate the selected plot based on user input.
        """
        global df
        if df is None:
            return
        plot_type = plot_type_var.get()
        x_col = x_col_var.get()
        y_col = y_col_var.get()
        group_col = group_col_var.get()
        symbol_col = symbol_col_var.get() if symbol_col_var.get() != "" else None  # Handle empty selection
        color_col = color_col_var.get() if color_col_var.get() != "" else group_col  # Default to group_col if not selected
        fit_line = fit_line_var.get()
        connect_points = connect_points_var.get()
        minor_ticks = minor_ticks_var.get()
        hollow = hollow_var.get()  # Get the hollow option
        save_figures = save_figures_var.get()  # Get the save figures option
        silent_mode = silent_mode_var.get()  # Get the silent mode option
        output_folder = output_folder_var.get()  # Get the output folder
        color_scheme = color_scheme_var.get()
        xscale = xscale_var.get()
        yscale = yscale_var.get()
        x_label = x_label_var.get()
        y_label = y_label_var.get()

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        colors = colormap(df[color_col], colorscheme=color_scheme)

        if plot_type == "nqplot":
            nqplot(df, x_col, group_col, color_col=color_col, colors=colors, symbol_col=symbol_col, fit_line=fit_line,
                   connect_points=connect_points, minor_ticks=minor_ticks, xscale=xscale, yscale=yscale, x_label=x_label,
                   y_label=y_label, hollow=hollow, save_figures=save_figures, silent_mode=silent_mode, output_folder=output_folder)
        elif plot_type == "xyplot":
            xyplot(df, x_col, y_col, group_col, color_col=color_col, symbol_col=symbol_col, colors=colors, fit_line=fit_line,
                   connect_points=connect_points, minor_ticks=minor_ticks, xscale=xscale, yscale=yscale, x_label=x_label,
                   y_label=y_label, hollow=hollow, save_figures=save_figures, silent_mode=silent_mode, output_folder=output_folder)

    root = tk.Tk()
    root.title("Plotting GUI")

    plot_type_var = tk.StringVar(value="nqplot")
    x_col_var = tk.StringVar()
    y_col_var = tk.StringVar()
    group_col_var = tk.StringVar()
    symbol_col_var = tk.StringVar()  # Variable for symbol column
    color_col_var = tk.StringVar()  # Variable for color column
    fit_line_var = tk.BooleanVar(value=True)
    connect_points_var = tk.BooleanVar(value=False)
    minor_ticks_var = tk.BooleanVar(value=True)
    hollow_var = tk.BooleanVar(value=False)  # Variable for hollow option
    save_figures_var = tk.BooleanVar(value=True)  # Variable for saving figures
    silent_mode_var = tk.BooleanVar(value=True)  # Variable for silent mode
    output_folder_var = tk.StringVar(value=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")  # Default output folder
    color_scheme_var = tk.StringVar(value="")
    xscale_var = tk.StringVar(value="linear")
    yscale_var = tk.StringVar(value="linear")
    x_label_var = tk.StringVar(value="")  # Default empty
    y_label_var = tk.StringVar(value="")  # Default empty

    # Add buttons for generating sample data and loading files
    ttk.Button(root, text="Generate Sample Data", command=generate_sample_data_gui).grid(row=0, column=0, columnspan=1, sticky=tk.W)
    ttk.Button(root, text="Load File", command=load_file).grid(row=0, column=1, columnspan=2, sticky=tk.W)

    # Add button to delete "plot_" folders
    ttk.Button(root, text="Delete 'plot_' Folders", command=delete_plot_folders).grid(row=0, column=3, columnspan=2, sticky=tk.W)

    # Add Exit button
    ttk.Button(root, text="Exit", command=exit_program).grid(row=0, column=5, columnspan=1, sticky=tk.W)

    ttk.Label(root, text="Plot Type:").grid(row=1, column=0, sticky=tk.W)
    ttk.Radiobutton(root, text="nqplot", variable=plot_type_var, value="nqplot").grid(row=1, column=1, sticky=tk.W)
    ttk.Radiobutton(root, text="xyplot", variable=plot_type_var, value="xyplot").grid(row=1, column=2, sticky=tk.W)

    ttk.Label(root, text="X Column:").grid(row=2, column=0, sticky=tk.W)
    x_col_combobox = ttk.Combobox(root, textvariable=x_col_var, values=[])
    x_col_combobox.grid(row=2, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="Y Column:").grid(row=3, column=0, sticky=tk.W)
    y_col_combobox = ttk.Combobox(root, textvariable=y_col_var, values=[])
    y_col_combobox.grid(row=3, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="Group Column:").grid(row=4, column=0, sticky=tk.W)
    group_col_combobox = ttk.Combobox(root, textvariable=group_col_var, values=[])
    group_col_combobox.grid(row=4, column=1, columnspan=2, sticky=tk.W)

    # Add dropdown for symbol column
    ttk.Label(root, text="Symbol Column:").grid(row=5, column=0, sticky=tk.W)
    symbol_col_combobox = ttk.Combobox(root, textvariable=symbol_col_var, values=[])
    symbol_col_combobox.grid(row=5, column=1, columnspan=2, sticky=tk.W)

    # Add dropdown for color column
    ttk.Label(root, text="Color Column:").grid(row=6, column=0, sticky=tk.W)
    color_col_combobox = ttk.Combobox(root, textvariable=color_col_var, values=[])
    color_col_combobox.grid(row=6, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="Color Scheme:").grid(row=7, column=0, sticky=tk.W)
    color_scheme_combobox = ttk.Combobox(root, textvariable=color_scheme_var, values=["", "Dark2", "tab10", "Set1", "Set3", "tab20", "plasma", "hsv"])
    color_scheme_combobox.grid(row=7, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="X Scale:").grid(row=8, column=0, sticky=tk.W)
    xscale_combobox = ttk.Combobox(root, textvariable=xscale_var, values=["linear", "log", "exp"])
    xscale_combobox.grid(row=8, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="Y Scale:").grid(row=9, column=0, sticky=tk.W)
    yscale_combobox = ttk.Combobox(root, textvariable=yscale_var, values=["linear", "log", "exp"])
    yscale_combobox.grid(row=9, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="X Axis Label:").grid(row=10, column=0, sticky=tk.W)
    x_label_entry = ttk.Entry(root, textvariable=x_label_var)
    x_label_entry.grid(row=10, column=1, columnspan=2, sticky=tk.W)

    ttk.Label(root, text="Y Axis Label:").grid(row=11, column=0, sticky=tk.W)
    y_label_entry = ttk.Entry(root, textvariable=y_label_var)
    y_label_entry.grid(row=11, column=1, columnspan=2, sticky=tk.W)

    # Add checkbox for hollow symbols
    ttk.Checkbutton(root, text="Hollow Symbols", variable=hollow_var).grid(row=12, column=0, sticky=tk.W)

    ttk.Checkbutton(root, text="Fit Line", variable=fit_line_var).grid(row=13, column=0, sticky=tk.W)
    ttk.Checkbutton(root, text="Connect Points", variable=connect_points_var).grid(row=13, column=1, sticky=tk.W)
    ttk.Checkbutton(root, text="Minor Ticks", variable=minor_ticks_var).grid(row=13, column=2, sticky=tk.W)

    # Add checkbox for saving figures
    ttk.Checkbutton(root, text="Save Figures", variable=save_figures_var).grid(row=14, column=0, sticky=tk.W)

    # Add checkbox for silent mode
    ttk.Checkbutton(root, text="Silent Mode", variable=silent_mode_var).grid(row=14, column=1, sticky=tk.W)

    # Add textbox for output folder
    ttk.Label(root, text="Output Folder:").grid(row=15, column=0, sticky=tk.W)
    output_folder_entry = ttk.Entry(root, textvariable=output_folder_var, width=30)
    output_folder_entry.grid(row=15, column=1, columnspan=2, sticky=tk.W)

    plot_button = ttk.Button(root, text="Plot", command=plot, state=tk.DISABLED)
    plot_button.grid(row=16, column=0, columnspan=3)

    root.mainloop()

if __name__ == "__main__":
    # df = generate_sample_data()
    create_gui()