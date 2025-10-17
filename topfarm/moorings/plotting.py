import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Necessario per la proiezione 3D
import matplotlib.cm as cm
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.patches import Polygon

# def create_optimization_gif_with_seabed(data, seabeds, foot_prints_all, n_moorings, output_filename='optimization_seabed.gif', interval=50):
#     """
#     Crea una GIF animata usando la funzione plot_anchor_wt_seabed per mostrare l'evoluzione
#     delle posizioni WT e ancore sopra i fondali marini.

#     Parameters
#     ----------
#     data : dict
#         Contiene le chiavi 'x', 'y', 'x_anchors', 'y_anchors', ognuna con una lista di frame.
#     seabeds : GeoDataFrame
#         Poligoni del fondale marino.
#     foot_prints_all : list
#         Lista dei dizionari `foot_print` per ogni frame.
#     n_moorings : int
#         Numero di ancore per turbina.
#     output_filename : str
#         Nome del file GIF di output.
#     interval : int
#         Tempo tra frame in millisecondi.
#     """

#     # Controllo consistenza
#     n_frames = len(data['x'])
#     assert all(len(data[key]) == n_frames for key in ['y', 'x_anchors', 'y_anchors']), "Inconsistent data lengths"
#     assert len(foot_prints_all) == n_frames, "Mismatch between frames and foot_prints"

#     frames_dir = "_temp_frames"
#     os.makedirs(frames_dir, exist_ok=True)

#     frame_paths = []

#     for frame in tqdm(range(n_frames)):
#         wt_x = data['x'][frame]
#         wt_y = data['y'][frame]
#         foot_print = foot_prints_all[frame]

#         fig, ax = plt.subplots(figsize=(10, 8))
#         plot_anchor_wt_seabed(seabeds, foot_print, n_moorings, wt_x, wt_y, base_cmap='tab20', ax=ax)
#         ax.set_title(f"Iteration: {frame+1}/{n_frames}")
        
#         frame_path = os.path.join(frames_dir, f"frame_{frame:04d}.png")
#         plt.savefig(frame_path)
#         plt.close(fig)
#         frame_paths.append(frame_path)

#     with imageio.get_writer(output_filename, mode='I', duration=interval / 1000) as writer:
#         for path in frame_paths:
#             image = imageio.imread(path)
#             writer.append_data(image)

    
#     for path in frame_paths:
#         os.remove(path)
#     os.rmdir(frames_dir)




def create_optimization_gif(data, output_filename='optimization.gif', ax=None, seabeds=None, forbidden_areas=None, seabed_color_map=None, anchor_color_map=None, interval=50):

    anchor_legend_lines = [
        Line2D([0], [0], marker='o', color='w', label=f"anchors", 
               markerfacecolor='red', markersize=8)
    ]


    wt_legend = [
        Line2D([0], [0], marker='o', color='w', label='WT', 
               markerfacecolor='blue', markersize=8)]
    all_legend_lines = wt_legend + anchor_legend_lines 
    
    # dict check
    n_frames = len(data['x'])
    assert all(len(data[key]) == n_frames for key in ['y', 'x_anchors', 'y_anchors']), "not valid data"
    base_cmap='tab20'
    
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
        
    # setting
    if seabeds is not None:
        # Seabed colors
        seabed_types = sorted(seabeds['seabed'].unique())
    
        if seabed_color_map is not None:
            seabed_colors = {stype: seabed_color_map.get(stype, 'gray') for stype in seabed_types}
        else:
            cmap = cm.get_cmap(base_cmap, len(seabed_types))
            seabed_colors = {stype: cmap(i) for i, stype in enumerate(seabed_types)}
    
        # Plot seabed areas
        for stype in seabed_types:
            subset = seabeds[seabeds['seabed'] == stype]
            subset.plot(ax=ax, color=seabed_colors[stype], alpha=0.6)
    
        seabed_legend_lines = [
            Line2D([0], [0], color=seabed_colors[stype], lw=4, alpha=0.6, label=stype)
            for stype in seabed_types
        ]
        all_legend_lines += seabed_legend_lines
        
   
    if forbidden_areas is not None:
 
        # Plot forbidden areas
        for i in range(len(forbidden_areas)):
            poly = Polygon(forbidden_areas[i], closed=True, edgecolor='blue', facecolor='lightblue', lw=2, alpha=0.8)
            ax.add_patch(poly)
    
        forbidden_legend_lines = [
            Line2D([0], [0], marker='o', color='w', label='shipwrecks', 
                   markerfacecolor='lightblue', markeredgecolor='blue', markersize=8)]    
        
        all_legend_lines += forbidden_legend_lines
    
    
    ax.set_title("Optimization history", fontsize=14)
    ax.set_xlabel("X [m]", fontsize=12)
    ax.set_ylabel("Y [m]", fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    
    turbine_scatter = ax.scatter([], [], c='blue', label='WT', s=30)
    anchor_scatter = ax.scatter([], [], c='red', label='Anchors', s=30)

    # set bounds
    all_x = np.concatenate(data['x'] + data['x_anchors'])
    all_y = np.concatenate(data['y'] + data['y_anchors'])


    def update(frame):
        turbine_scatter.set_offsets(np.c_[data['x'][frame], data['y'][frame]])
        anchor_scatter.set_offsets(np.c_[data['x_anchors'][frame], data['y_anchors'][frame]])
        ax.set_title(f"Iteration: {frame + 1}/{n_frames}", fontsize=12)
        return turbine_scatter, anchor_scatter
    
    # Limiti dinamici con margine
    margin = 3600
    x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
    y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin

    # ax.set_ylim(y_min, 4.511e+06)    

    # ax.set_xlim(x_min, 2.8500e+05)  
    ax.set_ylim(y_min, 4.429e+06)    

    ax.set_xlim(x_min-1300, 3.0380e+05)  

        
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    
    ax.legend(handles=all_legend_lines, fontsize=12)
    ax.grid(linestyle='dotted')

    ani.save(output_filename, writer=PillowWriter(fps=50))


# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np

# def create_optimization_gif(data, output_filename='optimization.gif', interval=50):
#     """
#     Crea una GIF che mostra la storia dell'ottimizzazione delle posizioni delle turbine.

#     Parameters
#     ----------
#     data : dict
#         Dizionario con chiavi 'x' e 'y', contenenti liste di liste delle posizioni delle turbine.
#     output_filename : str
#         Nome file per la GIF in uscita.
#     interval : int
#         Tempo in millisecondi tra i frame.
#     """
    
#     # Verifica dati
#     n_frames = len(data['x'])
#     assert all(len(data[key]) == n_frames for key in ['y']), "Data inconsistente: lunghezze diverse tra 'x' e 'y'"

#     # Flatten temporaneo per calcolare bounds
#     all_x = np.concatenate(data['x'])
#     all_y = np.concatenate(data['y'])

#     # Inizializza figura
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title("Optimization history", fontsize=14)
#     ax.set_xlabel("X [m]", fontsize=12)
#     ax.set_ylabel("Y [m]", fontsize=12)
#     ax.tick_params(axis='both', labelsize=11)

#     # Scatter iniziale (vuoto)
#     turbine_scatter = ax.scatter([], [], c='blue', label='WT', s=30)
#     ax.legend(fontsize=12)

#     # Limiti dinamici con margine
#     margin = 200
#     x_min, x_max = np.min(all_x) - margin, np.max(all_x) + margin
#     y_min, y_max = np.min(all_y) - margin, np.max(all_y) + margin
#     ax.set_xlim(x_min, x_max)
#     ax.set_ylim(y_min, y_max)

#     # Imposta xticks e yticks da 0 in avanti
#     x_tick_positions = np.arange(x_min, x_max, 3500)
#     y_tick_positions = np.arange(y_min, y_max, 3500)

#     x_tick_labels = x_tick_positions - x_min  # parti da 0
#     y_tick_labels = y_tick_positions - y_min
    
#     x_tick_positions = np.arange(x_min-4000, x_max+2300, 3500)
#     y_tick_positions = np.arange(y_min-1700, y_max+2700, 3500)

#     x_tick_labels = x_tick_positions - (x_min-4000)  # parti da 0
#     y_tick_labels = y_tick_positions - (y_min-1700)

#     ax.set_xticks(ticks=x_tick_positions)
#     ax.set_xticklabels(np.round(x_tick_labels).astype(int))
#     ax.set_yticks(ticks=y_tick_positions)
#     ax.set_yticklabels(np.round(y_tick_labels).astype(int))

#     # Funzione di aggiornamento frame
#     def update(frame):
#         turbine_scatter.set_offsets(np.c_[data['x'][frame], data['y'][frame]])
#         ax.set_title(f"Iteration: {frame + 1}/{n_frames}", fontsize=12)
#         return turbine_scatter,

#     # Crea animazione
#     ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)

#     # Salva GIF
#     ani.save(output_filename, writer='pillow')



def plot_anchor_wt_seabed(seabeds, foot_print, n_moorings, wt_x, wt_y, base_cmap='tab20'):
    """
    Plot seabed zones, anchors, and wind turbines with auto-generated legends and colors.

    Parameters
    ----------
    seabeds : GeoDataFrame, GeoDataFrame containing seabed polygons with 'seabed' column.
    foot_print : dict, Dictionary containing mooring and anchor data per turbine.
    n_moorings : int, Number of moorings per turbine.
    wt_x : array-like, X coordinates of wind turbines.
    wt_y : array-like, Y coordinates of wind turbines.
    base_cmap : str, Matplotlib colormap name used to auto-generate colors (default: 'tab20').
    
    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color seabed
    seabed_types = sorted(seabeds['seabed'].unique())
    cmap = cm.get_cmap(base_cmap, len(seabed_types))
    seabed_colors = {stype: cmap(i) for i, stype in enumerate(seabed_types)}

    # Plot seabed areas
    for stype in seabed_types:
        subset = seabeds[seabeds['seabed'] == stype]
        subset.plot(ax=ax, color=seabed_colors[stype], alpha=0.7)

    seabed_legend_lines = [
        Line2D([0], [0], color=seabed_colors[stype], lw=4, alpha=0.7, label=stype)
        for stype in seabed_types
    ]

    # Color anchor types
    anchor_seabed_types = set()
    for i in range(len(foot_print)):
        for j in range(n_moorings):
            try:
                seabed = foot_print[i]['anchors'][j]['seabed']
                anchor_seabed_types.add(seabed)
            except:
                continue

    anchor_seabed_types = sorted(anchor_seabed_types)
    anchor_cmap = cm.get_cmap('Set1', len(anchor_seabed_types))
    anchor_colors = {stype: anchor_cmap(i) for i, stype in enumerate(anchor_seabed_types)}

    # Plot anchors
    for i in range(len(foot_print)):
        for j in range(n_moorings):
            try:
                coords = foot_print[i]['anchors'][j]['coords']
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
                seabed = foot_print[i]['anchors'][j]['seabed']
                color = anchor_colors.get(seabed, 'black')
                ax.scatter(x, y, color=color, s=12)
            except Exception as e:
                print(f'{e}: Turbine {i}, anchor {j} missing.')

    anchor_legend_lines = [
        Line2D([0], [0], marker='o', color='w', label=f"{stype} Anchor", 
               markerfacecolor=anchor_colors[stype], markersize=8)
        for stype in anchor_seabed_types
    ]


    ax.scatter(wt_x, wt_y, color='black', s=10)
    wt_legend = [Line2D([0], [0], marker='o', color='w', label='WT', markerfacecolor='red', markersize=8)]

    all_legend_lines = seabed_legend_lines + anchor_legend_lines + wt_legend
    all_legend_labels = [h.get_label() for h in all_legend_lines]

    ax.legend(handles=all_legend_lines, labels=all_legend_labels)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_seabed(seabeds, base_cmap='tab20'):
    """
    Plot seabed zones, anchors, and wind turbines with auto-generated legends and colors.

    Parameters
    ----------
    seabeds : GeoDataFrame, GeoDataFrame containing seabed polygons with 'seabed' column.
    foot_print : dict, Dictionary containing mooring and anchor data per turbine.
    n_moorings : int, Number of moorings per turbine.
    wt_x : array-like, X coordinates of wind turbines.
    wt_y : array-like, Y coordinates of wind turbines.
    base_cmap : str, Matplotlib colormap name used to auto-generate colors (default: 'tab20').
    
    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color seabed
    seabed_types = sorted(seabeds['seabed'].unique())
    cmap = cm.get_cmap(base_cmap, len(seabed_types))
    seabed_colors = {stype: cmap(i) for i, stype in enumerate(seabed_types)}

    # Plot seabed areas
    for stype in seabed_types:
        subset = seabeds[seabeds['seabed'] == stype]
        subset.plot(ax=ax, color=seabed_colors[stype], alpha=0.7)

    seabed_legend_lines = [
        Line2D([0], [0], color=seabed_colors[stype], lw=4, alpha=0.7, label=stype)
        for stype in seabed_types
    ]


    all_legend_lines = seabed_legend_lines 
    all_legend_labels = [h.get_label() for h in all_legend_lines]

    ax.legend(handles=all_legend_lines, labels=all_legend_labels)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_mooring_footprint_3d(foot_print, indices_to_plot=None, colormap='viridis', title="Anchoring and mooring footprint"):
    """
    Plot 3D mooring footprint using matplotlib.

    Parameters
    ----------
    foot_print : dict,  Dictionary containing mooring data, with keys being indices and values containing
        'mooring_footprint' as Nx3 numpy arrays.
        
    indices_to_plot : list of int, optional, List of keys (integers) from foot_print to plot. If None, all keys are plotted.
        
    colormap : str, optional,  Name of a matplotlib colormap (default: 'viridis').

    title : str, optional, Title of the plot.

    Returns
    -------
    None
    """
    if indices_to_plot is None:
        indices_to_plot = list(foot_print.keys())

    n_colors = max(indices_to_plot) + 1
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_colors))

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in indices_to_plot:
        coords = foot_print[i]['mooring_footprint']
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=2, color=colors[i], label=f'Mooring {i}')

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=11, azim=170)
    plt.tight_layout()
    plt.legend()
    plt.show()


    


