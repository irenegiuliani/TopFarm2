import numpy as np
import openmdao.api as om
import pandas as pd
import statsmodels.api as sm
from shapely.geometry import Polygon, Point, MultiPolygon
from typing import Any
import random
from topfarm.moorings.moorings_footprint_calculator import moorings_footprint
from scipy.interpolate import griddata


def get_r2(x, y):
    x_const = sm.add_constant(x)   
    model = sm.OLS(y, x_const).fit()
    return model.rsquared


def load_OMsql(log):
    print('loading {}'.format(log))
    cr = om.CaseReader(log)
    rec_data = {}
    driver_cases = cr.list_cases('driver')
    cases = cr.get_cases('driver')
    for case in cases:
        for key in case.outputs.keys():
            if key not in rec_data:
                rec_data[key] = []     
            rec_data[key].append(case[key])
    return rec_data



def make_speedup_map(
    txt_path,
    grid_x,
    grid_y,
    method="linear+nearest",
):
    """
    Crea una mappa 2D di speedup su una griglia regolare.

    Parameters
    ----------
    txt_path : str
        Path al file .txt con colonne: x, y, speedup
    xmin, xmax, ymin, ymax : float
        Estensione del rettangolo di interesse (stesso sistema di coord. di x,y)
    dx, dy : float
        Spaziatura della griglia in x e y (risoluzione)
    method : str
        "linear+nearest" (default): interpola linearmente e riempie i buchi
        "nearest": solo nearest neighbour
        "linear": solo interpolazione lineare (NaN fuori dall’area dei dati)

    Returns
    -------
    X, Y : 2D np.ndarray
        Griglia delle coordinate (shape = (ny, nx))
    S : 2D np.ndarray
        Speedup su griglia (shape = (ny, nx)), pronto per PyWake
    """


    df = pd.read_csv(txt_path, delim_whitespace=True, engine="python")
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    s = (df["speedup"].to_numpy()/100)+1

    X, Y = np.meshgrid(grid_x, grid_y)


    from scipy.interpolate import Rbf

    rbf = Rbf(x, y, s, kernel="thin_plate_spline")  # prova anche 'linear', 'gaussian', ...
    S = rbf(X, Y)  # interpola + extrapola

    return X, Y, S



def generate_random_points_in_polygon(
    inclusion_polygon: Polygon,
    exclusion_areas: MultiPolygon,
    n: int,
    min_distance: float,
    site: Any,
    max_attempts: int = 100000,
):
    """
    Generate random turbine locations within a polygon with spacing and mooring constraints.

    Randomly samples points within the given inclusion polygon until `n` valid points are
    found or the maximum number of attempts is reached. A point is accepted if:
    - it lies inside `inclusion_polygon`,
    - it is at least `min_distance` away from all previously accepted points, and
    - all mooring anchors associated with that point (computed via `moorings_footprint`)
      lie outside the `exclusion_areas` multipolygon.

    Parameters
    ----------
    inclusion_polygon : shapely.geometry.Polygon
        Polygon defining the area within which turbine locations can be placed.
    exclusion_areas : shapely.geometry.MultiPolygon
        Multipolygon defining regions that mooring anchors must avoid. Can be None.
    n : int
        Number of random points (turbine locations) to generate.
    min_distance : float
        Minimum allowed distance between any two accepted points.
    site : Any
        Site object passed to `moorings_footprint` to compute the mooring layout.
    max_attempts : int, optional
        Maximum number of random samples to draw before giving up, by default 100000.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 2) containing the (x, y) coordinates of the accepted points.

    Raises
    ------
    RuntimeError
        If fewer than `n` valid points are found after `max_attempts` trials.
    """

    points = []
    attempts = 0

    minx, miny, maxx, maxy = inclusion_polygon.bounds

    while len(points) < n and attempts < max_attempts:
        attempts += 1

        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        wt_p = Point(x, y)

        if not inclusion_polygon.contains(wt_p):
            continue

        if any(wt_p.distance(other) < min_distance for other in points):
            continue

        footprint = moorings_footprint(
            np.array([x]), np.array([y]), site,
            beta=60,
            resolution_factor=1,
            max_d=260 * 2,
            mooring_type='taught',
            n_moorings=3,
            plot=False,   
        )

        x_anchors = np.asarray([anchor['coords'][0][0] for item in footprint for anchor in item['anchors']])
        y_anchors = np.asarray([anchor['coords'][0][1] for item in footprint for anchor in item['anchors']])

        anchors_ok = True
        for x_anc, y_anc in zip(x_anchors, y_anchors):
            anc_pt = Point(x_anc, y_anc)
            if exclusion_areas is not None and exclusion_areas.contains(anc_pt):
                anchors_ok = False
                break

        if not anchors_ok:
            continue

        points.append(wt_p)

    if len(points) < n:
        raise RuntimeError(
            f"Max number of attempts ({max_attempts}) reached. "
            f"{len(points)} points generated."
        )

    coords = np.array([[pt.x, pt.y] for pt in points], dtype=float)
    return coords

#create df for data
# def create_final_states_dataframe(list_of_dicts):
#     """
#     Converts a list of dictionaries (optimization histories) 
#     into a pandas DataFrame containing only the final state of each run.

#     Args:
#         list_of_dicts (list): 
#             A list of dictionaries. It is assumed that each dictionary 
#             represents one run, has the same keys, and its values 
#             are lists or arrays (the optimization history).

#     Returns:
#         pd.DataFrame: 
#             A DataFrame where each row corresponds to a run (one dictionary
#             from the input list) and each column corresponds to a key. 
#             The cells contain the last element of the history for that
#             key and run.
#     """
    
#     # List to hold the "final" state dictionaries
#     final_states = []

#     # Iterate over each "run" (each dictionary in the list)
#     for run_history in list_of_dicts:
        
#         # Dictionary for the final state of this single run
#         run_final_state = {}
        
#         # Iterate over each key (e.g., 'x', 'y', 'aep', 'cost') in the dictionary
#         for key, history in run_history.items():
#             try:
#                 # Extract the last element of the history (e.g., history[-1])
#                 # We use np.asarray to ensure it's an array
#                 # and to handle simple lists as well.
#                 last_element = np.asarray(history)[-1]
                
#                 # Assign the last element to the new dictionary
#                 run_final_state[key] = last_element
                
#             except (IndexError, TypeError) as e:
#                 # Handle cases where the history might be empty or not indexable
#                 print(f"Warning: Could not find the last element for key '{key}'. Inserting NaN.")
#                 run_final_state[key] = np.nan
        
#         # Add the dictionary containing only the final state to our list
#         final_states.append(run_final_state)

#     # Create the final DataFrame from the list of dictionaries
#     final_df = pd.DataFrame(final_states)
    
#     return final_df

def concat_dicts(dicts):
    """
    Concatena, per ogni chiave, le liste contenute in più dizionari.

    Ogni dizionario deve avere come valori delle liste.
    L'ordine di concatenazione segue l'ordine di dicts.
    """
    out = {}

    all_keys = set()
    for d in dicts:
        all_keys |= set(d.keys())

    for k in all_keys:
        out[k] = []
        for d in dicts:
            if k in d and d[k] is not None:
                out[k].extend(d[k])
    return out

def create_final_states_dataframe(list_of_dicts):
    """
    Estrae lo stato finale da ogni 'run' (dizionario) e restituisce
    un DataFrame in cui ogni cella è numerica:
      - float se lo stato finale è uno scalare
      - np.ndarray(float) se lo stato finale è un vettore/array
    """

    def _to_numeric_scalar_or_array(v):
        """Converte v in float o in np.ndarray(float); altrimenti NaN."""
        # Caso array/lista/tupla -> array di float (con NaN se serve)
        if isinstance(v, (list, tuple, np.ndarray)):
            arr_obj = np.array(v, dtype=object)
            flat = pd.to_numeric(arr_obj.reshape(-1), errors="coerce")  # ndarray di float/NaN
            arr_float = np.asarray(flat, dtype=float).reshape(arr_obj.shape)
            # se è un array 0-D o un solo elemento, diventa float
            if arr_float.size == 1:
                return float(arr_float.item())
            return arr_float

        # Caso scalare -> prova a convertirlo in float (anche da stringa)
        try:
            return float(pd.to_numeric(v, errors="coerce"))
        except Exception:
            return np.nan

    final_states = []

    for run_history in list_of_dicts:
        run_final_state = {}
        for key, history in run_history.items():
            try:
                # prendo l'ultimo elemento della storia
                last_element = np.asarray(history, dtype=object)[-1]
                # lo porto a float o array di float
                run_final_state[key] = _to_numeric_scalar_or_array(last_element)
            except Exception:
                # storia vuota/non indicizzabile o conversione impossibile
                run_final_state[key] = np.nan

        final_states.append(run_final_state)

    final_df = pd.DataFrame(final_states)
    return final_df
