import sys
import os
sys.path.append(r'D:\Giuliani\Projects\NADARA\Topfarm')
sys.path.append(r'D:\Giuliani\Projects\NADARA\Tool')
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point, Polygon, MultiPolygon
from matplotlib.lines import Line2D
import numpy as np
from shapely.geometry import Polygon, LineString, Point
import math
from pyproj import CRS
import xarray as xr
import shutil

def extract_boundaries_from_poly(geom):
    obstacles = []

    if geom.geom_type == 'Polygon':
        coords = np.array(geom.exterior.coords[:-1])  # rimuove punto duplicato finale
        obstacles.append(coords)

    elif geom.geom_type == 'MultiPolygon':
        for poly in geom.geoms:
            coords = np.array(poly.exterior.coords[:-1])
            obstacles.append(coords)

    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    
    return obstacles

def add_offset(gdf, offset):
    """
    Add a new polygonal area offset from the centroid of existing geometries 
    and subtract existing geometries from it to create a "hole".

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame containing Polygon or MultiPolygon geometries.
    offset : float
        The distance to offset from the centroid to define the new square polygon.

    Returns
    -------
    gdf_out : geopandas.GeoDataFrame
        The output GeoDataFrame with the additional "outside" polygon added.
    """
    
    geoms = [
        geom 
        for geometry in gdf.geometry
        for geom in (geometry.geoms if geometry.geom_type == 'MultiPolygon' else [geometry])
    ]

    multipoly = MultiPolygon(geoms)
    centroid = multipoly.centroid

    square = Polygon([
        (centroid.x - offset, centroid.y - offset),
        (centroid.x + offset, centroid.y - offset),
        (centroid.x + offset, centroid.y + offset),
        (centroid.x - offset, centroid.y + offset)
    ])

    outarea = square.difference(multipoly)


    def plot_geom(geom, title, color):
        x, y = geom.exterior.xy
        plt.fill(x, y, alpha=0.5, fc=color, ec='black')
        plt.axis('equal')
        plt.title(title)
        plt.show()

    plot_geom(square, "Offset Polygon", "lightblue")

    plt.figure()
    for geom in multipoly.geoms:
        x, y = geom.exterior.xy
        plt.fill(x, y, alpha=0.5, fc='lightgreen', ec='black')
    plt.axis('equal')
    plt.title("MultiPolygon")
    plt.show()


    new_row = gpd.GeoDataFrame({
        'seabed': ['forbidden'],
        'max_depth': [0.0],
        'name': ['outside'],
        'geometry': [outarea],
        'Shape_Area': [outarea.area]
    }, crs=CRS.from_epsg(32634))

    gdf_out = pd.concat([gdf, new_row], ignore_index=True)
    return gdf_out
    


def generate_seabed_gdf(gdb_path,
                        output_path,
                        key_layers,
                        seabed_type,
                        max_depth,
                        epsg=32634,
                        conc_area_path=None,):
    """
    Generate a GeoDataFrame from selected layers in a GDB file, assigning seabed properties,
    and optionally saving to file and plotting over a concession area.
    """

    assert len(key_layers) == len(seabed_type) == len(max_depth), \
        "All input lists (key_layers, seabed_type, max_depth) must have the same length"

    crs = CRS.from_epsg(epsg)
    seabed_features = []

    for key, s_type, depth in zip(key_layers, seabed_type, max_depth):
        gdf = gpd.read_file(gdb_path, layer=key).to_crs(crs).copy()

        # Drop problematic or duplicate fields
        gdf = gdf.drop(columns=[col for col in gdf.columns if col.lower() in ['name', 'shape_length']], errors='ignore')

        # Add required fields
        gdf['name'] = key
        gdf['seabed'] = s_type
        gdf['depth'] = depth
        gdf['shp_area'] = gdf.geometry.area.round(0).astype(int)  # Safer for shapefiles

        seabed_features.append(gdf)

    seabed_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # If saving to .shp, ensure all column names are ≤10 chars
        if output_path.endswith(".shp"):
            seabed_gdf = seabed_gdf.rename(columns=lambda x: x[:10])

        driver = "GPKG" if output_path.endswith(".gpkg") else None
        output_dir = os.path.dirname(output_path)
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        seabed_gdf.to_file(output_path, driver=driver)

    return seabed_gdf


# def process_seabed_layers(
#     gdb_path,
#     output_path,
#     key_layers,
#     seabed_types,
#     max_depths,
#     base_layer,
#     conc_area_path=None,
#     epsg=32634,):
#     """
#     Processes seabed layers from a GDB, computes area differences, and writes a unified seabed GeoDataFrame.

#     Parameters
#     ----------
#     gdb_path : str, Path to the input geodatabase (.gdb).
#     output_path : str, Path where the output GeoDataFrame should be saved.
#     key_layers : list of str, Layer names in the GDB to use.
#     seabed_types : list of str, Seabed types (e.g. 'rock', 'sand') for each layer.
#     max_depths : list of float, Max depth values for each seabed layer.
#     base_layer : str, Name of the base layer to use for difference operation.
#     conc_area_path : str, optional, Path to the concession area shapefile.
#     epsg : int, optional, Target EPSG for CRS conversion (default: 32634).

#     Returns
#     -------
#     seabed_gdf : geopandas.GeoDataFrame
#         Combined seabed features GeoDataFrame.
#     """

#     crs = CRS.from_epsg(epsg)

#     # Load and tag seabed layers
#     seabed_features = []
#     for layer, seabed, depth in zip(key_layers, seabed_types, max_depths):
#         gdf = gpd.read_file(gdb_path, layer=layer).to_crs(crs)
#         gdf['seabed'] = seabed
#         gdf['max_depth'] = depth
#         gdf['name'] = layer
#         gdf['Shape_Area'] = gdf.geometry.area.round(1)
#         seabed_features.append(gdf)

#     combined_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)

#     # Calculate difference to get 'sand' area
#     base_gdf = gpd.read_file(gdb_path, layer=base_layer).to_crs(crs)
#     clip_union = combined_gdf.unary_union
#     outside = base_gdf.copy()
#     outside["geometry"] = base_gdf.geometry.difference(clip_union)
#     outside = outside[~outside.is_empty & outside.geometry.notnull()].copy()

#     # Add metadata to sand area
#     outside['seabed'] = 'sand'
#     outside['max_depth'] = -9999
#     outside['name'] = 'sandarea'
#     outside['Shape_Area'] = outside.geometry.area.round(1)
#     seabed_features.append(outside)

#     # Merge all features and export
#     seabed_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     seabed_gdf.to_file(output_path)


#     return seabed_gdf


# def process_seabed_layers(
#     gdb_path,
#     output_path,
#     key_layers,
#     seabed_types,
#     max_depths,
#     base_layer,
#     conc_area_path=None,
#     epsg=32634,):
#     """
#     Processes seabed layers from a GDB, computes area differences, and writes a unified seabed GeoDataFrame.

#     Parameters
#     ----------
#     gdb_path : str, Path to the input geodatabase (.gdb).
#     output_path : str, Path where the output GeoDataFrame should be saved.
#     key_layers : list of str, Layer names in the GDB to use.
#     seabed_types : list of str, Seabed types (e.g. 'rock', 'sand') for each layer.
#     max_depths : list of float, Max depth values for each seabed layer.
#     base_layer : str, Name of the base layer to use for difference operation.
#     conc_area_path : str, optional, Path to the concession area shapefile.
#     epsg : int, optional, Target EPSG for CRS conversion (default: 32634).

#     Returns
#     -------
#     seabed_gdf : geopandas.GeoDataFrame
#         Combined seabed features GeoDataFrame.
#     """

#     crs = CRS.from_epsg(32634)
#     # Load and tag seabed layers
#     seabed_features = []          
#     for layer, seabed, depth in zip(key_layers, seabed_types, max_depths ):
#         gdf = gpd.read_file(gdb_path, layer=layer).to_crs(crs)
#         gdf['seabed'] = seabed
#         gdf['max_depth'] = depth
#         gdf['name'] = layer
#         gdf['Shape_Area'] = gdf.geometry.area.round(1)
#         seabed_features.append(gdf)
    
#     combined_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
    
#     # Calculate difference to get 'sand' area
#     base_gdf = gpd.read_file(gdb_path, layer=base_layer).to_crs(crs)
#     clip_union = combined_gdf.unary_union
#     outside = base_gdf.copy()
#     outside["geometry"] = base_gdf.geometry.difference(clip_union)
#     outside = outside[~outside.is_empty & outside.geometry.notnull()].copy()
    
#     # Add metadata to sand area
#     outside['seabed'] = 'sand'
#     outside['max_depth'] = -9999
#     outside['name'] = 'sandarea'
#     outside['Shape_Area'] = outside.geometry.area.round(1)
#     seabed_features.append(outside)
    
#     # Merge all features and export
#     seabed_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
    
#     a=seabed_gdf['seabed'].unique()
#     merge_df=[]
#     for i in seabed_gdf['seabed'].unique():
#         merged_geom = seabed_gdf[seabed_gdf['seabed'] == i].union_all()
        
#         if merged_geom.geom_type == 'Polygon':
#             merged_geom = MultiPolygon([merged_geom])
            
#         elif merged_geom.geom_type == 'GeometryCollection':
#             merged_geom = MultiPolygon([
#                 geom for geom in merged_geom.geoms 
#                 if geom.geom_type in ['Polygon', 'MultiPolygon']
#             ])
      
#         merge_df.append(gpd.GeoDataFrame({
#             'seabed': i,
#             'max_depth': [seabed_gdf[seabed_gdf['seabed'] == i]['max_depth'].iloc[0]],
#             'name': [i + ' union'],
#             'geometry': merged_geom,
#             'Shape_Area': merged_geom.area,
#         }, crs=gdf.crs))
        
#     seabed_gdf = gpd.GeoDataFrame(pd.concat(merge_df, ignore_index=True), crs=crs)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     seabed_gdf.to_file(output_path)
#     return seabed_gdf


def process_seabed_layers_from_list(
    seabed_layers_info,
    output_path,
    epsg=32634
):
    """
    Processes and merges seabed layers from multiple spatial data sources (GeoDatabase or Shapefile)
    into a unified GeoDataFrame, applying optional buffers, coordinate system harmonization, 
    and area computation for each seabed category.

    This function is designed to flexibly handle multiple seabed data inputs provided as a list of
    dictionaries, where each entry may reference:
    - a file geodatabase (.gdb) or geopackage (.gpkg) containing multiple layers, or
    - a single shapefile (.shp).
    
    For each layer, attributes such as seabed type, maximum depth, and an optional buffer distance
    can be specified. All geometries are reprojected to a common coordinate reference system (CRS),
    merged by seabed type, and finally written to a single output shapefile.

    Parameters
    ----------
    seabed_layers_info : list of dict
        A list containing one dictionary for each data source to be processed. 
        Each dictionary must include the following keys:
        
        - **path** (`str`): Path to the data source. Can be either:
          - a *File Geodatabase* (`.gdb` or `.gpkg`) containing one or more layers, or
          - a *Shapefile* (`.shp`).
        
        - **key_layers** (`list of str`, optional): Names of layers to be read from the GDB/GPKG.  
          Required only if `path` points to a GDB or GPKG. Ignored for shapefiles.

        - **seabed_type** (`list of str` or `str`): Seabed category (e.g., `'rock'`, `'sand'`, `'mud'`) 
          to associate with each layer. If a list is provided, it must have the same length as `key_layers`.

        - **max_depth** (`list of float` or `float`, optional): Maximum depth values (in meters) 
          corresponding to each layer. Defaults to `0` for all layers if not provided.

        - **add_buffer** (`list of float` or `float`, optional): Buffer distances (in meters) 
          to apply around geometries for each layer. Defaults to `0` (no buffering).

    output_path : str
        Path where the combined shapefile will be saved. The directory will be recreated if it exists.
        Example: `"C:/project/output/seabed_combined.shp"`

    epsg : int, optional
        EPSG code of the coordinate reference system (CRS) to which all layers are reprojected.
        Default is `32634` (UTM Zone 34N).

    Returns
    -------
    seabed_gdf : geopandas.GeoDataFrame
        A unified GeoDataFrame containing all processed seabed features.
        Columns:
        - `'seabed'`: Seabed type (str)
        - `'max_depth'`: Maximum depth associated with the seabed (float)
        - `'name'`: Original layer or file name (str)
        - `'geometry'`: Polygon or multipolygon geometry
        - `'Shape_Area'`: Computed area of each polygon (float, CRS units)

    Workflow
    --------
    1. **Input Parsing**:  
       Iterates through the list of seabed sources (`seabed_layers_info`).
       For each entry, determines if the source is a `.gdb`, `.gpkg`, or `.shp`.
       
    2. **Geometry Extraction**:  
       Reads each specified layer into a GeoDataFrame, reprojects it to the target CRS,
       and attaches metadata (`seabed`, `max_depth`, `Shape_Area`).

    3. **Optional Buffering**:  
       If a non-zero `add_buffer` value is provided, applies a buffer to each geometry
       using the specified distance (in CRS units).

    4. **Merging and Union**:  
       Concatenates all individual GeoDataFrames and performs a geometric union for each
       unique seabed type to create a single feature per category.

    5. **Output Export**:  
       The final unified GeoDataFrame is saved as a shapefile at `output_path`.

    Notes
    -----
    - All input layers are automatically reprojected to the specified `epsg`.
    - Buffer distances are applied in the same CRS units (e.g., meters for UTM projections).
    - The output directory is recreated to ensure a clean save environment.
    - The function is suitable for combining heterogeneous data sources (e.g., mixed GDB and shapefiles).

    Examples
    --------
    >>> seabed_layers_info = [
    ...     {
    ...         "path": "data/seabed_zones.gdb",
    ...         "key_layers": ["rock_zone", "sand_zone"],
    ...         "seabed_type": ["rock", "sand"],
    ...         "max_depth": [50, 30],
    ...         "add_buffer": [0, 500]
    ...     },
    ...     {
    ...         "path": "data/mud_area.shp",
    ...         "seabed_type": "mud",
    ...         "max_depth": 20,
    ...         "add_buffer": 0
    ...     }
    ... ]
    >>> output_path = "output/combined_seabed.shp"
    >>> result_gdf = process_seabed_layers_from_list(seabed_layers_info, output_path, epsg=32633)
    >>> result_gdf.head()
           seabed  max_depth           name                 geometry  Shape_Area
    0        rock       50.0    rock_zone_union  MULTIPOLYGON (...)    2.1e+06
    1        sand       30.0    sand_zone_union  MULTIPOLYGON (...)    1.8e+06
    2         mud       20.0     mud_area_union  MULTIPOLYGON (...)    0.9e+06
    """
    crs = CRS.from_epsg(epsg)
    seabed_features = []
    base_features = []

    # --- Loop through all seabed layer definitions ---
    for info in seabed_layers_info:
        path = info['path']
        layers = info.get('key_layers', None)
        seabed_type = info['seabed_type']
        max_depth = info.get('max_depth', None)
        add_buffer = info.get('add_buffer', None)
        is_base = info.get('is_base', None)

        # Handle GDB/GPKG
        if path.endswith(('.gdb', '.gpkg')):
            if not layers:
                raise ValueError(f"key_layers must be provided for GDB: {path}")

            n_layers = len(layers)
            max_depth = max_depth or [0] * n_layers
            add_buffer = add_buffer or [0] * n_layers
            is_base = is_base or [False] * n_layers

            for i, layer in enumerate(layers):
                gdf = gpd.read_file(path, layer=layer).to_crs(crs).copy()

                gdf['seabed'] = seabed_type[i] if isinstance(seabed_type, list) else seabed_type
                gdf['max_depth'] = max_depth[i]
                gdf['name'] = layer
                gdf['Shape_Area'] = gdf.geometry.area.round(1)

                if add_buffer[i] != 0:
                    gdf['geometry'] = gdf['geometry'].buffer(add_buffer[i])

                if is_base[i]:
                    base_features.append(gdf)
                else:
                    seabed_features.append(gdf)

        # Handle shapefiles
        elif path.endswith('.shp'):
            gdf = gpd.read_file(path).to_crs(crs).copy()

            gdf['seabed'] = seabed_type if not isinstance(seabed_type, list) else seabed_type[0]
            gdf['max_depth'] = max_depth if not isinstance(max_depth, list) else max_depth[0]
            gdf['name'] = os.path.splitext(os.path.basename(path))[0]
            gdf['Shape_Area'] = gdf.geometry.area.round(1)

            if add_buffer:
                buffer_val = add_buffer if isinstance(add_buffer, (int, float)) else add_buffer[0]
                if buffer_val != 0:
                    gdf['geometry'] = gdf['geometry'].buffer(buffer_val)

            if isinstance(is_base, list):
                base_flag = is_base[0]
            else:
                base_flag = bool(is_base)

            if base_flag:
                base_features.append(gdf)
            else:
                seabed_features.append(gdf)

    # --- Combine seabed layers ---
    combined_gdf = (
        gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
        if seabed_features else gpd.GeoDataFrame(columns=['geometry'], crs=crs)
    )

    # --- Compute base layer difference if exists ---
    if base_features:
        base_gdf = gpd.GeoDataFrame(pd.concat(base_features, ignore_index=True), crs=crs)
        clip_union = combined_gdf.unary_union if not combined_gdf.empty else None

        outside = base_gdf.copy()
        if clip_union:
            outside['geometry'] = base_gdf.geometry.difference(clip_union)
        outside = outside[~outside.is_empty & outside.geometry.notnull()].copy()

        outside['seabed'] = 'sand'
        outside['max_depth'] = -9999
        outside['name'] = 'outside_area'
        outside['Shape_Area'] = outside.geometry.area.round(1)
        seabed_features.append(outside)

    # --- Merge everything ---
    if seabed_features:
        seabed_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
    else:
        seabed_gdf = gpd.GeoDataFrame(columns=['geometry'], crs=crs)

    # --- Merge by seabed type ---
    merged_layers = []
    for stype in seabed_gdf['seabed'].unique():
        sub_gdf = seabed_gdf[seabed_gdf['seabed'] == stype]
        merged_geom = sub_gdf.unary_union
        if merged_geom.geom_type == 'Polygon':
            merged_geom = MultiPolygon([merged_geom])
        elif merged_geom.geom_type == 'GeometryCollection':
            merged_geom = MultiPolygon([
                g for g in merged_geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']
            ])

        merged_layers.append(gpd.GeoDataFrame({
            'seabed': stype,
            'max_depth': [sub_gdf['max_depth'].iloc[0]],
            'name': [stype + '_union'],
            'geometry': [merged_geom],
            'Shape_Area': [merged_geom.area]
        }, crs=crs))

    seabed_gdf = gpd.GeoDataFrame(pd.concat(merged_layers, ignore_index=True), crs=crs)

    # --- Save output ---
    output_dir = os.path.dirname(output_path)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seabed_gdf.to_file(output_path)

    return seabed_gdf
    


def process_seabed_layers(
    gdb_path,
    output_path,
    key_layers,
    seabed_types,
    max_depths,
    base_layer,
    conc_area_path=None,
    epsg=32634,
    add_buffer=None):
    """
    Processes seabed layers from a GDB, computes area differences, and writes a unified seabed GeoDataFrame.

    Parameters
    ----------
    gdb_path : str, Path to the input geodatabase (.gdb).
    output_path : str, Path where the output GeoDataFrame should be saved.
    key_layers : list of str, Layer names in the GDB to use.
    seabed_types : list of str, Seabed types (e.g. 'rock', 'sand') for each layer.
    max_depths : list of float, Max depth values for each seabed layer.
    base_layer : str, Name of the base layer to use for difference operation.
    conc_area_path : str, optional, Path to the concession area shapefile.
    epsg : int, optional, Target EPSG for CRS conversion (default: 32634).

    Returns
    -------
    seabed_gdf : geopandas.GeoDataFrame
        Combined seabed features GeoDataFrame.
    """

    crs = CRS.from_epsg(32634)
    # Load and tag seabed layers
    seabed_features = []          
    if add_buffer is not None:
        for bufferlayer in add_buffer:
            for layer, seabed, depth in zip(key_layers, seabed_types, max_depths ):
                if layer == bufferlayer:
                    gdf = gpd.read_file(gdb_path, layer=layer).to_crs(crs).copy()
                    gdf['seabed'] = seabed
                    gdf['max_depth'] = depth
                    gdf['name'] = layer
                    gdf['Shape_Area'] = gdf.geometry.area.round(1)
                    gdf['geometry'] = gdf['geometry'].buffer(add_buffer[layer]) 
                    seabed_features.append(gdf)

    for layer, seabed, depth in zip(key_layers, seabed_types, max_depths ):
        gdf = gpd.read_file(gdb_path, layer=layer).to_crs(crs).copy()
        gdf['seabed'] = seabed
        gdf['max_depth'] = depth
        gdf['name'] = layer
        gdf['Shape_Area'] = gdf.geometry.area.round(1)
        seabed_features.append(gdf)
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
    
    # Calculate difference to get 'sand' area
    base_gdf = gpd.read_file(gdb_path, layer=base_layer).to_crs(crs).copy()
    clip_union = combined_gdf.unary_union
    outside = base_gdf.copy()
    outside["geometry"] = base_gdf.geometry.difference(clip_union)
    outside = outside[~outside.is_empty & outside.geometry.notnull()].copy()
    
    # Add metadata to sand area
    outside['seabed'] = 'sand'
    outside['max_depth'] = -9999
    outside['name'] = 'sandarea'
    outside['Shape_Area'] = outside.geometry.area.round(1)
    seabed_features.append(outside)
    
    # Merge all features and export
    seabed_gdf = gpd.GeoDataFrame(pd.concat(seabed_features, ignore_index=True), crs=crs)
    
    merge_df=[]
    for i in seabed_gdf['seabed'].unique():

        merged_geom = seabed_gdf[seabed_gdf['seabed'] == i].union_all()
        
        if merged_geom.geom_type == 'Polygon':
            merged_geom = MultiPolygon([merged_geom])
            
        elif merged_geom.geom_type == 'GeometryCollection':
            merged_geom = MultiPolygon([
                geom for geom in merged_geom.geoms 
                if geom.geom_type in ['Polygon', 'MultiPolygon']
            ])
      
        merge_df.append(gpd.GeoDataFrame({
            'seabed': i,
            'max_depth': [seabed_gdf[seabed_gdf['seabed'] == i]['max_depth'].iloc[0]],
            'name': [i + ' union'],
            'geometry': merged_geom,
            'Shape_Area': merged_geom.area,
        }, crs=gdf.crs))
        
    seabed_gdf = gpd.GeoDataFrame(pd.concat(merge_df, ignore_index=True), crs=crs)
    output_dir = os.path.dirname(output_path)
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    seabed_gdf.to_file(output_path)
    return seabed_gdf



# def offset_bounds_from_bathymetry(polygon: Polygon, bathymetry: xr.DataArray, beta_deg: float = 70.0):
#     """
#         Offsets the boundary of a polygon based on bathymetry data.
    
#         This function calculates an offset for each vertex of the input polygon.
#         The offset magnitude is determined by the local water depth (interpolated
#         from the bathymetry data) and a given angle beta. The direction of
#         the offset is radial, pointing away from the polygon's centroid. This can
#         be useful for creating a buffer zone or a new polygon that follows the
#         depth contour.
    
#         Args:
#             polygon (Polygon): The input polygon whose boundary will be offset.
#             bathymetry (xr.DataArray): A 2D xarray DataArray containing the
#                                        bathymetry (depth) values. The coordinates
#                                        of this DataArray should be compatible with
#                                        the polygon's coordinates.
#             beta_deg (float, optional): The angle in degrees used to calculate
#                                         the offset distance. The offset is proportional
#                                         to depth * tan(beta). Defaults to 70.0.
    
#         Returns:
#             np.ndarray: An array of coordinates for the new, offset polygon.
#     """
#     beta_rad = math.radians(beta_deg)
#     centroid = polygon.centroid
#     bounds = polygon.exterior.coords

#     restrict_coords = []

#     for pt in bounds:
#         x_pt, y_pt = pt

#         depth_val = float(bathymetry.interp(x=x_pt, y=y_pt, method="linear").values)

#         dx = centroid.x - x_pt
#         dy = centroid.y - y_pt
#         length = np.hypot(dx, dy)

#         if length == 0:
#             ux, uy = 0.0, 0.0
#         else:
#             ux = dx / length
#             uy = dy / length

#         offset_val = -depth_val * math.tan(beta_rad)
#         new_x = x_pt + offset_val * ux
#         new_y = y_pt + offset_val * uy

#         restrict_coords.append((new_x, new_y))

#     return np.array(restrict_coords)


def offset_bounds_from_bathymetry(polygon: Polygon, bathymetry: xr.DataArray, beta_deg: float = 70.0): 
    beta_rad = math.radians(beta_deg)
    coords = list(polygon.exterior.coords)

    if coords[0] == coords[-1]:
        coords = coords[:-1]  

    restrict_coords = []
    n = len(coords)

    for i in range(n):
        p_prev = coords[i - 1]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]

        # Tangenti
        vec1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]])
        vec2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

        # Normali (90° in senso orario)
        def normal(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return np.array([0.0, 0.0])
            return np.array([v[1], -v[0]]) / norm

        n1 = normal(vec1)
        n2 = normal(vec2)

        # Normale media
        n_avg = n1 + n2
        norm = np.linalg.norm(n_avg)
        if norm == 0:
            n_avg = n1  # fallback
        else:
            n_avg /= norm

        # Interpolazione batimetria
        try:
            depth_val = float(bathymetry.interp(x=p_curr[0], y=p_curr[1], method="linear").values)
            if np.isnan(depth_val):
                raise ValueError
        except:
            depth_val = 0.0  # fallback se fuori raster

        # Offset
        offset_val = -depth_val * math.tan(beta_rad)
        new_x = p_curr[0] + offset_val * n_avg[0]
        new_y = p_curr[1] + offset_val * n_avg[1]

        restrict_coords.append((new_x, new_y))

    # Chiude il poligono ripetendo il primo punto alla fine
    if restrict_coords[0] != restrict_coords[-1]:
        restrict_coords.append(restrict_coords[0])

    return np.array(restrict_coords)

