# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:04:02 2025

@author: Giuliani
"""
import sys
import topfarm
from topfarm.moorings.seabed_features import seabed_features
import numpy as np
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
import geopandas as gpd
from shapely.geometry import box
import rioxarray
from pyproj import CRS
import miniball



def fine_mesh(x, y, z, resolution_factor):
    
    """
    Create a finer meshgrid by interpolating a given 2D surface.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x coordinates.
    y : np.ndarray
        1D array of y coordinates.
    z : np.ndarray
        2D array of z values corresponding to the grid defined by x and y.
    resolution_factor : int
        Factor by which the resolution of the grid will be increased.
    
    Returns
    -------
    X_fine : np.ndarray
        2D array of refined x coordinates (meshgrid).
    Y_fine : np.ndarray
        2D array of refined y coordinates (meshgrid).
    Z_fine : np.ndarray
        2D array of interpolated z values on the refined grid.
    """
    nx, ny = len(x) * resolution_factor, len(y) * resolution_factor
    interp = RegularGridInterpolator((y, x), z, method='linear', bounds_error=False, fill_value=np.nan)
    x_fine = np.linspace(x.min(), x.max(), nx)
    y_fine = np.linspace(y.min(), y.max(), ny)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    Z_fine = interp(np.column_stack([Y_fine.ravel(), X_fine.ravel()])).reshape(Y_fine.shape)
    return X_fine, Y_fine, Z_fine



def surf_clipping(surface, beta, alpha, x_t, y_t, mooring_type, n_moorings, plot):
    
    """
    Generate a conical mooring footprint and compute anchor points on a surface.
    
    Parameters
    ----------
    surface : pv.StructuredGrid
        PyVista surface mesh representing bathymetry.
    beta : float
        Cone half-angle (degrees) defining the mooring spread.
    alpha : float
        Orientation angle of the first mooring (degrees from North).
    x_t : float
        X coordinate of the turbine position.
    y_t : float
        Y coordinate of the turbine position.
    mooring_type : str
        Type of mooring ('taught' supported).
    n_moorings : int
        Number of mooring lines to simulate.
    plot : bool
        If True, plots the cone, surface, and anchors.
    
    Returns
    -------
    np.ndarray
        Coordinates of the edge points of the clipped cone.
    list of dict
        Anchor definitions with keys: 'name', 'coords', and 'mooring_type'.
    """
    
    # if mooring_type == 'catenary':
    #     anchors = []
        
    if mooring_type != 'taught':
        print('Only "taught" type implemented for now')
        return None, None
    
    if mooring_type == 'taught':
        beta_rad = np.radians(beta)
        
        try:
            # height of cone equal to max bathymetry plus offset (1000 m) for PyVista scale problem
            h = abs(np.min(surface.points[:, 2])) + 800
            r = h * np.tan(beta_rad)
            vertex = np.array([x_t, y_t, 0])
            direction = np.array([0, 0, 1])
            center = vertex - direction * (h / 2)
        
            cone = pv.Cone(center=center.tolist(), direction=direction.tolist(),
                           height=h, radius=r, resolution=350, capping=True) #300
                    
            clipped_plane = surface.triangulate().clean().clip_surface(cone.triangulate().clean(), invert=True)         # intersection surface between cone and bathymetry
            
            edges = clipped_plane.extract_surface().extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                non_manifold_edges=False, manifold_edges=False)
            
            # point at maximum distance from cone vertex, used for calculating max footprint radius
            p_max_d = edges.points[np.argmax(np.linalg.norm(edges.points - vertex, axis=1))]
            max_radius = np.linalg.norm(p_max_d - np.array([x_t, y_t, p_max_d[2]]))
            
        except Exception as e:
            print(f'{e}: error during mooring footprint clipping with bathymetry')
        
       
        # find anchoring points
        anchors = []
        try:
            for i in range(n_moorings):
                # for each mooring calculates alpha
                alpha_rad = np.radians(alpha + (360 / n_moorings) * i)
                dx = h * np.tan(beta_rad) * np.sin(alpha_rad)
                dy = h * np.tan(beta_rad) * np.cos(alpha_rad)
                dz = -h
                
                # cone vertex = starting point. For creating a line PyVista needs starting and ending point          
                end = vertex + np.array([dx, dy, dz])
                point, _ = surface.extract_surface().triangulate().clean().ray_trace(vertex, end)     # finds intersection between surface (bathymetry) and PyVista line
                anchor_coords = pv.PolyData(point).points
                anchor_point = anchor_coords[0]                
                length = np.linalg.norm(anchor_point - vertex)
                anchors.append({
                    'name': f'Anchor{i}', 
                    'coords': anchor_coords, 
                    'mooring_type': mooring_type,
                    'length': length,
                    })
                   
            for i in range(n_moorings):
                try_except=anchors[i]['coords'][0]
                
        except Exception as e:
            print(f'{e}: error during anchoring point clipping with bathymetry')


    if plot:
        plotter = pv.Plotter()
        plotter.add_mesh(surface, show_edges=True, color='lightblue', )
        plotter.add_mesh(cone, show_edges=True, color='red', opacity=0.7)
        plotter.add_mesh(clipped_plane, show_edges=True, color='blue')
        plotter.show()

    return edges.points, anchors, max_radius



def footprint(xarray, shape_crs, wt_x, wt_y, beta, alpha, max_d, resolution_factor, mooring_type, n_moorings, plot):
    """
    Generate the mooring footprint and anchors for a set of turbine positions.
    
    Parameters
    ----------
    xarray : xarray.DataArray
        Bathymetric raster data.
    shape_crs : CRS
        Coordinate reference system of the site boundary.
    wt_x : array-like
        X coordinates of the wind turbines.
    wt_y : array-like
        Y coordinates of the wind turbines.
    beta : float
        Cone spread angle for mooring lines.
    alpha : float
        Initial angle orientation (degrees).
    max_d : float
        Maximum distance to clip raster around each turbine.
    resolution_factor : int
        Factor for increasing grid resolution.
    mooring_type : str
        Type of mooring ('taught' supported).
    n_moorings : int
        Number of mooring lines per turbine.
    plot : bool
        If True, plot each step for visualization.
    
    Returns
    -------
    list of dict
        Each item contains:
            - 'mooring_footprint': array of clipped surface points.
            - 'anchors': list of anchor dictionaries for each turbine.
    """
    
    foot_print = []
    
    for x, y in zip(wt_x, wt_y):
        rect = box(x - max_d, y - max_d, x + max_d, y + max_d)
        rect_gdf = gpd.GeoDataFrame({'geometry': [rect]}, crs=shape_crs)

        try:
            clipped = xarray.rio.clip(rect_gdf.geometry, rect_gdf.crs, drop=True)
            # clipped = xarray.rio.clip_box(  minx=x - max_d,
            #                                 miny=y - max_d,
            #                                 maxx=x + max_d,
            #                                 maxy=y + max_d,
            #                                 crs=shape_crs,
            #                             )
            
            # interpolating if required for higher resolution 
            if resolution_factor != 1:
                X_fine, Y_fine, Z_fine = fine_mesh(clipped.x.data, clipped.y.data, clipped.data, resolution_factor)
                surface = pv.StructuredGrid(X_fine, Y_fine, Z_fine)

            else:
                X, Y =np.meshgrid(clipped.x.data, clipped.y.data)
                surface = pv.StructuredGrid(X, Y, clipped.data)
                
            # calling clipping function
            edges, anchors, max_radius = surf_clipping(surface, beta, alpha, x, y, mooring_type, n_moorings, plot)
            foot_print.append({'mooring_footprint': edges, 'anchors': anchors, 'max_radius': max_radius})
        
        except Exception as e:           
            print(f'{e} occurred while clipping bathymetry with max_d')
            edges, anchors, max_radius = [], [], []
            foot_print.append({'mooring_footprint': edges, 'anchors': anchors, 'max_radius': max_radius})
        
    return foot_print



def moorings_footprint(x, y, site, beta, resolution_factor, max_d, mooring_type, n_moorings, plot):
    
    # finds wd with higher frequency
    alpha = site.ds.Sector_frequency.wd.data[np.argmax(site.ds.Sector_frequency.data)]
    plot=False
    return seabed_features(site, footprint(site.water_depth, site.bounds_shape.crs,           
                                             np.array(x),            # wt_x
                                             np.array(y),            # wt_y
                                             beta, alpha, max_d, resolution_factor,
                                             mooring_type, n_moorings, plot))



    
    
    
    
    