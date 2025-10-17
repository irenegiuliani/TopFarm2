# site_data_loader.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geopandas as gpd
import xarray as xr
import rasterio
from topfarm.moorings.check_crs import check_crs
from pyproj import CRS
from py_wake import np
from shapely.geometry import MultiPolygon, Polygon, Point

def load_site_data(file_paths, epsg, name, reprojected_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reprojected_dir'), idx=None, exclusion_flag=False):

    ref_crs = CRS.from_epsg(epsg)
    sitename = name

    # Check/reproject only once
    check_crs(file_paths, ref_crs, sitename, reprojected_dir)

    
    if exclusion_flag:
        exclusion_shape=gpd.read_file(file_paths['exclusion_shape']).copy()
        exclusion = []
        for ind, row in exclusion_shape.iterrows():
            exclusion.append(np.array(row['geometry'].buffer(200).exterior.coords))
    else:
        exclusion = []
        
    
    wt_shape = gpd.read_file(file_paths['wt_shape']).copy()
    wt_x = []
    wt_y = []
    poly = []
    
    for geom in wt_shape.geometry:
        if isinstance(geom, MultiPolygon):
            for polygon in geom.geoms:
                centroid = polygon.centroid
                poly.append({'poly': polygon, 'x': centroid.x, 'y': centroid.y})
                wt_x.append(centroid.x)
                wt_y.append(centroid.y)
        elif isinstance(geom, Polygon):
            centroid = geom.centroid
            poly.append({'poly': geom, 'x': centroid.x, 'y': centroid.y})
            wt_x.append(centroid.x)
            wt_y.append(centroid.y)
        
        elif isinstance(geom, Point):
            poly.append({'poly': geom, 'x': geom.x, 'y': geom.y})
            wt_x.append(geom.x)
            wt_y.append(geom.y)
        else:
            raise ValueError(f"Not supported: {type(geom)}")
    
    wt_x = np.asarray(wt_x)
    wt_y = np.asarray(wt_y)
    
    if idx is not None:
        wt_x = np.delete(wt_x, idx).tolist()
        wt_y = np.delete(wt_y, idx).tolist()
    else:
        wt_x = wt_x.tolist()
        wt_y = wt_y.tolist()

    # Load bounds
    bounds_shape = gpd.read_file(file_paths['bounds_shape']).copy()
    geom = bounds_shape.geometry[0]
    try:
        boundary = np.array(geom.exterior.coords)
        
    except AttributeError:
        boundary = np.array(Polygon(geom).exterior.coords)
    
    if boundary.shape[1] == 3:
        boundary = np.delete(boundary, 2, axis=1)

    # Load bathymetry raster
    with rasterio.open(file_paths['bathymetry']) as src:
        data = src.read(1)
        transform = src.transform
        x_coords = np.arange(src.width) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(src.height) * transform.e + transform.f + transform.e / 2
        bathymetry = xr.DataArray(
            data=data,
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            name='Bathymetry',
            attrs={
                "transform": transform,
                "crs": src.crs.to_string(),
                "res": src.res,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
                "bounds": src.bounds,
                "driver": src.driver,
                "count": src.count,
                "width": src.width,
                "height": src.height
            }
        )

    seabeds_shape = gpd.read_file(file_paths['seabeds_shape']).copy()

    return {
        "wt_x": wt_x,
        "wt_y": wt_y,
        "bathymetry": bathymetry,
        "raster_path": file_paths['bathymetry'],
        "bounds_shape": bounds_shape,
        "boundary": boundary,
        "seabeds_shape": seabeds_shape,
        "sitename": sitename,
        "resolution_factor": 1,
        "exclusion_zones" : exclusion,
    }
