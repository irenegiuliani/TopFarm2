# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 08:30:04 2025

@author: Giuliani
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import geopandas as gpd
import fiona
from osgeo import gdal 
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import gc  


def check_crs(file_paths, ref_crs, sitename, reprojected_dir):
    
    os.makedirs(reprojected_dir, exist_ok=True)

    for key, path in file_paths.items():
        ext = os.path.splitext(path)[1].lower()

        if ext == '.shp':
            with fiona.open(path, 'r') as src:
                file = gpd.GeoDataFrame.from_features(src, crs=src.crs).to_crs(ref_crs)
            
            out_path = os.path.join(
                reprojected_dir,
                f"{sitename}{key}_reprojected{ref_crs.to_string().replace(':', '')}{ext}"
            )
            file.to_file(out_path)
            file_paths[key] = out_path
            del file
            gc.collect()

        elif ext == '.tif':
            with rasterio.open(path) as src:
                dst_crs = ref_crs.to_string()
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                out_path = os.path.join(
                    reprojected_dir,
                    f"{sitename}{key}_reprojected{ref_crs.to_string().replace(':', '')}{ext}"
                )

                with rasterio.open(out_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.nearest
                        )
            file_paths[key] = out_path
            gc.collect()

        elif ext == '.gdb':
            layers = fiona.listlayers(path)
            out_path = os.path.join(
                reprojected_dir,
                f"{sitename}{key}_reprojected{ref_crs.to_string().replace(':', '')}.gpkg"
            )
            for layer in layers:
                with fiona.open(path, layer=layer) as src:
                    file = gpd.GeoDataFrame.from_features(src, crs=src.crs).to_crs(ref_crs)
                    file.to_file(out_path, driver='GPKG', layer=layer)
                    del file
                    gc.collect()
            
            file_paths[key] = out_path

