# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 08:47:54 2025

@author: Giuliani
"""
import sys
sys.path.append(r'D:\Giuliani\Projects\NADARA\Topfarm')
sys.path.append(r'D:\Giuliani\Projects\NADARA\Tool')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.plot import show, plotting_extent
from rasterio.enums import Resampling
from Raster_tools import reproject_and_downsample, fill_holes, merge_raster, create_raster_from_template
import math

#%% Prepare Odra Stie
input_tif = r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\Fugro_Bathymetry_Rev02_raster.tif'
downsampled_tif = r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\Fugro_Bathymetry_Rev02_raster_downsampled.tif'
filled_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\Fugro_Bathymetry_Rev02_raster_filled.tif'
buffer_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\Fugro_Bathymetry_Rev02_raster_buffered.tif'
target_crs_epsg = 'EPSG:32634'
target_resolution_meters = (5, 5)

# Reproject and downsample
reproject_and_downsample(input_tif, downsampled_tif, target_crs_epsg, target_resolution_meters, resampling_method=Resampling.average)

# Fill holes
max_search_dist=10
smoothing_it=0
fill_holes(downsampled_tif, filled_tif, max_search_dist, smoothing_it)

with rasterio.open(filled_tif) as src:
    data_array=src.read(1)
create_raster_from_template(filled_tif, buffer_tif, np.nan_to_num(data_array, nan=-300))


# plot raster
with rasterio.open(buffer_tif) as src:
    data=src.read(1)
    extent=plotting_extent(src)
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # image = ax.imshow(data,
    #                   cmap='viridis',
    #                   extent=extent,
    #                   vmin=np.nanmin(data) if np.nanmin(data) < np.inf else None,
    #                   vmax=np.nanmax(data) if np.nanmax(data) > -np.inf else None,
    #                   )
    # cbar = plt.colorbar(image, ax=ax, label='Depth')
    # ax.grid(True, linestyle=':', alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    
    show(src, title='Odra bathymetry', cmap='viridis')
plt.show()


#%% Prepare Kailia site
input_tif = r'D:/Giuliani/Projects/NADARA/UNIFI_condivisa/JVinputs/Kailia/Fugro_Kailia_Bathy_2024_raster.tif'
downsampled_tif = r'D:/Giuliani/Projects/NADARA/UNIFI_condivisa/JVinputs/Kailia/Fugro_Kailia_Bathy_2024_raster_downsampled.tif'
input_tif_EMOD=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\Fugro_Kailia_Bathy_2024_MaGIC_EMODnet.tif'
resampled_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\Fugro_Kailia_Bathy_2024_MaGIC_EMODnet_res.tif'
merged_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\merged_bathymetry.tif'
filled_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\filled_bathymetry.tif'
buffer_tif=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\buffer_bathymetry.tif'
target_crs_epsg = 'EPSG:32634'
target_resolution_meters = (5, 5)

# Reproject and downsample original raster
reproject_and_downsample(input_tif, downsampled_tif, target_crs_epsg, target_resolution_meters, resampling_method=Resampling.average)

# Reproject and resample EMOD data
reproject_and_downsample(input_tif_EMOD, resampled_tif, target_crs_epsg, target_resolution_meters, resampling_method=Resampling.average)

# Merge
src_files = [rasterio.open(resampled_tif, masked=True), rasterio.open(downsampled_tif)]
merge_raster(src_files, merged_tif)

# Fill holes
max_search_dist=30
smoothing_it=0
fill_holes(merged_tif, filled_tif, max_search_dist, smoothing_it)

with rasterio.open(filled_tif) as src:
    data_array=src.read(1)
create_raster_from_template(filled_tif, buffer_tif, np.nan_to_num(data_array, nan=-300))

# plot raster
with rasterio.open(filled_tif) as src:
    data=src.read(1)
    extent=plotting_extent(src)
    
    # fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # image = ax.imshow(data,
    #                   cmap='viridis',
    
    #                   extent=extent,
    #                   vmin=np.nanmin(data) if np.nanmin(data) < np.inf else None,
    #                   vmax=np.nanmax(data) if np.nanmax(data) > -np.inf else None,
    #                   )
    # cbar = plt.colorbar(image, ax=ax, label='Depth')
    # ax.grid(True, linestyle=':', alpha=0.5)
    # plt.tight_layout()
    # plt.show()
    
    show(src, title='Kailia bathymetry', cmap='viridis')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import rasterio

with rasterio.open(filled_tif) as src:
    data = src.read(1)
    height, width = data.shape

    # Risoluzione (metri/pixel)
    pixel_width, pixel_height = src.res

    # Definisci extent in metri, da 0 a dimensione fisica reale
    extent = (0, width * (pixel_width-0.9), 0, height * (pixel_height-0.9))

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        data,
        cmap='viridis',
        extent=extent,
        origin='upper',  # immagine orientata come nei GIS
        vmin=np.nanmin(data) if np.nanmin(data) < np.inf else None,
        vmax=np.nanmax(data) if np.nanmax(data) > -np.inf else None,
    )

    cbar = plt.colorbar(image, ax=ax, shrink=0.7, aspect=15, anchor=(-0.25, 0.5))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Depth", fontsize=12)

    ax.set_title('Bathymetry', fontsize=12)
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(r'D:\Giuliani\Projects\NADARA\Topfarm\seabed.png', dpi=700, bbox_inches='tight')
    plt.show()



