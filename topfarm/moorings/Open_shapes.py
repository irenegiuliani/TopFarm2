# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:45:23 2025
@author: Giuliani
"""
import sys
sys.path.append(r'D:\Giuliani\Projects\NADARA\Topfarm')
sys.path.append(r'D:\Giuliani\Projects\NADARA\Tool')
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from pyproj import CRS
from shape_tools import process_seabed_layers, generate_seabed_gdf, add_offset, process_seabed_layers_from_list
from plotting import plot_seabed

#%% Prepare Odra Site
seabed_layers_info = [
    {
        "path": r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\20251024_Geohazards\Geowynd_Geohazard.gpkg',
        "key_layers": ['Landslide_area_500m_buffer', 'Pockmark_buffer'],
        "seabed_type": ['forbidden', 'forbidden'],
        "max_depth": [0, 0],
        "add_buffer": [0, 0],
        'is_base': [False, False],
    },
    {
        "path": r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\GroundModel-DataandReports\BFR_ODR_ENG_DAT_0001_IGM_GISRev02\BFR_ODR_ENG_DAT_0001_IGM_Rev02.gdb',
        "key_layers" :['Odra_OWF_Survey_Area'],
        "seabed_type": ['sand'],
        "max_depth": [-9999],
        'is_base': [True],
    },
    {
        "path": r'D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Odra\ODRA_within_16km_from_coast.shp',
        "seabed_type": ['forbidden'],
        "max_depth": [0],
    }
]

output_path=r'D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Odra\Odra_seabed.gdb'

seabed_Odra=process_seabed_layers_from_list(
    seabed_layers_info,
    output_path=output_path
)


# crs = CRS.from_epsg(32634)
# gdf = gpd.read_file(output_path).to_crs(crs)

# fig, ax = plt.subplots(figsize=(8, 6))
# gdf.plot(ax=ax, color='lightblue', edgecolor='black')
# plt.title("MultiPolygon")
# plt.grid(True)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

plot_seabed(seabed_Odra)



#%% Prepare Kailia Site

seabed_layers_info = [
    {
        "path": r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\GroundModelDataandReports\BFR_KAI_ENG_DAT_0002_IGMGISv4\2022-060_GW_Kailia_IGM_Rev03.gdb',
        "key_layers": ['OWF_Area_25102024_prj', 'SP2', 'SP3a', 'SP3b', 'SP5'],
        "seabed_type": ['sand', 'rock', 'rock', 'rock', 'forbidden'],
        "max_depth": [-9999, -15, -15, -15, 0],
        "add_buffer": [0, 0, 0, 0, 10],
        'is_base': [True, False, False, False, False],
    },
    {
        "path": r'D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Kailia\KAILIA_within_15km_from_coast.shp',
        "seabed_type": ['forbidden'],
        "max_depth": [0],
    }
]

output_path= r'D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Kailia\Kailia_seabed.gdb'

seabed_Kailia=process_seabed_layers_from_list(
    seabed_layers_info,
    output_path=output_path
)


plot_seabed(seabed_Kailia)

#%%Odra distance
# --- 1. Leggi i shapefile ---
shore = gpd.read_file(filename=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\20251024_Geohazards\Geowynd_Geohazard.gpkg', layer='Reg01012023_WGS84').to_crs(CRS.from_epsg(32634))   
shore.plot()

layers = fiona.listlayers(r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\20251024_Geohazards\Geowynd_Geohazard.gpkg')         # shape della costa
wf = gpd.read_file(filename=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\GroundModel-DataandReports\BFR_ODR_ENG_DAT_0001_IGM_GISRev02\BFR_ODR_ENG_DAT_0001_IGM_Rev02.gdb', layer='Odra_OWF_Survey_Area').to_crs(CRS.from_epsg(32634))   
wf.plot()


buffer_16km = shore.buffer(16000)  # buffer in metri
buffer_union = buffer_16km.unary_union  # unisce tutti i poligoni del buffer in uno solo

windfarm_within_16km = wf.copy()
windfarm_within_16km['geometry'] = wf.intersection(buffer_union)

windfarm_within_16km = windfarm_within_16km[~windfarm_within_16km.is_empty]

windfarm_within_16km.plot()

windfarm_within_16km.to_file(r"D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Odra\ODRA_within_16km_from_coast.shp")

#%%kailia distance
# --- 1. Leggi i shapefile ---
shore = gpd.read_file(filename=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\20251024_Geohazards\Geowynd_Geohazard.gpkg', layer='Reg01012023_WGS84').to_crs(CRS.from_epsg(32634))   
shore.plot()

layers = fiona.listlayers(r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\GroundModelDataandReports\BFR_KAI_ENG_DAT_0002_IGMGISv4\2022-060_GW_Kailia_IGM_Rev03.gdb')         # shape della costa
wf = gpd.read_file(filename=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Kailia\GroundModelDataandReports\BFR_KAI_ENG_DAT_0002_IGMGISv4\2022-060_GW_Kailia_IGM_Rev03.gdb', layer='OWF_Area_25102024_prj').to_crs(CRS.from_epsg(32634))   
wf.plot()


buffer_16km = shore.buffer(15000)  # buffer in metri
buffer_union = buffer_16km.unary_union  # unisce tutti i poligoni del buffer in uno solo

windfarm_within_16km = wf.copy()
windfarm_within_16km['geometry'] = wf.intersection(buffer_union)

windfarm_within_16km = windfarm_within_16km[~windfarm_within_16km.is_empty]

windfarm_within_16km.plot()

windfarm_within_16km.to_file(r"D:\Giuliani\Projects\NADARA\Tool\tmp_files\Seabed_features_Kailia\KAILIA_within_15km_from_coast.shp")



#%% Odra ship wrecks
wrecks_path=r'D:\Giuliani\Projects\NADARA\UNIFI_condivisa\JVinputs\Odra\Environmental_Constraints\Wrecks\RELITTI ODRA.shp'
output_path=r'D:\Giuliani\Projects\NADARA\Tool\tmp_files\Forbidden_areas_Odra\Odra_shipwrecks.gdb'
crs=CRS.from_epsg(32634)

wrecks = gpd.read_file(wrecks_path).to_crs(crs)

merged_geom = wrecks.union_all()

if merged_geom.geom_type == 'Polygon':
    merged_geom = MultiPolygon([merged_geom])
    
elif merged_geom.geom_type == 'GeometryCollection':
    merged_geom = MultiPolygon([
        geom for geom in merged_geom.geoms 
        if geom.geom_type in ['Polygon', 'MultiPolygon']
    ])


merged_gdf = gpd.GeoDataFrame(geometry=[merged_geom], crs=crs)

fig, ax = plt.subplots(figsize=(8, 6))
merged_gdf.plot(ax=ax, color='lightblue', edgecolor='black')
plt.title("MultiPolygon")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

merged_gdf.to_file(output_path)









