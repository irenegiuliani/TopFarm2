# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:47:46 2025

@author: Giuliani
"""

from shapely.ops import unary_union
from shapely.geometry import Point
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# def moorings_cost(mooring_anchoring, anchoring_cost, moorings_meter_cost):
#     costs={
#         'moorings': [],
#         'anchoring': [],
#         }
    
#     tmp=0
    
#     for i in range(len(mooring_anchoring)):
#         cost_moor = 0
#         cost_anchor = 0
#         for j in range(len(mooring_anchoring[i]['anchors'])):
#             cost_moor += mooring_anchoring[i]['anchors'][j]['length']*moorings_meter_cost  
#             for at in anchoring_cost:
#                 if at == mooring_anchoring[i]['anchors'][j]['seabed']:
#                     cost_anchor += anchoring_cost[at]
#                     tmp=1
#                     break
          
#             if tmp == 0:
#                 print(f'anchoring type of turbine {i} anchor {j} not in cost database')      
            
#             tmp=0   
            
#         costs['moorings'].append(cost_moor)   
#         costs['anchoring'].append(cost_anchor)
        
#     costs['moorings']=np.array(costs['moorings'])
#     costs['anchoring']=np.array(costs['anchoring'])
#     return costs

def moorings_cost_gaussian_map(mooring_anchoring, anchoring_cost, moorings_meter_cost, xs, ys, cost_grid):
    mooring_costs = []
    anchoring_costs = []

    for i, turbine in enumerate(mooring_anchoring):
        cost_moor = 0.0
        cost_anchor = 0.0

        for j, anchor in enumerate(turbine['anchors']):

            cost_moor += anchor['length'] * moorings_meter_cost  
            cost_anchor += cost_map_interpolator(anchor['coords'][0][0], anchor['coords'][0][1], xs, ys, cost_grid)


        mooring_costs.append(cost_moor)
        anchoring_costs.append(cost_anchor)

    return {
        'moorings': np.array(mooring_costs),
        'anchoring': np.array(anchoring_costs),
    }


def cost_map_interpolator(x, y, xs, ys, cost_grid):
    interp = RegularGridInterpolator((ys, xs), cost_grid, method='linear', bounds_error=False, fill_value=10000000)
    point=np.column_stack([y, x])  
    return float(interp(point))

def get_cost_map_gaussian(gdf, cost_map, res=5, sigma=2, cost_col='cost', seabed_col='seabed'):

    gdf[cost_col] = gdf[seabed_col].map(cost_map)
    minx, miny, maxx, maxy = gdf.total_bounds

    nx = int(np.ceil((maxx - minx) / res))
    ny = int(np.ceil((maxy - miny) / res))

    xs = minx + res * (0.5 + np.arange(nx))
    ys = miny + res * (0.5 + np.arange(ny))

    X, Y = np.meshgrid(xs, ys)  

    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(X.ravel(), Y.ravel()),
        crs=gdf.crs,
    )

    joined = points.sjoin(
        gdf[[cost_col, "geometry"]],
        how="left",
        predicate="within",  
    )
    
    cost_grid = joined[cost_col].to_numpy().reshape(ny, nx)
    max_cost = max(cost_map.values())
    cost_grid = np.where(np.isnan(cost_grid), max_cost, cost_grid)
    gaussian_grid = gaussian_filter(cost_grid, sigma=sigma, mode='nearest')
    return xs, ys, gaussian_grid


def moorings_cost_step(mooring_anchoring, anchoring_cost, moorings_meter_cost):
    mooring_costs = []
    anchoring_costs = []

    for i, turbine in enumerate(mooring_anchoring):
        cost_moor = 0.0
        cost_anchor = 0.0

        for j, anchor in enumerate(turbine['anchors']):

            cost_moor += anchor['length'] * moorings_meter_cost  

            seabed_type = anchor['seabed']

            if seabed_type in anchoring_cost:
                cost_anchor += anchoring_cost[seabed_type]
            else:
                print(
                    f"Anchoring type of turbine {i} anchor {j} "
                    f"('{seabed_type}') not in cost database"
                )

        mooring_costs.append(cost_moor)
        anchoring_costs.append(cost_anchor)

    return {
        'moorings': np.array(mooring_costs),
        'anchoring': np.array(anchoring_costs),
    }


def get_nearest_cost(cost_zones_gdf, point, cost_col: str = "cost"):
    if cost_zones_gdf.empty:
        raise ValueError("ERROR: gdf is empty.")
    dists = cost_zones_gdf.geometry.distance(point)
    idx_min = dists.idxmin()
    return cost_zones_gdf.loc[idx_min, cost_col]


import geopandas as gpd
from typing import List, Dict, Any

def moorings_cost_poly(
    moorings: List[Dict[str, Any]],
    gdf: gpd.GeoDataFrame,
    anchoring_cost_map: Dict[str, float],
    moorings_meter_cost: float = 500.0,
    seabed_col: str = "seabed",
    cost_col: str = "cost",
    buffer: float = 10.0,
    penalty_coeff: float = 150.0,
    i: int = 0,
    activate_extra_penalty: bool = False,
) -> Dict[str, np.ndarray]:
    
    gdf = gdf.copy()
    gdf[cost_col] = gdf[seabed_col].map(anchoring_cost_map)

    forbidden_gdf = gdf[gdf[seabed_col] == "forbidden"]
    cost_zones = gdf[gdf[seabed_col] != "forbidden"]
    
    forbidden_geom = None
    forbidden_boundary = None
    
    if not forbidden_gdf.empty:
        forbidden_geom = unary_union(forbidden_gdf.geometry.values)
        forbidden_boundary = forbidden_geom.boundary
    
    if activate_extra_penalty:
        # coeff = np.exp(0.001*i)
        coeff = (1 + 100) / (1 + np.exp(-(np.log(100)/2500) * (i - 2500))) - 1

    else:
        coeff = 1
    
    mooring_costs = []
    anchoring_costs = []
    anchoring_dist = []

    for i, turbine in enumerate(moorings):
        current_moor_cost = 0.0
        current_anchor_cost = 0.0

        for j, anchor in enumerate(turbine['anchors']):

            current_moor_cost += anchor['length'] * moorings_meter_cost
            
            seabed_type = anchor['seabed']
            
            if seabed_type not in anchoring_cost_map:
                print(f"WARNING: Turbine {i}, Anchor {j} - Type '{seabed_type}' not in cost DataBase. Skipping.")
                continue
            
            base_cost = anchoring_cost_map[seabed_type]
            anchor_cost_component = base_cost 

            if forbidden_geom is not None:
                coords = anchor['coords'][:, :2].flatten()
                pt = Point(coords[0], coords[1])

                dist = pt.distance(forbidden_geom)

                is_inside = (dist == 0)
                if is_inside:
                    dist = -pt.distance(forbidden_boundary)

                anchoring_dist.append(dist)
                
                if dist < buffer:
                    nearest_cost = get_nearest_cost(cost_zones, pt, cost_col=cost_col)
                
                    if dist < 0:
                        penalty = penalty_coeff * (buffer ** 3)
                        anchor_cost_component = nearest_cost + ((penalty - (200 * dist)) * coeff)
                    else:
                        violation = buffer - dist  
                        penalty = penalty_coeff * (violation ** 3)
                        anchor_cost_component = nearest_cost + (penalty * coeff)


                # if dist < buffer:
                #     if dist<0:
                #         nearest_cost = get_nearest_cost(cost_zones, pt, cost_col=cost_col)
                #         penalty = penalty_coeff * ((0 - buffer) ** 2)
                #         anchor_cost_component = nearest_cost + penalty - (50*dist)
                #     else:
                #         nearest_cost = get_nearest_cost(cost_zones, pt, cost_col=cost_col)
                #         penalty = penalty_coeff * ((dist - buffer) ** 2)
                #         anchor_cost_component = nearest_cost + penalty
            
            current_anchor_cost += anchor_cost_component

        mooring_costs.append(current_moor_cost)
        anchoring_costs.append(current_anchor_cost)

    return {
        'moorings': np.array(mooring_costs),
        'anchoring': np.array(anchoring_costs),
        'anchoring_dist_from_forbidden': np.array(anchoring_dist),
    }

# def moorings_cost(
#     moorings,
#     gdf,
#     anchoring_cost_map,
#     moorings_meter_cost: float = 500.0,
#     seabed_col: str = "seabed",
#     cost_col: str = "cost",
#     buffer: float = 10.0,
#     penalty_coeff: float = 50.0,
# ):


#     gdf["cost"] = gdf["seabed"].map(anchoring_cost_map)

#     mooring_costs = []
#     anchoring_costs = []
#     forbidden_gdf = gdf[gdf[seabed_col] == "forbidden"]
#     cost_zones = gdf[gdf[seabed_col] != "forbidden"]
    
#     for i, turbine in enumerate(moorings):
#         cost_moor = 0.0
#         cost_anchor = 0.0
    
#         for j, anchor in enumerate(turbine['anchors']):
#             cost_moor += anchor['length'] * moorings_meter_cost  
#             seabed_type = anchor['seabed']
         
#             if not forbidden_gdf.empty:
#                 forbidden_geom = unary_union(forbidden_gdf.geometry.values)   
#                 pt=Point(anchor['coords'][:, :2])
#                 dist = pt.distance(forbidden_geom)
                
#                 if dist == 0:
#                     dist = -pt.distance(forbidden_geom.boundary)
    
#                 if dist < buffer:
#                     if seabed_type in anchoring_cost_map:
#                         cost_anchor += get_nearest_cost(cost_zones, pt, cost_col="cost")+penalty_coeff*(dist-buffer)**2
#                     else:
#                         print(
#                             f"Anchoring type of turbine {i} anchor {j} "
#                             f"('{seabed_type}') not in cost database"
#                         )
                    
#                 elif dist > buffer:       
#                     if seabed_type in anchoring_cost_map:
#                         cost_anchor += anchoring_cost_map[seabed_type]
#                     else:
#                         print(
#                             f"Anchoring type of turbine {i} anchor {j} "
#                             f"('{seabed_type}') not in cost database"
#                         )
    
#             else:
#                 if seabed_type in anchoring_cost_map:
#                     cost_anchor += anchoring_cost_map[seabed_type]
#                 else:
#                     print(
#                         f"Anchoring type of turbine {i} anchor {j} "
#                         f"('{seabed_type}') not in cost database"
#                     )
#         mooring_costs.append(cost_moor)
#         anchoring_costs.append(cost_anchor)
    
#     cost = {
#         'moorings': np.array(mooring_costs),
#         'anchoring': np.array(anchoring_costs),
#     }
#     return cost
