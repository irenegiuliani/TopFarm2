# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 13:37:05 2025

@author: Giuliani
"""

import numpy as np
from shapely.geometry import Point


def seabed_features(site, foot_print):
    """Assigns seabed feature information to anchors within a footprint.

    This function iterates through each anchor point in the provided footprint and
    determines which seabed feature it falls within. The 'Feature' name of the
    containing seabed is then added to the anchor's dictionary.

    Args:
        site (object): An object representing the site. It is expected to have an
            attribute `seabeds` which is a GeoDataFrame. This GeoDataFrame should
            contain a 'geometry' column with geometric shapes representing the
            seabed features and a 'Feature' column with the name of each feature.
        foot_print (list): A list of dictionaries. Each dictionary represents a
            component of the footprint and contains an 'anchors' key. The value of
            'anchors' is a list of dictionaries, with each dictionary representing
            an anchor. Each anchor dictionary must have a 'coords' key, containing
            a list of coordinates [x, y, z].

    Returns:
        None: The function modifies the `foot_print` list in-place by adding a
            'seabed' key to each anchor's dictionary if a containing seabed
            feature is found.
    """

    seabeds=site.seabeds
    for j in range(len(foot_print)):
    
        for i, anchor in enumerate(foot_print[j]['anchors']):
       
            for s, seabed in enumerate(seabeds.iterrows()):
                try:
                    x, y, z = anchor['coords'][0][0], anchor['coords'][0][1], anchor['coords'][0][2]
                    point = Point(x, y)
                    
                       
                    if seabeds.loc[s, 'geometry'].contains(point):
                        foot_print[j]['anchors'][i]['seabed']=seabeds.loc[s, 'seabed']
                        break
                        
                    else:
                        foot_print[j]['anchors'][i]['seabed']='Nodata'
                        
                except Exception as e:
                    print(f'{e}: Turbine number {j} has no Anchoring {i}')
                    
    return foot_print