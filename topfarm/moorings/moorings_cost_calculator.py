# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:47:46 2025

@author: Giuliani
"""

import sys
import numpy as np


def moorings_cost(mooring_anchoring, anchoring_cost, moorings_meter_cost):
    costs={
        'moorings': 0,
        'anchoring': 0,
        }
    
    tmp=0
    for i in range(len(mooring_anchoring)):
        for j in range(len(mooring_anchoring[i]['anchors'])):
            costs['moorings']+=mooring_anchoring[i]['anchors'][j]['length']*moorings_meter_cost
            
            for at in anchoring_cost:
                
                if at == mooring_anchoring[i]['anchors'][j]['seabed']:
                    costs['anchoring']+=anchoring_cost[at]
                    tmp=1
                    break
          
            if tmp == 0:
                print(f'anchoring type of turbine {i} anchor {j} not in cost database')      
            
            tmp=0    
    return costs