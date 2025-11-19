# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 20:47:46 2025

@author: Giuliani
"""

import sys
import numpy as np


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


def moorings_cost(mooring_anchoring, anchoring_cost, moorings_meter_cost):
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
