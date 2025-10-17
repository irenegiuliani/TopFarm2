#%%
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  
    import numpy as np
    import os
    import openmdao.api as om
    import pandas as pd
    import time
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import openmdao.utils.logger_utils as logger_utils
    logger = logger_utils.get_logger('openmdao', level='INFO')
    
    
    import numpy as np
    import os
    import pandas as pd
    import time
    
    import topfarm
    from NADARA.Turbines.T18MW260 import T18MW260 #18MW 260m 
    from NADARA.Sites.Odra_Site import Odra_Site #statistic Kailia intallation site
    
    from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014 as BGD #wake model
    from py_wake.superposition_models import LinearSum #wake superposition model
    from py_wake.rotor_avg_models import GaussianOverlapAvgModel #rotor averaging model
    from py_wake.turbulence_models import GCLTurbulence #turbulence model
    
    # Import DTU Cost and Scaling model (dtu_wind_cm_main)
    from dtu_wind_cm_main_FOWT import economic_evaluation as ee_2
    
    # Import Topfarm constraints for site boundary and spacing
    from topfarm.constraint_components.boundary import XYBoundaryConstraint, ExclusionZone, InclusionZone #spatial constraint for sea lot
    from topfarm.constraint_components.spacing import SpacingConstraint #WT minimum distance
    from topfarm.constraint_components.constraint_aggregation import DistanceConstraintAggregation
    # Import Topfarm support classes for setting up problem and workflow
    from topfarm.cost_models.cost_model_wrappers import CostModelComponent
    from topfarm.cost_models.py_wake_wrapper import PyWakeAEPCostModelComponent
    from topfarm.cost_models.electrical.simple_msp import  XYCablePlotComp#, ElNetCost,
    from optiwindnet.api import WindFarmNetwork
    from optiwindnet.api import EWRouter, HGSRouter, MILPRouter
    from optiwindnet.augmentation import poisson_disc_filler
    
    from topfarm import TopFarmGroup, TopFarmProblem
    from topfarm.plotting import XYPlotComp, NoPlot
    
    # Import Topfarm implementation of or Scipy drivers
    from topfarm.easy_drivers import EasyScipyOptimizeDriver, EasySGDDriver, EasyRandomSearchDriver
    from multiprocessing import freeze_support
    from topfarm.moorings.moorings_footprint_calculator import moorings_footprint
    from topfarm.moorings.moorings_cost_calculator import moorings_cost
    from topfarm.moorings.plotting import plot_anchor_wt_seabed
    from topfarm.moorings.shape_tools import offset_bounds_from_bathymetry, extract_boundaries_from_poly
    from shapely.geometry import Polygon, Point, MultiPolygon
    from shapely.ops import unary_union
    from topfarm.moorings.load_site_data import load_site_data
    from topfarm.easy_drivers import EasySimpleGADriver
    from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle, RandomizeTurbinePosition



    #%% Setting
    start_time = time.time()
    file_paths={
        'wt_shape' : r'NADARA\Sites\Odra_inputs\BFR_ODR_SICH_MAP_0001_7-9_WTG.shp',
        'bounds_shape' : r'NADARA\Sites\Odra_inputs\BFR_ODR_SICH_MAP_0001_1-9_Concession Area.shp',
        'bathymetry' : r'NADARA\Sites\Odra_inputs\Fugro_Bathymetry_Rev02_raster_buffered.tif',
        'seabeds_shape' : r'NADARA\Sites\Odra_inputs\Odra_seabed.gdb',
        'exclusion_shape': r'NADARA\Sites\Odra_inputs\RELITTI_ODRA.shp',
        }
    
    
    epsg = 32634
    name = 'Odra'
    idx=np.arange(0, 34)
    site_data = load_site_data(file_paths, epsg, name, idx = idx, exclusion_flag = True)   
    site = Odra_Site(site_data)  

    n_wt = 39
    opt_res={}
    windTurbines = T18MW260()

    max_WT_conn = 5
    
    WM = 'BGD'
    nit = 1e06
    tollerance = 1e-7
        
    boundary_tr=site.boundary
    
    wake_model = BGD(site, windTurbines, k=0.04, use_effective_ws=True, 
                                     rotorAvgModel = GaussianOverlapAvgModel(), 
                                     superpositionModel = LinearSum(),  
                                     turbulenceModel = GCLTurbulence()) 
    
    
    '___________________________input data___________________________'
    # vectors for turbine properties: diameter, rated power and hub height. these are inputs to the cost model
    Drotor_vector = [windTurbines.diameter()] * n_wt                    #[m]
    Pitching_moment = [windTurbines.pit_moment] * n_wt                  #
    power_rated_vector = [float(windTurbines.power(20))*1e-6] * n_wt    #[W]
    hub_height_vector = [windTurbines.hub_height()] * n_wt              #[m]
    
    # add additional cost model inputs for shore distance, energy price, project lifetime, rated rotor speed and water depth
    distance_from_shore = site.ex_distance              #[km]
    project_duration = 30                               #[years]
    discount_rate=0.07                                  # [-]
    rated_rpm_array = [windTurbines.rated_rpm] * n_wt   # [rpm]
    
    #specify minimum spacing in diameters between turbines
    min_spacing = 4 #[-] ref:https://doi.org/10.1016/j.renene.2022.03.104 , cita anche [42] e [4] nel paper. Cita anche https://guidetofloatingoffshorewind.com/guide/b-balance-of-plant/b-3-mooring-system/, qua dice che il movimento consentito è di 30-35% della profondità, che sommata alla distanza fra il centro della piattaforma e il punto dove è collegato l'ormeggio (837 per la 15 MW) viene circa 3.9D (~937m)
    
    #specify the cable cost
    cables = np.array([(1, 358+260), (2, 358+386), (5, 358+650)])  #cable to use (n of turbines, price €/m)
    var_cable_cost          = 2.70144e06    #[€/km] specific cost for export cables (scaling the value with respct to plant capacity (/125*1005) from: A life cycle cost model for floating offshore wind farms)
    
    # set up function for new cost model with initial inputs as set above
    eco_eval = ee_2(distance_from_shore, project_duration, discount_rate, var_cable_cost)
    
    #mooring function setup
    n_moorings=3
    mooring_type='taught'
    beta=60
    resolution_factor=site.resolution_factor
    max_d=windTurbines.diameter()*2
    exclusion_zones = site.exclusion_zones

    anchoring_cost_ud={
        'sand': 200000,
        'rock': 400000,
        'forbidden': 400000,
        }
    moorings_meter_cost_ud=500
    mooring_footprint_opt = []


    #%%Cables setting

    border_poly = Polygon(site.boundary).buffer(-0.5)
    filtered_obstacles = []

    for obs in extract_boundaries_from_poly(unary_union(site.seabeds[site.seabeds['seabed'] == 'forbidden'].geometry[0])):
        poly = Polygon(obs)
        clipped = poly.intersection(border_poly)
        if clipped.is_empty:
            continue
        
        elif clipped.geom_type == 'Polygon':
            filtered_obstacles.append(np.array(clipped.exterior.coords))
            
        elif clipped.geom_type == 'MultiPolygon':
            for part in clipped.geoms:
                filtered_obstacles.append(np.array(part.exterior.coords))
    
        
    router = EWRouter()
    
    #%% Python cost functions 

    #AEP calculator                
    def aep_func(x, y, **kwargs):
        sim_stat = wake_model(x, y, n_cpu=1)    
        res = sim_stat.aep().sum(['wd','ws']).values*10**6  
        return res


    #cable length optimizer
    class WFNComponent(CostModelComponent):
        def __init__(self, turbines_pos, substations_pos, cables, border, obstacles, router, **kwargs):
            self.wfn = WindFarmNetwork(
                turbinesC=turbines_pos,
                substationsC=substations_pos,
                cables=cables,
                router=router,
                borderC=border,
                # obstacleC_=obstacles,
                )
    
            def compute(x, y, xs, ys):
                tin=time.time()
                
                self.wfn.optimize(turbinesC=np.column_stack((x, y)),
                                  substationsC=np.column_stack((xs, ys)),
                                  router = EWRouter()
                                  )
                
                self.wfn.optimize(router = MILPRouter(solver_name='ortools', time_limit=60, mip_gap=0.005, verbose=True))
                
                tend=time.time()
                
                print('Cables optimization took: {:.0f}s'.format(tin-tend))
                
                return self.wfn.cost(), {
                    'network_length': self.wfn.length(),
                    'terse_links': self.wfn.terse_links(),
                }

    
            def compute_partials(x, y, xs, ys):
                tin=time.time()
                grad_wt, grad_ss = self.wfn.gradient(
                    turbinesC=np.column_stack((x, y)),
                    substationsC=np.column_stack((xs, ys)),
                )
                dc_dx, dc_dy = grad_wt[:, 0], grad_wt[:, 1]
                dc_dxss, dc_dyss = grad_ss[:, 0], grad_ss[:, 1]
                tend=time.time()
                print('Cables partials took: {:.0f}s'.format(tin-tend))
                return [dc_dx, dc_dy, dc_dxss, dc_dyss]

    
            x_init, y_init = turbines_pos.T
            x_ss_init, y_ss_init = substations_pos.T
            super().__init__(
                input_keys=[('x', x_init), ('y', y_init),
                            ('xs', x_ss_init), ('ys', y_ss_init)],
                n_wt=turbines_pos.shape[0],
                cost_function=compute,
                cost_gradient_function=compute_partials,
                objective=False,
                output_keys=[('cabling_cost', 0.0)],
                additional_output=[
                    ('network_length', 0.0),
                    ('terse_links', np.zeros(turbines_pos.shape[0])),
                ],
                **kwargs,
            )
            
    #mooring cost def
    def mooring_func(x, y, **kwargs):
        tin=time.time()       
        print(f'mooring funct evaluated at {x[0]}, {y[0]}')
        moorings_anchoring = moorings_footprint(x, y, site, beta, resolution_factor, max_d, mooring_type, n_moorings, plot=False)  
        mooring_footprint_opt.append(moorings_anchoring)
        total_length = 0.0
        for i in range(len(moorings_anchoring)):
            for j in range(len(moorings_anchoring[i]['anchors'])):
                total_length += moorings_anchoring[i]['anchors'][j]['length']                
        cost_dict = moorings_cost(moorings_anchoring, anchoring_cost_ud, moorings_meter_cost_ud)
        moorings_cost_val = float(cost_dict['moorings'])
        anchoring_cost_val = float(cost_dict['anchoring'])
        total_length_val = float(total_length) 
        x_anchors=np.asarray([anchor['coords'][0][0] for item in moorings_anchoring for anchor in item['anchors']])
        y_anchors=np.asarray([anchor['coords'][0][1] for item in moorings_anchoring for anchor in item['anchors']])
        z_anchors=np.asarray([anchor['coords'][0][2] for item in moorings_anchoring for anchor in item['anchors']])
        max_radius=np.asarray([item['max_radius'] for item in moorings_anchoring ])
        
        tend=time.time()
        print('Moorings funct took: {:.0f}s'.format(tin-tend))
        
        return [moorings_cost_val, anchoring_cost_val], {'moorings_lengths': total_length_val,
                                                         'x_anchors': x_anchors,
                                                         'y_anchors': y_anchors,
                                                         'z_anchors': z_anchors,
                                                         'max_radius': max_radius,
                                                         }


    # function for calculating simplified Levelized Cost of Energy (sLCOE)
    def lcoe_func(aep, cabling_cost, moorings_cost, anchoring_cost, **kwargs):
        eco_eval.calculate_sLCOE(
            rated_rpm_array, 
            Drotor_vector, 
            power_rated_vector, 
            hub_height_vector,
            aep,  
            cabling_cost, 
            Pitching_moment,
            moorings_cost,
            anchoring_cost)
       
        return eco_eval.sLCOE
            


    #%% Cost model components                         

    # create an openmdao component for aep and lcoe to add to the problem
    aep_comp = CostModelComponent(input_keys=['x','y'],
                                  n_wt=n_wt,
                                  cost_function=aep_func,
                                  output_keys="aep",
                                  output_unit="kWh",
                                  objective=False,
                                  output_vals=np.zeros(n_wt)
                                  # cost_gradient_function=fd,
                                  # num_par_fd=5
                                  )

    # Cables
    cable_comp = WFNComponent(
                turbines_pos=site.initial_position,
                substations_pos=np.column_stack((site.initial_position.T[0].mean(), site.initial_position.T[1].mean())),
                cables=cables,
                router=router,
                border=site.boundary,
                obstacles=filtered_obstacles,
            )


    # moorings
    mooring_comp = CostModelComponent(input_keys=['x', 'y'],                                     
                                      n_wt=n_wt,
                                      cost_function=mooring_func,
                                      objective=False,
                                      output_keys=[
                                          ('moorings_cost', 0.0),         
                                          ('anchoring_cost', 0.0)         
                                          ],
                                      additional_output=[
                                          ('moorings_lengths', 0.0),
                                          ('x_anchors', np.zeros(n_wt*n_moorings)),
                                          ('y_anchors', np.zeros(n_wt*n_moorings)),
                                          ('z_anchors', np.zeros(n_wt*n_moorings)),
                                          ('max_radius', np.zeros(n_wt)),
                                        ])


    sLCOE_comp = CostModelComponent(input_keys=['aep', ('cabling_cost', 0.0), ('moorings_cost', 0.0), ('anchoring_cost', 0.0)],
                                  n_wt=n_wt,
                                  cost_function=lcoe_func,
                                  output_keys="sLCOE",
                                  output_unit="€/MWh",
                                  objective=True,
                                  maximize=False, 
                                  # cost_gradient_function=fd,
                                  # num_par_fd=30
                                  )

    #%% Constraints definition
    zones=[]
    for obs in filtered_obstacles:
        zones.append(ExclusionZone(obs))
    
    for obs in site.exclusion_zones:
        zones.append(ExclusionZone(obs))
        
    zones.append(InclusionZone(site.boundary))
    
    anchor_constr = XYBoundaryConstraint(zones, n_var=n_wt*n_moorings, boundary_type='multi_polygon', x_key='x_anchors', y_key='y_anchors')
    spacing_constr = SpacingConstraint(min_spacing * windTurbines.diameter())


    #%% Problem Assembly
    # create a group for the aep and LCOE components that links their common input/output (aep)

    lcoe_group = TopFarmGroup([aep_comp, cable_comp, mooring_comp, sLCOE_comp])


    #%% Topfarm problem and optimization

    problem = TopFarmProblem(
                design_vars={
                            **dict(zip('xy', site.initial_position.T)),
                            'xs': site.initial_position.T[0].mean(),
                            'ys': site.initial_position.T[1].mean()
                        },
                cost_comp=lcoe_group,
                constraints=[anchor_constr, spacing_constr],
                # driver=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=1000000, disp=True),
                driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition(max_step=6), max_iter=100000, max_time=100000, disp=False),
                plot_comp=XYPlotComp())
    
    fsql = "ODRA_optiwind.sql"
    output_dir = r'recorder_output'

    recorder2 = om.SqliteRecorder(os.path.join(output_dir, fsql))
    problem.driver.add_recorder(recorder2)                      # record optimization data (DVs, constraints, objective, etc...)
    problem.add_recorder(recorder2)                             # record ALL model data (opt data + model variables)

    #recorder options
    problem.driver.recording_options["record_constraints"] = True
    problem.driver.recording_options["record_desvars"] = True
    problem.driver.recording_options["record_objectives"] = True


    x = np.linspace(site.boundary.T[0].min(), site.boundary.T[0].max(), 20, endpoint=False)
    y = np.linspace(site.boundary.T[1].min(), site.boundary.T[1].max(), 20, endpoint=False)
    YY, XX = np.meshgrid(y, x)
    
    aep_ss = PyWakeAEPCostModelComponent(windFarmModel=wake_model, n_wt=n_wt, wd=np.linspace(0.0, 360.0, 12, endpoint=False), objective=False)
    problem.smart_start(XX, YY, aep_ss.get_aep4smart_start(), plot=0, seed=10)
    cost_ev, state_ev = problem.evaluate()

    cost, state, recorder = problem.optimize()


    








# %%
