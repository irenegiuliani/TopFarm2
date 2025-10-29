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
    import random
    import matplotlib.pyplot as plt
    plt.style.use('default')

    def generate_random_points_in_polygon(inclusion_polygon: Polygon, 
                                          exclusion_areas: MultiPolygon, 
                                          n: int, 
                                          min_distance: float,
                                          max_attempts: int = 100000):
        """
        Random points generator
        
        """
        points = []
        attempts = 0
    
        minx, miny, maxx, maxy = inclusion_polygon.bounds
    
        while len(points) < n and attempts < max_attempts:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            p = Point(x, y)
    
            if inclusion_polygon.contains(p) and not exclusion_areas.contains(p):
                # Check min distance
                if all(p.distance(other) >= min_distance for other in points):
                    points.append(p)
    
            attempts += 1
    
        if len(points) < n:
            raise RuntimeError(f"Max number of attempts reached. "
                               f"{len(points)} points generated.")
    
        coords = np.array([[pt.x, pt.y] for pt in points], dtype=float)
        return coords
    
    
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
    site_data = load_site_data(file_paths, epsg, name, idx=idx, exclusion_flag=True)   
    site = Odra_Site(site_data)  
    
    windTurbines = T18MW260()
    n_wt = 39   
    
    layout=generate_random_points_in_polygon(Polygon(site.boundary), 
                                                          site.seabeds[site.seabeds['seabed'] == 'forbidden'].geometry[0], 
                                                          n_wt, 
                                                          4*windTurbines.diameter(),
                                                          10000)
    #%%
    site                      = site
    beta                      = 60
    windTurbines              = T18MW260()
    n_wt                      = len(layout)
    smart_start               = False
    cables                    = np.array([(1, 358+260), (2, 358+386), (5, 358+650)])
    min_spacing               = 4
    project_duration          = 30
    discount_rate             = 0.07
    var_cable_cost            = 900e03
    n_moorings                = 3
    mooring_type              = 'taught'
    anchoring_cost_ud         = { 'sand': 200000,
                                    'rock': 200000,
                                    'forbidden': 1000000}
    moorings_meter_cost_ud    = 500
    router                    = EWRouter()
    fsql_recorder             = "ODRA_optiwind.sql"
    output_dir_csv            = r'csv_output'
    output_dir_recorder       = r'recorder_output'
    opt_driver                = EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=400, disp=True, auto_scale=True)  
    # opt_driver                = EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition(max_step=None), max_iter=1, max_time=None, disp=False)
    wake_model                = BGD(site, windTurbines, k=0.04, use_effective_ws=True, 
                                      rotorAvgModel = GaussianOverlapAvgModel(), 
                                      superpositionModel = LinearSum(),  
                                      turbulenceModel = GCLTurbulence())
    
    
    site.change_initial_position(layout)
    # vectors for turbine properties: diameter, rated power and hub height. these are inputs to the cost model
    Drotor_vector = [windTurbines.diameter()] * n_wt                    #[m]
    Pitching_moment = [windTurbines.pit_moment] * n_wt                  #
    power_rated_vector = [float(windTurbines.power(20))*1e-6] * n_wt    #[W]
    hub_height_vector = [windTurbines.hub_height()] * n_wt              #[m]
    
    # add additional cost model inputs for shore distance, energy price, project lifetime, rated rotor speed and water depth
    distance_from_shore = site.ex_distance              #[km]
    
    rated_rpm_array = [windTurbines.rated_rpm] * n_wt   # [rpm]
    
    #specify the cable cost
    
    # set up function for new cost model with initial inputs as set above
    eco_eval = ee_2(distance_from_shore, project_duration, discount_rate, var_cable_cost)
    
    #mooring function setup
    
    resolution_factor=site.resolution_factor
    max_d=windTurbines.diameter()*2
    
    mooring_footprint_opt = []
    
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
           
    
    
    #AEP calculator                
    def aep_func(x, y, **kwargs):
        sim_stat = wake_model(x, y, n_cpu=1)    
        res = sim_stat.aep().sum(['wd','ws']).values*10**6  
        res_nowake = sim_stat.aep(with_wake_loss=False).sum(['wd','ws']).values*10**6 
        loss = 100 * (res_nowake - res) / res_nowake
        print(f'{sum(res)} AEP calculated with {sum(loss)/39} % losses')                                
        return [res], {'aep_loss': loss}
    
    
    #cable length optimizer
    class WFNComponent(CostModelComponent):
        def __init__(self, turbines_pos, substations_pos, cables, border, obstacles, router, **kwargs):
            self.wfn = WindFarmNetwork(
                turbinesC=turbines_pos,
                substationsC=substations_pos,
                cables=cables,
                router=router,
                borderC=border,
                obstacleC_=obstacles,
                )
    
            def compute(x, y, xs, ys):
                tin=time.time()
                try: 
                    self.wfn.merge_obstacles_into_border()
                    
                                                                        
                                                                             
                                                         
                                       
                    
                    self.wfn.optimize(turbinesC=np.column_stack((x, y)),
                                      substationsC=np.column_stack((xs, ys)),
                                      router = HGSRouter(time_limit=5)
                                      )
        
                except Exception as e:
                    print(f'{e} - negleting obstacbels')
                    self.wfn = WindFarmNetwork(
                        turbinesC=turbines_pos,
                        substationsC=substations_pos,
                        cables=cables,
                        router=router,
                        )
                    self.wfn.merge_obstacles_into_border()
                    self.wfn.optimize(turbinesC=np.column_stack((x, y)),
                                      substationsC=np.column_stack((xs, ys)),
                                                                             
                                                         
                                       
                   
                                                                        
                                      router = HGSRouter(time_limit=5)
                                      )
    
                tend=time.time()
                
                print('Cables optimization took: {:.0f}s'.format(tin-tend))
                print(f'Cable length: {self.wfn.length()} m')
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
        moorings_anchoring = moorings_footprint(x, y, site, beta, resolution_factor, max_d, mooring_type, n_moorings, False)  
        mooring_footprint_opt.append(moorings_anchoring)
        total_length = 0.0
        for i in range(len(moorings_anchoring)):
            for j in range(len(moorings_anchoring[i]['anchors'])):
                total_length += moorings_anchoring[i]['anchors'][j]['length']                
        cost_dict = moorings_cost(moorings_anchoring, anchoring_cost_ud, moorings_meter_cost_ud)
        moorings_cost_val = np.array(cost_dict['moorings'])
        anchoring_cost_val = np.array(cost_dict['anchoring'])
        total_length_val = float(total_length) 
        x_anchors=np.asarray([anchor['coords'][0][0] for item in moorings_anchoring for anchor in item['anchors']])
        y_anchors=np.asarray([anchor['coords'][0][1] for item in moorings_anchoring for anchor in item['anchors']])
        z_anchors=np.asarray([anchor['coords'][0][2] for item in moorings_anchoring for anchor in item['anchors']])
        max_radius=np.asarray([item['max_radius'] for item in moorings_anchoring ])
        
        tend=time.time()
        print('Moorings funct took: {:.0f}s'.format(tin-tend))
        print(f'Moorings cost: {moorings_cost_val} €')
        print(f'Anchors cost: {anchoring_cost_val} €')                                                                                              
        return [moorings_cost_val, anchoring_cost_val], {'moorings_lengths': total_length_val,
                                                         'x_anchors': x_anchors,
                                                         'y_anchors': y_anchors,
                                                         'z_anchors': z_anchors,
                                                         'max_radius': max_radius,
                                                         }
    
    
    # function for calculating simplified Levelized Cost of Energy (sLCOE)
    def lcoe_func(aep, cabling_cost, moorings_cost, anchoring_cost, network_length, **kwargs):
        eco_eval.calculate_sLCOE(
            rated_rpm_array, 
            Drotor_vector, 
            power_rated_vector, 
            hub_height_vector,
            aep,  
            cabling_cost, 
            Pitching_moment,
            moorings_cost,
            anchoring_cost,
            network_length)
        print(f"sLCOE calculated: {eco_eval.sLCOE} €/MWh")                                                                                             
        return eco_eval.sLCOE, {'array_of_cables' : eco_eval.project_costs['BOP']["array_of_cables"],
                                  'cables_export' : eco_eval.project_costs['BOP']['cables_export'],
                                  'substation': eco_eval.project_costs['BOP']['substation'],
                                  'CAPEX': eco_eval.project_costs_sums['CAPEX'],
                                  'OPEX': eco_eval.project_costs_sums['OPEX'],
                                  'BOP': eco_eval.project_costs_sums['BOP'],
                                  'DEVEX': eco_eval.project_costs_sums['DEVEX'],
                                  'ABEX' : eco_eval.project_costs_sums['ABEX']
                                  }
                                 
    
    # create an openmdao component for aep and lcoe to add to the problem
    aep_comp = CostModelComponent(input_keys=['x','y'],
                                  n_wt=n_wt,
                                  cost_function=aep_func,
                                  output_keys="aep",
                                  output_unit="kWh",
                                  objective=False,
                                  output_vals=np.zeros(n_wt),
                                  additional_output=[
                                        ('aep_loss', np.zeros(n_wt)),
                                ])
    
    
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
                                          ('moorings_cost', np.zeros(n_wt)),         
                                          ('anchoring_cost', np.zeros(n_wt))         
                                          ],
                                      additional_output=[
                                          ('moorings_lengths', 0.0),
                                          ('x_anchors', np.zeros(n_wt*n_moorings)),
                                          ('y_anchors', np.zeros(n_wt*n_moorings)),
                                          ('z_anchors', np.zeros(n_wt*n_moorings)),
                                          ('max_radius', np.zeros(n_wt)),
                                        ])
    
    
    sLCOE_comp = CostModelComponent(input_keys=['aep', ('cabling_cost', 0.0), ('moorings_cost', np.zeros(n_wt)), ('anchoring_cost', np.zeros(n_wt)), ('network_length', 0.0)],
                                  n_wt=n_wt,
                                  cost_function=lcoe_func,
                                  output_keys="sLCOE",
                                  output_unit="€/MWh",
                                  objective=True,
                                  maximize=False, 
                                  additional_output = [('array_of_cables' , 0.0),
                                                       ('cables_export', 0.0),
                                                       ('substation', 0.0),
                                                       ('CAPEX', 0.0),
                                                       ('OPEX', 0.0),
                                                       ('BOP', 0.0),
                                                       ('DEVEX', 0.0),
                                                       ('ABEX' , 0.0)]
                                  )
    
    
    zones=[]
    for obs in filtered_obstacles:
        zones.append(ExclusionZone(obs))
    
    for obs in site.exclusion_zones:
        zones.append(ExclusionZone(obs))
        
    zones.append(InclusionZone(site.boundary))
    
    anchor_constr = XYBoundaryConstraint(zones, boundary_type='multi_polygon')
    spacing_constr = SpacingConstraint(min_spacing * windTurbines.diameter())
    
    
    lcoe_group = TopFarmGroup([aep_comp, cable_comp, mooring_comp, sLCOE_comp])
    
    
    problem = TopFarmProblem(
                design_vars={
                            **dict(zip('xy', site.initial_position.T)),
                            # 'xs': site.initial_position.T[0].mean(),
                            # 'ys': site.initial_position.T[1].mean()
                        },
                cost_comp=lcoe_group,
                constraints=[anchor_constr, spacing_constr],
                driver=opt_driver,
                plot_comp=XYPlotComp()) 
    
    
    # recorder2 = om.SqliteRecorder(os.path.join(output_dir_recorder, fsql_recorder))
    # problem.driver.add_recorder(recorder2)                      # record optimization data (DVs, constraints, objective, etc...)
    # problem.add_recorder(recorder2)                             # record ALL model data (opt data + model variables)
    
    # #recorder options
    # problem.driver.recording_options["record_constraints"] = True
    # problem.driver.recording_options["record_desvars"] = True
    # problem.driver.recording_options["record_objectives"] = True
    # problem.driver.recording_options['record_inputs'] = True
    # problem.driver.recording_options['record_outputs'] = True
    
    # cost, state = problem.evaluate()
     
    cost, state, recorder = problem.optimize(recorder_as_list=True)
    # cable_comp.wfn.plot(legend=True, infobox=False, landscape=True, )
    
    #'array initializaion and output saving'
    WTS_save =              []
    site_save =             []
    wakeMod_save =          []
    AEP_save =              []
    CF_save =               []
    Wakel_save =            []
    LCOE_save =             []
    LCOE_save =             []
    el_con_save =           []
    IA_costs_save =         []
    IA_length_save =        []
    EX_costs_save =         []
    IA_weight_save =        []
    CAPEX_save =            []
    Turbine_total_save =    []
    Turbine_comp_save =     []
    Turbine_weight_save =   []
    Platform_total_save =   []
    Platform_comp_save =    []
    Moorings_save =         []
    Anchoring_save =        []
    Substation_save =       []
    Platform_weight_save =  []
    BOP_save =              []
    OPEX_save =             []
    DEVEX_save =            []
    ABEX_save =             []
    n_it_save =             []

    AEP = aep_func(state['x'], state['y'])[0][0] #[kW] WF annual energy production
    AEP_gross = wake_model(state['x'], state['y']).aep(with_wake_loss=False).sum().data #[kW] WF annual energy production
    Wake_eff = (AEP_gross - (sum(AEP) / 1e06)) / AEP_gross *100
    
    CF = sum(AEP) / (n_wt*power_rated_vector[0] * 1e03 *8760) #[-] WF capacity factor
    
    # convergence history
    # conv_h = recorder['sLCOE']
    # conv_h_df = pd.DataFrame(conv_h)
    state_df = pd.DataFrame({
        "x": state.get("x"),
        "y": state.get("y")
    })
    
    OSS_df = pd.DataFrame({
        "x": state.get("xs"),
        "y": state.get("ys")
    })
    
    if not os.path.exists(os.path.join(os.getcwd(), output_dir_csv)):
        os.makedirs(os.path.join(os.getcwd(), output_dir_csv))
        
    # conv_h_df.to_csv(os.getcwd() + output_dir_csv + str(windTurbines.name()) + '_' + site.name + '_' + '_conv_h.csv')
    state_df.to_csv(os.path.join(os.getcwd(), output_dir_csv, f"{windTurbines.name()}_{site.name}_layout.csv"), float_format="%.8e", index=False)
    OSS_df.to_csv(os.path.join(os.getcwd(), output_dir_csv, f"{windTurbines.name()}_{site.name}_OSS.csv"), float_format="%.8e", index=False)
        
    WTS_save.append(windTurbines.name())
    site_save.append(site.name)
                           
    AEP_save.append(sum(AEP)/1e06)
    CF_save.append(CF)
    Wakel_save.append(Wake_eff)
                                          
    LCOE_save.append(cost)
    el_connection_cost=eco_eval.project_costs['BOP']["array_of_cables"]#IA_length['elnet_length'] * cable_cost_per_meter
    IA_costs_save.append(el_connection_cost)
                                                                  
    EX_costs_save.append(eco_eval.project_costs['BOP']['cables_export'])#+eco_eval.project_costs['BOP']['cables_export_installation'])
    IA_weight_save.append(el_connection_cost/eco_eval.project_costs_sums['CAPEX']*100)
    CAPEX_save.append(eco_eval.project_costs_sums['CAPEX'])
    Turbine_total_save.append(eco_eval.turbine_general_costs_sums["TOTAL"][0])
    Turbine_comp_save.append(eco_eval.turbine_general_costs_sums["bill_of_material"][0])
    Turbine_weight_save.append(eco_eval.turbine_general_costs_sums["TOTAL"].sum()/eco_eval.project_costs_sums['CAPEX']*100)
    Platform_total_save.append(eco_eval.foundation_general_costs_sums["TOTAL"][0])
    Platform_comp_save.append(eco_eval.foundation_general_costs_sums["bill_of_material"][0])
    Moorings_save.append(sum(eco_eval.foundation_general_costs["moorings_anchoring"]["moorings"]))
    Anchoring_save.append(sum(eco_eval.foundation_general_costs["moorings_anchoring"]["anchoring"]))
    Substation_save.append(eco_eval.project_costs['BOP']['substation'])
    Platform_weight_save.append(eco_eval.foundation_general_costs_sums["TOTAL"].sum()/eco_eval.project_costs_sums['CAPEX']*100)
    BOP_save.append(eco_eval.project_costs_sums['BOP'])
    OPEX_save.append(eco_eval.project_costs_sums['OPEX'])
    DEVEX_save.append(eco_eval.project_costs_sums['DEVEX'])
    ABEX_save.append(eco_eval.project_costs_sums['ABEX'])
    # n_it_save.append(len(conv_h))
    
    
    outputs=pd.DataFrame()
    
    outputs['Site [-]']                 = [item for item in site_save]
    outputs['WindTurbine [-]']          = [item for item in WTS_save]
                                                                         
    outputs['AEP [GWh]']				= [item for item in AEP_save]
    outputs['CF [-]']					= [item for item in CF_save]
    outputs['Wake losses [%]']			= [item for item in Wakel_save]
    outputs['LCOE [€/MWh]']				= [item for item in LCOE_save] 
    outputs['cable_IA_costs [€]']		= [item for item in IA_costs_save]
                                                                       
    outputs['cable_EX_costs [€]']		= [item for item in EX_costs_save]
    outputs['cable_IA_weight [%]']		= [item for item in IA_weight_save]
    outputs['CAPEX [€]']				= [item for item in CAPEX_save]
    outputs['Turbine_total [€]']		= [item for item in Turbine_total_save]
    outputs['Turbine_comp [€]']			= [item for item in Turbine_comp_save]
    outputs['Turbine_weight [%]']		= [item for item in Turbine_weight_save]
    outputs['Platform_total [€]']		= [item for item in Platform_total_save]
    outputs['Platform_comp [€]']		= [item for item in Platform_comp_save]
    outputs['Moorings [€]']				= [item for item in Moorings_save]
    outputs['Anchoring [€]']			= [item for item in Anchoring_save]
    outputs['Substation [€]']			= [item for item in Substation_save]
    outputs['Platform_weight [%]']		= [item for item in Platform_weight_save]
    outputs['BOP [€]']					= [item for item in BOP_save]
    outputs['OPEX [€]']					= [item for item in OPEX_save]
    outputs['DEVEX [€]']				= [item for item in DEVEX_save]
    outputs['ABEX [€]']					= [item for item in ABEX_save]
                                                                      
    
    outputs.to_csv(os.path.join(os.getcwd(), output_dir_csv, f"{windTurbines.name()}_{site.name}_outputs.csv"))
                                
                       
    
    print('--- LCOE= %s' % cost)
