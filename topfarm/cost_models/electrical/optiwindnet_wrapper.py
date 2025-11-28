from optiwindnet.api import WindFarmNetwork
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
import numpy as np

#cable length optimizer
class WFNComponent(CostModelComponent):
    def __init__(self, turbines_pos, substations_pos, cables, border, obstacles, router, objective=False, **kwargs):
        #self.wfn = wfn

        def compute(x, y, xs, ys):           
            try: 
                self.wfn = WindFarmNetwork(
                    turbinesC=np.column_stack((x,y)),
                    substationsC=np.column_stack((xs,ys)),
                    cables=cables,
                    router=router,
                    borderC=border,
                    obstacleC_=obstacles,
                    )                                                                                                     
                self.wfn.optimize()

            except Exception as e:
                print(f'{e} - negleting obstacles')
                self.wfn = WindFarmNetwork(
                    turbinesC=np.column_stack((x,y)),
                    substationsC=np.column_stack((xs,ys)),
                    cables=cables,
                    router=router,
                    )
                self.wfn.optimize()

            print(f'Cable cost: {self.wfn.cost()} m')
            return self.wfn.cost(), {
                'network_length': self.wfn.length(),
                'terse_links': self.wfn.terse_links(),
            }

        def compute_partials(x, y, xs, ys):
            grad_wt, grad_ss = self.wfn.gradient(
                turbinesC=np.column_stack((x, y)),
                substationsC=np.column_stack((xs, ys)),
            )
            dc_dx, dc_dy = grad_wt[:, 0], grad_wt[:, 1]
            dc_dxss, dc_dyss = grad_ss[:, 0], grad_ss[:, 1]
            return [dc_dx, dc_dy, dc_dxss, dc_dyss]


        x_init, y_init = turbines_pos.T
        x_ss_init, y_ss_init = substations_pos.T
        super().__init__(
            input_keys=[('x', x_init), ('y', y_init),
                        ('xs', x_ss_init), ('ys', y_ss_init)],
            n_wt=turbines_pos.shape[0],
            cost_function=compute,
            cost_gradient_function=compute_partials,
            objective=objective,
            maximize=False,
            output_keys=[('cabling_cost', 0.0)],
            additional_output=[
                ('network_length', 0.0),
                ('terse_links', np.zeros(turbines_pos.shape[0])),
            ],
            **kwargs,
        )
