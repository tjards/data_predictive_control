import json
import numpy as np

class Disturbance:

    def __init__(self):

        # load parameters from config
        with open('configs/config_disturbance.json') as f:
            cfg = json.load(f)

        # get disturbance configs 
        self.dist_type = cfg["type"]
        if self.dist_type == 'linear':
            self.d   = np.array(cfg['d_lin'])
        else:
            # only supports linear for now, field later 
            self.d = None

    def evolve(self, t):

        if self.dist_type == 'linear':
            pass
        else:
            # placeholder for more complex disturbances
            self.d = None

        return self.d