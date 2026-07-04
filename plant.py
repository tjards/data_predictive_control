import numpy as np
import json

class Plant:

    def __init__(self):

        # load parameters from config
        with open('configs/config_plant.json') as f:
            cfg = json.load(f)

        A   = np.array(cfg['A'])
        B   = np.array(cfg['B'])
        constraints = cfg['constraints']
        d   = np.array(cfg['d'])
        x0  = np.array(cfg['x0'], dtype=float)

        self.A = A
        self.B = B
        self.constraints = constraints
        self.d = d
        self.x0 = x0

    def evolve(self, x, u, disturb = True):
        
        if not disturb:
            x_next = self.A @ x + self.B @ (u)
        else:
            x_next = self.A @ x + self.B @ (u + self.d)
            
        # soften the constraint
        #softener = 1
        #x_next[2] = np.clip(x_next[2], softener*constraints['x_min'][2], softener*constraints['x_max'][2])
        #x_next[3] = np.clip(x_next[3], softener*constraints['x_min'][3], softener*constraints['x_max'][3])

        return x_next