import numpy as np
import json

# load parameters from config
with open('config_plant.json') as f:
    cfg = json.load(f)

A   = np.array(cfg['A'])
B   = np.array(cfg['B'])
constraints = cfg['constraints']
d   = np.array(cfg['d'])

def debug():
    return A

def evolve(x, u):
        
    x_next = A @ x + B @ (u + d)

    # soften the constraint
    #softener = 1

    #x_next[2] = np.clip(x_next[2], softener*constraints['x_min'][2], softener*constraints['x_max'][2])
    #x_next[3] = np.clip(x_next[3], softener*constraints['x_min'][3], softener*constraints['x_max'][3])


    return x_next