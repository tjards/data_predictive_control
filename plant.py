import numpy as np
import json

# load parameters from config
with open('config.json') as f:
    cfg = json.load(f)

A   = np.array(cfg['A'])
B   = np.array(cfg['B'])
constraints = cfg['constraints']

def evolve(x, u):
        
    x_next = A @ x + B @ u

    x_next[2] = np.clip(x_next[2], constraints['x_min'][2], constraints['x_max'][2])
    x_next[3] = np.clip(x_next[3], constraints['x_min'][3], constraints['x_max'][3])

    return x_next, u 