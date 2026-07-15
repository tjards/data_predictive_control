
'''
Generate a nonlinear input-channel disturbance field

        x[k+1] = A x[k] + B (u[k] + d(p[k], t))

        where A,B,x,u are the plant and
        d(p,t) is a nonlinear 2D field composed of:
        
        1. slowly rotating background bias (wind), and
        2. several vortex-like spatial disturbances (defined by centers)


        derivation in /docs/disturbance/nonlinear.md


'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

class VortexConfig():

    def __init__(self):

        # these are the centers of the vortices (hard code now, make random later)
        self.vortex_centers = np.array([
            [-2.5, 2.0],
            [2.0, 2.0],
            [2.5, -2.0],
        ]) 
        
        # corresponding size(s) and amplitudes(s)
        self.vortex_sigmas = np.array([1.5, 
                                       1.8, 
                                       1.4])
        self.vortex_amps = np.array([0.65, 
                                     1.00,
                                     -0.75])
        self.background_amp = 0.35
        self.omega = 0.35
        self.moving_centers = True
        self.center_motion_amp = 0.35
        self.center_motion_rate = 0.2

class VortexField:

    def __init__(self):

        self.config = VortexConfig()
        print('vortex initialized')

    def evolve_centers(self, t):

        # pull out initial vortex centers
        centers = np.asarray(self.config.vortex_centers).copy()

        if not self.config.moving_centers:
            return centers
        
        # else, iterate through each center
        for i in range(len(centers)):

            phase = i * np.pi / 2.0
            centers[i, 0] += self.config.center_motion_amp * np.sin(self.config.center_motion_rate * t + phase)
            centers[i, 1] += self.config.center_motion_amp * np.cos(self.config.center_motion_rate * t + phase)

        return centers 
    
    def evolve_background(self, t):

        b = self.config.background_amp * np.array([
            np.cos(self.config.omega * t),
            np.sin(self.config.omega * t)
            ])
        
        return b

    def compute_gain(self, c, t):

        return 1.0 + 0.30 * np.sin(0.5 * self.config.omega * t + np.linalg.norm(c))

    
    def compute_disturbance(self, x, t):

        # strip out x/y position
        p = np.asarray(x)[:2]

        # background wind
        b = self.evolve_background(t)
        
        # get the updated centers
        centers = self.evolve_centers(t)

        # initialize total disturbance
        d_actual = b.copy()

        for c, sigma, amp in zip(centers, self.config.vortex_sigmas, self.config.vortex_amps):

            diff = p - c
            spacial_influence = np.exp(-np.dot(diff, diff) / (sigma**2))
            vortex_direction = np.array([-diff[1], diff[0]])
            gain = self.compute_gain(c, t)

            d_actual += amp * gain * vortex_direction * spacial_influence

        return d_actual
    
    def grid_for_plots(self, xlim, ylim, n, t):

        xs = np.linspace(xlim[0], xlim[1], n)
        ys = np.linspace(ylim[0], ylim[1], n)

        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for row in range(n):
            for col in range(n):
                d = self.compute_disturbance(np.array([X[row, col], Y[row, col]]), t)
                U[row, col] = d[0]
                V[row, col] = d[1]

        speed = np.sqrt(U**2 + V**2)
        return X, Y, U, V, speed        


    def plot_field_at_t(self, plot_field_path, xlim, ylim, n, t):

        X, Y, U, V, speed = self. grid_for_plots(xlim=xlim, ylim=ylim, n=n, t=t)
        centers = self.evolve_centers(t)

        fig, ax = plt.subplots(figsize=(8,7))
        contour = ax.contourf(X,Y, speed, levels = 30, alpha = 0.75)
        fig.colorbar(contour, ax=ax, label = 'Disturbance Intensity')
        ax.streamplot(X, Y, U, V, density = 1.2, linewidth = 1.0, arrowsize = 1.1)

        # quiver may look nice later (optional)
        #skip = 3
        #ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip], angles="xy", scale_units="xy", scale=1.5, width=0.003)

        #ax.scatter(centers[:, 0], centers[:, 1], marker="x", facecolors = 'red', s=90, label="Vortices Centers")
        #ax.scatter(feature_map.centers[:, 0], feature_map.centers[:, 1], marker="o", s=45, facecolors="none", label="CALA will go here)

        ax.set_title(f"Disturbance field at t = {t:.2f}s")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        #ax.legend(loc="upper right", framealpha=0.9)
        plt.tight_layout()

        fig.savefig(plot_field_path, dpi=180)
        plt.close(fig)


# testing 
field = VortexField()
field.plot_field_at_t('visualization/plots/field.png', xlim=(-5.0, 5.0), ylim=(-5.0, 5.0), n = 35, t =0.0)

















