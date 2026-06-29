'''
Produced by Claude Sonnet 4.6

Minor refinements made by author: tjards

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def _cost_grid(X, Y, V):
    """
    Evaluate J(x) = x^T V x over a 2D (x1, x2) grid.
    States x3, x4, ... are held at zero (position-plane slice).
    """
    pts = np.stack([X, Y], axis=-1)   # (..., 2)
    V2  = V[:2, :2]                   # upper-left 2x2 block
    return np.einsum('...i,ij,...j->...', pts, V2, pts)


def animate_trajectory(state_history, predicted_sequences, V,
                       x_target=None, filename='trajectory.gif'):
    
    """
    Produce a GIF of the MPC closed-loop trajectory overlaid on a cost
    function contour map.

    Parameters
    ----------
    state_history : list[ndarray (nx,)]
        Actual state at each simulation step (length T+1).
    predicted_sequences : list[ndarray (h, nx)]
        MPC-predicted state sequences at each step (length T).
    V : ndarray (nx, nx)
        Positive-definite matrix for cost contours (e.g. DARE solution).
    x_target : ndarray (nx,), optional
        Target state; defaults to the zero vector.
    filename : str
        Output filename for the animated GIF.
    """
    states = np.array(state_history)        # (T+1, nx)
    T      = len(predicted_sequences)       # number of MPC steps
    nx     = V.shape[0]

    if x_target is None:
        x_target = np.zeros(nx)
    x_target = np.asarray(x_target, dtype=float)

    # ------------------------------------------------------------------
    # Grid bounds: cover all actual + predicted states with margin
    # ------------------------------------------------------------------
    all_pts = np.vstack([states[:, :2]] + [p[:, :2] for p in predicted_sequences])
    margin  = 0.5

    x1_min = min(all_pts[:, 0].min(), x_target[0]) - margin
    x1_max = max(all_pts[:, 0].max(), x_target[0]) + margin
    x2_min = min(all_pts[:, 1].min(), x_target[1]) - margin
    x2_max = max(all_pts[:, 1].max(), x_target[1]) + margin

    # Square up the axes so the contours look right
    r1, r2 = x1_max - x1_min, x2_max - x2_min
    if r1 > r2:
        pad = (r1 - r2) / 2
        x2_min -= pad; x2_max += pad
    else:
        pad = (r2 - r1) / 2
        x1_min -= pad; x1_max += pad

    # ------------------------------------------------------------------
    # Cost contour grid
    # ------------------------------------------------------------------
    N_grid = 300
    xx = np.linspace(x1_min, x1_max, N_grid)
    yy = np.linspace(x2_min, x2_max, N_grid)
    X, Y = np.meshgrid(xx, yy)
    Z    = _cost_grid(X, Y, V)

    # Logarithmic levels give dark centre, light exterior
    z_lo   = max(float(Z.min()), 1e-8)
    z_hi   = float(Z.max())
    levels = np.logspace(np.log10(z_lo), np.log10(z_hi), 30)

    # ------------------------------------------------------------------
    # Static figure elements
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.contourf(X, Y, Z, levels=levels, cmap='gray_r')
    ax.contour(X, Y, Z, levels=levels, colors='dimgray',
               linewidths=0.4, linestyles='--', alpha=0.5)

    ax.plot(x_target[0], x_target[1], 'g+', markersize=14, markeredgewidth=2,
            label='target', zorder=6)
    ax.plot(states[0, 0], states[0, 1], 'ro', markersize=8,
            label='start', zorder=6)

    # Dynamic elements
    trace_line, = ax.plot([], [], color='royalblue', lw=2.0,
                          label='trajectory', zorder=5)
    cur_dot,    = ax.plot([], [], 'o', color='royalblue', markersize=8, zorder=7)
    pred_line,  = ax.plot([], [], color='royalblue', lw=1.0, alpha=0.75, ls=':',
                          label='predicted', zorder=4)

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel('$x_1$  (position)', fontsize=12)
    ax.set_ylabel('$x_2$  (position)', fontsize=12)
    ax.set_title('MPC Closed-Loop Trajectory', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_aspect('equal')
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Animation callbacks
    # ------------------------------------------------------------------
    def init():
        trace_line.set_data([], [])
        cur_dot.set_data([], [])
        pred_line.set_data([], [])
        return trace_line, cur_dot, pred_line

    def update(frame):
        # Actual trajectory up to this frame
        trace_line.set_data(states[:frame + 1, 0], states[:frame + 1, 1])
        cur_dot.set_data([states[frame, 0]], [states[frame, 1]])

        # Predicted horizon from current state (if still within sim window)
        if frame < T:
            pred = predicted_sequences[frame]             # (h, nx)
            px   = np.concatenate([[states[frame, 0]], pred[:, 0]])
            py   = np.concatenate([[states[frame, 1]], pred[:, 1]])
            pred_line.set_data(px, py)
        else:
            pred_line.set_data([], [])

        return trace_line, cur_dot, pred_line

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(states),
        init_func=init,
        blit=True,
        interval=150,
    )

    ani.save(filename, writer=animation.PillowWriter(fps=8))
    print(f"Animation saved to '{filename}'")
    plt.close(fig)


def plot_inputs(input_history, constraints, filename='inputs.png'):
    """
    Static plot of applied control inputs over time vs. constraint bounds.

    Parameters
    ----------
    input_history : list[ndarray (nu,)]
        Applied input at each step (length T).
    constraints : dict
        The constraints dict from config (box or lmi supported for bounds).
    filename : str
        Output filename for the saved figure.
    """
    inputs = np.array(input_history)   # (T, nu)
    T, nu  = inputs.shape
    steps  = np.arange(T)

    # Extract scalar bounds per input channel (box constraints only)
    u_min = u_max = None
    if constraints.get('type') == 'box':
        u_min = np.array(constraints['u_min'], dtype=float)
        u_max = np.array(constraints['u_max'], dtype=float)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(nu, 1, figsize=(9, 3 * nu), sharex=True)
    if nu == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        color = colors[i % len(colors)]
        ax.plot(steps, inputs[:, i], color=color, lw=1.8, label=f'$u_{i+1}$')

        if u_min is not None and u_max is not None:
            ax.axhline(u_min[i], color=color, lw=1.2, ls='--', alpha=0.7,
                       label=f'$u_{{{i+1},min}}$ = {u_min[i]:.2g}')
            ax.axhline(u_max[i], color=color, lw=1.2, ls=':',  alpha=0.7,
                       label=f'$u_{{{i+1},max}}$ = {u_max[i]:.2g}')
            # Shade the feasible band
            ax.axhspan(u_min[i], u_max[i], color=color, alpha=0.06)

        ax.set_ylabel(f'$u_{i+1}$', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Time step $k$', fontsize=12)
    fig.suptitle('Control Inputs vs. Constraints', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Input plot saved to '{filename}'")
    plt.close(fig)


def plot_velocities(state_history, constraints, vel_indices=None, filename='velocities.png'):
    """
    Static plot of velocity states over time vs. state constraint bounds.

    Parameters
    ----------
    state_history : list[ndarray (nx,)]
        Actual state at each simulation step (length T+1).
    constraints : dict
        The constraints dict from config.
    vel_indices : list[int], optional
        Indices of velocity states in the state vector. Defaults to the
        second half (e.g. [2, 3] for nx=4).
    filename : str
        Output filename for the saved figure.
    """
    states = np.array(state_history)   # (T+1, nx)
    T, nx  = states.shape
    steps  = np.arange(T)

    if vel_indices is None:
        vel_indices = list(range(nx // 2, nx))

    # Extract bounds for velocity indices (box constraints only)
    x_min = x_max = None
    if constraints.get('type') == 'box':
        x_min = np.array(constraints['x_min'], dtype=float)
        x_max = np.array(constraints['x_max'], dtype=float)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    nv = len(vel_indices)

    fig, axes = plt.subplots(nv, 1, figsize=(9, 3 * nv), sharex=True)
    if nv == 1:
        axes = [axes]

    for plot_i, (ax, state_i) in enumerate(zip(axes, vel_indices)):
        color = colors[plot_i % len(colors)]
        label = f'$x_{{{state_i + 1}}}$'
        ax.plot(steps, states[:, state_i], color=color, lw=1.8, label=label)

        if x_min is not None and x_max is not None:
            ax.axhline(x_min[state_i], color=color, lw=1.2, ls='--', alpha=0.7,
                       label=f'$x_{{{state_i + 1},min}}$ = {x_min[state_i]:.2g}')
            ax.axhline(x_max[state_i], color=color, lw=1.2, ls=':',  alpha=0.7,
                       label=f'$x_{{{state_i + 1},max}}$ = {x_max[state_i]:.2g}')
            ax.axhspan(x_min[state_i], x_max[state_i], color=color, alpha=0.06)

        ax.set_ylabel(label, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

    axes[-1].set_xlabel('Time step $k$', fontsize=12)
    fig.suptitle('Velocity States vs. Constraints', fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Velocity plot saved to '{filename}'")
    plt.close(fig)
