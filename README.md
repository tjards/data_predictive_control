
# Convex Data-driven Predictive Control (DPC) 

A convex data-driven predictive control (DPC) framework for systems operating in biased conditions such as wind. System matrices are identified online from input-output data, disturbances are estimated from prediction residuals, and [cvxpy](https://www.cvxpy.org/) solves the resulting convex optimization at each step.

## Formulation

At each time step $k$, the following problem is solved over prediction horizon $h$:

$$\min_{U} \; J = \sum_{i=0}^{h-1} \Bigl[ x(k{+}i)^\top Q\, x(k{+}i) + u(k{+}i)^\top R\, u(k{+}i) \Bigr] + x(k{+}h)^\top P\, x(k{+}h)$$

subject to:

$$x(k{+}i{+}1) = \hat{A}\,x(k{+}i) + \hat{B}\bigl(u(k{+}i) + \hat{d}(k)\bigr), \quad i = 0,\ldots,h{-}1$$

$$x_{\min} \leq x(k{+}i) \leq x_{\max}, \quad i = 1,\ldots,h$$

$$u_{\min} \leq u(k{+}i) \leq u_{\max}, \quad i = 0,\ldots,h{-}1$$

where $Q \succeq 0$, $R \succ 0$, and $P \succeq 0$ are state, input, and terminal cost weights ($P$ is the solution to the [Discrete Algebraic Riccati Equation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html)). The three data-driven quantities are updated online at every step:

**Model** — $\hat{A},\hat{B}$ are identified from a sliding window of $N$ input-output pairs via least-squares (see [Data-driven Modelling](#data-driven-modelling)):

$$[\hat{A},\,\hat{B},\,\hat{c}] = \arg\min \sum_{j=1}^{N} \bigl\| x(j{+}1) - \hat{A}\,x(j) - \hat{B}\,u(j) - \hat{c} \bigr\|^2$$

**Disturbance** — $\hat{d}$ is inferred from the one-step prediction residual via the Moore-Penrose pseudoinverse $\hat{B}^\dagger$ (see [Bias Rejection](#bias-rejection)):

$$\hat{d}(k) = \hat{B}^{\dagger} \Bigl[ x(k) - \hat{A}\,x(k{-}1) - \hat{B}\,u(k{-}1) \Bigr]$$

**Stacked prediction** — the dynamics are lifted over the full horizon for the optimizer:

$$X_{\text{pred}} = \hat{\mathcal{A}}\,x(k) + \hat{\mathcal{B}}\,U + \hat{\mathcal{D}}\,\hat{d}$$

where $\hat{\mathcal{A}} \in \mathbb{R}^{h n_x \times n_x}$, $\hat{\mathcal{B}} \in \mathbb{R}^{h n_x \times h n_u}$, $\hat{\mathcal{D}} \in \mathbb{R}^{h n_x \times n_u}$, and $U = [u(k)^\top, \ldots, u(k{+}h{-}1)^\top]^\top$.

The MPC engages only once $(\hat{A},\hat{B})$ satisfies stability (spectral radius $< 1.1$) and full controllability ($\text{rank}[\hat{B},\,\hat{A}\hat{B},\,\ldots,\,\hat{A}^{n_x-1}\hat{B}] = n_x$).

## Data-driven Modelling

The plant is modelled as $x(k{+}1) = \hat{A}\,x(k) + \hat{B}\,u(k) + \hat{c}$, with $\hat{A}$ and $\hat{B}$ estimated online.

### Excitation

The plant is first excited with zero-mean random bounded inputs: each input drawn uniformly from $[u_{\min}, u_{\max}]$ is immediately followed by its negation (a *rockback* step). Trajectories are stored in a sliding window of length $N$.

### Residual (delta-x) Formulation

Only the *state increment* is regressed, rather than the full next state:

$$\Delta x(k) = x(k{+}1) - x(k) = \tilde{A}\,x(k) + \hat{B}\,u(k)$$

This improves conditioning since $\Delta x$ is typically small relative to $x$. The full matrix is recovered as $\hat{A} = I + \tilde{A}$.

### Internal Batch Normalization

Each batch is normalized by per-feature standard deviations from the current window before fitting:

$$x_n = \frac{x}{\sigma_x}, \quad u_n = \frac{u}{\sigma_u}, \quad \Delta x_n = \frac{\Delta x}{\sigma_{\Delta x}}$$

### Least Squares Fit

A stacked regressor with a bias column is formed:

$$Z_n = \begin{bmatrix} X_n \\ U_n \\ \mathbf{1}^\top \end{bmatrix} \in \mathbb{R}^{(n_x + n_u + 1) \times N}$$

$\Phi_n$ is found by solving $\Delta X_n \approx \Phi_n Z_n$ via [numpy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), then un-normalized:

$$\tilde{A} = \text{diag}(\sigma_{\Delta x})\;\Phi_n^{(x)}\;\text{diag}(\sigma_x^{-1}), \qquad \hat{B}_{\text{new}} = \text{diag}(\sigma_{\Delta x})\;\Phi_n^{(u)}\;\text{diag}(\sigma_u^{-1})$$

### Exponential Smoothing Update

New estimates are blended into the running model with learning rate $\alpha \in (0,1]$:

$$\hat{A} \leftarrow (1-\alpha)\,\hat{A} + \alpha\,(I + \tilde{A}), \qquad \hat{B} \leftarrow (1-\alpha)\,\hat{B} + \alpha\,\hat{B}_{\text{new}}$$

### Results

The animation shows the excitation phase followed by the controlled trajectory once the model passes the viability checks stated in [Formulation](#formulation).

![Modeller excitation and convergence](docs/modeller/trajectory.gif)

| Control Inputs | Velocity States |
|:---:|:---:|
| ![Control inputs within bounds](docs/modeller/inputs.png) | ![Velocity states within bounds](docs/modeller/velocities.png) |


## Bias Rejection

Wind is treated as an unknown slowly-varying disturbance $d$ entering through the input channel:

$$x(k+1) = \hat{A}\,x(k) + \hat{B}\bigl(u(k) + d\bigr)$$

At each step, $\hat{d}$ is estimated from the one-step prediction residual via $\hat{B}^\dagger$ (as shown in [Formulation](#formulation)) and fed forward into the stacked prediction used by the optimizer.

### Results

| Without Disturbance Rejection | With Disturbance Rejection |
|:---:|:---:|
| ![Trajectory without disturbance rejection](docs/windy/trajectory_withoutdist.gif) | ![Trajectory with disturbance rejection](docs/windy/trajectory_withdist.gif) |

Without rejection the trajectory drifts from the origin; with rejection the estimated disturbance is cancelled in the predictions, restoring convergence.

| Control Inputs | Velocity States |
|:---:|:---:|
| ![Control inputs within bounds](docs/windy/inputs_withoutdist.png) | ![Velocity states within bounds](docs/windy/velocities_withoutdist.png) |



## Use

Install dependencies and run:

```bash
pip install -r requirements.txt
python main.py
```

Parameters are configured in `configs/`.

## References

- Parts of this project were developed with the assistance of Claude Sonnet 4.6
- Solving the optimization: [cvxpy](https://www.cvxpy.org/)
- Terminal cost via Discrete Algebraic Riccati Equation: [scipy.linalg.solve_discrete_are](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html)
- Disturbance estimation via Moore-Penrose pseudoinverse: [numpy.linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html) 
