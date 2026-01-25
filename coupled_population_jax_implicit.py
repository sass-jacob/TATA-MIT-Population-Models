"""
Coupled population model - Fully Implicit Implementation.
Backward Euler time stepping + Newton-Raphson solver.

Physics:
- Two populations: f1 (untransformed), f2 (transforming)
- Advection velocity R(c) from linearized Butler-Volmer
- Concentration-dependent transition rate K12(c)
- Constraint: Total current I = Q * integral(R1*f1 + R2*f2)
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import matplotlib.pyplot as plt
import time

jax.config.update("jax_enable_x64", True)

#####################
# Physics functions #
#####################
def exchange_current_density(c):
    """Exchange current density function (ECIT form): j0 = sqrt(c)*(1-c)"""
    return jnp.sqrt(c) * (1 - c)

def transition_rate(c, Kut):
    """Concentration-dependent transition rate K12(c)"""
    return Kut * (1 + jnp.tanh(50 * (c - 0.05)))

def chemical_potential_rs(c, omega):
    """Regular solution chemical potential: mu = ln(c/(1-c)) + omega*(1-2c)"""
    return jnp.log(c / (1 - c)) + omega * (1 - 2 * c)

##########################
# Spatial discretization #
##########################
def create_grid(n_points, c_min=1e-5, c_max=1-1e-5):
    """Create finite volume grid."""
    c_edges = jnp.linspace(c_min, c_max, n_points + 1)
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    dc = c_edges[1] - c_edges[0]
    return c_centers, c_edges, dc

def compute_fluxes_fv(f, R, D, dc):
    """
    Compute fluxes at interior cell faces.
    Flux J = R*f - D*grad(f)
    
    Advection: Upwind
    Diffusion: Central difference
    """
    # R at edges (arithmetic mean)
    R_edges = 0.5 * (R[:-1] + R[1:])
    
    # Upwind advection
    # If R > 0, flux from left (i). If R < 0, flux from right (i+1).
    f_upwind = jnp.where(R_edges >= 0, f[:-1], f[1:])
    flux_adv = R_edges * f_upwind
    
    # Diffusion (central difference)
    flux_diff = -D * (f[1:] - f[:-1]) / dc
    
    return flux_adv + flux_diff

###################
# Residual System #
###################
def compute_reaction_rates_and_voltage_coeffs(c, V, params):
    """
    Compute R1, R2 for a given Voltage V.
    R = -j0 * ( (V - phi_ref)/VT + mu )
    """
    V_T = params["V_T"]
    phi_ref = params["phi_ref"]
    omega = params["omega"]
    k0 = params["k0"]
    
    mu_1 = chemical_potential_rs(c, omega)
    mu_2 = 0.0
    
    j0_1 = k0 * exchange_current_density(c)
    j0_2 = k0 * exchange_current_density(0.5)
    
    delta_mu_1 = (V - phi_ref) / V_T + mu_1
    delta_mu_2 = (V - phi_ref) / V_T + mu_2
    
    R1 = -j0_1 * delta_mu_1
    R2 = -j0_2 * delta_mu_2
    
    return R1, R2

def system_residual(y, y_old, dt, params, c_centers, dc):
    """
    Compute residual for the fully implicit system.
    
    y: [f1_0, ..., f1_{N-1}, f2_0, ..., f2_{N-1}, V]
    """
    n = c_centers.shape[0]
    f1 = y[:n]
    f2 = y[n:2*n]
    V = y[-1]
    
    f1_old = y_old[:n]
    f2_old = y_old[n:2*n]
    
    # Physics parameters
    D0 = params["D0"]
    Kut = params["Kut"]
    I_app = params["I_app"]
    Q_nom = params["Q_nom"]
    
    # 1. Calculate velocities
    R1, R2 = compute_reaction_rates_and_voltage_coeffs(c_centers, V, params)
    R2_vec = jnp.full_like(c_centers, R2)
    
    # 2. Fluxes at interior faces (size n-1)
    # Boundary fluxes are zero (no flux BC)
    J1_interior = compute_fluxes_fv(f1, R1, D0, dc)
    J2_interior = compute_fluxes_fv(f2, R2_vec, D0, dc)
    
    J1 = jnp.pad(J1_interior, (1, 1), constant_values=0.0)
    J2 = jnp.pad(J2_interior, (1, 1), constant_values=0.0)
    
    # Divergence: (J_right - J_left) / dc
    div_J1 = (J1[1:] - J1[:-1]) / dc
    div_J2 = (J2[1:] - J2[:-1]) / dc
    
    # Reaction term
    K12 = transition_rate(c_centers, Kut)
    trans_term = K12 * f1
    
    # Conservation Equations (Backward Euler)
    # (f - f_old)/dt + div(J) = source
    # Residual = (f - f_old)/dt + div(J) - source
    res_f1 = (f1 - f1_old) / dt + div_J1 + trans_term
    res_f2 = (f2 - f2_old) / dt + div_J2 - trans_term
    
    # Constraint Equation
    # integral(R1*f1 + R2*f2) dc = I/Q
    current_density = jnp.sum(R1 * f1 * dc) + jnp.sum(R2 * f2 * dc)
    res_V = current_density - (I_app / Q_nom)
    
    return jnp.concatenate([res_f1, res_f2, jnp.array([res_V])])

#################
# Newton Solver #
#################
@jax.jit
def newton_step(y, y_old, dt, params, c_centers, dc):
    """
    Perform one damped Newton step.
    y_new = y - J^{-1} * F(y)
    """
    # Calculate Residual and Jacobian
    res_fn = lambda y_curr: system_residual(y_curr, y_old, dt, params, c_centers, dc)
    F = res_fn(y)
    J = jax.jacobian(res_fn)(y)
    
    # Solve linear system
    delta = jnp.linalg.solve(J, -F)
    
    return y + delta, jnp.linalg.norm(delta), jnp.linalg.norm(F)

def solve_step_newton(y_guess, y_old, dt, params, c_centers, dc, max_iter=20, tol=1e-8):
    """
    Solve non-linear system for one time step.
    """
    y = y_guess
    for i in range(max_iter):
        y_next, diff_norm, res_norm = newton_step(y, y_old, dt, params, c_centers, dc)
        y = y_next
        # convergence check
        if res_norm < tol:
            return y, True, i+1
            
    return y, False, max_iter

##################
# Initialization #
##################
def solve_voltage_initial(c, f1, f2, I_app, params, dc):
    """Solve for initial V."""
    V_T = params["V_T"]
    phi_ref = params["phi_ref"]
    omega = params["omega"]
    k0 = params["k0"]
    Q_nom = params["Q_nom"]
    
    mu_1 = chemical_potential_rs(c, omega)
    mu_2 = 0.0
    j0_1 = k0 * exchange_current_density(c)
    j0_2 = k0 * exchange_current_density(0.5)
    
    # R = a*V + b
    # -j0 * ((V-phi)/Vt + mu)
    # = -j0/Vt * V + j0*(phi/Vt - mu)
    a_1 = -j0_1 / V_T
    b_1 = j0_1 * (phi_ref / V_T - mu_1)
    a_2 = -j0_2 / V_T
    b_2 = j0_2 * (phi_ref / V_T - mu_2)
    
    int_a = jnp.sum(a_1 * f1) * dc + a_2 * jnp.sum(f2) * dc
    int_b = jnp.sum(b_1 * f1) * dc + b_2 * jnp.sum(f2) * dc
    
    return (I_app / Q_nom - int_b) / int_a

def gaussian_ic(c, c0, width=0.02):
    f = jnp.exp(-((c - c0) ** 2) / (2 * width ** 2))
    return f / (width * jnp.sqrt(2 * jnp.pi))

###################
# Simulation Loop #
###################
def run_simulation(params_dict, I_app_val, start_SOC, t_final, n_points=200, dt_val=None):
    # Setup parameters
    params = params_dict.copy()
    params["I_app"] = I_app_val
    
    # Grid
    c, c_edges, dc = create_grid(n_points)
    n = n_points
    
    # Initial Conditions
    f1_init = gaussian_ic(c, start_SOC)
    f1_init = f1_init / (jnp.sum(f1_init) * dc)
    f2_init = jnp.zeros_like(c)
    
    # Initial Voltage Guess
    V_init = solve_voltage_initial(c, f1_init, f2_init, I_app_val, params, dc)
    
    # Pack state
    y_current = jnp.concatenate([f1_init, f2_init, jnp.array([V_init])])
    
    # Time stepping
    if dt_val is None:
        dt = 0.01
    else:
        dt = dt_val
        
    n_steps = int(np.ceil(t_final / dt))
    
    print(f"Grid: {n_points} points")
    print(f"Time: {t_final} h, dt={dt}, steps={n_steps}")
    
    t_hist = []
    V_hist = []
    SOC_hist = []
    mass_hist = []
    f1_hist = []
    f2_hist = []
    
    y = y_current
    
    start_time = time.time()
    
    for step in range(n_steps + 1):
        f1 = y[:n]
        f2 = y[n:2*n]
        V = y[-1]
        
        # Save history
        current_time = step * dt
        t_hist.append(current_time)
        V_hist.append(float(V))
        mass = float(jnp.sum(f1 + f2) * dc)
        mass_hist.append(mass)
        SOC = float(jnp.sum(c * (f1 + f2)) * dc)
        SOC_hist.append(SOC)
        f1_hist.append(np.array(f1))
        f2_hist.append(np.array(f2))
        
        if step % (n_steps // 10) == 0:
            print(f"Step {step}/{n_steps}: t={current_time:.2f}, V={V:.4f}, Mass={mass:.6f}, SOC={SOC:.4f}")
            
        if step == n_steps:
            break
            
        # Step forward
        # Use current solution as guess for next step
        y_next, converged, iters = solve_step_newton(y, y, dt, params, c, dc)
        
        if not converged:
            print(f"WARNING: Constant step Newton did not converge at step {step}!")
            
        y = y_next

    elapsed = time.time() - start_time
    print(f"Simulation finished in {elapsed:.2f}s")
    
    return {
        "t": np.array(t_hist),
        "V": np.array(V_hist),
        "SOC": np.array(SOC_hist),
        "mass": np.array(mass_hist),
        "f1": np.array(f1_hist),
        "f2": np.array(f2_hist),
        "c": c,
        "params": params
    }

def plot_results(res, filename="implicit_solver_results.png", diagnose=False):
    t, c = res["t"], res["c"]
    f1, f2 = res["f1"], res["f2"]
    V, SOC, mass = res["V"], res["SOC"], res["mass"]
    params = res["params"]
    I_app = params["I_app"]
    
    if diagnose:
        rows = 2
        figsize = (15, 9)
    else:
        rows = 1
        figsize = (15, 4.5)
        
    fig, axes = plt.subplots(rows, 3, figsize=figsize)
    
    if rows == 1:
        ax_row1 = axes
    else:
        ax_row1 = axes[0]
        ax_row2 = axes[1]
    
    # 1. f1 evolution
    ax = ax_row1[0]
    pcm = ax.pcolormesh(c, t, np.maximum(f1, 1e-6), 
                        cmap='seismic', shading='gouraud',
                        norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=f1.max()))
    ax.set_xlabel(r"$c$")
    ax.set_ylabel("Time [h]")
    ax.set_title(r"$f_1$ (untransformed)")
    fig.colorbar(pcm, ax=ax, label="Probability density")
    
    # 2. f2 evolution
    ax = ax_row1[1]
    if f2.max() > 1e-3:
        pcm = ax.pcolormesh(c, t, np.maximum(f2, 1e-6), 
                            cmap='seismic', shading='gouraud',
                            norm=plt.matplotlib.colors.LogNorm(vmin=1e-3, vmax=max(f2.max(), 1e-2)))
    else:
        pcm = ax.pcolormesh(c, t, f2, cmap='seismic', shading='gouraud')
    ax.set_xlabel(r"$c$")
    ax.set_ylabel("Time [h]")
    ax.set_title(r"$f_2$ (transforming)")
    fig.colorbar(pcm, ax=ax, label="Probability density")
    
    # 3. Voltage vs SOC
    ax = ax_row1[2]
    ax.plot(SOC, V)
    ax.set_title("Voltage vs SOC")
    ax.set_xlabel("SOC")
    ax.set_ylabel("Voltage [V]")
    
    # --- Row 2: Diagnostics (Optional) ---
    if diagnose:
        # 4. Final Distribution Snapshot
        ax = ax_row2[0]
        ax.plot(c, f1[-1], label=r'$f_1(T_{final})$')
        ax.plot(c, f2[-1], label=r'$f_2(T_{final})$')
        ax.set_xlabel(r"$c$")
        ax.set_title("Final Distribution")
        ax.legend()
        
        # 5. SOC vs Time
        ax = ax_row2[1]
        ax.plot(t, SOC, label="Simulated")
        ax.plot(t, SOC[0] + (I_app/params["Q_nom"])*t, 'r--', label="Theoretical")
        ax.set_title("SOC vs Time")
        ax.set_xlabel("Time [h]")
        ax.legend()
        
        # 6. Mass Conservation
        ax = ax_row2[2]
        ax.plot(t, mass)
        ax.set_title("Total Mass Conservation")
        ax.set_xlabel("Time [h]")
        ax.ticklabel_format(useOffset=False)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Results saved to {filename}")

if __name__ == "__main__":
    params = {
        "Q_nom": 2.3,
        "omega": 4.5,
        "k0": 0.5,
        "V_T": 0.0257,
        "D0": 2e-3,
        "phi_ref": 3.422,
        "Kut": 1.0,
    }
    
    I_app = 0.23
    start_SOC = 0.1
    t_final = 10.0
    
    dt = 0.05 
    
    res = run_simulation(params, I_app, start_SOC, t_final, n_points=100, dt_val=dt)
    
    # Plotting
    import sys
    diagnose = "--diagnose" in sys.argv
    plot_results(res, diagnose=diagnose)
