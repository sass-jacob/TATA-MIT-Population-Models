"""Dual Population Battery Model (LFP + Graphite)"""
from typing import Dict, Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# some default parameters
DEFAULT_PARAMS = {
    # Cathode (LFP)
    "Q_cat": 1.0,
    "omega_cat": 2.6,
    "k0_cat": 5.0,
    "D0_cat": 1e-2,
    "phi_ref_cat": 3.422,
    
    # Anode (Graphite)
    "Q_an": 1.0,
    "k0_an": 5.0,
    "D0_an": 1e-2,
    "phi_ref_an": 0.12, # approximate reference for graphite
    
    # Global
    "V_T": 0.0257,
    "Omga": 0.81169,
    "Omgb": 2.2214
}

class DualPopulationVariables(NamedTuple):
    """immutable state container"""
    f_cat: jnp.ndarray # cathode probability distribution
    f_an: jnp.ndarray  # anode probability distribution
    V_cell: float      # cell voltage (phi_cat - phi_an)
    phi_an: float      # anode potential vs Li/Li+
    t: float           # time

#####################
# Physics Functions #
#####################
@jax.jit
def chemical_potential_LFP(c: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    Regular solution chemical potential (LFP cathode).
    mu = ln(c/(1-c)) + omega(1-2c)
    """
    # Clip to avoid log(0)
    c = jnp.clip(c, 1e-6, 1.0 - 1e-6)
    return jnp.log(c/(1-c)) + omega*(1-2*c)

@jax.jit
def step_up(c, c0, width):
    return 0.5 * (1 + jnp.tanh((c - c0) / width))

@jax.jit
def step_down(c, c0, width):
    return 0.5 * (1 - jnp.tanh((c - c0) / width))

@jax.jit
@jax.jit
def graphite_5part_mu(c: jnp.ndarray, Omga:float, Omgb:float) -> jnp.ndarray:
    """ 
    Graphite chemical potential (dimensionless mu/kBT) based on detailed equations.
    mu_tot = mu1 + mu2 + mu3 + mu4 + mu5
    Returns mu_tot / (kB * T) which is dimensionless.
    """
    # Avoid singularities
    c = jnp.clip(c, 1e-4, 1.0-1e-4) # Slightly tighter clip for exp terms
    
    # mu1 [A.2]
    # term in bracket: [-30*exp(-c/0.025) - 2(1-c)] * SD(c, 0.38, 0.05)
    term1_inner = -30 * jnp.exp(-c / 0.025) - 2 * (1 - c)
    term1_blob = term1_inner * step_down(c, 0.38, 0.05)
    
    term1_tanh = (
        0.7 * (jnp.tanh((c - 0.37) / 0.075) - 1) +
        0.8 * (jnp.tanh((c - 0.2) / 0.06) - 1) +
        0.38 * (jnp.tanh((c - 0.14) / 0.015) - 1)
    )
    mu1 = term1_blob + term1_tanh # Dimensionless
    
    # mu2 [A.3]
    mu2 = -0.05 / (c**0.55) # Dimensionless
    
    # mu3 [A.4]
    mu3 = 10 * step_up(c, 1.0, 0.015)
    
    # mu4 [A.5]
    # 1.8 * Omga * (0.17 - c^0.98) * SD * SU
    mu4 = 1.8 * Omga * (0.17 - c**0.98) * step_down(c, 0.55, 0.045) * step_up(c, 0.38, 0.05)
    
    # mu5 [A.6]
    # [0.4*Omga*(0.74-c) + 0.55*Omgb - 2(1-c)] * SU(c, 0.6, 0.04)
    term5_bracket = 0.4 * Omga * (0.74 - c) + 0.55 * Omgb - 2 * (1 - c)
    mu5 = term5_bracket * step_up(c, 0.6, 0.04)
    
    mu_total_dimless = mu1 + mu2 + mu3 + mu4 + mu5
    return mu_total_dimless-0.02/0.0257

@jax.jit
def exchange_current_density(c: jnp.ndarray, k0: float) -> jnp.ndarray:
    """
    Exchange current density j0 = k0 * sqrt(c) * (1-c)
    """
    return k0 * jnp.sqrt(c) * (1-c)

@jax.jit
def reaction_rate_cathode(c: jnp.ndarray, omega: float, phi_cat: float, k0: float, phi_ref: float, V_T: float):
    # add a negative sign here for charging and positive sign for discharging
    mu_cat = -chemical_potential_LFP(c, omega)
    j0 = exchange_current_density(c, k0)
    delta_mu = (phi_cat - phi_ref) / V_T + mu_cat
    return -j0 * delta_mu

@jax.jit
def reaction_rate_anode(c: jnp.ndarray, Omga: float, Omgb: float, phi_an: float, k0: float, phi_ref: float, V_T: float):
    # add a positive sign here for charging and negative sign for discharging
    mu_an_dim = graphite_5part_mu(c, Omga, Omgb)
    j0 = exchange_current_density(c, k0)
    delta_mu = (phi_an - phi_ref) / V_T + mu_an_dim
    return -j0 * delta_mu

def finite_volume_fluxes(
    f: jnp.ndarray,
    R: jnp.ndarray,
    D: float,
    dc: float
) -> jnp.ndarray:
    """
    Compute fluxes at cell faces using finite volume method.
    J = R*f (advection with upwind) - D*df/dc (diffusion)
    """
    R_faces = 0.5 * (R[:-1] + R[1:])
    f_upwind = jnp.where(R_faces >= 0, f[:-1], f[1:])
    flux_advection = R_faces * f_upwind
    flux_diffusion = -D * (f[1:] - f[:-1]) / dc
    return flux_advection + flux_diffusion

def _create_residual(c_grid: jnp.ndarray, dc: float, params: Dict):
    Nc = len(c_grid)
    
    # Cathode Params
    omega_cat = params["omega_cat"]
    k0_cat = params["k0_cat"]
    D0_cat = params["D0_cat"]
    phi_ref_cat = params["phi_ref_cat"]
    Q_cat = params["Q_cat"]
    
    # Anode Params
    Omga = params["Omga"]
    Omgb = params["Omgb"]
    k0_an = params["k0_an"]
    D0_an = params["D0_an"]
    phi_ref_an = params["phi_ref_an"] 
    Q_an = params["Q_an"]
    
    V_T = params["V_T"]

    @jax.jit
    def residual(y: jnp.ndarray, y_old: jnp.ndarray, dt: float, I_app: float) -> jnp.ndarray:
        f_cat = y[:Nc]
        f_an = y[Nc:2*Nc]
        V_cell = y[2*Nc]
        phi_an = y[2*Nc+1]
        
        f_cat_old = y_old[:Nc]
        f_an_old = y_old[Nc:2*Nc]
        
        # Potentials
        phi_cat = V_cell + phi_an
        
        # --- Cathode Physics ---
        R_cat = reaction_rate_cathode(c_grid, omega_cat, phi_cat, k0_cat, phi_ref_cat, V_T)
        J_cat = finite_volume_fluxes(f_cat, R_cat, D0_cat, dc)
        J_cat = jnp.pad(J_cat, (1, 1), constant_values=0.0)
        div_J_cat = (J_cat[1:] - J_cat[:-1]) / dc
        res_f_cat = (f_cat - f_cat_old) / dt + div_J_cat
        
        # --- Anode Physics ---
        R_an = reaction_rate_anode(c_grid, Omga, Omgb, phi_an, k0_an, phi_ref_an, V_T)
        J_an = finite_volume_fluxes(f_an, R_an, D0_an, dc)
        J_an = jnp.pad(J_an, (1, 1), constant_values=0.0)
        div_J_an = (J_an[1:] - J_an[:-1]) / dc
        res_f_an = (f_an - f_an_old) / dt + div_J_an
        
        int_J_cat = jnp.sum(J_cat) * dc # Rate of change of total Li in Cathode
        res_I_cat = int_J_cat - (I_app / Q_cat)
        
        int_J_an = jnp.sum(J_an) * dc   # Rate of change of total Li in Anode
        # If I_app > 0 (Discharge), Li leaves Anode. d(Li_an)/dt should be negative.
        res_I_an = int_J_an - (-I_app / Q_an)
        
        return jnp.concatenate([
            res_f_cat, 
            res_f_an, 
            jnp.array([res_I_cat]), 
            jnp.array([res_I_an])
        ])
    
    return residual

#################
# Newton Solver #
#################
def _create_solver(residual_fn):
    @jax.jit
    def newton_step(y, y_old, dt, I_app):
        F = residual_fn(y, y_old, dt, I_app)
        J = jax.jacobian(lambda x: residual_fn(x, y_old, dt, I_app))(y)
        delta = jnp.linalg.solve(J, -F)
        return y + delta, jnp.linalg.norm(F)
    
    def solve(y_init, y_old, dt, I_app, tol=1e-8, max_iter=20):
        y = y_init
        for i in range(max_iter):
            y, norm = newton_step(y, y_old, dt, I_app)
            if norm < tol:
                break
        return y, i+1, norm
    
    return solve


####################
# Main Model Class #
####################
class DualPopulationModel:

    def __init__(
        self,
        params: Optional[Dict] = None,
        Nc: int = 100 # Increased for stability
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.Nc = Nc
        self.c_grid = jnp.linspace(1e-5, 1-1e-5, Nc)
        self.dc = float(self.c_grid[1] - self.c_grid[0])
        self._residual = _create_residual(self.c_grid, self.dc, self.params)
        self._solver = _create_solver(self._residual)

    def initialize(
        self,
        SOC_cat: float = 0.1,
        SOC_an: float = 0.9, # Initially full anode for discharge
        I_app: float = 0.0
    ) -> DualPopulationVariables:
        sigma = 0.05
        # Initialize Gaussians
        f_cat_init = jnp.exp(-(self.c_grid - SOC_cat)**2 / (2 * sigma**2))
        f_cat_init /= jnp.sum(f_cat_init) * self.dc
        
        f_an_init = jnp.exp(-(self.c_grid - SOC_an)**2 / (2 * sigma**2))
        f_an_init /= jnp.sum(f_an_init) * self.dc
        
        # Initial Voltage Guess
        # Rough OCP guess
        # V_cell ~ OCP_cat - OCP_an
        phi_ref_cat = self.params["phi_ref_cat"]
        omega_cat = self.params["omega_cat"]
        Omga = self.params["Omga"]
        Omgb = self.params["Omgb"]
        
        mu_cat_scalar = float(chemical_potential_LFP(jnp.array(SOC_cat), omega_cat))
        U_cat_approx = phi_ref_cat - mu_cat_scalar * self.params["V_T"] # Inverse of reaction rate logic roughly
        
        U_an_approx = float(graphite_5part_mu(jnp.array(SOC_an), Omga, Omgb))
        
        V_cell_init = U_cat_approx - U_an_approx
        phi_an_init = U_an_approx
        
        return DualPopulationVariables(
            f_cat=f_cat_init,
            f_an=f_an_init,
            V_cell=V_cell_init,
            phi_an=phi_an_init,
            t=0.0
        )
        
    def step(
        self,
        state: DualPopulationVariables,
        I_app: float,
        dt: float = 0.02
    ) -> Tuple[DualPopulationVariables, Dict]:
        y_old = jnp.concatenate([
            state.f_cat, 
            state.f_an, 
            jnp.array([state.V_cell]),
            jnp.array([state.phi_an])
        ])
        
        y_new, iters, norm = self._solver(y_old, y_old, dt, I_app)
        
        f_cat_new = y_new[:self.Nc]
        f_an_new = y_new[self.Nc:2*self.Nc]
        V_cell_new = float(y_new[2*self.Nc])
        phi_an_new = float(y_new[2*self.Nc+1])

        new_state = DualPopulationVariables(
            f_cat = f_cat_new,
            f_an = f_an_new,
            V_cell = V_cell_new,
            phi_an = phi_an_new,
            t = state.t + dt
        )

        soc_cat = float(jnp.sum(self.c_grid * f_cat_new) * self.dc)
        soc_an = float(jnp.sum(self.c_grid * f_an_new) * self.dc)

        info = {
            "SOC_cat": soc_cat,
            "SOC_an": soc_an,
            "V_cell": V_cell_new,
            "phi_an": phi_an_new,
            "iters": iters,
            "res": float(norm)
        }

        return new_state, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Initializing Dual Population Model...")
    model = DualPopulationModel(Nc=100)

    # Charging
    start_soc_cat = 0.95
    start_soc_an = 0.05
    # discharging
    # start_soc_cat = 0.05
    # start_soc_an = 0.95
    
    state = model.initialize(SOC_cat=start_soc_cat, SOC_an=start_soc_an)
    print(f"Initial State: SOC_cat={start_soc_cat:.2f}, SOC_an={start_soc_an:.2f}")
    print(f"Initial V_cell: {state.V_cell:.4f} V, phi_an: {state.phi_an:.4f} V")
    
    # Discharge Current (I > 0 for Discharge in our convention for Cathode)
    # Cathode gains Li -> I_app > 0

    # I_discharge would be negative for charging
    I_discharge = -0.1
    dt = 0.001
    n_steps = 9000 # Long run test (5 hours) similar duration
    
    times, socs_cat, socs_an, v_cells, phi_ans = [], [], [], [], []
    f_cat_history, f_an_history = [], []
    
    print("Starting simulation...")
    for step in range(n_steps):
        state, info = model.step(state, I_app=I_discharge, dt=dt)
        
        times.append(state.t)
        socs_cat.append(info["SOC_cat"])
        socs_an.append(info["SOC_an"])
        v_cells.append(info["V_cell"])
        phi_ans.append(info["phi_an"])
        
        # Store distributions every 100 steps to avoid memory overload
        if step % 100 == 0:
            f_cat_history.append(state.f_cat)
            f_an_history.append(state.f_an)
        
        if (step+1) % 100 == 0:
            print(f"t={state.t:.3f}, V={info['V_cell']:.4f}, Anode={info['phi_an']:.4f}, FF_c={info['SOC_cat']:.2f}")
            
    fig = plt.figure(figsize=(14, 10))
    
    # Calculate Cathode Potential
    phi_cats = [v + p for v, p in zip(v_cells, phi_ans)]

    # Top Left: Cell Voltage (2D)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(times, v_cells, 'b-', label='Cell Voltage')
    ax1.set_title('Cell Voltage')
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('Voltage [V]')
    ax1.legend()
    ax1.grid(True)
    
    # Top Right: Filling Fraction (2D)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(times, socs_cat, 'g-', label='Cathode Filling Fraction')
    ax2.plot(times, socs_an, 'k--', label='Anode Filling Fraction')
    ax2.set_title('Filling Fraction')
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Filling Fraction')
    ax2.legend()
    ax2.grid(True)
    
    # Bottom Left: Cathode Distribution (3D Waterfall)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    times_history = [times[i] for i in range(0, n_steps, 100)] if n_steps > 100 else times
    # Ensure lengths match
    if len(times_history) > len(f_cat_history): times_history = times_history[:len(f_cat_history)]

    for i, (t_snap, f) in enumerate(zip(times_history, f_cat_history)):
        # Plot x=c, y=time, z=f
        color = plt.cm.viridis(i / len(f_cat_history))
        ax3.plot(model.c_grid, [t_snap]*len(model.c_grid), f, color=color, alpha=0.8)
        
    ax3.set_title('Cathode Distribution Evolution')
    ax3.set_xlabel('Concentration')
    ax3.set_ylabel('Time')
    ax3.set_zlabel('Probability Density')
    
    # Bottom Right: Anode Distribution (3D Waterfall)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    for i, (t_snap, f) in enumerate(zip(times_history, f_an_history)):
        color = plt.cm.inferno(i / len(f_an_history))
        ax4.plot(model.c_grid, [t_snap]*len(model.c_grid), f, color=color, alpha=0.8)
        
    ax4.set_title('Anode Distribution Evolution')
    ax4.set_xlabel('Concentration')
    ax4.set_ylabel('Time')
    ax4.set_zlabel('Probability Density')
    
    plt.tight_layout()
    plt.savefig('dual_population_test.png')
    print("Test complete. Saved dual_population_test.png")
    plt.show()
    
    # Extra Plot: Potentials Side by Side
    fig_pot, (ax_an, ax_cat) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax_an.plot(times, phi_ans, 'r-', label='Anode Potential')
    ax_an.set_title('Anode Potential (Graphite)')
    ax_an.set_xlabel('Time [h]')
    ax_an.set_ylabel('Potential [V vs Li/Li+]')
    ax_an.grid(True)
    ax_an.legend()
    
    ax_cat.plot(times, phi_cats, 'g-', label='Cathode Potential')
    ax_cat.set_title('Cathode Potential (LFP)')
    ax_cat.set_xlabel('Time [h]')
    ax_cat.set_ylabel('Potential [V vs Li/Li+]')
    ax_cat.grid(True)
    ax_cat.legend()
    
    plt.tight_layout()
    plt.savefig('potentials_side_by_side.png')
    print("Saved potentials_side_by_side.png")
    plt.show()