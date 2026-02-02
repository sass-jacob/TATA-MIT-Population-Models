"""Single Population Battery Model"""
from typing import Dict, Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# some default parameters
DEFAULT_PARAMS = {
    "Q": 1.0,
    "omega": 2.6,
    "k0": 5.0,
    "D0": 1e-2,
    "phi_ref": 3.422,
    "V_T": 0.0257,
}

class SinglePopulationVariables(NamedTuple):
    """immutable state container"""
    f: jnp.ndarray # probability distribution function
    V: float       # voltage
    t: float       # time

#####################
# Physics Functions #
#####################
@jax.jit
def chemical_potential(c: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    Regular solution chemical potential.
    mu = ln(c/(1-c)) + omega(1-2c)
    """
    return jnp.log(c/(1-c)) + omega*(1-2*c)

@jax.jit
def exchange_current_density(c: jnp.ndarray, k0: float) -> jnp.ndarray:
    """
    Exchange current density j0 = k0 * sqrt(c) * (1-c)
    """
    return k0 * jnp.sqrt(c) * (1-c)

@jax.jit
def reaction_rate(c: jnp.ndarray, omega: float, k0: float, phi_ref: float, V_T: float, V: float):
    """
    Compute the reaction rate R = -delta_mu * j0 using linearized kinetics
    """
    mu = chemical_potential(c, omega)
    j0 = exchange_current_density(c, k0)
    delta_mu = V / V_T - phi_ref / V_T + mu
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
    # TODO: do upwinding based on sign of I_app
    f_upwind = jnp.where(R_faces >= 0, f[:-1], f[1:])
    flux_advection = R_faces * f_upwind
    flux_diffusion = -D * (f[1:] - f[:-1]) / dc
    return flux_advection + flux_diffusion

def _create_f_residual(c_grid: jnp.ndarray, dc: float, params: Dict):
    n = len(c_grid)
    omega = params["omega"]
    k0 = params["k0"]
    D0 = params["D0"]
    phi_ref = params["phi_ref"]
    V_T = params["V_T"]
    Q = params["Q"]

    @jax.jit
    def f_residual(y: jnp.ndarray, y_old: jnp.ndarray, dt: float, I_app: float) -> jnp.ndarray:
        f = y[:n]
        V = y[-1]
        f_old = y_old[:n]

        R = reaction_rate(c_grid, omega, k0, phi_ref, V_T, V)
        J = finite_volume_fluxes(f, R, D0, dc)
        # add no-flux boundary conditions
        J = jnp.pad(J, (1, 1), constant_values=0.0)
        # compute divergence of flux
        div_J = (J[1:] - J[:-1]) / dc
        # use backward euler for time integration for now
        res_f = (f - f_old) / dt + div_J
        # current constraint: flux integral = applied current / capacity
        int_J = jnp.sum(J) * dc
        res_V = int_J - (I_app / Q)
        # return residual vector [f_0, f_1, ..., f_{Nc-1}, V]
        return jnp.concatenate([res_f, jnp.array([res_V])])
    
    return f_residual

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
    
    def solve(y_init, y_old, dt, I_app, tol=1e-10, max_iter=20):
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
class SinglePopulationModel:

    def __init__(
        self,
        params: Optional[Dict] = None,
        Nc: int = 100
    ):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.Nc = Nc
        self.c_grid = jnp.linspace(1e-5, 1-1e-5, Nc)
        self.dc = float(self.c_grid[1] - self.c_grid[0])
        self._f_residual = _create_f_residual(self.c_grid, self.dc, self.params)
        self._solver = _create_solver(self._f_residual)
        self._jit_step = jax.jit(self._step)

    def initialize(
        self,
        SOC: float = 0.1,
        I_app: float = 0.0
    ) -> SinglePopulationVariables:
        sigma = 0.05
        f_init = jnp.exp(-(self.c_grid - SOC)**2 / (2 * sigma**2))
        f_init /= jnp.sum(f_init) * self.dc
        V_init = self._V_initial(f_init, I_app)
        return SinglePopulationVariables(f=f_init, V=V_init, t=0.0)
        
    def step(
        self,
        state: SinglePopulationVariables,
        I_app: float,
        dt: float = 0.02
    ) -> Tuple[SinglePopulationVariables, Dict]:
        y_old = jnp.concatenate([state.f, jnp.array([state.V])])
        y_new, iters, norm = self._solver(y_old, y_old, dt, I_app)
        f_new = y_new[:self.Nc]
        V_new = float(y_new[-1])

        new_state = SinglePopulationVariables(
            f = f_new,
            V = V_new,
            t = state.t + dt
        )

        soc = float(jnp.sum(self.c_grid * f_new) * self.dc)

        info = {
            "SOC": soc,
            "V": V_new,
            "iters": iters,
            "res": float(norm)
        }

        return new_state, info

    def _step(self, y_old: jnp.ndarray, dt: float, I_app: float):
        """JIT-compiled step implementation."""
        return self._solver(y_old, y_old, dt, I_app)


    def _V_initial(self, f: jnp.ndarray, I_app: float) -> float:
        omega = self.params["omega"]
        k0 = self.params["k0"]
        phi_ref = self.params["phi_ref"]
        V_T = self.params["V_T"]
        Q = self.params["Q"]
        dc = self.dc
        # solve sum(-j0 * ((V - phi_ref)/V_T + mu) * f * dc) = I_app / Q
        # rearrange to solve for V
        j0 = exchange_current_density(self.c_grid, k0)
        mu = chemical_potential(self.c_grid, omega)
        A = -jnp.sum(j0 * f * dc) / V_T
        B = -jnp.sum(j0 * mu * f * dc)
        return float(phi_ref + (I_app / Q - B) / A)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    model = SinglePopulationModel()
    
    start_soc = 0.1
    state = model.initialize(SOC=start_soc)
    print(f"Initial SOC: {start_soc:.2f}")
    print(f"Initial V: {state.V:.4f} V")
    
    I_discharge = 0.5           # 0.5C charge (negative = discharge)
    dt = 0.001                  # Time step [hours]
    n_steps = 1600              # ~1.6 hours
    
    times, socs, voltages = [0.0], [start_soc], [state.V]
    
    for step in range(n_steps):
        state, info = model.step(state, I_app=I_discharge, dt=dt)
        times.append(state.t)
        socs.append(info["SOC"])
        voltages.append(info["V"])
        
        if (step + 1) % 20 == 0:
            print(f"  t={state.t:.2f}h, SOC={info['SOC']:.3f}, V={info['V']:.4f}V")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(times, socs, 'b-', linewidth=2)
    ax1.set_xlabel('Time [h]')
    ax1.set_ylabel('SOC')
    ax1.set_title('State of Charge')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, voltages, 'r-', linewidth=2)
    ax2.set_xlabel('Time [h]')
    ax2.set_ylabel('Voltage [V]')
    ax2.set_title('Terminal Voltage')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discharge_simulation.png', dpi=150)
    plt.show()