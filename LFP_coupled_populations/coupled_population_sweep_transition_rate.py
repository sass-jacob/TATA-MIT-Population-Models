import pybamm
import numpy as np
import matplotlib.pyplot as plt


def create_model():
    model = pybamm.BaseModel(name="CoupledPopulation")

    # --- VARIABLES ---
    c = pybamm.SpatialVariable("c", domain="filling_fraction_space", coord_sys="cartesian")
    f_1 = pybamm.Variable("Probability Distribution Untransformed", domain="filling_fraction_space")
    f_2 = pybamm.Variable("Probability Distribution Transforming", domain="filling_fraction_space")
    V_cell = pybamm.Variable("Cell Voltage [V]")

    # --- PARAMETERS ---
    thermal_voltage = pybamm.Parameter("Thermal Voltage [V]")
    phi_ref = pybamm.Parameter("Reference Potential [V]")
    phi_phase_separation = pybamm.Parameter("Phase Separation Potential [V]")
    Omega = pybamm.Parameter("Interaction Parameter")
    k0 = pybamm.Parameter("Rate Constant")
    D = pybamm.Parameter("Fluctuations Strength")
    alpha = 0.5
    Q_nom = pybamm.Parameter("Nominal cell capacity [A.h]")
    nu = pybamm.Parameter("Numerical Viscosity")
    K_12 = pybamm.FunctionParameter("Transition Rate Untransformed to Transforming", {"cell voltage [V]": V_cell})
    ecd = pybamm.FunctionParameter("Exchange current density", {"c": c})

    # --- INPUTS ---
    I_app = pybamm.InputParameter("Current function [A]")

    # --- EQUATIONS ---
    mu_1 = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)
    delta_mu_1 = V_cell / thermal_voltage - phi_ref / thermal_voltage - mu_1
    R0_1 = k0 * ecd
    g_1 = pybamm.exp(alpha * delta_mu_1) - pybamm.exp(-(1 - alpha) * delta_mu_1)
    R_1 = R0_1 * g_1
    eps = 1e-10
    R_abs_1 = pybamm.sqrt(R_1**2 + eps)
    D_eff_1 = D + nu * R_abs_1
    diffusive_flux_1 = -D_eff_1 * pybamm.grad(f_1)
    advective_flux_1 = R_1 * f_1
    total_flux_1 = advective_flux_1 + diffusive_flux_1

    mu_2 = phi_phase_separation / thermal_voltage
    delta_mu_2 = V_cell / thermal_voltage - mu_2
    R0_2 = k0 * ecd
    g_2 = pybamm.exp(alpha * delta_mu_2) - pybamm.exp(-(1 - alpha) * delta_mu_2)
    R_2 = R0_2 * g_2
    R_abs_2 = pybamm.sqrt(R_2**2 + eps)
    D_eff_2 = nu * R_abs_2
    diffusive_flux_2 = -D_eff_2 * pybamm.grad(f_2)
    advective_flux_2 = R_2 * f_2
    total_flux_2 = advective_flux_2 + diffusive_flux_2

    model.rhs = {
        f_1: -pybamm.div(total_flux_1) - K_12 * f_1,
        f_2: -pybamm.div(total_flux_2) + K_12 * f_1
    }
    model.algebraic = {
        V_cell: pybamm.Integral(f_1 * R_1, c) + pybamm.Integral(f_2 * R_2, c) - I_app / Q_nom
    }

    model.boundary_conditions = {
        f_1: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")},
        f_2: {"left": (pybamm.Scalar(0), "Neumann"), "right": (pybamm.Scalar(0), "Neumann")}
    }

    start_c = pybamm.Parameter("Initial Filling Fraction")
    width = 0.01
    f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
    model.initial_conditions = {
        f_1: f_init / (width * np.sqrt(2 * np.pi)),
        f_2: 0,
        V_cell: pybamm.Scalar(3.422)
    }

    c_avg = pybamm.Integral(f_1 * c, c) + pybamm.Integral(f_2 * c, c)
    model.events = [
        pybamm.Event("Maximum stoichiometry", 0.95 - c_avg),
        pybamm.Event("Minimum stoichiometry", c_avg - 0.05),
        pybamm.Event("Maximum voltage", 4.2 - V_cell),
        pybamm.Event("Minimum voltage", V_cell - 2.5),
    ]

    return model, c


def run_simulation(K12_prefactor, K12_additive, constant_current_value, start_SOC, t_eval, n_points=200):
    """Run simulation with specified transition rate parameters."""
    model, c = create_model()
    geometry = {"filling_fraction_space": {c: {"min": pybamm.Scalar(1e-3), "max": pybamm.Scalar(1-1e-3)}}}
    dc = 1.0 / n_points

    def exchange_current_density(c_val):
        return np.sqrt(c_val * (1 - c_val))

    # Transition rate function with the specified parameters
    phi_ps = 3.422
    V_th = 0.025
    def transition_rate(V):
        return K12_prefactor * np.exp(V_th / ((V - phi_ps)**2 + K12_additive))

    params = pybamm.ParameterValues({
        "Nominal cell capacity [A.h]": 2.3,
        "Interaction Parameter": 4.5,
        "Rate Constant": 1.0,
        "Thermal Voltage [V]": V_th,
        "Fluctuations Strength": 2e-4,
        "Initial Filling Fraction": start_SOC,
        "Reference Potential [V]": 3.422,
        "Phase Separation Potential [V]": phi_ps,
        "Current function [A]": constant_current_value,
        "Numerical Viscosity": dc,
        "Transition Rate Untransformed to Transforming": transition_rate,
        "Exchange current density": exchange_current_density,
    })

    model_proc = params.process_model(model, inplace=False)
    submesh_types = {"filling_fraction_space": pybamm.Uniform1DSubMesh}
    var_pts = {c: n_points}
    mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
    spatial_methods = {"filling_fraction_space": pybamm.FiniteVolume()}
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model_proc)

    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)

    try:
        solution = solver.solve(
            model_proc,
            t_eval,
            inputs={"Current function [A]": constant_current_value}
        )
    except pybamm.SolverError as e:
        print(f"  Solver failed for K12_prefactor={K12_prefactor}, K12_additive={K12_additive}: {str(e)[:60]}...")
        return None

    c_vals = mesh["filling_fraction_space"].nodes
    dx = c_vals[1] - c_vals[0]
    f_1_sol = solution["Probability Distribution Untransformed"].entries
    f_2_sol = solution["Probability Distribution Transforming"].entries
    c_avg = np.sum(c_vals[:, np.newaxis] * f_1_sol, axis=0) * dx + np.sum(c_vals[:, np.newaxis] * f_2_sol, axis=0) * dx
    V_sol = solution["Cell Voltage [V]"].entries

    return {
        'c_avg': c_avg,
        'V_sol': V_sol,
        'K12_prefactor': K12_prefactor,
        'K12_additive': K12_additive
    }


if __name__ == "__main__":
    # Simulation parameters
    constant_current_value = 0.23 # 0.1C
    start_SOC = 0.05
    t_eval = np.linspace(0, 9.4, 1000)

    # Parameter sweep values
    K12_prefactor_values = [1e-3, 1e-2, 1e-1, 1.0]
    K12_additive_values = [1e-2, 1e-1, 1.0]

    n_prefactors = len(K12_prefactor_values)
    n_additives = len(K12_additive_values)

    print(f"Running parameter sweep: {n_prefactors} x {n_additives} = {n_prefactors * n_additives} simulations")
    print(f"  K12 prefactors: {K12_prefactor_values}")
    print(f"  K12 additives:  {K12_additive_values}")
    print()

    results = {}
    for i, K12_pre in enumerate(K12_prefactor_values):
        for j, K12_add in enumerate(K12_additive_values):
            print(f"  Running K12_prefactor={K12_pre}, K12_additive={K12_add}...")
            result = run_simulation(K12_pre, K12_add, constant_current_value, start_SOC, t_eval)
            if result is not None:
                results[(i, j)] = result

    print(f"\nCompleted {len(results)} / {n_prefactors * n_additives} simulations successfully!")

    fig, axes = plt.subplots(n_prefactors, n_additives, figsize=(4 * n_additives, 3 * n_prefactors), sharex=True, sharey=True)

    colors = plt.cm.viridis(np.linspace(0, 1, n_prefactors * n_additives))

    for i, K12_pre in enumerate(K12_prefactor_values):
        for j, K12_add in enumerate(K12_additive_values):
            ax = axes[i, j] if n_prefactors > 1 else axes[j]

            if (i, j) in results:
                res = results[(i, j)]
                ax.plot(res['c_avg'], res['V_sol'], 'b-', linewidth=1.5)
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')

            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel(f"$K_{{12,pre}}$={K12_pre}\nV [V]")

            if i == 0:
                ax.set_title(f"$K_{{12,add}}$={K12_add}")

            if i == n_prefactors - 1:
                ax.set_xlabel(r"$\langle c \rangle$")

    fig.suptitle(
        f"Voltage Response: Varying Transition Rate Parameters\n"
        f"Current = {constant_current_value} A, Initial SOC = {start_SOC}",
        fontsize=14
    )
    plt.tight_layout()
    # plt.savefig("transition_rate_sweep_grid.png", dpi=150, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    cmap = plt.cm.plasma
    norm = plt.Normalize(0, len(results) - 1)

    for idx, ((i, j), res) in enumerate(results.items()):
        K12_pre = res['K12_prefactor']
        K12_add = res['K12_additive']
        color = cmap(norm(idx))
        plt.plot(res['c_avg'], res['V_sol'], '-', color=color, linewidth=1.5,
                 label=f"pre={K12_pre}, add={K12_add}")

    plt.xlabel(r"Filling Fraction $\langle c \rangle$")
    plt.ylabel("Cell Voltage [V]")
    plt.title("Voltage Response: All Transition Rate Parameter Combinations")
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("transition_rate_sweep_overlay.png", dpi=150, bbox_inches='tight')
    plt.show()