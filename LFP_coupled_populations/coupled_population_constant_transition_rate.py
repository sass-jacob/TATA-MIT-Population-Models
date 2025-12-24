import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel(name="CoupledPopulation")

# --- VARIABLES ---
# filling fraction coordinate 'c' from 0 to 1
c = pybamm.SpatialVariable("c", domain="filling_fraction_space", coord_sys="cartesian")
geometry = {"filling_fraction_space": {c: {"min": pybamm.Scalar(1e-3), "max": pybamm.Scalar(1-1e-3)}}}
# f_1 is the fraction of active material particles that have filling fraction 'c' that are
# in the region where they follow the regular solution chemical potential
f_1 = pybamm.Variable("Probability Distribution Untransformed", domain="filling_fraction_space")
# f_2 is the fraction of active material particles that have filling fraction 'c' that are
# in the region where they follow the phase separated chemical potential
f_2 = pybamm.Variable("Probability Distribution Transforming", domain="filling_fraction_space")
# V_cell is the voltage of the cell
V_cell = pybamm.Variable("Cell Voltage [V]")

# --- PARAMETERS ---
thermal_voltage = pybamm.Parameter("Thermal Voltage [V]")
phi_ref = pybamm.Parameter("Reference Potential [V]")
# the potential at which the phase separation transition occurs
phi_phase_separation = pybamm.Parameter("Phase Separation Potential [V]")
# the interaction parameter that drives phase separation.
# if omega < 2, phase separation does not occur
Omega = pybamm.Parameter("Interaction Parameter")
# the kinetic rate constant
k0 = pybamm.Parameter("Rate Constant")
# parameter that controls the strength of fluctuations
D = pybamm.Parameter("Fluctuations Strength")
# alpha controls the symmetry between charge and discharge kinetics
alpha = 0.5
# the nominal cell capacity
Q_nom = pybamm.Parameter("Nominal cell capacity [A.h]")
# numerical viscosity for advection term
nu = pybamm.Parameter("Numerical Viscosity")
# the transition rate between untransformed to transformed
K_12 = pybamm.Parameter("Transition Rate Untransformed to Transforming")
# exchange current density
ecd = pybamm.FunctionParameter("Exchange current density", {"c": c})

# --- EXCHANGE CURRENT DENSITY FUNCTION ---
# general form of exchange current density
def exchange_current_density(c):
    """Exchange current density: sqrt(c * (1-c)) - symmetric Butler-Volmer form."""
    return np.sqrt(c * (1 - c))

# --- INPUTS ---
# the applied current
I_app = pybamm.InputParameter("Current function [A]")


# --- EQUATIONS ---
# regular solution chemical potential function (nondimensional)
mu_1 = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)
# delta_mu is the overpotential driving force for (de)intercalation reactions (nondimensional)
delta_mu_1 = V_cell / thermal_voltage - phi_ref / thermal_voltage - mu_1
# the exchange current density (nondimensional) - now a FunctionParameter
R0_1 = k0 * ecd
# the butler volmer driving term (nondimensional)
g_1 = pybamm.exp(alpha * delta_mu_1) - pybamm.exp(-(1 - alpha) * delta_mu_1)
# total reaction current density (nondimensional)
R_1 = R0_1 * g_1
# correction term for advection
eps = 1e-10
R_abs_1 = pybamm.sqrt(R_1**2 + eps)
# effective diffusion
D_eff_1 = D + nu * R_abs_1
# if curious, can look up the Lax-Friedrichs scheme for advection
# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms (nondimensional)
diffusive_flux_1 = -D_eff_1 * pybamm.grad(f_1)
# reactions drive the concentration changes within particles (nondimensional)
advective_flux_1 = R_1 * f_1
total_flux_1 = advective_flux_1 + diffusive_flux_1

mu_2 = phi_phase_separation / thermal_voltage
delta_mu_2 = V_cell / thermal_voltage - phi_ref / thermal_voltage - mu_2
# the exchange current density (nondimensional)
R0_2 = k0 * ecd
# the butler volmer driving term (nondimensional)
g_2 = pybamm.exp(alpha * delta_mu_2) - pybamm.exp(-(1 - alpha) * delta_mu_2)
# total reaction current density (nondimensional)
R_2 = R0_2 * g_2
# correction term for advection
eps = 1e-10
R_abs_2 = pybamm.sqrt(R_2**2 + eps)
# effective diffusion
D_eff_2 = nu * R_abs_2
# if curious, can look up the Lax-Friedrichs scheme for advection
# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms (nondimensional)
diffusive_flux_2 = -D_eff_2 * pybamm.grad(f_2)
# reactions drive the concentration changes within particles (nondimensional)
advective_flux_2 = R_2 * f_2
# total flux (nondimensional)
total_flux_2 = advective_flux_2 + diffusive_flux_2

# conservation law
model.rhs = {
    f_1: -pybamm.div(total_flux_1) - K_12 * f_1,
    f_2: -pybamm.div(total_flux_2) + K_12 * f_1
}

# current constraint
model.algebraic = {
    V_cell: pybamm.Integral(f_1 * R_1, c) + pybamm.Integral(f_2 * R_2, c) - I_app / Q_nom
}

# --- BOUNDARY AND INITIAL CONDITIONS ---
# zero flux at boundaries (Conservation of Probability)
model.boundary_conditions = {
    f_1: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    },
    f_2: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# initial condition: gaussian centered at start_c
start_c = pybamm.Parameter("Initial Filling Fraction")
width = 0.01
f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
model.initial_conditions = {
    f_1: f_init / (width * np.sqrt(2 * np.pi)), 
    f_2: 0,
    V_cell: pybamm.Scalar(3.422)
}

# add stopping events to prevent solver divergence near domain boundaries
# these events stop the simulation when average filling fraction approaches singularities
c_avg = pybamm.Integral(f_1 * c, c) + pybamm.Integral(f_2 * c, c)  # average filling fraction in the population

# set conservative limits based on domain boundaries (c ∈ [1e-4, 1-1e-4])
# stop when within safe margins of the boundaries to maintain numerical stability
model.events = [
    pybamm.Event("Maximum stoichiometry", 0.95 - c_avg),  # stop if <c> > 0.95
    pybamm.Event("Minimum stoichiometry", c_avg - 0.05),  # stop if <c> < 0.05
    pybamm.Event("Maximum voltage", 4.2 - V_cell),  # stop if V > 4.2V
    pybamm.Event("Minimum voltage", V_cell - 2.5),  # stop if V < 2.5V
]

# --- INPUT PARAMETERS ---
constant_current_value = 0.23   # input current in Amperes
start_SOC = 0.05                # initial SOC of the population
n_points = 200
dc = 1.0 / n_points


# --- PARAMETER VALUES ---
params = pybamm.ParameterValues({
    "Nominal cell capacity [A.h]": 2.3,
    "Interaction Parameter": 4.5,
    "Rate Constant": 1.0,
    "Thermal Voltage [V]": 0.025,
    "Fluctuations Strength": 2e-4,
    "Initial Filling Fraction": start_SOC,
    "Reference Potential [V]": 3.422,
    "Phase Separation Potential [V]": 3.422,
    "Current function [A]": constant_current_value,
    "Numerical Viscosity": dc,
    "Transition Rate Untransformed to Transforming": 1e-2,
    "Exchange current density": exchange_current_density,
})

# --- DISCRETIZATION ---
model_proc = params.process_model(model, inplace=False)
submesh_types = {"filling_fraction_space": pybamm.Uniform1DSubMesh}
var_pts = {c: n_points} # 200 grid points
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"filling_fraction_space": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model_proc)

t_eval = np.linspace(0, 9.4, 1000)

# --- SOLVE ---
solver = pybamm.IDAKLUSolver(
    atol=1e-6,
    rtol=1e-6
)
solution = solver.solve(
    model_proc, 
    t_eval,
    inputs={"Current function [A]": constant_current_value}
)

# plotting filling fraction of population versus SOC
c_vals = mesh["filling_fraction_space"].nodes  # filling fraction c (x-axis)
f_1_sol = solution["Probability Distribution Untransformed"].entries # probability density f (color)
f_2_sol = solution["Probability Distribution Transforming"].entries # probability density f (color)

# Compute average filling fraction (analogous to SOC) from the distribution
dx = c_vals[1] - c_vals[0]
c_avg = np.sum(c_vals[:, np.newaxis] * f_1_sol, axis=0) * dx + np.sum(c_vals[:, np.newaxis] * f_2_sol, axis=0) * dx

mass_1 = np.sum(f_1_sol, axis=0) * dx
f_1_sol = f_1_sol / mass_1[np.newaxis, :]
mass_2 = np.sum(f_2_sol, axis=0) * dx
f_2_sol = f_2_sol / mass_2[np.newaxis, :]

# lithiation or delithiation
if constant_current_value < 0:
    y_data = 1 - c_avg
    y_label = r"Filling Fraction $1 - \langle c \rangle$ (Delithiation)"
else:
    y_data = c_avg
    y_label = r"Filling Fraction $\langle c \rangle$ (Lithiation)"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot f_1 (Untransformed)
pcm1 = ax1.pcolormesh(
    c_vals, 
    y_data, 
    f_1_sol.T, 
    cmap='coolwarm',       
    shading='gouraud',
    vmin=0, 
    vmax=10
)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"Local Mole Fraction $c$")
ax1.set_ylabel(y_label)
ax1.set_title(r"$f_1$ Untransformed Population")
fig.colorbar(pcm1, ax=ax1, label=r"$f_1(c)$")

# Plot f_2 (Transforming)
pcm2 = ax2.pcolormesh(
    c_vals, 
    y_data, 
    f_2_sol.T, 
    cmap='coolwarm',       
    shading='gouraud',
    vmin=0, 
    vmax=10
)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r"Local Mole Fraction $c$")
ax2.set_ylabel(y_label)
ax2.set_title(r"$f_2$ Transforming Population")
fig.colorbar(pcm2, ax=ax2, label=r"$f_2(c)$")

fig.suptitle(
    fr"Coupled Population Dynamics" + f"\nCurrent={constant_current_value}[A], " + 
    fr"$\Omega={params['Interaction Parameter']}$, $K_{{12}}={params['Transition Rate Untransformed to Transforming']}$",
    fontsize=12
)

plt.tight_layout()
plt.show()

# plot cell voltage vs. filling fraction
plt.figure(figsize=(7, 5))
V_sol = solution["Cell Voltage [V]"]
plt.plot(c_avg, V_sol.entries, 'k-', linewidth=2)
plt.xlabel(r"Filling Fraction $\langle c \rangle$")
plt.ylabel(r"Cell Voltage [V]")
plt.title("Cell Voltage vs. Filling Fraction")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()