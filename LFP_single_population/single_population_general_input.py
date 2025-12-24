import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel(name="SinglePopulation")

# --- VARIABLES ---
# filling fraction coordinate 'c' from 0 to 1
c = pybamm.SpatialVariable("c", domain="filling_fraction_space", coord_sys="cartesian")
geometry = {"filling_fraction_space": {c: {"min": pybamm.Scalar(1e-3), "max": pybamm.Scalar(1-1e-3)}}}
# f is the fraction of active material particles that have filling fraction 'c'
f = pybamm.Variable("Probability Distribution", domain="filling_fraction_space")
# V_cell is the voltage of the cell
V_cell = pybamm.Variable("Cell Voltage [V]")

# --- PARAMETERS ---
thermal_voltage = pybamm.Parameter("Thermal Voltage [V]")
phi_ref = pybamm.Parameter("Reference Potential [V]")
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


# --- INPUTS ---
# the applied current
I_app = pybamm.InputParameter("Current function [A]")


# --- EQUATIONS ---
# regular solution chemical potential function (nondimensional)
mu_particle = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)
# delta_mu is the overpotential driving force for (de)intercalation reactions (nondimensional)
delta_mu = V_cell / thermal_voltage - phi_ref / thermal_voltage - mu_particle
# the exchange current density (nondimensional)
R0 = k0 * pybamm.sqrt((1 - c) * c)
# the butler volmer driving term (nondimensional)
g_driving = pybamm.exp(alpha * delta_mu) - pybamm.exp(-(1 - alpha) * delta_mu)
# total reaction current density (nondimensional)
R = R0 * g_driving

# correction term for advection
eps = 1e-10
R_abs = pybamm.sqrt(R**2 + eps)
# effective diffusion
D_eff = D + nu * R_abs
# if curious, can look up the Lax-Friedrichs scheme for advection

# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms (nondimensional)
diffusive_flux = -D_eff * pybamm.grad(f)
# reactions drive the concentration changes within particles (nondimensional)
advective_flux = R * f
total_flux = advective_flux + diffusive_flux

# conservation law
model.rhs = {f: -pybamm.div(total_flux)}

# current constraint
model.algebraic = {
    V_cell: pybamm.Integral(f * R, c) - I_app / Q_nom
}

# --- BOUNDARY AND INITIAL CONDITIONS ---
# zero flux at boundaries (Conservation of Probability)
model.boundary_conditions = {
    f: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# initial condition: gaussian centered at start_c
start_c = pybamm.Parameter("Initial Filling Fraction")
width = 0.01
f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
model.initial_conditions = {
    f: f_init / (width * np.sqrt(2 * np.pi)), 
    V_cell: pybamm.Scalar(3.422)
}

# add stopping events to prevent solver divergence near domain boundaries
# these events stop the simulation when average filling fraction approaches singularities
c_avg = pybamm.Integral(f * c, c)  # average filling fraction in the population

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
    "Current function [A]": constant_current_value,
    "Numerical Viscosity": dc,
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
f_sol = solution["Probability Distribution"].entries # probability density f (color)

# Compute average filling fraction (analogous to SOC) from the distribution
dx = c_vals[1] - c_vals[0]
c_avg = np.sum(c_vals[:, np.newaxis] * f_sol, axis=0) * dx

mass = np.sum(f_sol, axis=0) * dx
f_sol = f_sol / mass[np.newaxis, :]

# lithiation or delithiation
if constant_current_value < 0:
    y_data = 1 - c_avg
    y_label = r"Filling Fraction $1 - \langle c \rangle$ (Delithiation)"
else:
    y_data = c_avg
    y_label = r"Filling Fraction $\langle c \rangle$ (Lithiation)"

plt.figure(figsize=(7, 6))

C_grid, Y_grid = np.meshgrid(c_vals, y_data)

pcm = plt.pcolormesh(
    c_vals, 
    y_data, 
    f_sol.T, 
    cmap='coolwarm',       
    shading='gouraud',
    vmin=0, 
    vmax=10
)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r"Local Mole Fraction $c$")
plt.ylabel(y_label)
plt.title(fr"Population Dynamics" + f"\nCurrent={constant_current_value}[A], " + fr"$\Omega={params['Interaction Parameter']}$", pad=20)
plt.colorbar(pcm, label=r"f($c$) Probability Density of Particle Having Fraction $c$")

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

# we see here that the voltage is somewhat nonsensical because the effects of the onset
# of phase separation lead to a jump in chemical potential from the regular solution chemical potential to that of the
# phase separated particles. we will introduce a new population of particles that is the "phase separated" population
# instead of using just a single population of particles that follow the regular solution chemical potential we define the
# phase separated population chemical potential to have constant mu_ref_phase_separated since phase separation is a constant
# potential process.