import pybamm
import numpy as np
import matplotlib.pyplot as plt

model = pybamm.BaseModel(name="SinglePopulation_ConstantCurrent")

# filling fraction coordinate 'c' from 0 to 1
c = pybamm.SpatialVariable("c", domain="filling_fraction_space", coord_sys="cartesian")
geometry = {"filling_fraction_space": {c: {"min": pybamm.Scalar(1e-4), "max": pybamm.Scalar(1-1e-4)}}}

# solving for the probability density function f and the reservoir chemical potential mu_res
# f is the fraction of active material particles that have filling fraction 'c'
f = pybamm.Variable("Probability Distribution", domain="filling_fraction_space")
# mu_res is the chemical potential of the reservoir, which is related to the voltage of the cell
mu_res = pybamm.Variable("Reservoir Chemical Potential")

# the interaction parameter that drives phase separation.
# if omega < 2, phase separation does not occur
Omega = pybamm.Parameter("Interaction Parameter Omega")
# the applied current density (nondimensional)
I_app = pybamm.Parameter("Applied Current")
# the rate constant for the kinetics
k0 = pybamm.Parameter("Rate Constant")
# parameter that controls the strength of the thermal fluctuations
D = pybamm.Parameter("Fluctuation Strength D")
# alpha controls the symmetry between charge and discharge kinetics
alpha = 0.5 

# regular solution chemical potential function
mu_particle = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)

# delta_mu is the overpotential driving force for (de)intercalation reactions
delta_mu = mu_res - mu_particle
# the butler volmer symmetric exchage current density
R0 = k0 * pybamm.sqrt(c) * (1 - pybamm.sqrt(c))
# the butler volmer sinh term
g_driving = pybamm.exp(alpha * delta_mu) - pybamm.exp(-(1 - alpha) * delta_mu)
# total reaction current density
R = R0 * g_driving

# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms
# diffusive flux is related to thermal fluctuations
diffusive_flux = -D * pybamm.grad(f)
# reactions drive the concentration changes within particles
advective_flux = f * R
total_flux = advective_flux + diffusive_flux

# conservation Law
# df/dt = -div(j)
model.rhs = {f: -pybamm.div(total_flux)}

# constant current constraint
model.algebraic = {
    mu_res: pybamm.Integral(f * R, c) - I_app
}

# boundary and initial conditions
# zero flux at boundaries (Conservation of Probability)
model.boundary_conditions = {
    f: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# initial condition: gaussian centered at start_c
start_c = pybamm.Parameter("Initial Average Fraction")
width = 0.01
f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
model.initial_conditions = {
    f: f_init / (width * np.sqrt(2 * np.pi)), 
    mu_res: pybamm.Scalar(0)
}

# simulation parameters
current_value = 0.05  # this is the C-rate of the cell
start_val = 0.04      # what is the initial filling fraction of the population?
target_val = 0.99     # stop the simulation before the population is fully lithiated

params = pybamm.ParameterValues({
    "Interaction Parameter Omega": 4.5,    # omega > 2 for phase separation
    "Applied Current": current_value,
    "Rate Constant": 1.0,
    "Fluctuation Strength D": 2e-4,       # small thermal noise
    "Initial Average Fraction": start_val
})

model_proc = params.process_model(model, inplace=False)
submesh_types = {"filling_fraction_space": pybamm.Uniform1DSubMesh}
var_pts = {c: 200} # 200 grid points
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"filling_fraction_space": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model_proc)

# calculate simulation time for a single discharge
total_duration = abs(target_val - start_val) / abs(current_value)
t_eval = np.linspace(0, total_duration, 400)

solver = pybamm.CasadiSolver(mode="safe", dt_max=total_duration/100)
solution = solver.solve(model_proc, t_eval)

# plotting filling fraction of population versus SOC
c_vals = mesh["filling_fraction_space"].nodes  # filling fraction c (x-axis)
f_sol = solution["Probability Distribution"].entries # probability density f (color)

dx = c_vals[1] - c_vals[0]
c_avg = np.sum(c_vals[:, np.newaxis] * f_sol, axis=0) * dx

mass = np.sum(f_sol, axis=0) * dx
f_sol = f_sol / mass[np.newaxis, :]

# lithiation or delithiation
if current_value < 0:
    y_data = 1 - c_avg
    y_label = r"Average Fraction $1 - \langle c \rangle$ (Delithiation)"
else:
    y_data = c_avg
    y_label = r"Average Fraction $\langle c \rangle$ (Lithiation)"


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
plt.title(fr"Population Dynamics" + f"\nC-rate={current_value}, " + fr"$\Omega={params['Interaction Parameter Omega']}$", pad=20)
plt.colorbar(pcm, label=r"f($c$) Probability Density of Particle Having Fraction $c$")

plt.tight_layout()
plt.show()

# plot reservoir chemical potential vs. average fraction
plt.figure(figsize=(7, 5))
mu_res_sol = solution["Reservoir Chemical Potential"].entries
plt.plot(c_avg, mu_res_sol, 'k-', linewidth=2)
plt.xlabel(r"Average Fraction $\langle c \rangle$")
plt.ylabel(r"Reservoir Chemical Potential $\mu_{res}$")
plt.title("Reservoir Chemical Potential vs. Filling Fraction")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# we see here that the reservoir chemical potential (Voltage) is somewhat nonsensical because the effects of the onset
# of phase separation lead to a jump in chemical potential from the regular solution chemical potential to that of the
# phase separated particles. we will introduce a new population of particles that is the "phase separated" population
# instead of using just a single population of particles that follow the regular solution chemical potential we define the
# phase separated population chemical potential to have constant mu_ref_phase_separated since phase separation is a constant
# potential process.