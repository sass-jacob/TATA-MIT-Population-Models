# Change K_12 to be potential dependent
import pybamm
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("[CHECKPOINT 1] Imports complete")
print("=" * 50)

model = pybamm.BaseModel(name="CoupledPopulation")
print("[CHECKPOINT 2] Model created")

# --- VARIABLES ---
# filling fraction coordinate 'c' from 0 to 1
c = pybamm.SpatialVariable("c", domain="filling_fraction_space", coord_sys="cartesian")
geometry = {"filling_fraction_space": {c: {"min": pybamm.Scalar(1e-5), "max": pybamm.Scalar(1-1e-5)}}}
# f_1 is the fraction of active material particles that have filling fraction 'c' that are
# in the region where they follow the regular solution chemical potential
f_1 = pybamm.Variable("Probability Distribution Untransformed", domain="filling_fraction_space")
# f_2 is the fraction of active material particles that have filling fraction 'c' that are
# in the region where they follow the phase separated chemical potential
f_2 = pybamm.Variable("Probability Distribution Transforming", domain="filling_fraction_space")
# f_3 is the fraction of particles in the final "sink" state
f_3 = pybamm.Variable("Probability Distribution Sink", domain="filling_fraction_space")
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
# exchange current density
ecd = pybamm.FunctionParameter("Exchange current density", {"c": c})

# --- PARAMETERS FOR CURRENT-DEPENDENT TRANSITIONS ---
Da_p = pybamm.Parameter("Process Damkohler Number")  
Da_w = pybamm.Parameter("Wiring Damkohler Number")
c_s_star = pybamm.Parameter("Nomminal nucleation concentration")   # c_s*
delta_c_s = pybamm.Parameter("Miscibility Gap")      # Δc_s
delta_v_g = pybamm.Parameter("Voltage Gap")  # Δṽ

# --- EXCHANGE CURRENT DENSITY FUNCTION ---
# general form of exchange current density
def exchange_current_density(c):
    # ECIT form
    return np.sqrt(c)* (1 - c)

# --- INPUTS ---
# the applied current
I_app = pybamm.InputParameter("Current function [A]")

eps_int = 1e-10
t_ref = 2.3 / 0.23  
# --- EQUATIONS ---
# regular solution chemical potential function (nondimensional)
mu_1 = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)
# delta_mu is the overpotential driving force for (de)intercalation reactions (nondimensional)
delta_mu_1 = V_cell / thermal_voltage - phi_ref / thermal_voltage + mu_1
# the exchange current density (nondimensional) - now a FunctionParameter
R0_1 = k0 * ecd
# the butler volmer driving term (nondimensional)
g_1 = -delta_mu_1
# total reaction current density (nondimensional)
R_1 = R0_1 * g_1 *t_ref
# correction term for advection
eps = 1e-10
R_abs_1 = pybamm.sqrt(R_1**2 + eps)
# effective diffusion
D_eff_1 = D + nu * R_abs_1
# if curious, can look up the Lax-Friedrichs scheme for advection
# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms (nondimensional)
diffusive_flux_1 = -D_eff_1 * pybamm.grad(f_1) *t_ref
# reactions drive the concentration changes within particles (nondimensional)
advective_flux_1 = R_1 * f_1
total_flux_1 = advective_flux_1 + diffusive_flux_1

mu_2 = phi_phase_separation / thermal_voltage
delta_mu_2 = V_cell / thermal_voltage - phi_ref / thermal_voltage + mu_2
# the exchange current density (nondimensional)
R0_2 = k0 * exchange_current_density(0.5)
# the butler volmer driving term (nondimensional)
g_2 = -delta_mu_2
# total reaction current density (nondimensional)
R_2 = R0_2 * g_2 *t_ref
# correction term for advection
eps = 1e-10
R_abs_2 = pybamm.sqrt(R_2**2 + eps)
# effective diffusion
D_eff_2 = D + nu * R_abs_2
# if curious, can look up the Lax-Friedrichs scheme for advection
# for the fokker planck equation, the flux is defined by the sum of diffusive and advective terms (nondimensional)
diffusive_flux_2 = -D_eff_2 * pybamm.grad(f_2) *t_ref
# reactions drive the concentration changes within particles (nondimensional)
advective_flux_2 = R_2 * f_2
# total flux (nondimensional)
total_flux_2 = advective_flux_2 + diffusive_flux_2

# --- COMPUTE IMPEDANCES AND TRANSITION TERMS ---
# β coefficients
# beta_u = 1 / (Da_w * exchange_current_density(0.5) * c_s_star)
# beta_a = 1 / delta_c_s
beta_u = 100
beta_a = beta_u/(beta_u - 1)

# Regularized f values to prevent division by zero
f_1_reg = pybamm.Maximum(f_1, pybamm.Scalar(eps_int))
f_2_reg = pybamm.Maximum(f_2, pybamm.Scalar(eps_int))

lambda_u = pybamm.sqrt(Da_p * exchange_current_density(0.5) * f_1_reg)
lambda_a = pybamm.sqrt(Da_p * exchange_current_density(0.5) * f_2_reg)
# Z impedances (using regularized f values to avoid division by zero)
Z_u = (1 / (Da_p * exchange_current_density(0.5) * f_1_reg)) * (lambda_u / pybamm.tanh(lambda_u + eps_int))
Z_a = (1 / (Da_p * exchange_current_density(0.5) * f_2_reg)) * (lambda_a / pybamm.tanh(lambda_a + eps_int))
Z_a_p = Z_a / pybamm.cosh(lambda_a + eps_int)
# Current-dependent transition products (Ĩ = I_app)
I_tilde = I_app * t_ref / Q_nom  # normalized current
denominator = Z_u + Z_a_p + eps_int

# I_a * f_a and I_u * f_u products
Ia_fa_product = (I_tilde * Z_u + delta_v_g) / denominator
Iu_fu_product = (I_tilde * Z_a_p - delta_v_g) / denominator

# conservation law with current-dependent transitions
model.rhs = {
    f_1: -pybamm.div(total_flux_1) - beta_u * Iu_fu_product,
    f_2: -pybamm.div(total_flux_2) + beta_u * Iu_fu_product - beta_a * Ia_fa_product,
    f_3: beta_a * Ia_fa_product
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
    },
    f_3: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# initial condition: gaussian centered at start_c
start_c = pybamm.Parameter("Initial Filling Fraction")
width = 0.02
f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
model.initial_conditions = {
    f_1: f_init / (width * np.sqrt(2 * np.pi)), 
    f_2: pybamm.Scalar(1e-10),  # Small positive value to avoid division by zero
    f_3: pybamm.Scalar(1e-10),  # Small positive value
    V_cell: pybamm.Scalar(3.422)
}

# add stopping events to prevent solver divergence near domain boundaries
# these events stop the simulation when average filling fraction approaches singularities
c_avg = pybamm.Integral(f_1 * c, c) + pybamm.Integral(f_2 * c, c) + pybamm.Integral(f_3 * c, c)  # average filling fraction in the population

# set conservative limits based on domain boundaries (c ∈ [1e-4, 1-1e-4])
# stop when within safe margins of the boundaries to maintain numerical stability
model.events = [
    pybamm.Event("Maximum stoichiometry", 0.99 - c_avg),  # stop if <c> > 0.95
    pybamm.Event("Minimum stoichiometry", c_avg - 0.01),  # stop if <c> < 0.05
    pybamm.Event("Maximum voltage", 4.2 - V_cell),  # stop if V > 4.2V
    pybamm.Event("Minimum voltage", V_cell - 2.5),  # stop if V < 2.5V
]

print("[CHECKPOINT 3] Variables, parameters, and equations defined")

# --- INPUT PARAMETERS ---
constant_current_value = 0.23   # input current in Amperes
start_SOC = 0.02                # initial SOC of the population
n_points = 200
dc = 1.0 / n_points

# Calculate parameters for prediction
L = 100.0 * 1e-6
poros = 0.4
# t_ref = ((L/1e-4)**2)*(31.168)
R_p = 100e-9  # 100 nm particle radius
thickness = 20e-9  # 20 nm particle thickness
a_p = (2/(thickness))*(1-poros)

k0_val = 1.6e-1  # numeric value for calculations (different from pybamm.Parameter k0)
P_L = 0.69
rho_s = 1.3793e28

# Protect against division by zero for very small currents
# Da_proc = k0_val*a_p*3600/(P_L*abs(constant_current_value)*(1-poros)*rho_s*1.602e-19)
# Da_wire = k0_val*a_p*L**2/(poros**1.5*0.13)
# Da_proc_mod = k0_val*a_p*L**2*(1-0.38)/(poros**1.5*1000*96485*10**(-9.65))
# Da_lim = Da_p/Da_proc  # Commented out - mixing symbolic and numeric
# Q_nom / I_app in numeric form (t_ref is symbolic, need numeric version)

Da_proc = t_ref * k0_val * a_p / (P_L*96485*22897)
Da_wire = k0_val * a_p * L**2 / (poros**1.5 * 0.13)
c_nuc = 0.03
def LFP_OCV(x):
    OCV = np.zeros_like(x)
   
    OCV = 3.422 - 0.0257*np.log(x) + 0.0257*np.log(1-x) - 4.51*0.0257*(1-2*x)
    return OCV
Vg = 0.1*abs(LFP_OCV(c_nuc)-LFP_OCV(0.5))/0.0257
# --- PARAMETER VALUES ---
params = pybamm.ParameterValues({
    "Nominal cell capacity [A.h]": 2.3,
    "Interaction Parameter": 4.5,
    "Rate Constant": 0.5,
    "Thermal Voltage [V]": 0.0257,
    "Fluctuations Strength": 2e-3,
    "Initial Filling Fraction": start_SOC,
    "Reference Potential [V]": 3.422,
    "Phase Separation Potential [V]": 3.422,
    "Current function [A]": constant_current_value,
    "Numerical Viscosity": dc,
    "Exchange current density": exchange_current_density,
    # Parameters for current-dependent transitions
    "Process Damkohler Number": Da_proc,   
    "Wiring Damkohler Number": Da_wire,             
    "Nomminal nucleation concentration": 0.01,           
    "Miscibility Gap": (0.95 - 0.015),               
    "Voltage Gap": Vg                
})
print("[CHECKPOINT 4] Parameter values set")

# --- DISCRETIZATION ---
print("[CHECKPOINT 5] Starting discretization...")
model_proc = params.process_model(model, inplace=False)
print("  - Model processed")
submesh_types = {"filling_fraction_space": pybamm.Uniform1DSubMesh}
var_pts = {c: n_points} # 200 grid points
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)
spatial_methods = {"filling_fraction_space": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
print("  - Discretisation object created")
disc.process_model(model_proc)
print("  - Model discretized")
print("[CHECKPOINT 6] Discretization complete")

t_eval = np.linspace(0, 10*t_ref, 1000)
print(f"[CHECKPOINT 7] Time array created: {len(t_eval)} points, t_max = {t_eval[-1]:.2f}")
# --- SOLVE ---
print("[CHECKPOINT 8] Creating solver...")
solver = pybamm.IDAKLUSolver(
    atol=1e-6,
    rtol=1e-6
)
print("[CHECKPOINT 9] Starting solve... (this may take a while)")
solution = solver.solve(
    model_proc, 
    t_eval,
    inputs={"Current function [A]": constant_current_value}
)
print("[CHECKPOINT 10] Solve complete!")

print("[CHECKPOINT 11] Extracting solution data...")
# plotting filling fraction of population versus SOC
c_vals = mesh["filling_fraction_space"].nodes  # filling fraction c (x-axis)
f_1_sol = solution["Probability Distribution Untransformed"].entries # probability density f (color)
f_2_sol = solution["Probability Distribution Transforming"].entries # probability density f (color)
f_3_sol = solution["Probability Distribution Sink"].entries # probability density f (color)
# Compute average filling fraction (analogous to SOC) from the distribution
dx = c_vals[1] - c_vals[0]
c_avg = np.sum(c_vals[:, np.newaxis] * f_1_sol, axis=0) * dx + np.sum(c_vals[:, np.newaxis] * f_2_sol, axis=0) * dx + np.sum(c_vals[:, np.newaxis] * f_3_sol, axis=0) * dx

mass_1 = np.sum(f_1_sol, axis=0) * dx
f_1_sol = f_1_sol / mass_1[np.newaxis, :]
mass_2 = np.sum(f_2_sol, axis=0) * dx
f_2_sol = f_2_sol / mass_2[np.newaxis, :]
mass_3 = np.sum(f_3_sol, axis=0) * dx
f_3_sol = f_3_sol / mass_3[np.newaxis, :]

# lithiation or delithiation
if constant_current_value < 0:
    y_data = 1 - c_avg
    y_label = r"Filling Fraction $1 - \langle c \rangle$ (Delithiation)"
else:
    y_data = c_avg
    y_label = r"Filling Fraction $\langle c \rangle$ (Lithiation)"

print("[CHECKPOINT 12] Creating plots...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Plot f_1 (Untransformed)
pcm1 = ax1.pcolormesh(
    c_vals, 
    y_data, 
    f_1_sol.T, 
    cmap='coolwarm',       
    shading='auto',
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
    shading='auto',
    vmin=0, 
    vmax=10
)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel(r"Local Mole Fraction $c$")
ax2.set_ylabel(y_label)
ax2.set_title(r"$f_2$ Transforming Population")
fig.colorbar(pcm2, ax=ax2, label=r"$f_2(c)$")

# Plot f_3 (Sink)
pcm3 = ax3.pcolormesh(
    c_vals, 
    y_data, 
    f_3_sol.T, 
    cmap='coolwarm',       
    shading='auto',
    vmin=0, 
    vmax=10
)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_xlabel(r"Local Mole Fraction $c$")
ax3.set_ylabel(y_label)
ax3.set_title(r"$f_3$ Sink Population")
fig.colorbar(pcm3, ax=ax3, label=r"$f_3(c)$")

fig.suptitle(
    fr"Coupled Population Dynamics" + f"\nCurrent={constant_current_value}[A], " + 
    fr"$\Omega={params['Interaction Parameter']}$",
    fontsize=12
)

plt.tight_layout()
plt.show()

print("[CHECKPOINT 13] Creating voltage plot...")
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
print("=" * 50)
print("[CHECKPOINT 14] Script completed successfully!")
print("=" * 50)