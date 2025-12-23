import pybamm
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Model Definition ---
model = pybamm.BaseModel(name="Zhao2019_FokkerPlanck_ConstantCurrent")

# --- 2. Geometry and Variables ---
# The "space" is the mole fraction coordinate 'c' from 0 to 1
c = pybamm.SpatialVariable("c", domain="mole_fraction_space", coord_sys="cartesian")
geometry = {"mole_fraction_space": {c: {"min": pybamm.Scalar(1e-4), "max": pybamm.Scalar(1-1e-4)}}}

# Variables
f = pybamm.Variable("Probability Distribution", domain="mole_fraction_space")
mu_res = pybamm.Variable("Reservoir Chemical Potential") # Algebraic variable (Unknown)

# --- 3. Parameters ---
# Physical constants
# Physical constants
Omega = pybamm.Parameter("Interaction Parameter Omega")
# We use FunctionParameter to allow I_app to vary with time (or other variables)
I_app = pybamm.FunctionParameter("Applied Current", {"Time": pybamm.t}) # Target total rate d<c>/dt
k0 = pybamm.Parameter("Rate Constant")
D = pybamm.Parameter("Fluctuation Strength D")
alpha = 0.5 

# --- 4. Physics Equations ---

# Thermodynamic Chemical Potential (Regular Solution Model) [cite: 1294]
# mu = ln(c/(1-c)) + Omega*(1-2c)
mu_particle = pybamm.log(c / (1 - c)) + Omega * (1 - 2 * c)

# Reaction Kinetics (Butler-Volmer) [cite: 1266, 1278]
# Drift Velocity R = R0 * g(delta_mu)
delta_mu = mu_res - mu_particle
R0 = k0 * c * (1 - c) # Exchange current density
g_driving = pybamm.exp(alpha * delta_mu) - pybamm.exp(-(1 - alpha) * delta_mu)
R = R0 * g_driving

# Flux Definition (Eq. 20) [cite: 1204]
# Flux j = Advection (f*R) + Diffusion (-D*grad(f))
diffusive_flux = -D * pybamm.grad(f)
advective_flux = f * R
total_flux = advective_flux + diffusive_flux

# Conservation Law
# df/dt = -div(j)
model.rhs = {f: -pybamm.div(total_flux)}

# --- 5. Constant Current Constraint ---
# We need to find mu_res such that the total rate of change equals I_app
# d<c>/dt = Integral(total_flux) = I_app [cite: 1255]
# Note: Since boundaries are zero-flux, Integral(div(j)) is 0, but 
# the rate of change of the *mean* depends on the integral of the flux itself.
# We reformulate the flux integral to avoid summing over edges (which raises ShapeError):
# Integral(J) dc = Integral(f*R - D*grad(f)) dc = Integral(f*R) dc - D*(f(1) - f(0))
model.algebraic = {
    mu_res: pybamm.Integral(f * R, c) 
#    - D * (pybamm.boundary_value(f, "right") - pybamm.boundary_value(f, "left")) 
    - I_app
}

# --- 6. Boundary and Initial Conditions ---
# Zero Flux at boundaries (Conservation of Probability)
model.boundary_conditions = {
    f: {
        "left": (pybamm.Scalar(0), "Neumann"),
        "right": (pybamm.Scalar(0), "Neumann"),
    }
}

# Initial Condition: Gaussian centered at start_c
# We define start_c as a parameter to easily switch between Lithiation/Delithiation
start_c = pybamm.Parameter("Initial Average Fraction")
width = 0.01
f_init = pybamm.exp(-((c - start_c)**2) / (2 * width**2))
# Normalize initial guess roughly (solver adjusts)
model.initial_conditions = {
    f: f_init / (width * np.sqrt(2 * np.pi)), 
    mu_res: pybamm.Scalar(0)
}

# --- 7. Simulation Setup ---

# Simulation Parameters matching Figure 3 (Phase Separation) [cite: 1562]
current_value = 0.05  # Positive for Lithiation
start_val = 0.04      # Empty particle
target_val = 0.99     # Stop before "full" singularity at c=1

# Define a variable current function
def current_function(t):
    # Example: Sinusoidal current
    # Period T = 1, Amplitude = 0.5
    I_0 = 0.5
    omega = 2 * np.pi
    return I_0 * pybamm.sin(omega * t)

params = pybamm.ParameterValues({
    "Interaction Parameter Omega": 4.5,    # Omega > 2 for phase separation
    "Applied Current": current_function,   # Pass the function here
    "Rate Constant": 1.0,
    "Fluctuation Strength D": 2e-4,       # Low noise to see sharp peaks
    "Initial Average Fraction": start_val
})

# Process Model
model_proc = params.process_model(model, inplace=False)

# Meshing
submesh_types = {"mole_fraction_space": pybamm.Uniform1DSubMesh}
var_pts = {c: 200} # 200 grid points
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

# Discretization
spatial_methods = {"mole_fraction_space": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model_proc)

# Calculate Simulation Time
# Time = Capacity / Rate
total_duration = abs(target_val - start_val) / abs(current_value)
t_eval = np.linspace(0, total_duration, 400)

# Solve using Casadi (Required for DAEs with algebraic constraints)
solver = pybamm.CasadiSolver(mode="safe", dt_max=total_duration/100)
solution = solver.solve(model_proc, t_eval)

# --- 8. Data Post-Processing ---

# Extract Data
c_vals = mesh["mole_fraction_space"].nodes  # Local Fraction c (X-axis)
f_sol = solution["Probability Distribution"].entries # Probability f (Color)

# Calculate Average Fraction <c> over time (Y-axis)
# <c>(t) = Integral(c * f(c,t) dc)
dx = c_vals[1] - c_vals[0]
c_avg = np.sum(c_vals[:, np.newaxis] * f_sol, axis=0) * dx

# Normalize f for better plotting (optional, ensures integral is approx 1)
# The solver conserves mass, but finite volume approx might drift slightly
mass = np.sum(f_sol, axis=0) * dx
f_sol = f_sol / mass[np.newaxis, :]

# Determine Plot Axes based on Current Direction (Fig 3 style)
if current_value < 0:
    # Delithiation: Plot vs (1 - <c>) so axis goes 0 -> 1
    y_data = 1 - c_avg
    y_label = r"Average Fraction $1 - \langle c \rangle$ (Delithiation)"
else:
    # Lithiation: Plot vs <c>
    y_data = c_avg
    y_label = r"Average Fraction $\langle c \rangle$ (Lithiation)"

# --- 9. Plotting (Replicating Fig 3/5 Bottom Row) ---
plt.figure(figsize=(7, 6))

# pcolormesh requires X and Y to define the corners of the grid
# We construct a meshgrid for accurate mapping
C_grid, Y_grid = np.meshgrid(c_vals, y_data)

# Plot Heatmap
pcm = plt.pcolormesh(
    c_vals, 
    y_data, 
    f_sol.T, 
    cmap='coolwarm',       # Inverted grayscale (Dark = High Probability) [cite: 1476]
    shading='gouraud',   # Smooth interpolation
    vmin=0, 
    vmax=10              # Saturation limit as noted in Fig 3 caption [cite: 1561]
)

# Formatting
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(r"Local Mole Fraction $c$")
plt.ylabel(y_label)
plt.xlabel(r"Local Mole Fraction $c$")
plt.ylabel(y_label)
plt.title(fr"Population Dynamics Fingerprint" + f"\nVariable Current, " + fr"$\Omega={params['Interaction Parameter Omega']}$", pad=20)
plt.colorbar(pcm, label=r"Probability Density $f(c)$")

plt.tight_layout()

# --- 10. Reservoir Chemical Potential Plot ---
plt.figure(figsize=(7, 5))
mu_res_sol = solution["Reservoir Chemical Potential"].entries
plt.plot(c_avg, mu_res_sol, 'k-', linewidth=2)
plt.xlabel(r"Average Fraction $\langle c \rangle$")
plt.ylabel(r"Reservoir Chemical Potential $\mu_{res}$")
plt.title("Reservoir Chemical Potential vs. Filling Fraction")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()