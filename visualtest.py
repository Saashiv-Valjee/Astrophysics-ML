from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

# Schwarzschild Metric Parameters
M = 1  # Mass of the black hole
L = 4  # Angular momentum per unit mass of the photon

# Effective potential
def V_squared(r, M, L):
    return (1 - 2 * M / r) * (L ** 2) / (r ** 2)

# The system of ODEs
def equations(y, lambda_, M, L):
    r, dr_dlambda, phi = y
    d2r_dlambda2 = (2 * M / (r ** 2)) * (1 - 2 * M / r) * (L ** 2 / (r ** 2)) - (L ** 2) / (r ** 3) + (2 * M * L ** 2) / (r ** 4)
    dphi_dlambda = L / (r ** 2)
    return [dr_dlambda, d2r_dlambda2, dphi_dlambda]

# Initial conditions: [r, dr/dlambda, phi]
y0 = [5.0, 0.0, 0.0]  # Start at r=5, with dr/dlambda=0 (turning point), phi=0
lambda_ = np.linspace(0, 50, 500)  # Affine parameter range

# Numerical integration of the equations of motion
solution = odeint(equations, y0, lambda_, args=(M, L))

# Extract the results
r = solution[:, 0]
phi = solution[:, 2]

# Convert to Cartesian coordinates for plotting
x = r * np.cos(phi)
y = r * np.sin(phi)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.scatter([0], [0], color='black')  # Black hole at the origin
plt.xlabel('x')
plt.ylabel('y')
plt.title('Photon Trajectory Around a Black Hole')
plt.grid(True)
plt.show()
