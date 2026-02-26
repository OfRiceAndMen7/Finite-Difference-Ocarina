

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1.0        # Length of tube (meters)
T = 0.01       # Total simulation time (seconds)
c = 343        # Speed of sound in air (m/s)

nx = 1000       # Number of spatial steps
dx = L / nx
dt = 0.5 * dx / c  # Time step (CFL condition)
nt = int(T / dt)   # Number of time steps

# Initialization
u = np.zeros((nx+1, nt+1))  # u[x, t]

# Initial displacement: e.g., Gaussian pulse
x = np.linspace(0, L, nx+1)
u[:, 0] = np.exp(-1000 * (x - 0.3)**2)

# First time step using forward difference
u[1:-1, 1] = u[1:-1, 0] + 0.5 * (c*dt/dx)**2 * (u[2:, 0] - 2*u[1:-1, 0] + u[:-2, 0])

# Time stepping
for n in range(1, nt):
    # Neumann BC at x=0 (closed end)
    u[0, n] = u[1, n]

    # Dirichlet BC at x=L (open end)
    u[-1, n] = 0

    # Update interior points
    u[1:-1, n+1] = 2*u[1:-1, n] - u[1:-1, n-1] + (c*dt/dx)**2 * (u[2:, n] - 2*u[1:-1, n] + u[:-2, n])

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, u[:, 0])
ax.set_ylim(-1.1, 1.1)
ax.set_title("1D Wave in a Tube: One End Closed, One End Open")

def update(frame):
    line.set_ydata(u[:, frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=nt, interval=20)
plt.show()
