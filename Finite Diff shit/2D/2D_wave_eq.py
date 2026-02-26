import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# --- Simulation Parameters ---
Nx, Ny = 100, 100           # Grid size
Lx, Ly = 1.0, 1.0           # Physical size (meters)
c = 343                     # Speed of sound (m/s)
dx = Lx / Nx
dy = Ly / Ny
h = dx                      # Spatial step (assuming square cells)
k = 0.5 * h / c             # Time step for stability
T = 0.01                    # Total simulation time
Nt = int(T / k)             # Number of time steps
lambda_sq = (c * k / h) ** 2

# --- State Arrays ---
p = np.zeros((Nx, Ny))
p_prev = np.zeros_like(p)
p_next = np.zeros_like(p)

# --- Jet Input (like air blown into ocarina) ---
f = 1000                    # Frequency of jet (Hz)
A = 0.01                    # Amplitude
input_j = np.arange(Ny//3 + 2, Ny//3 + 4)  # vertical band of jet input

# --- Tone Hole Locations (along top wall for example) ---
tone_holes = [(Nx // 3, Ny - 1), (2 * Nx // 3, Ny - 1)]  # Two tone holes at top

# --- Output Data Storage ---
tone_hole_outputs = {th: [] for th in tone_holes}

# --- Plot Setup ---
fig, ax = plt.subplots()
img = ax.imshow(p.T, cmap='RdBu', vmin=-A, vmax=A, origin='lower', interpolation='none')
ax.set_title("Chladni-like figures")
plt.colorbar(img, ax=ax, label='Pressure')

def update(frame):
    global p, p_prev, p_next

    # FDTD update (interior points)
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            laplacian = (
                p[i+1, j] + p[i-1, j] +
                p[i, j+1] + p[i, j-1] -
                4 * p[i, j]
            )
            p_next[i, j] = 2 * p[i, j] - p_prev[i, j] + lambda_sq * laplacian

    # Neumann boundary conditions (rigid walls)
    p_next[0, :] = p_next[1, :]         # left wall
    p_next[-1, :] = p_next[-2, :]       # right wall
    p_next[:, 0] = p_next[:, 1]         # bottom wall
    p_next[:, -1] = p_next[:, -2]       # top wall

    # Jet input at left boundary (overwrite Neumann)
    t = frame * k
    jet_value = A * np.sin(2 * np.pi * f * t)
    for j in input_j:
        p_next[1, j] = jet_value

    # Tone holes: overwrite with Dirichlet (pressure = 0)
    for (i, j) in tone_holes:
        p_next[i, j] = 0

    # Record pressure at tone holes
    for (i, j) in tone_holes:
        tone_hole_outputs[(i, j)].append(p[i, j])

    # Rotate buffers
    p_prev, p, p_next = p, p_next, p_prev

    # Update plot
    img.set_data(p.T)
    return [img]

ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)
plt.show()

# --- Save Output to WAV Files ---
sample_rate = int(1 / k)
scaling_factor = 32767  # 16-bit PCM scale

for (i, j), data in tone_hole_outputs.items():
    audio = np.array(data)
    if np.max(np.abs(audio)) > 0:
        audio /= np.max(np.abs(audio))  # normalize to -1..1
    audio_int16 = (audio * scaling_factor).astype(np.int16)
    filename = f"tonehole_{i}_{j}.wav"
    write(filename, sample_rate, audio_int16)
    print(f"Saved tone hole output to: {filename}")
