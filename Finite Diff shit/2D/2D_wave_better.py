import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# --- Simulation Parameters ---
Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
c = 343
dx = Lx / Nx
dy = Ly / Ny
h = dx
k = 0.5 * h / c
T = 0.2  # simulation duration in seconds
Nt = int(T / k)
lambda_sq = (c * k / h) ** 2

# --- State Arrays ---
p = np.zeros((Nx, Ny))
p_prev = np.zeros_like(p)
p_next = np.zeros_like(p)

# --- Input and Output ---
f = 1000
A = 0.01
input_j = np.arange(Ny // 2 - 2, Ny // 2 + 2)
tone_holes = [(Nx // 3, Ny - 1), (2 * Nx // 3, Ny - 1)]
tone_hole_outputs = {th: [] for th in tone_holes}

# --- Optional: Store pressure frames for animation ---
store_frames = True
frames_to_plot = []

# --- Simulation Loop (no animation yet) ---
for n in range(Nt):
    # Update interior
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            laplacian = (
                p[i+1, j] + p[i-1, j] +
                p[i, j+1] + p[i, j-1] -
                4 * p[i, j]
            )
            p_next[i, j] = 2 * p[i, j] - p_prev[i, j] + lambda_sq * laplacian

    # Neumann BC (walls)
    p_next[0, :] = p_next[1, :]
    p_next[-1, :] = p_next[-2, :]
    p_next[:, 0] = p_next[:, 1]
    p_next[:, -1] = p_next[:, -2]

    # Jet input
    t = n * k
    jet_value = A * np.sin(2 * np.pi * f * t)
    for j in input_j:
        p_next[1, j] = jet_value

    # Record pressure
    for (i, j) in tone_holes:
        tone_hole_outputs[(i, j)].append(p[i, j])

    # Save frame for animation
    if store_frames and n % 20 == 0:  # downsample for speed
        frames_to_plot.append(p.copy())

    # Rotate buffers
    p_prev, p, p_next = p, p_next, p_prev

print("Simulation done.")

# --- Save WAV Files ---
sample_rate = int(1 / k)
for (i, j), data in tone_hole_outputs.items():
    audio = np.array(data)
    if np.max(np.abs(audio)) > 0:
        audio /= np.max(np.abs(audio))
    audio_int16 = (audio * 32767).astype(np.int16)
    write(f"tonehole_{i}_{j}_better.wav", sample_rate, audio_int16)
    print(f"Saved: tonehole_{i}_{j}_better.wav")

# --- Optional: Animate Stored Frames ---
print(f'I Lost The Game')
fig, ax = plt.subplots()
img = ax.imshow(frames_to_plot[0].T, cmap='RdBu', vmin=-A, vmax=A, origin='lower', interpolation='none')
ax.set_title("2D FDTD Ocarina Simulation")
plt.colorbar(img, ax=ax, label='Pressure')

def animate_plot(frame_idx):
    img.set_data(frames_to_plot[frame_idx].T)
    return [img]

ani = FuncAnimation(fig, animate_plot, frames=len(frames_to_plot), interval=30, blit=True)
plt.show()
