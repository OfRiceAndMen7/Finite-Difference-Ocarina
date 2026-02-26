import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write
from scipy.signal import resample_poly
from fractions import Fraction

# --- Simulation Parameters ---
Nx, Ny, Nz = 36, 53, 124
Lx, Ly, Lz = 0.108, 0.159, 0.372
c = 343
dx = Lx / Nx
h = dx

sigma_0 = 0.0
sigma_1 = 0

k = 0.99 * (h / c) * np.sqrt((2 - sigma_0) / 6.0)
T = 2.0
Nt = int(T / k)
lambda_sq = (c * k / h) ** 2

# Jet Input
f = 100
A = 1e-4
input_y = np.arange(20, 23)     
input_z = np.arange(38, 40)

# Tone Hole Locations and sizing
hole_rad = 2
tone_holes = [(34, 28, 23), (9, 28, 67)]

# Output Storage
tone_hole_outputs = {th: [] for th in tone_holes}
slice_frames = []
occ = np.load("occupancy.npy")  # shape (Nx, Ny, Nz), dtype=bool
for z in range(20, 25):
    occ[29, 25, z] = 0

# --- Air Neighbor Calculation ---
def compute_air_neighbors(occ, is_solid):
    Nx, Ny, Nz = occ.shape
    air_neighbors = np.zeros((Nx, Ny, Nz), dtype=np.int32)
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                if occ[i, j, k] == is_solid:  # solid or air
                    count = 0
                    count += not occ[i+1, j, k]
                    count += not occ[i-1, j, k]
                    count += not occ[i, j+1, k]
                    count += not occ[i, j-1, k]
                    count += not occ[i, j, k+1]
                    count += not occ[i, j, k-1]
                    air_neighbors[i, j, k] = count
    return air_neighbors


air_air = compute_air_neighbors(occ, 1)
solid_air = compute_air_neighbors(occ, 0)


# --- Numba FDTD Kernel ---
@njit(parallel=True)
def fdtd_update(p, p_prev, p_next, laplacian_prev, lambda_sq, sigma_0, sigma_1):
    Nx, Ny, Nz = p.shape
    laplacian = np.zeros((Nx-2, Ny-2, Nz-2), dtype=p.dtype)

    # laplacian
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                if not occ[i, j, k]:  # solid
                    lap = 0.0

                    # X direction - handle all 4 cases
                    has_xplus = not occ[i+1, j, k]
                    has_xminus = not occ[i-1, j, k]

                    if has_xplus and has_xminus:

                        lap += (p[i+1, j, k] - 2*p[i, j, k] + p[i-1, j, k])
                    elif has_xplus and not has_xminus:
                        lap += (p[i+1, j, k] - p[i, j, k])
                    elif not has_xplus and has_xminus:
                        lap += (p[i-1, j, k] - p[i, j, k])

                    # Y direction
                    has_yplus = not occ[i, j+1, k]
                    has_yminus = not occ[i, j-1, k]

                    if has_yplus and has_yminus:
                        lap += (p[i, j+1, k] - 2*p[i, j, k] + p[i, j-1, k])
                    elif has_yplus and not has_yminus:
                        lap += (p[i, j+1, k] - p[i, j, k])
                    elif not has_yplus and has_yminus:
                        lap += (p[i, j-1, k] - p[i, j, k])

                    # Z direction
                    has_zplus = not occ[i, j, k+1]
                    has_zminus = not occ[i, j, k-1]

                    if has_zplus and has_zminus:
                        lap += (p[i, j, k+1] - 2*p[i, j, k] + p[i, j, k-1])
                    elif has_zplus and not has_zminus:
                        lap += (p[i, j, k+1] - p[i, j, k])
                    elif not has_zplus and has_zminus:
                        lap += (p[i, j, k-1] - p[i, j, k])

                    laplacian[i-1, j-1, k-1] = lap

    # Update pressure
    for i in prange(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                if not occ[i, j, k]:

                    p_next[i, j, k] = (
                        (2 - sigma_0) * p[i, j, k]
                        - (1 - sigma_0) * p_prev[i, j, k]
                        + lambda_sq * laplacian[i-1, j-1, k-1]
                        - sigma_1 * (laplacian[i-1, j-1, k-1] - laplacian_prev[i-1, j-1, k-1])
                    )
                else:
                    p_next[i,j,k] = 0

    p_next[0, :, :] = 0
    p_next[-1, :, :] = 0
    p_next[:, 0, :] = 0
    p_next[:, -1, :] = 0
    p_next[:, :, 0] = 0
    p_next[:, :, -1] = 0
    return laplacian

# --- Main Simulation ---
def run_simulation():
    global tone_hole_outputs, slice_frames
    p = np.zeros((Nx, Ny, Nz), dtype=np.float32)
    p_prev = np.zeros_like(p)
    p_next = np.zeros_like(p)
    laplacian_prev = np.zeros((Nx-2, Ny-2, Nz-2), dtype=np.float32)
    for frame in range(Nt):

        # Apply jet to input region
        for j in input_y:
            for l in input_z:
                p[28, j, l] += A

        # FDTD update
        laplacian_prev = fdtd_update(p, p_prev, p_next, laplacian_prev, lambda_sq, sigma_0, sigma_1)
        # Tone holes
        for (i, j, l) in tone_holes:
            tone_hole_outputs[(i, j, l)].append(p[i, j, l])
        # Store slice
        if frame % 10 == 0:
            slice_frames.append(p[:, :, 38].copy())
        # Rotate buffers
        p_prev, p, p_next = p, p_next, p_prev
        if frame % 100 == 0:
            print(f"Frame {frame}, max pressure: {np.max(np.abs(p)):.5f}")

# --- Audio Output ---
def save_audio():
    sim_rate = int(round(1.0 / k))
    target_rate = 44100
    scaling_factor = 32767
    frac = Fraction(target_rate, sim_rate).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    print(f"Resampling from {sim_rate} Hz to {target_rate} Hz (up={up}, down={down})")
    for (i, j, l), data in tone_hole_outputs.items():
        audio = np.array(data, dtype=np.float64)
        if np.max(np.abs(audio)) > 0:
            audio /= np.max(np.abs(audio))
        audio_resampled = resample_poly(audio, up, down)
        audio_int16 = (audio_resampled * scaling_factor).astype(np.int16)
        filename = f"tonehole_{i}_{j}_{l}_44100.wav"
        write(filename, target_rate, audio_int16)
        print(f"Saved tone hole output to: {filename}")

# --- Animation ---
def animate_simulation():
    fig, ax = plt.subplots()
    img = ax.imshow(slice_frames[0], cmap='seismic', vmin=-A, vmax=A, origin='lower', interpolation='none')
    ax.set_title("le tube (top slice)")
    ax.set_ylabel('X')
    ax.set_xlabel('Z')
    plt.colorbar(img, ax=ax, label='Pressure')

    def update_anim(frame):
        img.set_data(slice_frames[frame])
        return [img]

    ani = FuncAnimation(fig, update_anim, frames=len(slice_frames), interval=1, blit=True)
    plt.show()

# --- Run Simulation ---
run_simulation()
save_audio()
animate_simulation()
