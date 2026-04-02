import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write
from dynamic_wall import DynamicWall

# --- Simulation Parameters ---
Nx, Ny = 30, 40          # Ny=40: fits trisegment (4+22+14=40)
Lx     = 1.0
c      = 343
h      = Lx / Nx
k      = 0.5 * h / c     # lambda_sq = 0.25 (stable in 2D)
T      = 2
Nt     = int(T / k)
lsq    = (c * k / h) ** 2

# --- Static occupancy mask (True = solid wall) ---
# Hollow open box: top wall (i=5), left wall (j=5), right wall (j=24).
# No bottom wall — row i=23 is the trisegment dynamic floor.
occ = np.zeros((Nx, Ny), dtype=bool)
for i in range(5, 23):          # i=5..22 (exclude i=23, trisegment row)
    for j in range(5, 25):      # j=5..24
        if i == 5 or j == 5 or j == 24:
            occ[i, j] = True
# Row i=23 is NOT pre-marked: it is replaced by the trisegment.

# --- DynamicWall: replaces row i=23 (dynamic sliding bottom wall) ---
# a: j=0..Ma-1          (air, grows as b slides right)
# b: j=Ma..Ma+Nb-1      (b[B_SOL0:B_SOL1] = solid floor; remainder air)
# c: j=Ma+Nb..Ny-1      (air, shrinks as b slides right)
Nb_ts  = 22
B_SOL0 = 1            # b[B_SOL0:B_SOL1] are solid
B_SOL1 = 20

b_occ_ts = np.zeros(Nb_ts, dtype=bool)
b_occ_ts[B_SOL0:B_SOL1] = True

Ma_ts_init = 4
Ma_ts_end  = 14           # slide right by 10 over Nt steps
pos_ab_ts  = np.linspace(Ma_ts_init, Ma_ts_end, Nt)

wall = DynamicWall(
    Na_init=Ma_ts_init,
    Nb=Nb_ts,
    Nc_init=Ny - Ma_ts_init - Nb_ts,
    b_occ=b_occ_ts,
    slide_schedule=pos_ab_ts,
    lsq=lsq,
)

# --- 2D pressure ---
p      = np.zeros((Nx, Ny))
p_prev = np.zeros_like(p)
p_next = np.zeros_like(p)

# --- Input / Output ---
f_in  = 440;   A_in = 0.01
inp_i = 6;      inp_j = np.arange(12, 18)

tone_holes        = [(25,25), (18, 12)]
tone_hole_outputs = {th: [] for th in tone_holes}

store_frames = True
frames_to_plot = []
Ma_anim        = []

# ==========================================================================
# Main loop
# ==========================================================================
for n in range(Nt):

    # ── Row-23 lookup arrays for 2D neighbor queries (time n) ────────────
    ts_arr   = wall.get_values()      # shape (Ny,), solid cells = 0
    ts_solid = wall.get_solid_mask()  # shape (Ny,), True = solid
    ts_solid |= occ[23, :]

    # ── 2D FDTD: all rows except i=23 ────────────────────────────────────
    for i in range(1, Nx - 1):
        if i == 23:
            continue
        for j in range(1, Ny - 1):
            if occ[i, j]:
                continue

            # i+1 neighbor (may be triseg row 23)
            if i + 1 == 23:
                x_plus = p[i, j] if ts_solid[j] else ts_arr[j]
            else:
                x_plus = p[i, j] if occ[i+1, j] else p[i+1, j]

            # i-1 neighbor (may be triseg row 23)
            if i - 1 == 23:
                x_minus = p[i, j] if ts_solid[j] else ts_arr[j]
            else:
                x_minus = p[i, j] if occ[i-1, j] else p[i-1, j]

            y_plus  = p[i, j] if occ[i, j+1] else p[i, j+1]
            y_minus = p[i, j] if occ[i, j-1] else p[i, j-1]

            laplacian = x_plus + x_minus + y_plus + y_minus - 4*p[i, j]
            p_next[i, j] = 2*p[i, j] - p_prev[i, j] + lsq * laplacian

    # Fully dissipative outer boundary (Dirichlet p=0)
    p_next[0, :]  = 0
    p_next[-1, :] = 0
    p_next[:, 0]  = 0
    p_next[:, -1] = 0

    # Jet input
    jv = A_in * np.sin(2 * np.pi * f_in * n * k)
    for j in inp_j:
        p_next[inp_i, j] = jv

    # Tone holes
    for (ti, tj) in tone_holes:
        tone_hole_outputs[(ti, tj)].append(p[ti, tj])

    # ── DynamicWall step using p at time n (before rotation) ─────────────
    wall.step([p[22, :], p[24, :]])

    # Store frame (embed triseg air values into row 23 for visualisation)
    if store_frames and n % 20 == 0:
        vis = p.copy()
        vis[23, :] = ts_arr
        frames_to_plot.append(vis)
        Ma_anim.append(wall.Ma)

    # Rotate 2D buffers
    p_prev, p, p_next = p, p_next, p_prev

print("Simulation done.")

# --- Save WAV ---
sample_rate = int(1 / k)
for (ti, tj), data in tone_hole_outputs.items():
    audio = np.array(data)
    if np.max(np.abs(audio)) > 0:
        audio /= np.max(np.abs(audio))
    write(f"tonehole_{ti}_{tj}_{Nx}x{Ny}.wav", sample_rate, (audio * 32767).astype(np.int16))
    print(f"Saved: tonehole_{ti}_{tj}_{Nx}x{Ny}.wav")

# --- Animation ---
print('I Lost The Game')
fig, ax = plt.subplots(figsize=(10, 8))

img = ax.imshow(frames_to_plot[0].T, cmap='RdBu',
                vmin=-1.5*A_in, vmax=1.5*A_in,
                origin='lower', interpolation='none', aspect='auto')

# Static wall overlay (gray, semi-transparent)
wall_rgba = np.zeros((Nx, Ny, 4))
wall_rgba[occ] = [0.3, 0.3, 0.3, 0.7]
ax.imshow(np.transpose(wall_rgba, (1,0,2)), origin='lower', interpolation='none', aspect='auto')

# Dynamic floor overlay (updated per frame)
dyn_rgba = np.zeros((Nx, Ny, 4))
dyn_img  = ax.imshow(np.transpose(dyn_rgba, (1,0,2)), origin='lower', interpolation='none', aspect='auto')

ax.set_xlabel('i (row)')
ax.set_ylabel('j (column)')
plt.colorbar(img, ax=ax, label='Pressure')


def animate_plot(fi):
    img.set_data(frames_to_plot[fi].T)
    Ma_f = Ma_anim[fi]
    dyn  = np.zeros((Nx, Ny, 4))
    for kb in range(B_SOL0, B_SOL1):
        jg = Ma_f + kb
        if 0 <= jg < Ny:
            dyn[23, jg] = [0.1, 0.1, 0.5, 0.85]   # blue = dynamic solid floor
    dyn_img.set_data(np.transpose(dyn, (1,0,2)))
    ax.set_title(
        f'2D FDTD + sliding trisegment floor  |  step {fi*20}/{Nt}'
        f'  |  Ma={Ma_f}  floor j=[{Ma_f+B_SOL0}, {Ma_f+B_SOL1-1}]'
    )
    return img, dyn_img


ani = FuncAnimation(fig, animate_plot, frames=len(frames_to_plot), interval=30, blit=True)
plt.show()
