import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.fft as fft

# Parameters
c = 343
N = 1000
L = 1.0
h = L / N
lambda_ = 1.0
k = lambda_ * h / c
TF = 2
NF = int(TF / k)

s0 = 2 * (1 - lambda_**2)  # = 0
s1 = lambda_**2            # = 1

x = np.linspace(0, L, N + 1)

# Pluck parameters
pluck_pos = 0.5        # position of pluck (0 to 1)
max_disp = .2


# Create initial displacement
initial_disp = np.zeros_like(x)

for i, xi in enumerate(x):
    # CHEEKS
    # if 0.2 < xi < 0.5:
    #     initial_disp[i] = -max(np.sqrt(.0625 - (xi-0.35)**2)-0.2, 0)
    # elif 0.5 < xi < 0.8:
    #     initial_disp[i] = -max(np.sqrt(.0625 - (xi-0.65)**2)-0.2, 0)
    # else:
    #     initial_disp[i] = 0

    # cock and balls
    # if 0.2 < xi < 0.4:
    #     initial_disp[i] = 1.5*max(np.sqrt((0.125)**2 - (xi-0.3)**2)-0.075, 0)
    # elif 0.4 < xi < 0.6:
    #     initial_disp[i] = -25*((xi-0.5)**2-0.01)
    # elif 0.6 < xi < 0.8:
    #     initial_disp[i] = 1.5*max(np.sqrt((0.125)**2 - (xi-0.7)**2)-0.075, 0)
    # else:
    #     initial_disp[i] = 0

    # #lol
    if xi <= 0.5:
        initial_disp[i] = max_disp * xi/0.5
    else:
        initial_disp[i] = max_disp* (1-xi)/0.5

    # # weird peak
    # if xi < 0.3:
    #     initial_disp[i] = (10**(xi/0.3) - 1)/10
    # elif 0.3 <= xi :
    #     initial_disp[i] = (10**(1-((xi-0.3)/0.7)) - 1)/10
    # else:
    #     initial_disp[i] = 0




# Initial conditions
u0 = initial_disp
u1 = u0.copy()  # zero initial velocity => u1 = u0

u2 = np.zeros_like(x)

history = [u0.copy(), u1.copy()]

for n in range(2, NF):
    u2[1:N] = -u0[1:N] + s0 * u1[1:N] + s1 * (u1[0:N-1] + u1[2:N+1])
    u2[0] = 0
    u2[N] = 0

    history.append(u2.copy())
    u0, u1 = u1, u2.copy()

history = np.array(history)

fig, ax = plt.subplots()
line, = ax.plot(x, history[0], 'r-', lw=2)
ax.set_ylim(-1.2 * max_disp, 1.2 * max_disp)
ax.set_xlim(0, L)
ax.set_xlabel('Position (x)')
ax.set_ylabel('Displacement (u)')
title = ax.set_title(f'Frame 1/{len(history)}')  # Initialize title object

def update(frame):
    line.set_ydata(history[frame])
    title.set_text(f'Frame {frame+1}/{len(history)}')
    return line, title

ani = animation.FuncAnimation(fig, update, frames=len(history), interval=5, blit=True)
plt.show()
# --- Extract time signal at a pickup point ---
pickup_x = 0.25
pickup_idx = np.argmin(np.abs(x - pickup_x))  # closest spatial index
signal = history[:, pickup_idx]  # displacement vs time

# Sampling rate (since each time step is k seconds)
fs = 1 / k
N_fft = 2**14

# FFT
fft_vals = fft.fft(signal, n=N_fft)
fft_freq = fft.fftfreq(N_fft, d=1/fs)

# Keep only positive frequencies
pos_mask = fft_freq > 0
fft_freq = fft_freq[pos_mask]
fft_mag = np.abs(fft_vals[pos_mask])

# --- Plot FFT ---
plt.figure(figsize=(10, 5))
plt.semilogx(fft_freq, 20 * np.log10(fft_mag + 1e-12))  # use fft_mag here
plt.xlim(20, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title(f'FFT Magnitude at Pickup Point (x={pickup_x:.2f})')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
