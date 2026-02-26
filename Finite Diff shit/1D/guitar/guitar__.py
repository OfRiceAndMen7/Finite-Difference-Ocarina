import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.fft as fft
from scipy.io.wavfile import write
from scipy.signal import resample

# --- Parameters ---
c = 1                   # Wave speed
kappa = 1e-5            # Stiffness
sigma0 = 0.9           # Damping (frequency-independent)
sigma1 = 1e-6           # Damping (frequency-dependent)

L = 1.0                 # String length
N = 5000
h = L / N
x = np.linspace(0, L, N + 1)

# Time step based on stability
k_wave = h / c
k_stiff = h**2 / (np.sqrt(2) * kappa)
k = 0.9 * min(k_wave, k_stiff)

lambda_ = c * k / h
mu = kappa * k / h**2

TF = 4
NF = int(TF / k)
print(f"Time step k = {k:.3e}, λ = {lambda_:.3f}, μ = {mu:.3e}, Frames = {NF}")

# --- Initial displacement: raised cosine pluck ---
max_disp = 1.0

initial_disp = np.zeros_like(x)

initial_disp = (np.sin(625 * 2 * np.pi * x))
initial_disp *= max_disp


# Initial conditions
u0 = initial_disp.copy()
u1 = u0.copy()
u2 = np.zeros_like(x)
history = [u0.copy(), u1.copy()]

# --- Time stepping ---
for n in range(2, NF):
    u2[2:N-1] = (
        (2 - 2 * sigma0 * k) * u1[2:N-1]
        - (1 - 2 * sigma0 * k) * u0[2:N-1]
        + lambda_**2 * (u1[3:N] - 2 * u1[2:N-1] + u1[1:N-2])
        - mu**2 * (u1[4:N+1] - 4 * u1[3:N] + 6 * u1[2:N-1] - 4 * u1[1:N-2] + u1[0:N-3])
        - 2 * sigma1 * k * (u1[3:N] - 2 * u1[2:N-1] + u1[1:N-2]) / h**2
    )
    # boundary conditions
    u2[0] = 0
    u2[1] = 0.5 * (u2[2] + u2[0])
    u2[N] = 0
    u2[N-1] = 0.5 * (u2[N] + u2[N-2])

    history.append(u2.copy())
    u0, u1 = u1, u2.copy()

history = np.array(history)

# # --- Animation ---
# fig, ax = plt.subplots()
# line, = ax.plot(x, history[0], 'r-', lw=2)
# ax.set_ylim(-1.2 * max_disp, 1.2 * max_disp)
# ax.set_xlim(0, L)
# ax.set_xlabel('Position (x)')
# ax.set_ylabel('Displacement (u)')
# title = ax.set_title(f'Frame 1/{len(history)}')

# def update(frame):
#     line.set_ydata(history[frame])
#     title.set_text(f'Frame {frame+1}/{len(history)}')
#     return line, title

# ani = animation.FuncAnimation(fig, update, frames=len(history), interval=5, blit=True)
# plt.show()

# --- Audio extraction & normalization ---
pickup = int(N * 0.15)
signal = history[:, pickup]
signal /= np.max(np.abs(signal))  # Normalize

# --- Optional trim after decay ---
trim_ratio = 0.95  # Keep 95% of the signal
signal = signal[:int(len(signal) * trim_ratio)]

# --- Resample to audio rate ---
fs = 44100
expected_samples = int(fs * (len(signal) * k))  # Adjusted duration
resampled = resample(signal, expected_samples)

# --- Save audio ---
audio_int16 = np.int16(resampled * 32767)
write("guitar_actually625hz.wav", fs, audio_int16)
print("✓ Audio written to 'guitar_like_string_stable.wav'")

# --- FFT Analysis ---
N_fft = 2**14
fft_vals = fft.fft(resampled, n=N_fft)
fft_freq = fft.fftfreq(N_fft, d=1/fs)
pos_mask = fft_freq > 0  # Exclude zero frequency for log scale
fft_freq = fft_freq[pos_mask]
fft_mag = np.abs(fft_vals[pos_mask])

plt.figure(figsize=(10, 5))
plt.semilogx(fft_freq, 20 * np.log10(fft_mag + 1e-12))
plt.xlim(20, 5000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('FFT Magnitude at Pickup Point (Log Frequency Scale)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
