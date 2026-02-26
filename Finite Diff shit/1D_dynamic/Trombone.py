import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# --- Parameters ---
drawThings = False  # We'll do our own animation at the end
drawStart = 0
drawSpeed = 1000

fs = 44100
k = 1 / fs
lengthSound = fs * 2

numFromBound = 1
if numFromBound == 0:
    raise ValueError("numFromBound = 0 is undefined")

Ninit = 60
Nend = 72
N = int(np.floor(Ninit))
NPrev = N

dispCorr = Ninit > Nend
L = 1.0

cInit = L / (Ninit * k)
cEnd = L / (Nend * k)
cVec = np.linspace(cInit, cEnd, lengthSound)

if numFromBound == -1:
    M = int(np.ceil(0.5 * N))
    Mw = int(np.floor(0.5 * N))
else:
    M = N - numFromBound
    Mw = numFromBound

uNext = np.zeros(M)
u = np.zeros(M)
u[0] = 1.0
uPrev = u.copy()

wNext = np.zeros(Mw)
w = np.zeros(Mw)
wPrev = np.zeros(Mw)

def create_Dxx(n):
    e = np.ones(n)
    diagonals = [e, -2*e, e]
    offsets = [-1, 0, 1]
    return sp.diags(diagonals, offsets, shape=(n, n), format='csr')

Dxxu = create_Dxx(len(u))
Dxxw = create_Dxx(len(w))

out = np.zeros(lengthSound)

# To store grid states for animation (store every Nth frame for memory efficiency)
storeEvery = 2000
u_store = []
w_store = []
h_store = []  # store grid spacing h at each saved frame

for n in range(lengthSound):
    c = cVec[n]
    h = c * k
    Nfrac = L / h
    N = int(np.floor(Nfrac))
    lambdaSq = (c**2) * (k**2) / (h**2)

    alf = Nfrac - N

    if abs(N - NPrev) > 1:
        raise RuntimeError("Can only add/remove one grid point at a time")

    if N != NPrev:
        if N > NPrev:
            cubicIp = np.array([
                alf * (alf + 1) / (- (alf + 2) * (alf + 3)),
                2 * alf / (alf + 2),
                2 / (alf + 2),
                2 * alf / (-(alf + 3) * (alf + 2))
            ])

            if numFromBound == -1:
                if N % 2 == 1:
                    uNext = np.append(uNext, 0)
                    u = np.append(u, cubicIp @ np.concatenate((u[-2:], w[:2])))
                    uPrev = np.append(uPrev, cubicIp @ np.concatenate((uPrev[-2:], wPrev[:2])))
                else:
                    wNext = np.insert(wNext, 0, 0)
                    w = np.concatenate((np.flip(cubicIp) @ np.concatenate((u[-2:], w[:2])), w))
                    wPrev = np.concatenate((np.flip(cubicIp) @ np.concatenate((uPrev[-2:], wPrev[:2])), wPrev))
            elif numFromBound == 1:
                uNext = np.append(uNext, 0)
                u = np.append(u, cubicIp[:3] @ np.concatenate((u[-2:], w[:1])))
                uPrev = np.append(uPrev, cubicIp[:3] @ np.concatenate((uPrev[-2:], wPrev[:1])))
            else:
                uNext = np.append(uNext, 0)
                u = np.append(u, cubicIp @ np.concatenate((u[-2:], w[:2])))
                uPrev = np.append(uPrev, cubicIp @ np.concatenate((uPrev[-2:], wPrev[:2])))
        else:
            if numFromBound == -1:
                if N % 2 == 0:
                    uNext = uNext[:-1]
                    u = u[:-1]
                    uPrev = uPrev[:-1]
                else:
                    wNext = wNext[1:]
                    w = w[1:]
                    wPrev = wPrev[1:]
            else:
                uNext = uNext[:-1]
                u = u[:-1]
                uPrev = uPrev[:-1]

        Dxxu = create_Dxx(len(u))
        Dxxw = create_Dxx(len(w))

        if numFromBound == -1:
            M = int(np.ceil(0.5 * N))
            Mw = int(np.floor(0.5 * N))
        else:
            M = N - numFromBound
            Mw = numFromBound

    ip = np.array([
        -(alf - 1) / (alf + 1),
        1,
        (alf - 1) / (alf + 1)
    ])

    if numFromBound == 1:
        virtualGridPoints = np.array([
            ip[2] * u[-1] + ip[1] * w[0] + 0,
            ip[0] * u[-2] + ip[1] * u[-1] + ip[2] * w[0]
        ])
    else:
        virtualGridPoints = np.array([
            ip[2] * u[-1] + ip[1] * w[0] + ip[0] * w[1],
            ip[0] * u[-2] + ip[1] * u[-1] + ip[2] * w[0]
        ])

    uNext = 2 * u + lambdaSq * Dxxu.dot(u) - uPrev
    uNext[-1] += lambdaSq * virtualGridPoints[0]

    wNext = 2 * w + lambdaSq * Dxxw.dot(w) - wPrev
    wNext[0] += lambdaSq * virtualGridPoints[1]

    if dispCorr:
        epsilon = 0
        sig0 = 1
        etaPrev = wPrev[0] - uPrev[-1]
        rForce = (1 - sig0 / k) / (1 + sig0 / k)
        oOP = (h * (1 + sig0 / k) * (1 - alf)) / (2 * h * (alf + epsilon) + 2 * k**2 * (1 + sig0 / k) * (1 - alf))
        F = (wNext[0] - uNext[-1] + rForce * etaPrev) * oOP

        uNext[-1] += k**2 / h * F
        wNext[0] -= k**2 / h * F

    out[n] = u[0]

    # Store for animation every storeEvery steps
    if n % storeEvery == 0:
        u_store.append(u.copy())
        w_store.append(w.copy())
        h_store.append(h)

    uPrev = u.copy()
    u = uNext.copy()
    wPrev = w.copy()
    w = wNext.copy()
    NPrev = N

# Save output waveform to WAV file (scaled to int16)
out_norm = out / np.max(np.abs(out))
write("trombone_dynamic_60-72.wav", fs, (out_norm * 32767).astype(np.int16))

# Plot waveform
time = np.arange(lengthSound) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, out, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title('Output Waveform')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Animate the grid ---

fig, ax = plt.subplots(figsize=(10, 4))

line_left, = ax.plot([], [], 'b.-', linewidth=2, markersize=10, label='Left system u')
line_right, = ax.plot([], [], 'ro-', linewidth=2, markersize=6, label='Right system w')
ax.set_xlim(0, L)
ax.set_ylim(-1, 1)
ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel("Displacement")
ax.set_title("Dynamic Finite Difference Grid")
ax.legend()
ax.grid(True)

def init():
    line_left.set_data([], [])
    line_right.set_data([], [])
    return line_left, line_right

def animate(i):
    u_frame = u_store[i]
    w_frame = w_store[i]
    h_frame = h_store[i]

    hLocsLeft = np.arange(len(u_frame) + 1) * h_frame
    hLocsRight = np.flip(L - np.arange(len(w_frame) + 1) * h_frame)

    # Left system with zero boundary at left end
    left_x = hLocsLeft
    left_y = np.concatenate(([0], u_frame))

    # Right system with zero boundary at right end
    right_x = hLocsRight
    right_y = np.concatenate((w_frame, [0]))

    line_left.set_data(left_x, left_y)
    line_right.set_data(right_x, right_y)
    ax.set_title(f"Dynamic Finite Difference Grid\nFrame {i * storeEvery} / {lengthSound}")

    return line_left, line_right

anim = FuncAnimation(fig, animate, init_func=init, frames=len(u_store), interval=50, blit=True)

plt.show()
