import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# --- Parameters ---
drawThings = False
drawStart = 0
drawSpeed = 1000

fs = 44100
k = 1 / fs
lengthSound = fs * 8
# speed of sound ~343 m/s
c = 331
sigma = 1e-12

numFromBound = -1
if numFromBound == 0:
    raise ValueError("numFromBound = 0 is undefined")

Ninit = 424
Narr = np.linspace(0, 4 * np.pi, lengthSound)
Nvec = Ninit + 0 * np.sin(Narr)
NPrev = Ninit

# -1: evenly split spatial grid (seems to be preferred)
if numFromBound == -1:
    M = int(np.ceil(0.5 * Ninit))
    Mw = int(np.floor(0.5 * Ninit))
else:
    M = Ninit - numFromBound
    Mw = numFromBound

wave_freq = 0 # sine wave IC with given spatial frequency, 0 for no IC
uNext = np.zeros(M)
x = np.linspace(0, M, M)
u = 1e-7* np.sin(wave_freq * 2 * np.pi * x/Ninit)
u = uNext.copy()
uPrev = u.copy()

wNext = np.zeros(Mw)
w = 1e-7* np.sin(wave_freq * 2 * np.pi * (x+M-1)/Ninit)
w = wNext.copy()
wPrev = np.zeros(Mw)

# centered difference approximation to du/dx
def first_derivative(f, h):
    df = np.zeros_like(f)
    df[1:-1] = (f[2:] - f[:-2]) / (2 * h)
    df[0] = (f[1] - f[0]) / h
    df[-1] = (f[-1] - f[-2]) / h
    return df

# centered difference approximation to du2/d2t using matrices;
# boundary conditions and virtual points added in recursion
def create_Dxx(n):
    e = np.ones(n)
    diagonals = [e, -2*e, e]
    offsets = [-1, 0, 1]
    return sp.diags(diagonals, offsets, shape=(n, n), format='csr')

# initialize matrices
Dxxu = create_Dxx(len(u))
Dxxw = create_Dxx(len(w))

out = np.zeros(lengthSound)
energy = np.zeros(lengthSound)


storeEvery = 100
energy_store = []
u_store = []
w_store = []
h_store = []
excite_store = []
y_store = []


# single-mass lip reed model (constants cannibalized from Willemsen et al.)
y = 4e-4             # lip displacement (initial and prev)
yPrev = 2e-4
mLip = 5e-3       # lip mass (kg)
kLip = 5.2e4       # lip stiffness (N/m)
rLip = 0.001           # lip damping coefficient
H0 = 9e-4         # equilibrium opening (m)
A_lip = 5e-5        # lip area (m^2)
rho = 1.179         # air density
Kc = 1e5       # collision stiffness
alpha_c = 2     # nonlinearity exponent (2 = quadratic)


epsilon = 0.05  # Nonlinear coefficient


for n in range(lengthSound):

    h =  c * k/0.999 # buffered courant number, ck/h = 0.999 for stability
    Nfrac = Nvec[n]
    N = int(np.floor(Nfrac))
    lambdaSq = (c**2) * (k**2) / (h**2)

    alf = Nfrac - N

    if abs(N - NPrev) > 1:
        raise RuntimeError("Can only add/remove one grid point at a time")

    if N != NPrev or n == 0:

        # cubic interpolant (Willemson et al., 2021)
        cubicIp = np.array([
            alf * (alf + 1) / (- (alf + 2) * (alf + 3)),
            2 * alf / (alf + 2),
            2 / (alf + 2),
            2 * alf / (-(alf + 3) * (alf + 2))
        ])


        if N > NPrev:
            if numFromBound == -1:
                if N % 2 == 1:
                    uNext = np.append(uNext, 0)
                    u = np.append(u, cubicIp @ np.concatenate((u[-2:], w[:2])))
                    uPrev = np.append(uPrev, cubicIp @ np.concatenate((uPrev[-2:], wPrev[:2])))
                else:
                    wNext = np.insert(wNext, 0, 0)
                    w = np.insert(w, 0, np.flip(cubicIp) @ np.concatenate((u[-2:], w[:2])))
                    wPrev = np.insert(wPrev, 0, np.flip(cubicIp) @ np.concatenate((uPrev[-2:], wPrev[:2])))
            elif numFromBound == 1:
                uNext = np.append(uNext, 0)
                u = np.append(u, cubicIp[:3] @ np.concatenate((u[-2:], w[:1])))
                uPrev = np.append(uPrev, cubicIp[:3] @ np.concatenate((uPrev[-2:], wPrev[:1])))
            else:
                uNext = np.append(uNext, 0)
                u = np.append(u, cubicIp @ np.concatenate(u[-2:], w[:2]))
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

        # update centered difference matrices when number of nodes changes
        Dxxu = create_Dxx(len(u))
        Dxxw = create_Dxx(len(w))

        if numFromBound == 1:
            M = N - numFromBound
            Mw = numFromBound

    ip = np.array([
        -(alf - 1) / (alf + 1),
        1,
        (alf - 1) / (alf + 1)
    ])

    if numFromBound == 1:
        virtualGridPoints = np.array([
            ip[2] * u[-1] + ip[1] * w[0],
            ip[0] * u[-2] + ip[1] * u[-1] + ip[2] * w[0]
        ])
    else:
        virtualGridPoints = np.array([
            ip[2] * u[-1] + ip[1] * w[0] + ip[0] * w[1],
            ip[0] * u[-2] + ip[1] * u[-1] + ip[2] * w[0]
        ])

    # single-mass lip reed (no collision)

    if n <= fs/500:
        Pm = 6e3 * 0.5**(1000*n/fs)
    else:
        Pm = 1.5e3  # Mouth pressure (Pa)

    # bore pressure at excitation point (right difference)
    pBore = -rho * c**2 * (u[1] - u[0]) / h
    dP = Pm - pBore

    # inverse of gap between lips
    eta = -y - H0

    # Bernoulli volume flow through lips
    if eta < 0 and dP != 0:
        UB = A_lip * np.abs(eta) * np.sign(dP) * np.sqrt(2 * np.abs(dP) / rho)
        phi = 0
    else:
        UB = 0.0
        phi = 2 * Kc * eta**alpha_c

    # lip motion volume flow
    Ur = A_lip * (y - yPrev) / k

    # total volume flow into bore
    U_total = UB + Ur

    vLip = (y - yPrev) / k
    aLip = ( -rLip * vLip - kLip * y + A_lip * dP - phi) / mLip
    yNext = 2 * y - yPrev + k**2 * aLip

    # Inject volume flow into FDTD as velocity at u[0]
    excitation = (k / A_lip) * U_total

    du_dx_u = first_derivative(u, h)
    nonlinear_term_u = first_derivative(du_dx_u**2, h)

    du_dx_w = first_derivative(w, h)
    nonlinear_term_w = first_derivative(du_dx_w**2, h)

    # 1D wave equation with nonlinear terms to simulate shockwaves
    uNext = 2 * u + lambdaSq * Dxxu.dot(u) + k**2 * epsilon * nonlinear_term_u - uPrev + (sigma/k * (u - uPrev))
    uNext[-1] += lambdaSq * virtualGridPoints[0]

    wNext = 2 * w + lambdaSq * Dxxw.dot(w) + k**2 * epsilon * nonlinear_term_w - wPrev + (sigma/k * (w - wPrev))
    wNext[0] += lambdaSq * virtualGridPoints[1]

    uNext[0] += excitation

    out[n] = w[-2]


    if n % storeEvery == 0:

        u_vel = (uNext - uPrev) / (2*k)
        w_vel = (wNext - wPrev) / (2*k)
        T_u = 0.5 * h * np.sum(u_vel**2)
        T_w = 0.5 * h * np.sum(w_vel**2)
        V_u = 0.5 * c**2 / h * np.sum((np.diff(np.concatenate(([0], u, [0])))**2))
        V_w = 0.5 * c**2 / h * np.sum((np.diff(np.concatenate(([0], w, [0])))**2))
        V_collision = 0
        if eta > 0:
            V_collision = (2 * Kc / (alpha_c + 1)) * eta**(alpha_c + 1)
        energy_store.append(T_u + T_w + V_u + V_w + V_collision)

        u_store.append(u.copy())
        w_store.append(w.copy())
        h_store.append(h)
        excite_store.append(excitation)
        y_store.append(-1 * eta)

    # update recursion
    uPrev = u.copy()
    u = uNext.copy()
    wPrev = w.copy()
    w = wNext.copy()
    NPrev = N
    yPrev = y
    y = yNext

# --- Save Output ---
audio_max = np.max(np.abs(out))
out_norm = out / audio_max
write("lipreed_dynamic_trombone.wav", fs, (out_norm * 32767).astype(np.int16))

# output
time = np.arange(lengthSound) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, out, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title('Output Waveform (with Lip Reed)')
plt.grid(True)
plt.tight_layout()
plt.show()

# energy
time_storeEvery = np.arange(int(np.ceil(lengthSound/storeEvery))) / fs
plt.figure(figsize=(10, 4))
plt.plot(time_storeEvery, energy_store, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('Energy of system')
plt.grid(True)
plt.tight_layout()
plt.show()

# lip position
plt.figure(figsize=(10,4))
plt.plot(time_storeEvery, y_store)
plt.xlabel('Time (s)')
plt.ylabel('Lip Displacement (m)')
plt.title('Lip Displacement/Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# excitation force
plt.figure(figsize = (10, 4))
plt.plot(time_storeEvery, excite_store)
plt.xlabel('Time (s)')
plt.ylabel('Excitation Force')
plt.title('Excitation Force over/Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Airflow ---

# plt.figure(figsize=(10, 4))
# plt.plot(time_storeEvery, U_store, label='Airflow U (mm^3/s)')
# plt.xlabel('Time (s)')
# plt.ylabel('Value')
# plt.title('Lip Displacement and Airflow Over Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# --- Animation ---
fig, ax = plt.subplots(figsize=(10, 4))
line_left, = ax.plot([], [], 'b.-', linewidth=2, markersize=10, label='u')
line_right, = ax.plot([], [], 'ro-', linewidth=2, markersize=6, label='w')
ax.set_xlim(0, 1.0)

if audio_max == 0:
    ax.set_ylim(-1, 1)
else:
    ax.set_ylim(-1.2 * audio_max, 1.2 * audio_max)

ax.set_xlabel(r"$x$ (m)")
ax.set_ylabel("Displacement")
ax.set_title("Dynamic FDTD Grid with Lip Reed")
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

    left_x = np.linspace(0, 0.5, len(u_frame))
    left_y = u_frame

    right_x = np.linspace(0.5, 1.0, len(w_frame))
    right_y = w_frame

    line_left.set_data(left_x, left_y)
    line_right.set_data(right_x, right_y)

    ax.set_title(f"Frame {i * storeEvery} / {lengthSound}")
    return line_left, line_right


anim = FuncAnimation(fig, animate, init_func=init, frames=len(u_store), interval=50, blit=True)
plt.show()
