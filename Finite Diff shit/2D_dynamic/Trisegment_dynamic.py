import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write

# --- Parameters ---
fs = 44100
k = 1 / fs
lengthSound = fs * 2
cs = 331.0          # speed of sound ('cs' to avoid collision with grid array 'c')
L = 1.0

h = cs * k / 0.95  # grid spacing (Courant number = 0.95)
N_total = int(np.floor(L / h))  # total grid points -- fixed for entire simulation
lambdaSq = (cs * k / h) ** 2

# --- State correction parameters (Willemsen et al. 2021, Sec. 4.4) ---
# beta_sc = (1-alf)/(alf+eps_sc) --> large when alf-->0 (just before grid resize).
# sigma_sc = 1 gives critical damping: gamma_minus = lambda^2 * beta * (1-sigma_sc)/2 = 0,
# so the previous-gap term drops out and the formula reduces to a single division.
eps_sc   = 1e-5   # regularisation; prevents division by zero at alf = 0
sigma_sc = 1.0    # spring damping coefficient (Table 2 in paper)

# --- b slides right: its integer left-edge index Ma increases over time ---
Nb = 20             # fixed size of b (never changes)
slide_amount = 10   # how far b slides (in grid spacings) over lengthSound steps

Ma_init = (N_total - Nb) // 3      # start with a smaller than c
Ma_end  = Ma_init + slide_amount
Mc_init = N_total - Ma_init - Nb   # Mc shrinks as b slides right

# fractional left-junction position (a|b) varies linearly with time
pos_ab_vec = np.linspace(Ma_init, Ma_end, lengthSound)

# --- Initial grids ---
Ma = Ma_init
Mc = Mc_init

a = np.zeros(Ma);  a[Ma // 4] = 0.01   # small initial pulse in a
b = np.zeros(Nb)
c = np.zeros(Mc)

aPrev = a.copy();  aNext = np.zeros_like(a)
bPrev = b.copy();  bNext = np.zeros_like(b)
cPrev = c.copy();  cNext = np.zeros_like(c)

MaPrev = Ma


def create_Dxx(n):
    e = np.ones(n)
    return sp.diags([e, -2*e, e], [-1, 0, 1], shape=(n, n), format='csr')

Dxxa = create_Dxx(Ma)
Dxxb = create_Dxx(Nb)
Dxxc = create_Dxx(Mc)

out    = np.zeros(lengthSound)
energy = np.zeros(lengthSound)

storeEvery = 2000
a_store, b_store, c_store, Ma_store = [], [], [], []

for n in range(lengthSound):

    pos_ab = pos_ab_vec[n]
    Ma_new = int(np.floor(pos_ab))
    alf    = pos_ab - Ma_new          # fractional offset (same at both junctions since Nb is integer)
    Mc_new = N_total - Ma_new - Nb

    dMa = Ma_new - MaPrev             # +1, -1, or 0

    if abs(dMa) > 1:
        raise RuntimeError("b moved more than one grid spacing in one step")

    if dMa != 0:
        # Cubic interpolant coefficients (Willemsen et al. 2021)
        cubicIp = np.array([
            alf * (alf + 1) / (-(alf + 2) * (alf + 3)),
            2 * alf / (alf + 2),
            2 / (alf + 2),
            2 * alf / (-(alf + 3) * (alf + 2))
        ])

        if dMa > 0:
            # b moves RIGHT:
            #   a gains one point at its right end (interpolated from a[-2:], b[:2])
            #   c loses its leftmost point
            new_a      = cubicIp @ np.concatenate((a[-2:],     b[:2]))
            new_a_prev = cubicIp @ np.concatenate((aPrev[-2:], bPrev[:2]))

            aNext = np.append(aNext, 0.0)
            a     = np.append(a,     new_a)
            aPrev = np.append(aPrev, new_a_prev)

            cNext = cNext[1:]
            c     = c[1:]
            cPrev = cPrev[1:]

        else:
            # b moves LEFT:
            #   c gains one point at its left end (interpolated from b[-2:], c[:2])
            #   a loses its rightmost point
            new_c      = np.flip(cubicIp) @ np.concatenate((b[-2:],     c[:2]))
            new_c_prev = np.flip(cubicIp) @ np.concatenate((bPrev[-2:], cPrev[:2]))

            cNext = np.insert(cNext, 0, 0.0)
            c     = np.concatenate(([new_c],      c))
            cPrev = np.concatenate(([new_c_prev], cPrev))

            aNext = aNext[:-1]
            a     = a[:-1]
            aPrev = aPrev[:-1]

        Ma = Ma_new
        Mc = Mc_new
        Dxxa = create_Dxx(len(a))
        Dxxc = create_Dxx(len(c))

    # --- Virtual grid points at both junctions ---
    # Both junctions share alf (Nb is integer, so both edges have the same fractional offset)
    ip = np.array([
        -(alf - 1) / (alf + 1),
         1,
         (alf - 1) / (alf + 1)
    ])

    # Left junction  (a | b):  pattern from Trombone_nonlinear numFromBound != 1
    vgp_a_right = ip[2]*a[-1]  + ip[1]*b[0]   + ip[0]*b[1]   # right virtual neighbor of a[-1]
    vgp_b_left  = ip[0]*a[-2]  + ip[1]*a[-1]  + ip[2]*b[0]   # left  virtual neighbor of b[0]

    # Right junction (b | c):  same pattern, u=b, w=c
    vgp_b_right = ip[2]*b[-1]  + ip[1]*c[0]   + ip[0]*c[1]   # right virtual neighbor of b[-1]
    vgp_c_left  = ip[0]*b[-2]  + ip[1]*b[-1]  + ip[2]*c[0]   # left  virtual neighbor of c[0]

    # --- Wave equation leapfrog update ---
    aNext = 2*a + lambdaSq * Dxxa.dot(a) - aPrev
    aNext[-1] += lambdaSq * vgp_a_right

    bNext = 2*b + lambdaSq * Dxxb.dot(b) - bPrev
    bNext[0]  += lambdaSq * vgp_b_left
    bNext[-1] += lambdaSq * vgp_b_right

    cNext = 2*c + lambdaSq * Dxxc.dot(c) - cPrev
    cNext[0]  += lambdaSq * vgp_c_left

    # --- State correction (Willemsen et al. 2021, Sec. 4.4) ---
    # Implicit spring connecting each pair of junction boundary points.
    # Solved explicitly via the 2x2 linear system:
    #   gap_new = (gap_bar - 2*gamma_minus*gap_prev) / (1 + 2*gamma_plus)
    # With sigma_sc = 1: gamma_minus = 0, gamma_plus = lambda^2 * beta,
    # so gap_new = gap_bar / (1 + 2*lambda^2*beta).
    # The 1/(1+2*gamma_plus) denominator guarantees stability for all beta.
    beta_sc  = (1.0 - alf) / (alf + eps_sc)
    gamma_sc = lambdaSq * beta_sc * (1.0 + sigma_sc) / 2.0   # = lambda^2 * beta when sigma_sc = 1

    gap_bar_L = bNext[0]  - aNext[-1]                 # free gap at left  junction (a|b)
    corr_L    = gamma_sc * gap_bar_L / (1.0 + 2.0 * gamma_sc)
    aNext[-1] += corr_L
    bNext[0]  -= corr_L

    gap_bar_R = cNext[0]  - bNext[-1]                 # free gap at right junction (b|c)
    corr_R    = gamma_sc * gap_bar_R / (1.0 + 2.0 * gamma_sc)
    bNext[-1] += corr_R
    cNext[0]  -= corr_R
    if (n <= lengthSound//2):
        bNext[5] = 0

    out[n] = a[5]

    # --- Energy (kinetic + potential) ---
    a_vel = (a - aPrev) / k
    b_vel = (b - bPrev) / k
    c_vel = (c - cPrev) / k

    T = 0.5 * h * (np.sum(a_vel**2) + np.sum(b_vel**2) + np.sum(c_vel**2))
    V = (0.5 * cs**2 / h) * (
        np.sum(np.diff(np.concatenate(([0], a, [0])))**2) +
        np.sum(np.diff(np.concatenate(([0], b, [0])))**2) +
        np.sum(np.diff(np.concatenate(([0], c, [0])))**2)
    )
    energy[n] = T + V

    if n % storeEvery == 0:
        a_store.append(a.copy())
        b_store.append(b.copy())
        c_store.append(c.copy())
        Ma_store.append(Ma)

    aPrev, a = a.copy(), aNext.copy()
    bPrev, b = b.copy(), bNext.copy()
    cPrev, c = c.copy(), cNext.copy()
    MaPrev = Ma

# --- Save WAV ---
audio_max = np.max(np.abs(out))
if audio_max > 0:
    out_norm = out / audio_max
    write("sliding_3grid.wav", fs, (out_norm * 32767).astype(np.int16))

# --- Plot waveform ---
time = np.arange(lengthSound) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, out, 'b')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title('Output Waveform -- sliding 3-grid')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot energy ---
plt.figure(figsize=(10, 4))
plt.plot(time, energy, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Energy')
plt.title('Total Energy -- sliding 3-grid')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Animation: show a, b, c with their correct physical positions ---
fig, ax = plt.subplots(figsize=(12, 4))

line_a, = ax.plot([], [], 'b.-', linewidth=2, markersize=8,  label='a (left)')
line_b, = ax.plot([], [], 'g.-', linewidth=2, markersize=8,  label='b (middle, sliding)')
line_c, = ax.plot([], [], 'r.-', linewidth=2, markersize=8,  label='c (right)')

# Vertical lines marking b's boundaries
vline_left  = ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
vline_right = ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)

ax.set_xlim(0, L)
ax.set_ylim(-0.015, 0.015)
ax.set_xlabel('x (m)')
ax.set_ylabel('Displacement')
ax.set_title('Sliding 3-grid FDTD')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.4)


def init():
    line_a.set_data([], [])
    line_b.set_data([], [])
    line_c.set_data([], [])
    return line_a, line_b, line_c


def animate(i):
    a_f  = a_store[i]
    b_f  = b_store[i]
    c_f  = c_store[i]
    Ma_f = Ma_store[i]
    Mc_f = N_total - Ma_f - Nb

    # Physical x-positions: a owns [0 .. Ma_f * h], b follows, c takes the rest
    x_a = np.arange(Ma_f) * h
    x_b = (Ma_f + np.arange(Nb)) * h
    x_c = (Ma_f + Nb + np.arange(Mc_f)) * h

    line_a.set_data(x_a, a_f)
    line_b.set_data(x_b, b_f)
    line_c.set_data(x_c, c_f)

    vline_left.set_xdata([x_b[0],  x_b[0]])
    vline_right.set_xdata([x_b[-1], x_b[-1]])

    ax.set_title(
        f'Sliding 3-grid  |  frame {i * storeEvery}/{lengthSound}'
        f'  |  |a|={Ma_f}  |b|={Nb} (fixed)  |c|={Mc_f}'
        f'  |  N_total={N_total} (fixed)'
    )
    return line_a, line_b, line_c


anim = FuncAnimation(fig, animate, init_func=init,
                     frames=len(a_store), interval=50, blit=True)
plt.show()
