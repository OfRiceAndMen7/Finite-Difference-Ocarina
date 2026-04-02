"""
dynamic_wall.py
---------------
Generalised sliding trisegment (a | b | c) dynamic wall for FDTD simulations.

The wall is a 1-D array of N_total = Na + Nb + Nc cells that slides along its
own axis (the 'j' direction).  It can be embedded in a 2-D or 3-D parent FDTD
grid by passing the adjacent perpendicular rows at each timestep.

Fractional coupling
-------------------
Because b is fractionally offset from the parent grid by alf (the fractional
part of the a|b junction position), each b cell at integer local index kb has
a physical j-position of  Ma + kb + alf.  When sampling a parent-grid row for
the perpendicular coupling, we interpolate:

    w_low  = 1 - alf   at j_low  = Ma + kb
    w_high = alf        at j_high = Ma + kb + 1

a and c cells sit at exact integer positions and always use w_low=1, w_high=0.

Usage example (2-D parent grid, wall at row i=23)
-------------------------------------------------
    b_occ = np.zeros(22, dtype=bool)
    b_occ[1:20] = True                 # solid floor

    wall = DynamicWall(
        Na_init=4, Nb=22, Nc_init=14,
        b_occ=b_occ,
        slide_schedule=np.linspace(4, 14, Nt),
        lsq=0.25,
    )

    for n in range(Nt):
        wall.step([p[22, :], p[24, :]])   # 2 perp rows for 2-D

        # Read wall state for 2-D neighbour lookups
        ts_vals  = wall.get_values()       # shape (N_total,)
        ts_solid = wall.get_solid_mask()   # shape (N_total,), bool
"""

import numpy as np
import scipy.sparse as sp


class DynamicWall:
    """
    Sliding trisegment (a | b | c) dynamic wall.

    Parameters
    ----------
    Na_init : int
        Initial number of cells in left segment a.
    Nb : int
        Fixed number of cells in middle segment b.
    Nc_init : int
        Initial number of cells in right segment c.
    b_occ : array_like of bool, shape (Nb,)
        Occupancy mask for b: True = solid (rigid), False = air.
    slide_schedule : array_like of float, shape (Nt,)
        Fractional position of the a|b junction at each timestep.
        Ma = floor(slide_schedule[n]) gives the integer number of a cells;
        alf = slide_schedule[n] - Ma is the fractional offset.
    lsq : float
        Squared Courant number  lambda^2 = (c * k / h)^2.
    eps_sc : float, optional
        Regularisation for the state-correction denominator (default 1e-5).
    sigma_sc : float, optional
        State-correction spring damping (default 1.0, critical damping).
    """

    def __init__(self, Na_init, Nb, Nc_init, b_occ, slide_schedule, lsq,
                 eps_sc=1e-5, sigma_sc=1.0):

        self.Nb       = int(Nb)
        self.b_occ    = np.asarray(b_occ, dtype=bool)
        self.slide    = np.asarray(slide_schedule, dtype=float)
        self.lsq      = float(lsq)
        self.eps_sc   = float(eps_sc)
        self.sigma_sc = float(sigma_sc)

        assert len(self.b_occ) == self.Nb, "b_occ must have length Nb"

        self._step    = 0
        self.Ma       = int(Na_init)
        self.Mc       = int(Nc_init)
        self._Ma_prev = self.Ma
        self._N_total = self.Ma + self.Nb + self.Mc  # fixed throughout

        self.a     = np.zeros(self.Ma);   self.aPrev = np.zeros(self.Ma)
        self.b     = np.zeros(self.Nb);   self.bPrev = np.zeros(self.Nb)
        self.c     = np.zeros(self.Mc);   self.cPrev = np.zeros(self.Mc)

        self._Dxxa = _make_Dxx(self.Ma)
        self._Dxxb = _make_Dxx(self.Nb)
        self._Dxxc = _make_Dxx(self.Mc)

        # Precompute which air cells in b need a Neumann correction because
        # one of their horizontal neighbours is solid.
        # Each entry: (local_k, correction_value_getter)
        # We store the index and add lsq * b[k] to bNext[k] per solid neighbour.
        self._b_neumann_k = []
        for k in range(self.Nb):
            if not self.b_occ[k]:
                count = 0
                if k + 1 < self.Nb and self.b_occ[k + 1]:
                    count += 1   # right neighbour solid
                if k - 1 >= 0 and self.b_occ[k - 1]:
                    count += 1   # left neighbour solid
                if count:
                    self._b_neumann_k.append((k, count))

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def N_total(self):
        """Total number of cells (fixed)."""
        return self._N_total

    @property
    def alf(self):
        """Fractional part of the current a|b junction position."""
        pab = self.slide[self._step]
        return pab - int(np.floor(pab))

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_values(self):
        """
        Current pressure values for all N_total cells as a 1-D array.
        Solid cells always return 0.
        """
        return np.concatenate([self.a, self.b, self.c])

    def get_solid_mask(self):
        """
        Boolean solid mask for all N_total cells.
        True = solid (no FDTD update, acts as rigid wall).
        """
        mask = np.zeros(self._N_total, dtype=bool)
        mask[self.Ma: self.Ma + self.Nb] = self.b_occ
        return mask

    # ------------------------------------------------------------------
    # Fractional coupling weights
    # ------------------------------------------------------------------

    def get_perp_coupling_weights(self, global_j):
        """
        Return (j_low, j_high, w_low, w_high) for bilinear interpolation
        into a parent-grid row when sampling the perpendicular coupling for
        the cell at integer global index `global_j`.

        * b cells (Ma <= global_j < Ma+Nb) are offset by alf → fractional
          interpolation between global_j and global_j+1.
        * a and c cells sit at exact integer positions → w_low=1, w_high=0.
        """
        if self.Ma <= global_j < self.Ma + self.Nb:
            alf = self.alf
            return global_j, global_j + 1, 1.0 - alf, alf
        else:
            return global_j, global_j, 1.0, 0.0

    # ------------------------------------------------------------------
    # Main timestep
    # ------------------------------------------------------------------

    def step(self, perp_neighbor_rows):
        """
        Advance one timestep.

        Parameters
        ----------
        perp_neighbor_rows : sequence of 1-D arrays
            Parent-grid rows that are adjacent to the wall in the
            perpendicular direction(s).  The Laplacian contribution from
            these rows is  sum(rows) - len(rows) * self_value,
            so pass:
              * [row_above, row_below]           for a 2-D parent grid
              * [row_above, row_below,
                 row_front, row_back]            for a 3-D parent grid
            Each array must be indexed by global j (length >= N_total + 1).
        """
        assert self._step < len(self.slide), "step() called beyond slide schedule"

        pab    = self.slide[self._step]
        Ma_new = int(np.floor(pab))
        alf    = pab - Ma_new
        Mc_new = self._N_total - Ma_new - self.Nb
        dMa    = Ma_new - self._Ma_prev

        if abs(dMa) > 1:
            raise RuntimeError(
                f"DynamicWall: b moved {dMa} grid spacings in one step (max ±1)")

        # ── Grid management ────────────────────────────────────────────
        if dMa != 0:
            cip = np.array([
                alf*(alf+1) / (-(alf+2)*(alf+3)),
                2*alf    / (alf+2),
                2        / (alf+2),
                2*alf    / (-(alf+3)*(alf+2))
            ])
            if dMa > 0:               # b slides right: a gains, c loses
                na  = cip @ np.concatenate([self.a[-2:],     self.b[:2]])
                nap = cip @ np.concatenate([self.aPrev[-2:], self.bPrev[:2]])
                self.a     = np.append(self.a,     na)
                self.aPrev = np.append(self.aPrev, nap)
                self.c     = self.c[1:]
                self.cPrev = self.cPrev[1:]
            else:                     # b slides left: c gains, a loses
                nc  = np.flip(cip) @ np.concatenate([self.b[-2:],     self.c[:2]])
                ncp = np.flip(cip) @ np.concatenate([self.bPrev[-2:], self.cPrev[:2]])
                self.c     = np.concatenate([[nc],  self.c])
                self.cPrev = np.concatenate([[ncp], self.cPrev])
                self.a     = self.a[:-1]
                self.aPrev = self.aPrev[:-1]
            self.Ma = Ma_new;  self.Mc = Mc_new
            self._Dxxa = _make_Dxx(len(self.a))
            self._Dxxc = _make_Dxx(len(self.c))

        # ── Virtual junction points ────────────────────────────────────
        ip = np.array([-(alf-1)/(alf+1), 1.0, (alf-1)/(alf+1)])

        vgp_ar = ip[2]*self.a[-1] + ip[1]*self.b[0]  + ip[0]*self.b[1]
        vgp_bl = ip[0]*self.a[-2] + ip[1]*self.a[-1] + ip[2]*self.b[0]
        vgp_br = ip[2]*self.b[-1] + ip[1]*self.c[0]  + ip[0]*self.c[1]
        vgp_cl = ip[0]*self.b[-2] + ip[1]*self.b[-1] + ip[2]*self.c[0]

        # ── 1-D leapfrog (horizontal) ──────────────────────────────────
        aNext = 2*self.a + self.lsq * self._Dxxa.dot(self.a) - self.aPrev
        aNext[-1] += self.lsq * vgp_ar

        bNext = 2*self.b + self.lsq * self._Dxxb.dot(self.b) - self.bPrev
        bNext[0]  += self.lsq * vgp_bl
        bNext[-1] += self.lsq * vgp_br

        cNext = 2*self.c + self.lsq * self._Dxxc.dot(self.c) - self.cPrev
        cNext[0]  += self.lsq * vgp_cl

        # ── Neumann corrections at solid-air interfaces inside b ───────
        # For each air cell k adjacent to a solid cell, Dxx used 0 for that
        # neighbour; correct it to b[k] (zero normal-gradient condition).
        for k, n_solid_nbrs in self._b_neumann_k:
            bNext[k] += self.lsq * n_solid_nbrs * self.b[k]

        # ── Perpendicular coupling (integer j positions) ───────────────
        n_perp = len(perp_neighbor_rows)

        def _sample(row, j_l, j_h, w_l, w_h):
            """Bilinearly-sample one parent-grid row at a fractional index."""
            v = 0.0
            if 0 <= j_l < len(row):
                v += w_l * row[j_l]
            if j_h != j_l and 0 <= j_h < len(row):
                v += w_h * row[j_h]
            return v

        # a cells — exact integer j positions
        for ka in range(len(self.a)):
            perp = sum(_sample(row, ka, ka, 1.0, 0.0) for row in perp_neighbor_rows)
            aNext[ka] += self.lsq * (perp - n_perp * self.a[ka])

        # b cells — integer j position (floor of fractional offset)
        for kb in range(self.Nb):
            if not self.b_occ[kb]:
                jg   = self.Ma + kb
                perp = sum(_sample(row, jg, jg, 1.0, 0.0) for row in perp_neighbor_rows)
                bNext[kb] += self.lsq * (perp - n_perp * self.b[kb])

        # c cells — exact integer j positions
        for kc in range(len(self.c)):
            jg   = self.Ma + self.Nb + kc
            perp = sum(_sample(row, jg, jg, 1.0, 0.0) for row in perp_neighbor_rows)
            cNext[kc] += self.lsq * (perp - n_perp * self.c[kc])

        # ── Solid cells stay at zero ───────────────────────────────────
        bNext[self.b_occ] = 0.0

        # ── State correction (Willemsen 2021 §4.4) ────────────────────
        beta_sc = (1.0 - alf) / (alf + self.eps_sc)
        gam_sc  = self.lsq * beta_sc * (1.0 + self.sigma_sc) / 2.0

        gL = bNext[0]  - aNext[-1]
        cL = gam_sc * gL / (1.0 + 2.0*gam_sc)
        aNext[-1] += cL;   bNext[0]  -= cL

        gR = cNext[0]  - bNext[-1]
        cR = gam_sc * gR / (1.0 + 2.0*gam_sc)
        bNext[-1] += cR;   cNext[0]  -= cR

        # ── Advance state ─────────────────────────────────────────────
        self.aPrev, self.a = self.a, aNext
        self.bPrev, self.b = self.b, bNext
        self.cPrev, self.c = self.c, cNext
        self._Ma_prev = self.Ma
        self._step += 1


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _make_Dxx(n):
    e = np.ones(n)
    return sp.diags([e, -2*e, e], [-1, 0, 1], shape=(n, n), format='csr')
