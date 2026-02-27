
"""
analyze_grid.py

Reads XSPEC log files named like:
  M12_R10_D56.txt   (meaning M=1.2, R=10, D=5.6)

Extracts:
- Best-fit LogT_eff values from the "!XSPEC12>show free" block
- Confidence intervals from the "!XSPEC12>uncertain 1.0 2 12 22 32 42" table

Outputs:
- grid_temps.csv (one row per file; includes per-group temps + combined)
- 3D grid plot similar to HETE2021 Fig. 4 style
- Optional MCMC posterior samples over (M, R, D) given an observed Teff constraint
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except Exception:
    go = None


# ---------- constants ----------
K_B_eV_per_K = 8.617333262e-5  # Boltzmann constant in eV/K


# ---------- parsing helpers ----------
FNAME_RE = re.compile(r"M(?P<M>\d+)_R(?P<R>\d+)_D(?P<D>\d+)\.txt$")

# In your logs, the "uncertain" table looks like:
#  # Parameter   Confidence Range (1)
#  #     2      6.26359      6.27568    (-0.006...,0.005...)
UNCER_ROW_RE = re.compile(
    r"^\s*#\s*(?P<par>\d+)\s+(?P<lo>[0-9]*\.?[0-9]+)\s+(?P<hi>[0-9]*\.?[0-9]+)"
)

# In your "show free" block, lines look like:
#  #   2    2   nsatmos    LogT_eff   K        6.27002      +/-  5.42326E-03
SHOWFREE_ROW_RE = re.compile(
    r"^\s*#\s*(?P<par>\d+)\s+\d+\s+\S+\s+LogT_eff\s+K\s+(?P<val>[0-9]*\.?[0-9]+)"
)

TARGET_PARS = [2, 12, 22, 32, 42]

DEFAULT_OUTDIR = Path("out")


@dataclass
class TempEntry:
    """Temperatures for one file / one (M,R,D) point."""
    M: float
    R: float
    D: float
    # per-group best-fit logT
    logT: Dict[int, float]
    # per-group (lo, hi) in logT from "uncertain"
    logT_ci: Dict[int, Tuple[float, float]]


def parse_filename_to_mrd(path: Path) -> Tuple[float, float, float]:
    m = FNAME_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename does not match pattern M##_R##_D##.txt: {path.name}")

    Mi = int(m.group("M"))   # e.g. 12 means 1.2
    R = float(m.group("R"))  # km
    Di = int(m.group("D"))   # e.g. 56 means 5.6

    M = Mi / 10.0
    D = Di / 10.0
    return M, R, D


def extract_showfree_logT(lines: List[str]) -> Dict[int, float]:
    """
    Extract best-fit logT values from the "!XSPEC12>show free" block.
    We search the entire file for matching 'LogT_eff' rows and keep par numbers.
    """
    out: Dict[int, float] = {}
    for ln in lines:
        m = SHOWFREE_ROW_RE.match(ln)
        if m:
            par = int(m.group("par"))
            if par in TARGET_PARS:
                out[par] = float(m.group("val"))
    return out


def extract_uncertain_ci(lines: List[str]) -> Dict[int, Tuple[float, float]]:
    """
    Extract (lo, hi) bounds for each parameter from the "!XSPEC12>uncertain ..." table.
    """
    out: Dict[int, Tuple[float, float]] = {}
    for ln in lines:
        m = UNCER_ROW_RE.match(ln)
        if m:
            par = int(m.group("par"))
            if par in TARGET_PARS:
                lo = float(m.group("lo"))
                hi = float(m.group("hi"))
                out[par] = (lo, hi)
    return out


def logT_to_kT_eV(logT: float) -> float:
    """Convert log10(T[K]) to kT in eV."""
    T = 10.0 ** logT
    return K_B_eV_per_K * T


def ci_logT_to_kT_eV(ci: Tuple[float, float]) -> Tuple[float, float]:
    """Convert (logT_lo, logT_hi) to (kT_lo_eV, kT_hi_eV)."""
    lo, hi = ci
    return logT_to_kT_eV(lo), logT_to_kT_eV(hi)


def weighted_mean_and_sigma(values: np.ndarray, sigmas: np.ndarray) -> Tuple[float, float]:
    """
    Inverse-variance weighted mean + 1-sigma uncertainty (approx).
    For asymmetric errors, we’ll precompute a symmetric sigma in eV.
    """
    w = 1.0 / np.clip(sigmas, 1e-30, np.inf) ** 2
    mu = np.sum(w * values) / np.sum(w)
    sigma = math.sqrt(1.0 / np.sum(w))
    return float(mu), float(sigma)


def read_one_file(path: Path) -> TempEntry:
    M, R, D = parse_filename_to_mrd(path)
    lines = path.read_text(errors="replace").splitlines()

    logT = extract_showfree_logT(lines)
    logT_ci = extract_uncertain_ci(lines)

    missing = [p for p in TARGET_PARS if p not in logT or p not in logT_ci]
    if missing:
        raise ValueError(f"{path.name}: missing entries for parameters {missing}. "
                         f"Check that 'show free' and 'uncertain' blocks exist in the log.")

    return TempEntry(M=M, R=R, D=D, logT=logT, logT_ci=logT_ci)


# ---------- interpolation on a (M,R,D) grid ----------
def build_grid(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a regular 3D grid of value_col over unique M, R, D values.
    Returns (Mvals, Rvals, Dvals, grid) where grid shape is (len(M), len(R), len(D)).
    """
    Mvals = np.sort(df["M"].unique())
    Rvals = np.sort(df["R"].unique())
    Dvals = np.sort(df["D"].unique())

    grid = np.full((len(Mvals), len(Rvals), len(Dvals)), np.nan, dtype=float)

    idxM = {v: i for i, v in enumerate(Mvals)}
    idxR = {v: i for i, v in enumerate(Rvals)}
    idxD = {v: i for i, v in enumerate(Dvals)}

    for _, row in df.iterrows():
        grid[idxM[row["M"]], idxR[row["R"]], idxD[row["D"]]] = row[value_col]

    if np.isnan(grid).any():
        raise ValueError("Grid has missing points (NaNs). Ensure you have a complete M-R-D set of files.")
    return Mvals, Rvals, Dvals, grid


def trilinear_interp(M: float, R: float, D: float,
                     Mvals: np.ndarray, Rvals: np.ndarray, Dvals: np.ndarray,
                     grid: np.ndarray) -> float:
    """
    Trilinear interpolation within the convex hull of the grid.
    Assumes M, R, D are within min/max bounds.
    """
    def find_bounds(x: float, arr: np.ndarray) -> Tuple[int, int, float]:
        if x <= arr[0]:
            return 0, 0, 0.0
        if x >= arr[-1]:
            return len(arr)-1, len(arr)-1, 0.0
        hi = int(np.searchsorted(arr, x))
        lo = hi - 1
        t = (x - arr[lo]) / (arr[hi] - arr[lo])
        return lo, hi, float(t)

    i0, i1, tx = find_bounds(M, Mvals)
    j0, j1, ty = find_bounds(R, Rvals)
    k0, k1, tz = find_bounds(D, Dvals)

    # corners
    c000 = grid[i0, j0, k0]
    c001 = grid[i0, j0, k1]
    c010 = grid[i0, j1, k0]
    c011 = grid[i0, j1, k1]
    c100 = grid[i1, j0, k0]
    c101 = grid[i1, j0, k1]
    c110 = grid[i1, j1, k0]
    c111 = grid[i1, j1, k1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    c = c0 * (1 - tz) + c1 * tz
    return float(c)


# ---------- MCMC (simple Metropolis) ----------
def log_prior(m: float, r: float, d: float,
              mmin: float, mmax: float, rmin: float, rmax: float, dmin: float, dmax: float) -> float:
    if (mmin <= m <= mmax) and (rmin <= r <= rmax) and (dmin <= d <= dmax):
        return 0.0  # uniform prior in box
    return -np.inf


def log_likelihood(teff_obs: float, sigma_obs: float, teff_model: float) -> float:
    # Gaussian likelihood
    return -0.5 * ((teff_obs - teff_model) / sigma_obs) ** 2 - math.log(sigma_obs * math.sqrt(2 * math.pi))


def compute_grid_posterior_weights(
    teff_obs: float,
    sigma_obs: float,
    grid: np.ndarray,
) -> np.ndarray:
    """Compute normalized posterior weights over a discrete (M,R,D) grid."""
    logw = -0.5 * ((teff_obs - grid) / sigma_obs) ** 2
    logw -= np.nanmax(logw)  # numerical stability
    w = np.exp(logw)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("Posterior weights are invalid; check teff_obs/sigma/grid values.")
    return w / total


def sample_from_grid_posterior(
    weights: np.ndarray,
    Mvals: np.ndarray,
    Rvals: np.ndarray,
    Dvals: np.ndarray,
    nsteps: int,
    burn: int,
    seed: int,
) -> np.ndarray:
    """Draw independent samples from a discrete posterior defined on the grid."""
    if burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps")

    rng = np.random.default_rng(seed)
    flat_w = weights.ravel()
    draw_idx = rng.choice(flat_w.size, size=nsteps - burn, p=flat_w)
    ii, jj, kk = np.unravel_index(draw_idx, weights.shape)
    return np.column_stack([Mvals[ii], Rvals[jj], Dvals[kk]])


def metropolis_mcmc(
    teff_obs: float,
    sigma_obs: float,
    Mvals: np.ndarray,
    Rvals: np.ndarray,
    Dvals: np.ndarray,
    grid: np.ndarray,
    nsteps: int = 20000,
    burn: int = 5000,
    step_scales: Tuple[float, float, float] = (0.05, 0.5, 0.3),
    seed: int = 0,
) -> np.ndarray:
    """
    Samples posterior p(M,R,D | teff_obs) where teff_model(M,R,D) is from interpolated grid.
    step_scales are proposal std devs in (Msun, km, kpc).
    """
    rng = np.random.default_rng(seed)

    mmin, mmax = float(Mvals.min()), float(Mvals.max())
    rmin, rmax = float(Rvals.min()), float(Rvals.max())
    dmin, dmax = float(Dvals.min()), float(Dvals.max())

    # start in the middle
    m = float(np.median(Mvals))
    r = float(np.median(Rvals))
    d = float(np.median(Dvals))

    def log_post(mm: float, rr: float, dd: float) -> float:
        lp = log_prior(mm, rr, dd, mmin, mmax, rmin, rmax, dmin, dmax)
        if not np.isfinite(lp):
            return -np.inf
        tm = trilinear_interp(mm, rr, dd, Mvals, Rvals, Dvals, grid)
        return lp + log_likelihood(teff_obs, sigma_obs, tm)

    cur = log_post(m, r, d)
    samples = np.zeros((nsteps, 3), dtype=float)
    accept = 0

    for t in range(nsteps):
        mp = m + rng.normal(0, step_scales[0])
        rp = r + rng.normal(0, step_scales[1])
        dp = d + rng.normal(0, step_scales[2])

        prop = log_post(mp, rp, dp)
        if np.isfinite(prop) and (math.log(rng.random()) < (prop - cur)):
            m, r, d = mp, rp, dp
            cur = prop
            accept += 1

        samples[t] = (m, r, d)

    samples = samples[burn:]
    print(f"[MCMC] acceptance fraction ~ {accept / nsteps:.3f}")
    return samples

def metropolis_mcmc_discrete(
    teff_obs: float,
    sigma_obs: float,
    Mvals: np.ndarray,
    Rvals: np.ndarray,
    Dvals: np.ndarray,
    grid: np.ndarray,
    nsteps: int = 20000,
    burn: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    i = len(Mvals) // 2
    j = len(Rvals) // 2
    k = len(Dvals) // 2

    def teff_at(ii, jj, kk):
        return float(grid[ii, jj, kk])

    def log_post(ii, jj, kk):
        tm = teff_at(ii, jj, kk)
        return log_likelihood(teff_obs, sigma_obs, tm)

    def proposal_log_prob(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        """q(b|a) for the discrete proposal used below."""
        ai, aj, ak = a
        bi, bj, bk = b

        # one third chance to choose each axis
        if aj == bj and ak == bk and abs(ai - bi) == 1:
            if ai in (0, len(Mvals) - 1):
                p = 1.0 / 3.0
            else:
                p = 1.0 / 6.0
            return math.log(p)

        if ai == bi and ak == bk and abs(aj - bj) == 1:
            if aj in (0, len(Rvals) - 1):
                p = 1.0 / 3.0
            else:
                p = 1.0 / 6.0
            return math.log(p)

        if ai == bi and aj == bj and bk != ak:
            p = (1.0 / 3.0) * (1.0 / (len(Dvals) - 1))
            return math.log(p)

        return -np.inf

    cur = log_post(i, j, k)
    samples = np.zeros((nsteps, 3), dtype=float)
    accept = 0
    proposals = 0

    for t in range(nsteps):
        # Propose a NEW state (no clipping-to-same allowed)
        ip, jp, kp = i, j, k
        move = rng.integers(0, 3)  # 0=M, 1=R, 2=D

        if move == 0:
            if i == 0:
                ip = 1
            elif i == len(Mvals) - 1:
                ip = len(Mvals) - 2
            else:
                ip = i + rng.choice([-1, 1])

        elif move == 1:
            if j == 0:
                jp = 1
            elif j == len(Rvals) - 1:
                jp = len(Rvals) - 2
            else:
                jp = j + rng.choice([-1, 1])

        else:
            # choose a different distance index
            choices = [kk for kk in range(len(Dvals)) if kk != k]
            kp = int(rng.choice(choices))

        # sanity: ensure it's a real move
        if (ip, jp, kp) == (i, j, k):
            samples[t] = (Mvals[i], Rvals[j], Dvals[k])
            continue

        proposals += 1
        prop = log_post(ip, jp, kp)

        fwd = proposal_log_prob((i, j, k), (ip, jp, kp))
        rev = proposal_log_prob((ip, jp, kp), (i, j, k))
        log_alpha = prop - cur + rev - fwd

        if np.isfinite(prop) and np.isfinite(log_alpha) and (math.log(rng.random()) < log_alpha):
            i, j, k = ip, jp, kp
            cur = prop
            accept += 1

        samples[t] = (Mvals[i], Rvals[j], Dvals[k])

    samples = samples[burn:]
    acc = accept / max(proposals, 1)
    print(f"[MCMC-discrete] acceptance fraction ~ {acc:.3f} (excluding no-op moves)")
    return samples

# ---------- plotting (Fig.4-like wireframes) ----------
def plot_3d_grid(df: pd.DataFrame, outpath: Path, zcol: str = "kT_comb_eV", title: str = "Temperature grid") -> None:
    """
    Plot Teff (kT in eV) vs (M,R) for each D separately, similar to HETE2021 Fig. 4.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Save a compact CSV of the grid used for this 3D plot (per-output)
    datafile = outpath.parent / f"{outpath.stem}_data.csv"
    df[["M", "R", "D", zcol]].to_csv(datafile, index=False)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Per-distance style mapping (user request)
    style_map = {
        5.6: {"color": "blue", "linestyle": ":"},
        7.3: {"color": "green", "linestyle": "--"},
        8.8: {"color": "black", "linestyle": "-"},
    }

    # For visibility, plot mesh lines explicitly (rows + cols) with thicker lines
    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].copy()
        Mvals = np.sort(sub["M"].unique())
        Rvals = np.sort(sub["R"].unique())
        Z = np.empty((len(Mvals), len(Rvals)), dtype=float)
        for i, m in enumerate(Mvals):
            for j, r in enumerate(Rvals):
                Z[i, j] = float(sub[(sub["M"] == m) & (sub["R"] == r)][zcol].iloc[0])

        # style lookup with tolerance for floating point representation
        sty = None
        for k, v in style_map.items():
            if abs(float(D) - float(k)) < 1e-6:
                sty = v
                break
        if sty is None:
            sty = {"color": "C0", "linestyle": "-"}

        color = sty["color"]
        ls = sty["linestyle"]

        # Draw lines along R for each M, and along M for each R
        for i in range(len(Mvals)):
            xs = np.full(len(Rvals), Mvals[i])
            ys = Rvals
            zs = Z[i, :]
            ax.plot(xs, ys, zs, color=color, linestyle=ls, linewidth=2.0, alpha=0.95)

        for j in range(len(Rvals)):
            xs = Mvals
            ys = np.full(len(Mvals), Rvals[j])
            zs = Z[:, j]
            ax.plot(xs, ys, zs, color=color, linestyle=ls, linewidth=2.0, alpha=0.95)

    ax.set_xlabel("Mass (Msun)")
    ax.set_ylabel("Radius (km)")
    ax.set_zlabel(f"{zcol}")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=250)
    plt.close(fig)

def plot_3d_interactive(df: pd.DataFrame, outpath: Path, zcol: str = "kT_comb_eV", title: str = "Temperature grid") -> None:
    """Create an interactive Plotly HTML showing 3D lines per distance D."""
    if go is None:
        raise ImportError("plotly is required for interactive plotting. Install with `pip install plotly`.")

    # Per-distance style mapping (matching static plot)
    style_map = {
        5.6: {"color": "blue", "dash": "dot"},
        7.3: {"color": "green", "dash": "dash"},
        8.8: {"color": "black", "dash": "solid"},
    }

    fig = go.Figure()

    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].copy()
        Mvals = np.sort(sub["M"].unique())
        Rvals = np.sort(sub["R"].unique())
        Z = np.empty((len(Mvals), len(Rvals)), dtype=float)
        for i, m in enumerate(Mvals):
            for j, r in enumerate(Rvals):
                Z[i, j] = float(sub[(sub["M"] == m) & (sub["R"] == r)][zcol].iloc[0])

        # lookup style
        sty = None
        for k, v in style_map.items():
            if abs(float(D) - float(k)) < 1e-6:
                sty = v
                break
        if sty is None:
            sty = {"color": "steelblue", "dash": "solid"}

        # add lines along R for each M
        for i in range(len(Mvals)):
            xs = np.full(len(Rvals), Mvals[i])
            ys = Rvals
            zs = Z[i, :]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=sty["color"], width=4, dash=sty["dash"]),
                    hoverinfo="text",
                    text=[f"D={D}, M={xs[0]}, R={y}, {zcol}={z:.3f}" for y, z in zip(ys, zs)],
                    name=f"D={D}",
                    showlegend=(i == 0),
                )
            )

        # add lines along M for each R (no duplicate legend)
        for j in range(len(Rvals)):
            xs = Mvals
            ys = np.full(len(Mvals), Rvals[j])
            zs = Z[:, j]
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=sty["color"], width=4, dash=sty["dash"]),
                    hoverinfo="text",
                    text=[f"D={D}, M={x}, R={ys[0]}, {zcol}={z:.3f}" for x, z in zip(xs, zs)],
                    name=f"D={D}",
                    showlegend=False,
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="Mass (Msun)",
            yaxis_title="Radius (km)",
            zaxis_title=zcol,
        ),
        title=title,
        legend=dict(title="Distance (kpc)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(outpath, include_plotlyjs="cdn")


def plot_3d_evolution(df: pd.DataFrame, outpath: Path, zcols: list, steps_per_transition: int = 20, title: str = "3D evolution") -> None:
    """Create a Plotly HTML animation that smoothly interpolates between multiple z-columns.

    zcols: ordered list of column names (e.g., ['kT_g1_eV', ..., 'kT_g5_eV'])
    steps_per_transition: number of intermediate frames between consecutive spectra
    """
    if go is None:
        raise ImportError("plotly is required for interactive animation. Install with `pip install plotly`.")

    # prepare grids (assume same M,R grid for all)
    Mvals = np.sort(df["M"].unique())
    Rvals = np.sort(df["R"].unique())

    # Build Z array for each spectrum: shape (n_spec, len(M), len(R))
    Zs = []
    for zcol in zcols:
        Z = np.empty((len(Mvals), len(Rvals)), dtype=float)
        for i, m in enumerate(Mvals):
            for j, r in enumerate(Rvals):
                Z[i, j] = float(df[(df["M"] == m) & (df["R"] == r)][zcol].iloc[0])
        Zs.append(Z)

    # Create sequence of frames by linear interpolation between consecutive Zs
    frames = []
    spec_times = []
    for s in range(len(Zs) - 1):
        Z0 = Zs[s]
        Z1 = Zs[s + 1]
        for t in range(steps_per_transition):
            alpha = t / float(steps_per_transition)
            Zt = (1 - alpha) * Z0 + alpha * Z1
            frames.append(Zt)
            spec_times.append(s + alpha)
    # append final spectrum frame
    frames.append(Zs[-1])
    spec_times.append(len(Zs) - 1)

    # Build initial traces (use first frame)
    initZ = frames[0]

    # Compute global axis ranges so axes remain static during animation
    all_z_min = min([Z.min() for Z in Zs])
    all_z_max = max([Z.max() for Z in Zs])
    zpad = 0.05 * (all_z_max - all_z_min) if all_z_max > all_z_min else 0.1 * all_z_max
    zmin = all_z_min - zpad
    zmax = all_z_max + zpad

    xmin = float(Mvals[0])
    xmax = float(Mvals[-1])
    xpad = 0.02 * (xmax - xmin) if xmax > xmin else 0.1 * max(1.0, xmax)
    xmin -= xpad
    xmax += xpad

    ymin = float(Rvals[0])
    ymax = float(Rvals[-1])
    ypad = 0.02 * (ymax - ymin) if ymax > ymin else 0.1 * max(1.0, ymax)
    ymin -= ypad
    ymax += ypad
    traces = []
    trace_meta = []
    style_map = {
        5.6: {"color": "blue", "dash": "dot"},
        7.3: {"color": "green", "dash": "dash"},
        8.8: {"color": "black", "dash": "solid"},
    }

    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].copy()
        Mvals_local = np.sort(sub["M"].unique())
        Rvals_local = np.sort(sub["R"].unique())

        sty = None
        for k, v in style_map.items():
            if abs(float(D) - float(k)) < 1e-6:
                sty = v
                break
        if sty is None:
            sty = {"color": "steelblue", "dash": "solid"}

        # lines along R for each M
        for i in range(len(Mvals_local)):
            xs = np.full(len(Rvals_local), Mvals_local[i])
            ys = Rvals_local
            zs = initZ[i, :]
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=sty["color"], width=4, dash=sty["dash"]),
                                       name=f"D={D}", showlegend=(i == 0)))
            trace_meta.append((D, "MR", i))

        # lines along M for each R
        for j in range(len(Rvals_local)):
            xs = Mvals_local
            ys = np.full(len(Mvals_local), Rvals_local[j])
            zs = initZ[:, j]
            traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=sty["color"], width=4, dash=sty["dash"]),
                                       name=f"D={D}", showlegend=False))
            trace_meta.append((D, "RM", j))

    # Build plot and frames data: each frame must supply updated 'z' for each trace
    fig = go.Figure(data=traces)
    plot_frames = []
    Mvals_sorted = np.sort(df["M"].unique())
    Rvals_sorted = np.sort(df["R"].unique())
    for idx, Zt in enumerate(frames):
        data = []
        # build full Scatter3d traces for this frame in the same order as `traces`/`trace_meta`
        for ti, meta in enumerate(trace_meta):
            Dmeta, typ, ind = meta
            orig = traces[ti]
            if typ == "MR":
                xs = np.full(len(Rvals_sorted), Mvals_sorted[ind])
                ys = Rvals_sorted
                zs = Zt[ind, :]
            else:
                xs = Mvals_sorted
                ys = np.full(len(Mvals_sorted), Rvals_sorted[ind])
                zs = Zt[:, ind]

            trace = go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                 line=orig.line, name=orig.name, showlegend=orig.showlegend)
            data.append(trace)

        plot_frames.append(go.Frame(data=data, name=str(idx), layout=dict(title=f"{title} (t={spec_times[idx]:.2f})")))

    fig.frames = plot_frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=1.1,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
        scene=dict(
            xaxis=dict(title="Mass (Msun)", range=[xmin, xmax], autorange=False),
            yaxis=dict(title="Radius (km)", range=[ymin, ymax], autorange=False),
            zaxis=dict(title=zcols[0], range=[zmin, zmax], autorange=False),
        ),
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    fig.write_html(outpath, include_plotlyjs="cdn")


def analyze_cooling_rates(df: pd.DataFrame, outdir: Path) -> None:
    """Compute linear cooling rate across the five spectra for each (M,R,D).

    - Fits slope vs spectrum index (1..5) for columns kT_g1_eV..kT_g5_eV
    - Saves `outdir/data/cooling_rates.csv` with slopes per (M,R,D)
    - Produces boxplot per distance saved to `outdir/3d/cooling_rates_boxplot.png`
    - Runs basic statistical tests (ANOVA + pairwise t-tests) if scipy is available
    """
    required = [f"kT_g{g}_eV" for g in [1, 2, 3, 4, 5]]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Missing required column {c} for cooling analysis")

    x = np.array([1, 2, 3, 4, 5], dtype=float)
    rows = []
    for _, r in df.iterrows():
        y = np.array([r[f"kT_g{g}_eV"] for g in [1, 2, 3, 4, 5]], dtype=float)
        # linear fit slope per spectrum index
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({"M": r["M"], "R": r["R"], "D": r["D"], "slope_eV_per_spec": float(slope), "intercept": float(intercept)})

    outdf = pd.DataFrame(rows).sort_values(["D", "M", "R"]).reset_index(drop=True)
    outcsv = outdir / "data" / "cooling_rates.csv"
    outdf.to_csv(outcsv, index=False)

    # boxplot of slopes per distance
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = []
    labels = []
    for D in sorted(outdf["D"].unique()):
        vals = outdf[outdf["D"] == D]["slope_eV_per_spec"].to_numpy()
        groups.append(vals)
        labels.append(str(D))

    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_xlabel("Distance (kpc)")
    ax.set_ylabel("Cooling rate (eV per spectrum index)")
    ax.set_title("Cooling-rate distribution by distance")
    fig.tight_layout()
    outpng = outdir / "cooling" / "cooling_rates_boxplot.png"
    fig.savefig(outpng, dpi=200)
    plt.close(fig)

    print(f"[OK] Wrote {outcsv} and {outpng}")

    # Statistical tests if scipy available
    try:
        import scipy.stats as stats
    except Exception:
        print("[WARN] scipy not available; skipping formal statistical tests. Slopes saved to CSV for offline analysis.")
        return

    # ANOVA across distances
    samples = [outdf[outdf["D"] == D]["slope_eV_per_spec"].to_numpy() for D in sorted(outdf["D"].unique())]
    try:
        fval, pval = stats.f_oneway(*samples)
        print(f"ANOVA F={fval:.4g}, p={pval:.4g}")
    except Exception:
        print("[WARN] ANOVA failed; possibly unequal sizes or constant groups")

    # Pairwise t-tests with Bonferroni
    Ds = sorted(outdf["D"].unique())
    ncomb = 0
    for i in range(len(Ds)):
        for j in range(i + 1, len(Ds)):
            a = outdf[outdf["D"] == Ds[i]]["slope_eV_per_spec"].to_numpy()
            b = outdf[outdf["D"] == Ds[j]]["slope_eV_per_spec"].to_numpy()
            try:
                t, p = stats.ttest_ind(a, b, equal_var=False)
            except Exception:
                t, p = (np.nan, np.nan)
            ncomb += 1
            print(f"t-test D={Ds[i]} vs D={Ds[j]}: t={t:.4g}, p={p:.4g}")


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Folder with XSPEC log files (e.g. grid_logs/)")
    ap.add_argument("--pattern", type=str, default="M*_R*_D*.txt", help="Glob pattern for log files")
    ap.add_argument("--outcsv", type=str, default="grid_temps.csv", help="Output CSV")
    ap.add_argument("--outplot", type=str, default="grid_3d.png", help="Output 3D plot")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Output folder")
    ap.add_argument("--run-mcmc", action="store_true", help="Run MCMC over (M,R,D)")
    ap.add_argument("--teff-obs-ev", type=float, default=None, help="Observed kT (eV) for MCMC")
    ap.add_argument("--sigma-ev", type=float, default=None, help="1-sigma uncertainty on observed kT (eV) for MCMC")
    ap.add_argument("--mcmc-steps", type=int, default=30000)
    ap.add_argument("--mcmc-burn", type=int, default=8000)
    ap.add_argument("--mcmc-seed", type=int, default=0)
    ap.add_argument(
        "--sampler",
        choices=["grid", "mh-discrete"],
        default="grid",
        help="Posterior sampler: exact grid sampling (recommended) or discrete MH.",
    )
    ap.add_argument("--interactive", action="store_true", help="Write an interactive Plotly HTML 3D plot to out/3d/")
    ap.add_argument("--animate", action="store_true", help="Write an animated Plotly HTML showing time evolution between spectra")
    ap.add_argument("--anim-steps", type=int, default=20, help="Interpolation steps per transition for animation")
    ap.add_argument("--analyze-cooling", action="store_true", help="Perform statistical analysis of cooling rates across distances")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Create structured output subfolders
    data_dir = outdir / "data"
    d3_dir = outdir / "3d"
    corner_dir = outdir / "corner"
    for d in (data_dir, d3_dir, corner_dir):
        d.mkdir(parents=True, exist_ok=True)

    indir = Path(args.indir)
    files = sorted(indir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files found in {indir} matching {args.pattern}")

    rows = []
    for fp in files:
        entry = read_one_file(fp)

        # per-group temps
        kT = {}
        kT_lo = {}
        kT_hi = {}
        kT_sig = {}

        for p in TARGET_PARS:
            kT[p] = logT_to_kT_eV(entry.logT[p])
            lo_eV, hi_eV = ci_logT_to_kT_eV(entry.logT_ci[p])
            kT_lo[p] = lo_eV
            kT_hi[p] = hi_eV
            # symmetric sigma approx from half-width
            kT_sig[p] = 0.5 * (hi_eV - lo_eV)

        # combine 5 groups into one "representative" temperature for that (M,R,D)
        vals = np.array([kT[p] for p in TARGET_PARS], dtype=float)
        sigs = np.array([kT_sig[p] for p in TARGET_PARS], dtype=float)
        kT_comb, kT_comb_sig = weighted_mean_and_sigma(vals, sigs)

        row = {
            "file": fp.name,
            "M": entry.M,
            "R": entry.R,
            "D": entry.D,
            "kT_comb_eV": kT_comb,
            "kT_comb_sigma_eV": kT_comb_sig,
        }

        # store per-group columns (best + CI)
        for p, grp in zip(TARGET_PARS, [1, 2, 3, 4, 5]):
            row[f"logT_g{grp}"] = entry.logT[p]
            row[f"logTlo_g{grp}"] = entry.logT_ci[p][0]
            row[f"logThi_g{grp}"] = entry.logT_ci[p][1]
            row[f"kT_g{grp}_eV"] = kT[p]
            row[f"kTlo_g{grp}_eV"] = kT_lo[p]
            row[f"kThi_g{grp}_eV"] = kT_hi[p]

        rows.append(row)

    outcsv_path = data_dir / args.outcsv
    outplot_path = d3_dir / args.outplot
    df = pd.DataFrame(rows).sort_values(["D", "M", "R"]).reset_index(drop=True)
        # ---- DEBUG/REPORT: print the exact grid used ----
    cols = ["M", "R", "D", "kT_comb_eV", "kT_comb_sigma_eV"]
    print("\n=== GRID POINTS USED (M, R, D, kT[eV], sigma[eV]) ===")
    print(df[cols].to_string(index=False))

    print("\n=== UNIQUE GRID VALUES DETECTED ===")
    print("M unique:", sorted(df["M"].unique()))
    print("R unique:", sorted(df["R"].unique()))
    print("D unique:", sorted(df["D"].unique()))
    print("kT range (eV):", float(df["kT_comb_eV"].min()), "to", float(df["kT_comb_eV"].max()))
    df.to_csv(outcsv_path, index=False)
    print(f"[OK] Wrote {outcsv_path} with {len(df)} grid points")

    # Produce per-spectrum 3D plots (one per group / spectrum)
    for grp in [1, 2, 3, 4, 5]:
        zcol = f"kT_g{grp}_eV"
        title = f"XTE J1701-462 - spectrum {grp}"
        outplot = d3_dir / f"grid_3d_spec{grp}.png"
        plot_3d_grid(df, outplot, zcol=zcol, title=title)
        print(f"[OK] Wrote {outplot}")

        # Optional interactive Plotly HTML per-spectrum
        if args.interactive:
            interactive_path = d3_dir / f"grid_3d_spec{grp}_interactive.html"
            try:
                plot_3d_interactive(df, interactive_path, zcol=zcol, title=title)
                print(f"[OK] Wrote {interactive_path}")
            except ImportError as e:
                print(f"[WARN] Could not create interactive plot: {e}")

    # Optional animation across spectra
    if args.animate:
        zcols = [f"kT_g{grp}_eV" for grp in [1, 2, 3, 4, 5]]
        anim_path = d3_dir / "grid_3d_evolution.html"
        try:
            plot_3d_evolution(df, anim_path, zcols=zcols, steps_per_transition=args.anim_steps,
                              title="XTE J1701-462 - time evolution")
            print(f"[OK] Wrote {anim_path}")
        except ImportError as e:
            print(f"[WARN] Could not create animation: {e}")

    # Optional cooling-rate analysis
    if args.analyze_cooling:
        try:
            analyze_cooling_rates(df, outdir)
        except Exception as e:
            print(f"[ERROR] Cooling analysis failed: {e}")

    # Optional MCMC: infer (M,R,D) given a measured kT (eV) with uncertainty
    if args.run_mcmc:
        if args.teff_obs_ev is None or args.sigma_ev is None:
            raise SystemExit("--run-mcmc requires --teff-obs-ev and --sigma-ev")

        Mvals, Rvals, Dvals, grid = build_grid(df, "kT_comb_eV")

        if args.sampler == "grid":
            weights = compute_grid_posterior_weights(
                teff_obs=float(args.teff_obs_ev),
                sigma_obs=float(args.sigma_ev),
                grid=grid,
            )
            samples = sample_from_grid_posterior(
                weights=weights,
                Mvals=Mvals,
                Rvals=Rvals,
                Dvals=Dvals,
                nsteps=args.mcmc_steps,
                burn=args.mcmc_burn,
                seed=args.mcmc_seed,
            )
            print("[Sampler] drew independent samples from exact discrete grid posterior")
        else:
            samples = metropolis_mcmc_discrete(
                teff_obs=float(args.teff_obs_ev),
                sigma_obs=float(args.sigma_ev),
                Mvals=Mvals,
                Rvals=Rvals,
                Dvals=Dvals,
                grid=grid,
                nsteps=args.mcmc_steps,
                burn=args.mcmc_burn,
                seed=args.mcmc_seed,
            )

        # Save samples + quick corner-like summaries (no extra deps)
        samp_df = pd.DataFrame(samples, columns=["M", "R", "D"])
        samp_df.to_csv(corner_dir / "mcmc_samples.csv", index=False)
        print(f"[OK] Wrote {corner_dir / 'mcmc_samples.csv'}")

        for col in ["M", "R", "D"]:
            q16, q50, q84 = np.percentile(samp_df[col], [16, 50, 84])
            print(f"[MCMC] {col}: {q50:.4g} (+{q84-q50:.4g}/-{q50-q16:.4g})")

        # Simple pair plots
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        axes[0].hist(samp_df["M"], bins=40)
        axes[0].set_xlabel("M (Msun)")
        axes[1].hist(samp_df["R"], bins=40)
        axes[1].set_xlabel("R (km)")
        axes[2].hist(samp_df["D"], bins=40)
        axes[2].set_xlabel("D (kpc)")
        fig.tight_layout()
        fig.savefig(corner_dir / "XTE_mcmc_hist.png", dpi=200)
        plt.close(fig)
        print(f"[OK] Wrote {corner_dir / 'XTE_mcmc_hist.png'}")

        # Corner plot (ONLY here!)
        # ---- MCMC diagnostics ----
        print("\n=== MCMC SAMPLE DIAGNOSTICS ===")
        for col in ["M", "R", "D"]:
            arr = samp_df[col].to_numpy()
            print(
                f"{col}: min={arr.min():.6g} max={arr.max():.6g} "
                f"unique={len(np.unique(arr))}"
            )

        # If chain got stuck, print first few rows
        print("\nFirst 10 samples:")
        print(samp_df.head(10).to_string(index=False))

        # If all stuck, stop before corner to avoid crash
        stuck = all(samp_df[c].nunique() <= 1 for c in ["M", "R", "D"])
        if stuck:
            raise RuntimeError(
                "MCMC chain has no dynamic range (all samples identical). "
                "This usually means acceptance ~ 0 or your likelihood is rejecting proposals."
            )
        corner_ranges = [
            (1.2, 2.2),   # M
            (10.0, 16.0), # R
            (5.6, 8.8),   # D
        ]

        try:
            import corner
        except ImportError:
            print("[WARN] corner is not installed; skipping corner_plot.png")
            return

        fig = corner.corner(
            samp_df[["M", "R", "D"]],
            labels=[r"$M\ (M_\odot)$", r"$R\ (\mathrm{km})$", r"$D\ (\mathrm{kpc})$"],
            range=corner_ranges,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            levels=[0.68, 0.95],
            fill_contours=True,
            smooth=1.0,
        )
        fig.savefig(corner_dir / "corner_plot.png", dpi=200)
        print(f"[OK] Wrote {corner_dir / 'corner_plot.png'}")


if __name__ == "__main__":
    main()


# python MCMC.py --indir ../out/grid_logs --animate --anim-steps 30 --analyze-cooling --run-mcmc --teff-obs-ev 150 --sigma-ev 2