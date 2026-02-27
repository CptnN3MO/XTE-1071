
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
import corner
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTDIR = Path("/home/cptnn3mo/Desktop/Neutron_stars/XTE_1701-462/out")
OUTDIR.mkdir(parents=True, exist_ok=True)


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

        if np.isfinite(prop) and (math.log(rng.random()) < (prop - cur)):
            i, j, k = ip, jp, kp
            cur = prop
            accept += 1

        samples[t] = (Mvals[i], Rvals[j], Dvals[k])

    samples = samples[burn:]
    acc = accept / max(proposals, 1)
    print(f"[MCMC-discrete] acceptance fraction ~ {acc:.3f} (excluding no-op moves)")
    return samples

# ---------- plotting (Fig.4-like wireframes) ----------
def plot_3d_grid(df: pd.DataFrame, outpath: Path, zcol: str = "kT_comb_eV") -> None:
    """
    Plot Teff (kT in eV) vs (M,R) for each D separately, similar to HETE2021 Fig. 4.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for D in sorted(df["D"].unique()):
        sub = df[df["D"] == D].copy()
        # Create mesh for wireframe
        Mvals = np.sort(sub["M"].unique())
        Rvals = np.sort(sub["R"].unique())
        Z = np.empty((len(Mvals), len(Rvals)), dtype=float)
        for i, m in enumerate(Mvals):
            for j, r in enumerate(Rvals):
                Z[i, j] = float(sub[(sub["M"] == m) & (sub["R"] == r)][zcol].iloc[0])

        MM, RR = np.meshgrid(Mvals, Rvals, indexing="ij")
        ax.plot_wireframe(MM, RR, Z, linewidth=1)

    ax.set_xlabel("Mass (Msun)")
    ax.set_ylabel("Radius (km)")
    ax.set_zlabel(f"{zcol}")
    ax.set_title("Temperature grid (wireframes per distance)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True, help="Folder with XSPEC log files (e.g. grid_logs/)")
    ap.add_argument("--pattern", type=str, default="M*_R*_D*.txt", help="Glob pattern for log files")
    ap.add_argument("--outcsv", type=str, default="grid_temps.csv", help="Output CSV")
    ap.add_argument("--outplot", type=str, default="grid_3d.png", help="Output 3D plot")
    ap.add_argument("--run-mcmc", action="store_true", help="Run MCMC over (M,R,D)")
    ap.add_argument("--teff-obs-ev", type=float, default=None, help="Observed kT (eV) for MCMC")
    ap.add_argument("--sigma-ev", type=float, default=None, help="1-sigma uncertainty on observed kT (eV) for MCMC")
    ap.add_argument("--mcmc-steps", type=int, default=30000)
    ap.add_argument("--mcmc-burn", type=int, default=8000)
    ap.add_argument("--mcmc-seed", type=int, default=0)
    args = ap.parse_args()

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

    outcsv_path = OUTDIR / args.outcsv
    outplot_path = OUTDIR / args.outplot
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
    print(f"[OK] Wrote {outcsv_path.name} with {len(df)} grid points")

    # 3D plot like Fig.4 style (wireframe per distance)
    plot_3d_grid(df, outplot_path,  zcol="kT_comb_eV")
    print(f"[OK] Wrote {outplot_path.name}")

    # Optional MCMC: infer (M,R,D) given a measured kT (eV) with uncertainty
    if args.run_mcmc:
        if args.teff_obs_ev is None or args.sigma_ev is None:
            raise SystemExit("--run-mcmc requires --teff-obs-ev and --sigma-ev")

        Mvals, Rvals, Dvals, grid = build_grid(df, "kT_comb_eV")

        samples = metropolis_mcmc_discrete(
            teff_obs=float(args.teff_obs_ev),
            sigma_obs=float(args.sigma_ev),
            Mvals=Mvals, Rvals=Rvals, Dvals=Dvals, grid=grid,
            nsteps=args.mcmc_steps,
            burn=args.mcmc_burn,
            seed=args.mcmc_seed,
        )

        # Save samples + quick corner-like summaries (no extra deps)
        samp_df = pd.DataFrame(samples, columns=["M", "R", "D"])
        samp_df.to_csv(OUTDIR / "mcmc_samples.csv", index=False)
        print(f"[OK] Wrote {OUTDIR / 'mcmc_samples.csv'}")

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
        fig.savefig(OUTDIR / "XTE_mcmc_hist.png", dpi=200)
        plt.close(fig)
        print(f"[OK] Wrote {OUTDIR / 'XTE_mcmc_hist.png'}")

        samp_df = pd.DataFrame(samples, columns=["M", "R", "D"])
        samp_df.to_csv(OUTDIR / "mcmc_samples.csv", index=False)
        print(f"[OK] Wrote {OUTDIR / 'mcmc_samples.csv'}")

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
        fig.savefig(OUTDIR / "corner_plot.png", dpi=200)
        print(f"[OK] Wrote {OUTDIR / 'corner_plot.png'}")


if __name__ == "__main__":
    main()