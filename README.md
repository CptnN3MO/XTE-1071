# XTE-1071 MCMC

Small workflow to:
1) run an XSPEC grid over **(M, R, D)** for `tbabs*(nsatmos+pegpwrlw)`,
2) parse XSPEC logs to build a `(M,R,D) -> kT` grid,
3) (optionally) infer a posterior over **(M, R, D)** given an observed `kT` using a simple discrete Metropolis sampler.


---

## What it does

### XSPEC side
- Loads 5 spectra and fits a shared model `tbabs*(nsatmos+pegpwrlw)`. :contentReference[oaicite:0]{index=0}
- Runs a grid over:
  - `M ∈ {1.2, 1.4, 1.6, 1.8, 2.0, 2.2} Msun`
  - `R ∈ {10, 12, 14, 16} km`
  - `D ∈ {5.6, 7.3, 8.8} kpc` :contentReference[oaicite:1]{index=1}
- For each grid point it logs:
  - `show free` (to capture best-fit `LogT_eff`)
  - `uncer 1.0 2 12 22 32 42` (to capture confidence ranges)
  - plus flux/error outputs. :contentReference[oaicite:2]{index=2}

### Python side (`MCMC.py`)
- Reads log files named like `M12_R10_D56.txt` (meaning `M=1.2, R=10, D=5.6`). :contentReference[oaicite:3]{index=3}
- Extracts `LogT_eff` best-fit values from the `show free` block and CI bounds from the `uncer` table for params `[2, 12, 22, 32, 42]` (5 spectra/groups). :contentReference[oaicite:4]{index=4}
- Converts `log10(T[K])` → `kT[eV]` and computes a combined `kT` per grid point using an inverse-variance weighted mean (sigma estimated from CI half-width). :contentReference[oaicite:5]{index=5}
- Produces:
  - `grid_temps.csv`
  - a 3D wireframe plot `grid_3d.png`
  - optionally: `mcmc_samples.csv`, `XTE_mcmc_hist.png`, `corner_plot.png`. :contentReference[oaicite:6]{index=6}

---

## Repo contents

- `base_xspec.xcm` — loads data + defines the model and baseline parameters. :contentReference[oaicite:7]{index=7}
- `grid_xspec.xcm` — loops over the `(M,R,D)` grid and writes per-point log files into `grid_logs/`. :contentReference[oaicite:8]{index=8}
- `xspec.tcl` — one-shot XSPEC command sequence (fit + uncer + show/free + flux). :contentReference[oaicite:9]{index=9}
- `MCMC.py` — log parsing, grid building, plotting, and optional discrete MCMC. :contentReference[oaicite:10]{index=10}
- `grid_3d.png` — example output plot (in repo root).

---

## Requirements

### XSPEC
- HEASOFT/XSPEC with Tcl scripting support.
- Your grouped spectra must match the filenames used in the scripts (edit as needed):
  - `26976_grp.pi`, `26977_grp.pi`, `26978_grp.pi`, `obs4_grp.pi`, `26980_grp.pi` :contentReference[oaicite:11]{index=11}

### Python
Install the typical stack:
- `numpy`, `pandas`, `matplotlib`
- `corner` (only needed if you want the corner plot) :contentReference[oaicite:12]{index=12}

---

## Usage

### 1) Run the XSPEC grid

From an XSPEC prompt:

```tcl
@grid_xspec.xcm
