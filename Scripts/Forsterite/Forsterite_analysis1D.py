import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import os

# =============================================================================
# USER SETTINGS
# =============================================================================
FILE   = r"C:\Users\noahm\OneDrive\Desktop\REU\Forsterite\1D\forsterite_1D.h5"
OUTDIR = r"C:\Users\noahm\OneDrive\Desktop\REU\Forsterite\1D"

NX, NY, NZ = 10, 1, 1
CELL_LENGTH = 1.0
X_CENTERS = np.arange(0.5, NX * CELL_LENGTH, CELL_LENGTH)

SNAP_TIMES = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]

INJECTION_OFF = 2.0

RHO_FORSTERITE = 3222.0;  MW_FORSTERITE = 0.14069
RHO_MAGNESITE  = 2958.0;  MW_MAGNESITE  = 0.08431
RHO_SIO2AM     = 2200.0;  MW_SIO2AM     = 0.06008

os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# HDF5 UTILITIES
# =============================================================================
def get_time_years(key):
    m = re.search(r"Time:\s*([0-9.Ee+\-]+)\s*y", key)
    return float(m.group(1)) if m else None


def sorted_time_keys(f):
    keys = [k for k in f.keys() if get_time_years(k) is not None]
    return sorted(keys, key=get_time_years)


def reshape_1d(arr):
    return arr.squeeze().reshape((NX, NY, NZ), order="F")[:, 0, 0]


def load_variable(h5file, varname):
    frames, times = [], []
    with h5py.File(h5file, "r") as f:
        for tk in sorted_time_keys(f):
            grp = f[tk]
            if varname not in grp:
                continue
            arr = np.array(grp[varname][:])
            if arr.size != NX * NY * NZ:
                continue
            frames.append(reshape_1d(arr))
            times.append(get_time_years(tk))
    if not frames:
        raise ValueError(f"Variable '{varname}' not found.")
    return np.array(frames), np.array(times)


def select_snapshots(times, snap_times):
    idx_list, t_list = [], []
    for st in snap_times:
        if st > times[-1]:
            continue
        idx = int(np.argmin(np.abs(times - st)))
        if idx not in idx_list:
            idx_list.append(idx)
            t_list.append(times[idx])
    return idx_list, t_list


# =============================================================================
# SPATIAL PROFILE PLOTS (line plots along x at key times)
# =============================================================================
def plot_spatial_profiles(h5file):
    VARS = [
        ("pH",                                   "pH",                  "RdYlBu"),
        ("Forsterite_VF [m^3 mnrl_m^3 bulk]",   "Forsterite VF [m3/m3]","plasma_r"),
        ("Magnesite_VF [m^3 mnrl_m^3 bulk]",    "Magnesite VF [m3/m3]", "YlOrRd"),
        ("SiO2(am)_VF [m^3 mnrl_m^3 bulk]",     "SiO2(am) VF [m3/m3]", "YlOrBr"),
        ("Total_Mg++ [M]",                       "Mg2+ [mol/L]",         "viridis"),
    ]

    n = len(VARS)
    fig, axes = plt.subplots(n, 1, figsize=(10, n * 3.0), sharex=True)
    fig.suptitle("1D Spatial Profiles Along Column", fontsize=13, fontweight="bold")

    cmap_time = plt.cm.plasma
    n_snaps = len(SNAP_TIMES)

    for ax, (varname, label, _) in zip(axes, VARS):
        try:
            data, times = load_variable(h5file, varname)
        except ValueError as e:
            ax.set_ylabel(label, fontsize=9)
            ax.text(0.5, 0.5, f"Not found: {e}",
                    transform=ax.transAxes, ha="center", fontsize=8, color="red")
            continue

        idx_list, t_list = select_snapshots(times, SNAP_TIMES)
        for rank, (idx, t) in enumerate(zip(idx_list, t_list)):
            color = cmap_time(rank / max(n_snaps - 1, 1))
            ax.plot(X_CENTERS, data[idx], color=color,
                    linewidth=1.8, label=f"t = {t:.3f} y")

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        # Injection point marker
        ax.axvline(0.5, color="gray", linestyle=":", linewidth=1.0,
                   alpha=0.7, label="Injection cell")

    axes[-1].set_xlabel("x along column (m)")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "spatial_profiles.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG : {path}")


# =============================================================================
# STRIP SNAPSHOT PLOTS  (1 x NX imshow — where in the column things happen)
# =============================================================================
def plot_strip_snapshots(h5file):
    VARS = [
        ("pH",                                   "pH",                   "RdYlBu"),
        ("Forsterite_VF [m^3 mnrl_m^3 bulk]",   "Forsterite VF [m3/m3]","plasma_r"),
        ("Magnesite_VF [m^3 mnrl_m^3 bulk]",    "Magnesite VF [m3/m3]", "YlOrRd"),
        ("SiO2(am)_VF [m^3 mnrl_m^3 bulk]",     "SiO2(am) VF [m3/m3]", "YlOrBr"),
        ("Total_Mg++ [M]",                       "Mg2+ [mol/L]",         "viridis"),
    ]

    for varname, label, cmap in VARS:
        print(f"    Strip snapshot: {varname}")
        try:
            data, times = load_variable(h5file, varname)
        except ValueError as e:
            print(f"      SKIPPED — {e}")
            continue

        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-10

        idx_list, t_list = select_snapshots(times, SNAP_TIMES)
        n = len(idx_list)
        if n == 0:
            print(f"      No snapshot times in range — skipping.")
            continue

        fig, axes = plt.subplots(n, 1,
                                 figsize=(max(8, NX * 0.9), n * 1.4),
                                 squeeze=False)
        fig.suptitle(f"{label}  —  column strip snapshots",
                     fontsize=11, fontweight="bold")

        for i, (idx, t) in enumerate(zip(idx_list, t_list)):
            ax = axes[i][0]
            strip = data[idx].reshape(1, NX)
            im = ax.imshow(strip, aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[0, NX * CELL_LENGTH, -0.5, 0.5])
            ax.set_yticks([])
            ax.set_ylabel(f"t={t:.3f}y", fontsize=8, rotation=0,
                          labelpad=45, va="center")
            # Mark injection cell
            ax.axvline(CELL_LENGTH, color="white", linewidth=1.0,
                       linestyle="--", alpha=0.7)
            plt.colorbar(im, ax=ax, shrink=0.8,
                         pad=0.01).set_label(label, fontsize=7)

        axes[-1][0].set_xlabel("x along column (m)", fontsize=9)
        fig.tight_layout()
        safe_name = varname.split(" ")[0].replace("/", "").replace("(", "").replace(")", "")
        path = os.path.join(OUTDIR, f"strip_{safe_name}.png")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"      PNG : {path}")


def read_timeseries(h5file):
    times_l    = []
    moles_mg_l = []
    vf_fors_l, vf_mag_l, vf_sio2_l = [], [], []
    mean_por_l, mean_ph_l           = [], []

    with h5py.File(h5file, "r") as f:
        tkeys = sorted_time_keys(f)

        cell_vol       = np.array(f[tkeys[0]]["Volume [m^3]"][:]).squeeze()
        cell_vol       = cell_vol.reshape((NX, NY, NZ), order="F")[:, 0, 0]
        total_bulk_vol = float(cell_vol.sum())
        w              = cell_vol / total_bulk_vol

        for tk in tkeys:
            grp = f[tk]

            def load(name):
                return reshape_1d(np.array(grp[name][:]))

            mg_molar = load("Total_Mg++ [M]")        # mol/L
            ph       = load("pH")
            por      = load("Porosity")
            sat_l    = load("Liquid_Saturation")
            fors     = load("Forsterite_VF [m^3 mnrl_m^3 bulk]")
            mag      = load("Magnesite_VF [m^3 mnrl_m^3 bulk]")
            sio2     = load("SiO2(am)_VF [m^3 mnrl_m^3 bulk]")

            L_solution = cell_vol * por * sat_l * 1000.0
            moles_mg_l.append(float(np.sum(mg_molar * L_solution)))

            times_l.append(get_time_years(tk))
            vf_fors_l.append(float(fors @ w))
            vf_mag_l.append(float(mag  @ w))
            vf_sio2_l.append(float(sio2 @ w))
            mean_por_l.append(float(por  @ w))
            mean_ph_l.append(float(ph    @ w))

    times    = np.array(times_l)
    vf_fors  = np.array(vf_fors_l)
    vf_mag   = np.array(vf_mag_l)
    vf_sio2  = np.array(vf_sio2_l)
    mean_por = np.array(mean_por_l)
    mean_ph  = np.array(mean_ph_l)
    moles_mg = np.array(moles_mg_l)

    moles_fors_diss = -(vf_fors - vf_fors[0]) * total_bulk_vol * RHO_FORSTERITE / MW_FORSTERITE
    moles_mag_ppt   =  (vf_mag  - vf_mag[0])  * total_bulk_vol * RHO_MAGNESITE  / MW_MAGNESITE
    moles_sio2_ppt  =  (vf_sio2 - vf_sio2[0]) * total_bulk_vol * RHO_SIO2AM     / MW_SIO2AM
    delta_por       =  mean_por - mean_por[0]

    return {
        "times"          : times,
        "moles_mg"       : moles_mg,
        "moles_fors_diss": moles_fors_diss,
        "moles_mag_ppt"  : moles_mag_ppt,
        "moles_sio2_ppt" : moles_sio2_ppt,
        "delta_por"      : delta_por,
        "mean_ph"        : mean_ph,
        "total_bulk_vol" : total_bulk_vol,
    }


# =============================================================================
# TIME-SERIES PLOTS
# =============================================================================
def _vline(ax, t_end):
    if t_end >= INJECTION_OFF:
        ax.axvline(INJECTION_OFF, color="gray", ls="--", lw=1.2, alpha=0.7,
                   label=f"Injection off (t = {INJECTION_OFF} y)")


def plot_timeseries(d):
    times = d["times"]
    t_end = times[-1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("1D Domain-Integrated Reaction Summary", fontsize=13,
                 fontweight="bold")

    ax = axes[0, 0]
    ax.plot(times, d["moles_fors_diss"], lw=2, color="darkorange",
            label="Forsterite dissolved")
    ax.plot(times, d["moles_mag_ppt"],   lw=2, color="mediumseagreen",
            label="Magnesite precipitated")
    ax.plot(times, d["moles_sio2_ppt"],  lw=2, color="slateblue",
            label="SiO\u2082(am) precipitated")
    _vline(ax, t_end)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Cumulative moles")
    ax.set_title("Mineral Reactions Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, d["mean_ph"], lw=2, color="steelblue")
    _vline(ax, t_end)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Domain-mean pH")
    ax.set_title("pH Evolution (domain average)")
    ax.grid(True, alpha=0.3)
    if t_end >= INJECTION_OFF:
        ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(times, d["moles_mg"], lw=2, color="teal")
    _vline(ax, t_end)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Total Mg\u00b2\u207a in domain (mol)")
    ax.set_title("Total Dissolved Mg\u00b2\u207a in Pore Fluid\n"
                 "(molarity [mol/L] \u00d7 L solution per cell, summed)")
    ax.grid(True, alpha=0.3)
    if t_end >= INJECTION_OFF:
        ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(times, d["delta_por"] * 100, lw=2, color="dimgray")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    _vline(ax, t_end)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mean porosity change (\u0394 %)")
    ax.set_title("Domain-Averaged Porosity Change")
    ax.grid(True, alpha=0.3)
    if t_end >= INJECTION_OFF:
        ax.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "timeseries_summary.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG : {path}")


def plot_mass_balance(d):
    times       = d["times"]
    mg_released = 2.0 * d["moles_fors_diss"]
    mg_fluid_d  = d["moles_mg"] - d["moles_mg"][0]
    mg_in_mag   = d["moles_mag_ppt"]
    mg_sum      = mg_fluid_d + mg_in_mag

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, mg_released, lw=2, color="darkorange",
            label="Mg released (2 \u00d7 forsterite dissolved)")
    ax.plot(times, mg_fluid_d,  lw=2, color="steelblue",
            label="\u0394Mg in pore fluid vs t\u202f=\u202f0")
    ax.plot(times, mg_in_mag,   lw=2, color="mediumseagreen",
            label="Mg in magnesite")
    ax.plot(times, mg_sum,      lw=2, color="black", ls="--",
            label="Fluid \u0394 + magnesite")
    ax.axhline(0, color="k", lw=0.8, ls=":")
    _vline(ax, times[-1])
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mg (mol)")
    ax.set_title("Mg Mass Balance\n"
                 "(open system \u2014 east-face pressure BC allows Mg to exit)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "Mg_mass_balance.png")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  PNG : {path}")


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_summary(d):
    times = d["times"]
    print("\n" + "=" * 65)
    print("FORSTERITE 1D — SIMULATION SUMMARY")
    print("=" * 65)
    print(f"  Grid                           : {NX} x {NY} x {NZ} cells")
    print(f"  Domain bulk volume             : {d['total_bulk_vol']:.1f} m3")
    print(f"  Simulation end time            : {times[-1]:.2f} y")
    print(f"  Total Mg2+ in domain (final)   : {d['moles_mg'][-1]:.4e} mol")
    print(f"  Forsterite dissolved           : {d['moles_fors_diss'][-1]:.4e} mol")
    print(f"  Magnesite precipitated         : {d['moles_mag_ppt'][-1]:.4e} mol")
    print(f"  SiO2(am) precipitated          : {d['moles_sio2_ppt'][-1]:.4e} mol")
    print(f"  Mean porosity change           : {d['delta_por'][-1]*100:+.5f} %")
    print(f"  Domain-mean pH (t = 0)         : {d['mean_ph'][0]:.3f}")
    print(f"  Domain-mean pH (final)         : {d['mean_ph'][-1]:.3f}")

    mg_released  = 2.0 * d["moles_fors_diss"][-1]
    mg_accounted = (d["moles_mg"][-1] - d["moles_mg"][0]) + d["moles_mag_ppt"][-1]
    if mg_released > 0:
        print(f"  Mg mass balance closure        : {100*mg_accounted/mg_released:.1f}%")
        print(f"    (deficit = Mg advected out via east-face BC)")

    if d["moles_fors_diss"][-1] > 0:
        si_ratio = d["moles_sio2_ppt"][-1] / d["moles_fors_diss"][-1]
        print(f"  Si captured / forsterite diss. : {si_ratio:.3f}  (stoich. max = 1.0)")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    print("\n=== Spatial profiles ===")
    plot_spatial_profiles(FILE)

    print("\n=== Strip snapshots ===")
    plot_strip_snapshots(FILE)

    print("\n=== Time-series ===")
    d = read_timeseries(FILE)
    print(f"  Timesteps loaded : {len(d['times'])}")
    print(f"  Bulk volume      : {d['total_bulk_vol']:.1f} m3")

    print("\n=== Plots ===")
    plot_timeseries(d)
    plot_mass_balance(d)

    print_summary(d)
    print("\nDone.")