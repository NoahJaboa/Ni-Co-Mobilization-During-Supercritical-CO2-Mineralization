import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os

# =============================================================================
# USER SETTINGS
# =============================================================================
FILE   = r"C:\Users\noahm\OneDrive\Desktop\REU\Calcite\Calcite_diss.h5"
OUTDIR = r"C:\Users\noahm\OneDrive\Desktop\REU\Calcite"

NX, NY, NZ = 200, 1, 60
EXTENT     = [0, 200, -60, 0]

RHO_CALCITE = 2710.0   # kg/m3
MW_CALCITE  = 0.10009  # kg/mol  (CaCO3)

# =============================================================================
# UTILITIES
# =============================================================================
def get_time_years(key):
    m = re.search(r"Time:\s*([0-9.Ee+-]+)\s*y", key)
    if m:
        return float(m.group(1))
    return None


def sorted_time_keys(f):
    keys = [k for k in f.keys() if "Time" in k]
    return sorted(keys, key=get_time_years)


def reshape_2d(arr):
    return arr.reshape((NX, NY, NZ), order="F")[:, 0, :].T


def load_2d_variable(h5file, variable):
    data_all, times = [], []
    with h5py.File(h5file, "r") as f:
        for tk in sorted_time_keys(f):
            if variable not in f[tk]:
                continue
            arr = np.array(f[tk][variable][:]).squeeze()
            if arr.size != NX * NY * NZ:
                continue
            data_all.append(reshape_2d(arr))
            times.append(get_time_years(tk))
    if not data_all:
        raise ValueError(f"Variable not found in file: {variable}")
    return np.array(data_all), np.array(times)


# =============================================================================
# ANIMATIONS
# =============================================================================
ANIMATIONS = [
    {
        "variable": "pH",
        "label":    "pH",
        "outfile":  "pH_evolution.gif",
        "cmap":     "RdYlBu"
    },
    {
        "variable": "Total_Ca++ [M]",
        "label":    "Total Ca²⁺ [M]",
        "outfile":  "Ca_spatial.gif",
        "cmap":     "viridis"
    },
    {
        "variable": "Calcite_VF [m^3 mnrl_m^3 bulk]",
        "label":    "Calcite volume fraction",
        "outfile":  "Calcite_VF.gif",
        "cmap":     "plasma"
    },
]


def make_animation(variable, label, outfile, cmap="viridis"):
    print(f"  Animating: {variable}")
    data_all, times = load_2d_variable(FILE, variable)

    if "VF" in variable:
        plot_data = data_all - data_all[0]
        label = f"Δ {label} (change from initial)"
    else:
        plot_data = data_all

    vmin = np.nanmin(plot_data)
    vmax = np.nanmax(plot_data)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-10

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(
        plot_data[0], origin="lower", aspect="auto",
        extent=EXTENT, vmin=vmin, vmax=vmax, cmap=cmap
    )
    plt.colorbar(im, ax=ax).set_label(label)
    ax.set_xlabel("x distance (m)")
    ax.set_ylabel("z depth (m)")
    title = ax.set_title(f"{label}  —  t = {times[0]:.4f} y")

    def update(i):
        im.set_data(plot_data[i])
        title.set_text(f"{label}  —  t = {times[i]:.4f} y")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(times),
                                  interval=300, blit=False)
    path = os.path.join(OUTDIR, outfile)
    ani.save(path, writer="pillow", fps=4)
    print(f"    Saved: {path}")
    plt.close(fig)


# =============================================================================
# TIME-SERIES EXTRACTION
# =============================================================================
def read_timeseries(h5file):
    """
    Returns per-timestep domain-integrated quantities:
      times              (y)
      moles_ca           total Ca2+ in pore fluid (mol)
      moles_calcite_diss cumulative calcite dissolved (mol)
      delta_porosity     mean volume-weighted porosity change
    """
    times     = []
    moles_ca  = []
    vf_calc   = []
    mean_por  = []

    with h5py.File(h5file, "r") as f:
        tkeys = sorted_time_keys(f)
        cell_vol = f[tkeys[0]]["Volume [m^3]"][:].ravel()
        total_bulk_vol = cell_vol.sum()
        w = cell_vol / total_bulk_vol

        for tk in tkeys:
            grp = f[tk]

            ca   = grp["Total_Ca++ [M]"][:].ravel()
            por  = grp["Porosity"][:].ravel()
            sat  = grp["Liquid_Saturation"][:].ravel()
            calc = grp["Calcite_VF [m^3 mnrl_m^3 bulk]"][:].ravel()

            # Molar: mol/L * pore volume in L = mol
            pore_vol_L = cell_vol * por * sat * 1000.0
            moles_ca.append(np.sum(ca * pore_vol_L))

            times.append(get_time_years(tk))
            vf_calc.append(calc @ w)
            mean_por.append(por @ w)

    times    = np.array(times)
    vf_calc  = np.array(vf_calc)
    mean_por = np.array(mean_por)

    moles_calcite_diss = -(vf_calc - vf_calc[0]) * total_bulk_vol * RHO_CALCITE / MW_CALCITE
    delta_porosity     =   mean_por - mean_por[0]

    return times, np.array(moles_ca), moles_calcite_diss, delta_porosity


def dissolution_rate(times, moles_dissolved):
    dt    = np.diff(times)
    dm    = np.diff(moles_dissolved)
    rate  = dm / dt
    t_mid = 0.5 * (times[:-1] + times[1:])
    return t_mid, rate


# =============================================================================
# STATIC PLOTS
# =============================================================================
def plot_ca_over_time(times, moles_ca):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, moles_ca, linewidth=2, color="steelblue")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Total Ca²⁺ in domain (mol)")
    ax.set_title("Total Dissolved Ca²⁺ Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "Ca_moles_over_time.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def plot_calcite_dissolution(times, moles_diss):
    t_mid, rate = dissolution_rate(times, moles_diss)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(times, moles_diss, linewidth=2, color="darkorange")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Cumulative calcite dissolved (mol)")
    axes[0].set_title("Cumulative Calcite Dissolution")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_mid, rate, linewidth=2, color="firebrick")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_ylabel("Dissolution rate (mol/y)")
    axes[1].set_title("Instantaneous Calcite Dissolution Rate")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTDIR, "calcite_dissolution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def plot_mass_balance(times, moles_diss, moles_ca):
    """
    CaCO3 -> Ca2+ + CO3--, so Ca released = moles calcite dissolved.
    Check that Ca in solution tracks calcite dissolved.
    """
    ca_released   = moles_diss
    ca_in_solution = moles_ca - moles_ca[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, ca_released,    linewidth=2, color="darkorange", label="Ca released (calcite dissolved)")
    ax.plot(times, ca_in_solution, linewidth=2, color="steelblue",  label="Ca in solution (Δ from initial)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Ca (mol)")
    ax.set_title("Ca Mass Balance Check")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "Ca_mass_balance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def plot_porosity_change(times, delta_por):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, delta_por * 100, linewidth=2, color="dimgray")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Mean porosity change (Δ%)")
    ax.set_title("Domain-Averaged Porosity Change Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTDIR, "porosity_change.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.show()


def print_summary(times, moles_ca, moles_diss, delta_por):
    print("\nFinal summary")
    print("-" * 60)
    print(f"  Simulation end time          : {times[-1]:.2f} y")
    print(f"  Total Ca2+ in domain (final) : {moles_ca[-1]:.4e} mol")
    print(f"  Calcite dissolved            : {moles_diss[-1]:.4e} mol")
    print(f"  Mean porosity change         : {delta_por[-1]*100:+.4f} %")
    ca_accounted = moles_ca[-1] - moles_ca[0]
    ca_released  = moles_diss[-1]
    if ca_released > 0:
        print(f"  Ca mass balance closure      : {100*ca_accounted/ca_released:.1f}%")
    print("-" * 60)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # --- Variable name check ---
    print("=== Variables at first timestep ===")
    with h5py.File(FILE, "r") as f:
        keys = sorted([k for k in f.keys() if "Time" in k], key=get_time_years)
        for v in f[keys[0]].keys():
            print(v)

    # --- Animations ---
    print("\n=== Generating animations ===")
    for item in ANIMATIONS:
        try:
            make_animation(
                variable=item["variable"],
                label=item["label"],
                outfile=item["outfile"],
                cmap=item.get("cmap", "viridis")
            )
        except ValueError as e:
            print(f"  Skipping animation: {e}")

    # --- Time series ---
    print("\n=== Extracting time series ===")
    times, moles_ca, moles_diss, delta_por = read_timeseries(FILE)

    # --- Static plots ---
    print("\n=== Generating static plots ===")
    plot_ca_over_time(times, moles_ca)
    plot_calcite_dissolution(times, moles_diss)
    plot_mass_balance(times, moles_diss, moles_ca)
    plot_porosity_change(times, delta_por)

    print_summary(times, moles_ca, moles_diss, delta_por)
