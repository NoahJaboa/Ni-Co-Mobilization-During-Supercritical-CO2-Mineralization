import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# =========================
# USER SETTINGS
# =========================
h5file = Path(r"F:\PFLOTRAN\REU\Forsterite\forsterite_base.h5")

TARGET_TIMES = [0.01, 1.0, 1.5, 2.0]

VARS = [
    ("pH", "pH"),
    ("Gas_Saturation", "Gas saturation"),
    ("Forsterite_VF [m^3 mnrl_m^3 bulk]", "Forsterite VF"),
    ("Magnesite_VF [m^3 mnrl_m^3 bulk]", "Magnesite VF"),
    ("SiO2(am)_VF [m^3 mnrl_m^3 bulk]", "SiO2(am) VF"),
    ("Total_Mg++ [M]", "Mg++ [M]"),
]

# =========================
# HELPERS
# =========================
def get_time(name):
    m = re.search(r"Time:\s*([0-9.Ee+\-]+)\s*y", name)
    return float(m.group(1)) if m else None

def sorted_time_groups(f):
    groups = [k for k in f.keys() if get_time(k) is not None]
    return sorted(groups, key=get_time)

def nearest_time_group(time_groups, target):
    times = np.array([get_time(g) for g in time_groups])
    idx = np.argmin(np.abs(times - target))
    return time_groups[idx], times[idx]

def to_plot_array(arr):
    arr = np.array(arr).squeeze()

    # Your PFLOTRAN file gives arrays as (NX, NZ), e.g. (1000, 100).
    # Transpose to plot as z vs x.
    if arr.ndim == 2:
        return arr.T

    raise ValueError(f"Unexpected array shape: {arr.shape}")

# =========================
# LOAD + PLOT
# =========================
with h5py.File(h5file, "r") as f:
    time_groups = sorted_time_groups(f)

    print("\nAvailable times:")
    for g in time_groups:
        print(f"  {get_time(g):.5f} y -> {g}")

    selected = []
    for target in TARGET_TIMES:
        group, actual_time = nearest_time_group(time_groups, target)
        selected.append((group, actual_time))

    print("\nSelected times:")
    for group, actual_time in selected:
        print(f"  target -> actual: {actual_time:.5f} y   group: {group}")

    for varname, label in VARS:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)

        maps = []
        times = []

        for group, actual_time in selected:
            if varname not in f[group]:
                print(f"Skipping {label}: variable not found.")
                maps = []
                break

            arr_raw = np.array(f[group][varname][:]).squeeze()
            arr_plot = to_plot_array(arr_raw)

            maps.append(arr_plot)
            times.append(actual_time)

            print(
                f"t = {actual_time:.5f} y | "
                f"min = {np.nanmin(arr_raw):.6g}, "
                f"max = {np.nanmax(arr_raw):.6g}, "
                f"mean = {np.nanmean(arr_raw):.6g}"
            )

        if not maps:
            continue

        n = len(maps)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        axes = axes.flatten()

        for i, arr in enumerate(maps):
            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)

            im = axes[i].imshow(
                arr,
                origin="lower",
                extent=[0, 1000, 0, 100],
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            axes[i].set_title(
                f"{times[i]:.3f} y\n{label} = {vmin:.3g}–{vmax:.3g}",
                fontsize=10
            )
            axes[i].set_xlabel("Distance x [m]")
            axes[i].set_ylabel("Elevation z [m]")

            plt.colorbar(im, ax=axes[i], shrink=0.8, label=label)

        fig.suptitle(f"{label} maps", fontsize=14)
        plt.tight_layout()
        plt.show()