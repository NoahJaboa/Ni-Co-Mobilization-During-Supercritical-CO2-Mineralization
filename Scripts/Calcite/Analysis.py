import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os

# ============================================================
# USER SETTINGS
# ============================================================
file = r"F:\PFLOTRAN\REU\Calcite\Calcite_diss.h5"
outdir = r"F:\PFLOTRAN\REU\Calcite"

NX, NY, NZ = 200, 1, 60

extent = [0, 200, -60, 0]  # x_min, x_max, z_min, z_max

variables_to_plot = [
    {
        "name": "pH",
        "label": "pH evolution",
        "outfile": "pH_evolution.gif"
    },
    {
        "name": "Total_Ca++ [M]",
        "label": "Calcium release: Total Ca++ [M]",
        "outfile": "calcium_release.gif"
    }
]

# ============================================================
# FUNCTIONS
# ============================================================
def get_time_years(key):
    m = re.search(r"Time:\s*([0-9.Ee+-]+)\s*y", key)
    if m:
        return float(m.group(1))
    return None


def load_variable(h5file, variable):
    data_all = []
    times = []

    with h5py.File(h5file, "r") as f:
        time_keys = [k for k in f.keys() if "Time" in k]
        time_keys = sorted(time_keys, key=get_time_years)

        print("\nAvailable variables:")
        for v in f[time_keys[0]].keys():
            print(v)

        for tk in time_keys:
            if variable not in f[tk]:
                print(f"Skipping {tk}: variable not found")
                continue

            arr = np.array(f[tk][variable][:]).squeeze()

            if arr.size != NX * NY * NZ:
                print(f"Skipping {tk}: wrong size {arr.size}")
                continue

            arr = arr.reshape((NX, NY, NZ), order="F")[:, 0, :].T
            print(f"\n{variable} at time {get_time_years(tk)} years")
            data_all.append(arr)
            times.append(get_time_years(tk))

    if len(data_all) == 0:
        raise ValueError(f"No data found for variable: {variable}")

    return np.array(data_all), np.array(times)


def make_animation(variable, label, outfile):
    print(f"\nCreating animation for: {variable}")

    data_all, times = load_variable(file, variable)

    vmin = np.nanmin(data_all)
    vmax = np.nanmax(data_all)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    im = ax.imshow(
        data_all[0],
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel("x distance (m)")
    ax.set_ylabel("z depth (m)")

    title = ax.set_title(f"{label}, time = {times[0]:.4f} years")

    def update(i):
        im.set_data(data_all[i])
        title.set_text(f"{label}, time = {times[i]:.4f} years")
        return im, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=300,
        blit=False
    )

    save_path = os.path.join(outdir, outfile)
    ani.save(save_path, writer="pillow", fps=4)

    print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# MAIN SCRIPT
# ============================================================
for item in variables_to_plot:
    make_animation(
        variable=item["name"],
        label=item["label"],
        outfile=item["outfile"]
    )



import h5py
import numpy as np
import matplotlib.pyplot as plt

files = {
    "Normal injection": r"F:\PFLOTRAN\REU\Calcite\Calcite_diss.h5",
    "Higher injection": r"F:\PFLOTRAN\REU\Calcite\Calcite_diss_higher.h5",
    "Higher reaction rate": r"F:\PFLOTRAN\REU\Calcite\Calcite_diss_rate.h5",
}

rho_calcite = 2710.0
MW_calcite = 0.10009
porosity_0 = 0.20

def read_case(file):
    times, mean_vf, mean_ca = [], [], []
    cell_vol = None

    with h5py.File(file, "r") as f:
        keys = sorted(
            [k for k in f.keys() if k.startswith("Time:")],
            key=lambda k: float(k.split()[1])
        )

        for key in keys:
            t = float(key.split()[1])
            unit = key.split()[2]

            if unit in ("s", "sec"):
                t /= 3.156e7
            elif unit in ("d", "day"):
                t /= 365.25
            elif unit in ("h", "hr"):
                t /= 8766.0

            grp = f[key]

            if cell_vol is None:
                cell_vol = grp["Volume [m^3]"][:].ravel()

            vf = grp["Calcite_VF [m^3 mnrl_m^3 bulk]"][:].ravel()
            ca = grp["Total_Ca++ [M]"][:].ravel()

            w = cell_vol / cell_vol.sum()

            times.append(t)
            mean_vf.append(vf @ w)
            mean_ca.append(ca @ w)

    times = np.array(times)
    mean_vf = np.array(mean_vf)
    mean_ca = np.array(mean_ca)

    total_bulk_vol = cell_vol.sum()
    pore_vol_L = total_bulk_vol * porosity_0 * 1000.0

    delta_vf = mean_vf - mean_vf[0]
    delta_ca = mean_ca - mean_ca[0]

    moles_calcite_dissolved = -delta_vf * total_bulk_vol * rho_calcite / MW_calcite
    moles_ca_in_domain = delta_ca * pore_vol_L

    return times, delta_vf, moles_calcite_dissolved, delta_ca, moles_ca_in_domain


results = {}

for label, path in files.items():
    results[label] = read_case(path)

plt.figure(figsize=(8, 5))
for label, (t, dvf, mdiss, dca, mca) in results.items():
    plt.plot(t, mdiss, linewidth=2, label=label)

plt.xlabel("Time (years)")
plt.ylabel("Cumulative calcite dissolved (mol)")
plt.title("Effect of Injection Rate on Calcite Dissolution")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
for label, (t, dvf, mdiss, dca, mca) in results.items():
    plt.plot(t, dca * 1e3, linewidth=2, label=label)

plt.xlabel("Time (years)")
plt.ylabel("ΔTotal Ca²⁺ in domain (mM)")
plt.title("Effect of Injection Rate on Dissolved Ca²⁺")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


print("\nFinal comparison")
print("-" * 75)
print(f"{'Case':<20} {'Calcite dissolved (mol)':>25} {'ΔCa (mM)':>15} {'Ca in domain (mol)':>20}")
print("-" * 75)

for label, (t, dvf, mdiss, dca, mca) in results.items():
    print(f"{label:<20} {mdiss[-1]:>25.4e} {dca[-1]*1e3:>15.4f} {mca[-1]:>20.4e}")