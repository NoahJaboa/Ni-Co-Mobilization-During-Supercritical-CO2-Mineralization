#!/usr/bin/env python
# coding: utf-8

# In[2]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import re

# ============================================================
# FILE PATHS
# ============================================================
file = r"C:\Users\wnm9245\Desktop\REU\Pure\Calcite_diss_annotated.h5"

# ============================================================
# FUNCTIONS
# ============================================================
def get_time_years(key):
    m = re.search(r"Time:\s*([0-9.Ee+-]+)\s*y", key)
    if m:
        return float(m.group(1))
    return None


def calculate_pore_volume(vol, por, sat):
    return vol * por * sat * 1000


def total_ca_moles(h5file):
    times = []
    moles_ca = []

    with h5py.File(h5file, "r") as f:
        time_keys = [k for k in f.keys() if "Time" in k]
        time_keys = sorted(time_keys, key=get_time_years)

        cell_vol = f[time_keys[0]]["Volume [m^3]"][:].ravel()

        for tk in time_keys:
            grp = f[tk]

            ca      = grp["Total_Ca++ [M]"][:].ravel()
            por     = grp["Porosity"][:].ravel()
            sat     = grp["Liquid_Saturation"][:].ravel()

            pore_vol_L = calculate_pore_volume(cell_vol, por, sat)

            moles = np.sum(ca * pore_vol_L)

            times.append(get_time_years(tk))
            moles_ca.append(moles)

    return np.array(times), np.array(moles_ca)


# ============================================================
# MAIN SCRIPT
# ============================================================
times, moles_ca = total_ca_moles(file)

plt.figure(figsize=(8, 5))
plt.plot(times, moles_ca, linewidth=2)
plt.xlabel("Time (years)")
plt.ylabel("Total Ca²⁺ in domain (mol)")
plt.title("Total Dissolved Ca²⁺ Over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\wnm9245\Desktop\REU\Pure\ca_moles_over_time.png", dpi=300, bbox_inches="tight")
plt.show()



# In[ ]:




