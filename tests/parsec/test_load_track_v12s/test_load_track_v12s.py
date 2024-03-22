"""
Test loading v1.2s and v2.0 PARSEC tracks.
"""
from berliner.parsec import load_track_v12s, load_track_v20

t_v12s = load_track_v12s(
    "exdata/parsec-v1.2s/Z0.014Y0.273/Z0.014Y0.273OUTA1.74_F7_M001.000.DAT"
)
thb_v12s = load_track_v12s(
    "exdata/parsec-v1.2s/Z0.014Y0.273/Z0.014Y0.273OUTA1.74_F7_M1.000.HB.DAT"
)


t_v20s = load_track_v20(
    "exdata/parsec-v2.0s/ALL_ROT_Z0.014_Y0.273/VAR_ROT0.00_SH_Z0.014_Y0.273/Z0.014Y0.273O_IN0.00OUTA1.74_F7_M1.00.TAB"
)
thb_v20s = load_track_v20(
    "exdata/parsec-v2.0s/ALL_ROT_Z0.014_Y0.273/VAR_ROT0.00_SH_Z0.014_Y0.273/Z0.014Y0.273O_IN0.00OUTA1.74_F7_M1.00.TAB.HB"
)


import matplotlib.pyplot as plt

kwargs = dict(
    lw=2,
)
plt.plot(
    t_v12s["LOG_TE"],
    t_v12s["LOG_L"],
    label="Z0.014Y0.273OUTA1.74_F7_M002.000.DAT",
    **kwargs,
)
plt.plot(
    thb_v12s["LOG_TE"],
    thb_v12s["LOG_L"],
    label="Z0.014Y0.273OUTA1.74_F7_M2.000.HB.DAT",
    linestyle="--",
    **kwargs,
)

plt.plot(
    t_v20s["LOG_TE"],
    t_v20s["LOG_L"],
    label="Z0.014Y0.273O_IN0.00OUTA1.74_F7_M1.00.TAB",
    **kwargs,
)
plt.plot(
    thb_v20s["LOG_TE"],
    thb_v20s["LOG_L"],
    label="Z0.014Y0.273O_IN0.00OUTA1.74_F7_M1.00.TAB.HB",
    linestyle="--",
    **kwargs,
)
plt.legend()
plt.xlabel("LOG_TE")
plt.ylabel("LOG_L")
