from berliner.parsec import CMD
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# initialize it
c = CMD()

logages = np.arange(6, 10.5, 0.1)
isochrones = joblib.Parallel(n_jobs=2, verbose=10)(
    joblib.delayed(
        c.get_one_isochrone,
    )(logage=_, z=0.0152, mh=None, photsys_file="2mass_spitzer")
    for _ in logages
)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# for i in range(0, len(logages), 4):
for i in [0, 2]:
    isoc = isochrones[i]
    # ax.plot(isoc["logTe"], isoc["logg"], lw=2)
    ax.plot(isoc["logTe"], isoc["logg"], np.log10(isoc["Mini"]), c="gray")
    h = ax.scatter(
        isoc["logTe"],
        isoc["logg"],
        np.log10(isoc["Mini"]),
        c=isoc["label"],
        s=5,
        cmap=plt.cm.jet,
    )
    # for i in range(len(isoc)):
    #     ax.text(isoc["logTe"][i], isoc["logg"][i], isoc["label"][i])
ax.set_xlabel("LOGTE")
ax.set_ylabel("LOGG")
ax.set_zlabel("LOGM")
plt.colorbar(h, ax=ax)

fig.tight_layout()
ax.view_init(elev=90, azim=50)  # 仰角为20度，方位角为30度
fig.show()

plt.close("all")
