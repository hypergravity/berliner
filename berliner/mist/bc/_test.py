
#%%
#import glob
#fps = glob.glob(data_dir+"/*")

#%%
bc_dir = "/hydrogen/mist/1.2/bc/WISE"
from berliner.mist.bc import make_bc_dir
make_bc_dir(bc_dir)

#%%
from berliner.mist.bc._bc import make_bc_all
bc_dir = "/hydrogen/mist/1.2/bc"
make_bc_all(bc_dir,n_jobs=1)

#%%
from berliner.mist.bc import BCI
bci = BCI(bc_dir)
bci.load_bc(["WISE_W1"])
bci.interp_mag(0, 5750, 4.35, 0.01, 0)
