## berliner
Tools for stellar tracks & isochrones.

## author
Bo Zhang, [bozhang@nao.cas.cn](mailto:bozhang@nao.cas.cn)

## home page
- [https://github.com/hypergravity/berliner](https://github.com/hypergravity/berliner)
- [https://pypi.org/project/berliner/](https://pypi.org/project/berliner/)

## install
- for the latest **stable** version: `pip install berliner`
- for the latest **github** version: `pip install git+git://github.com/hypergravity/berliner`

## doc (TODO...)

### How to download isochrones from CMD 3.2
```python
# import CMD
from berliner.parsec import CMD
# initialize it
c = CMD()

# define your grid of logAge and [M/H] as tuple(lower, upper, step)
grid_logage = (6, 10.2, 0.1)
grid_mh = (-2.6, 0.5, 0.1)

# download isochrones in parallel
isoc_lgage, isoc_mhini, isoc_list_2mass_wise = c.get_isochrone_grid_mh(
    grid_logage=grid_logage, grid_mh=grid_mh, photsys_file="2mass_spitzer_wise",
    n_jobs=50, verbose=10)
```