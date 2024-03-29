## berliner
Tools for stellar tracks & isochrones.

What is berliner? See below↓
![](berliner.png)
Picture from [https://restaurantguru.com/Oben2-Heidelberg](https://restaurantguru.com/Oben2-Heidelberg).

## author
Bo Zhang, [bozhang@nao.cas.cn](mailto:bozhang@nao.cas.cn)

## home page
- [https://github.com/hypergravity/berliner](https://github.com/hypergravity/berliner)
- [https://pypi.org/project/berliner/](https://pypi.org/project/berliner/)

## Installation
```
pip install git+https://github.com/hypergravity/berliner.git
```

## How to download isochrones from CMD 3.7

last tested: 2022-11-09

```python
# import CMD
from berliner import CMD
# initialize it
c = CMD()

# Example 1: download one isochrone
c.get_one_isochrone(
    logage=9.0,     # log age
    z=0.0152,       # if [M/H] is not set, z is used
    mh=None,        # [M/H]
    photsys_file='2mass_spitzer', # photometric system
    )

# Example 2: download a grid of isochrones
# define your grid of logAge and [M/H] as tuple(lower, upper, step)
grid_logage = (6, 10.2, 0.1)
grid_mh = (-2.6, 0.5, 0.1)
# download isochrones in parallel
isoc_lgage, isoc_mhini, isoc_list_2mass_wise = c.get_isochrone_grid_mh(
    grid_logage=grid_logage, grid_mh=grid_mh, photsys_file="2mass_spitzer_wise",
    n_jobs=50, verbose=10)

# More ... 
c.help()            # take a look at the output, it may help you!
```