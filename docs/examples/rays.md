# Rays and autodiff

## 2D ray bending

Trace 2D rays through a smooth speed field and plot them over `c(x)`. Requires
`beamax[viz-mpl]`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_ray_bending.ipynb)

```python
--8<-- "examples/rays/2d_ray_bending.py"
```

## 2D rays autodiff

Optimize a neural `c(x)` field through the ray ODE and plot the rays, loss, and
$\Delta c$. Requires `beamax[viz-mpl,autodiff]`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_rays_autodiff.ipynb)

```python
--8<-- "examples/rays/2d_rays_autodiff.py"
```
