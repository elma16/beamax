# Rays and autodiff

## 2D ray bending

Launch 60 parallel rays towards a smooth sound-speed lens. Follow their
trajectories on the map of `c(x)` and compare their final directions with
their common launch direction. Requires `beamax[viz-mpl]`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/rays/2d_ray_bending.ipynb)

```python
--8<-- "examples/rays/2d_ray_bending.py"
```

## 2D rays autodiff

Choose a focal point and optimize a neural sound-speed field by differentiating
through the ray ODE. Compare the ray endpoints before and after optimization,
then inspect the change in `c(x)` that produced the focusing. Requires
`beamax[viz-mpl,autodiff]`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/v0.4.0/examples/rays/2d_rays_autodiff.ipynb)

```python
--8<-- "examples/rays/2d_rays_autodiff.py"
```
