# Rays and autodiff

Static ray-tracing examples for Gaussian beam trajectories and differentiable
ray objectives.

---

## 2D ray bending

Trace a small fan of 2D rays through a smooth analytic speed field. The script
overlays ray paths on `c(x)` and reports the observed lateral displacement and
direction change.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_ray_bending.ipynb)

```python
--8<-- "examples/rays/2d_ray_bending.py"
```

---

## 2D rays autodiff

Port the thesis ray-focusing example: represent `c(x)` with a small neural
field, optimize it with autodiff through the Gaussian beam ray ODE, and save
the before/after rays, loss curve, and $\Delta c$ panels.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/2d_rays_autodiff.ipynb)

```python
--8<-- "examples/rays/2d_rays_autodiff.py"
```
