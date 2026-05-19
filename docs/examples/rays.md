# Rays and autofocus

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

## Neural sound speed autofocus

Represent `c(x)` with a tiny Equinox neural field and optimize the field
parameters with autodiff through the Gaussian beam ray ODE. Source locations and
launch directions stay fixed; the example changes the medium itself.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elma16/beamax/blob/main/examples/rays/neural_sound_speed_autofocus.ipynb)

```python
--8<-- "examples/rays/neural_sound_speed_autofocus.py"
```
