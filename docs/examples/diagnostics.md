# Diagnostics

## Memory planning

Before attempting a 3D run, estimate the memory needed for forward propagation
and time reversal on a 128 × 128 × 128 grid. Set the budget in GiB, then
compare the estimates for streamed and materialized coefficient selection.
The script prints estimates without allocating the simulated 3D wavefield.

```sh
python examples/diagnostics/memory_planning.py --budget-gib 8
```

```python
--8<-- "examples/diagnostics/memory_planning.py"
```
