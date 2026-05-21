# `beamax.solvers.hybrid_solver`

High-level hybrid solver combining low-frequency and high-frequency backends.

## Key Objects

- `HybridSolver`: orchestrates LF/HF split, optional downsampling, and merge logic.
- `HybridBackend`: adapter for arbitrary LF solver operations. It accepts any
  subset of `forward`, `time_reversal`, and `adjoint` callables, as long as at
  least one is present.
- `HybridContext`: stable payload passed to LF backend callables. It contains
  domains, masks, sensors, time grids, WPT objects, target shape, and split
  configuration.
- `HybridSolverConfig`: typed configuration for interpolation, cutoff behavior, and runtime options.

LF backends do not need to subclass `Solver`. Each operation callable receives
`(component_array, context)` and may adapt that context into whatever argument
order the underlying LF solver expects. Missing operations fail only when that
operation is called.

For existing beamax-style solvers, use
`HybridBackend.from_beamax_solver(existing_solver)`.

## API Reference

::: beamax.solvers.hybrid_solver
