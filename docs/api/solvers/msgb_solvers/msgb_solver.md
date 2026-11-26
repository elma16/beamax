# `beamax.solvers.msgb_solvers.msgb_solver`

Forward, time-reversal, and adjoint MSGB orchestration, including stage
overrides and optional beam sharding.

Forward sample times are measured from the initial state at time zero.
You can request a delayed sample or a uniform grid beginning after zero.
Direct Gaussian beam integrators take the initial state at the first time
in their input grid.

With periodic boundaries and scalar sound speed, forward propagation
polarises initial velocity in Fourier space. Its spatial mean contributes
a uniform field. Real renderers can select one packet from each conjugate
pair when the MSWPT frame supports it. With other frames, the renderer
selects from the full frame and accounts for that in its amplitude.

::: beamax.solvers.msgb_solvers.msgb_solver
