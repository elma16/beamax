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

Planar inverse solves use one launch time per boundary packet and
integrate backward from the acquisition time to zero. Supply an
acquisition time grid starting at zero and spanning the boundary-data
time axis. Cartesian detector channels follow increasing tangential
grid order with spacing from `data_domain.dx[1:]`.

For real data on a conjugate frame, inverse `top_n` counts one
representative per conjugate pair. Complex renderers select signed
packets from the full frame and retain complex phase.

::: beamax.solvers.msgb_solvers.msgb_solver
