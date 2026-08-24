# `beamax.gb.PallasConfig`

Static launch, layout, and periodic-wrapping options for experimental Pallas
kernels.

Use the defaults unless profiling identifies a reason to tune them:

- `sensor_block_size` controls detector tiling;
- `gpu_time_block_size`, `gpu_num_warps`, and `gpu_num_stages` control GPU
  launches;
- `rounding_mode` and `periodic_axes` control minimum-image wrapping; and
- the `tpu_*` fields select TPU layouts and block sizes.

::: beamax.gb.pallas_config.PallasConfig
