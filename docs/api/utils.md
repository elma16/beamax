# `beamax.utils`

Shared array, FFT, interpolation, indexing, batching, device, and memory-planning
utilities.

## Arrays

::: beamax.utils.arrays
    options:
      show_root_heading: false
      members:
        - interpolate_nearest
        - pad_array
        - pad_zero
        - pad_edge
        - crop_centered
        - interpolate_fourier
        - extract_centered_box
        - rel_l2

## Fourier transforms

::: beamax.utils.fft
    options:
      show_root_heading: false
      members:
        - unitary_fft
        - unitary_ifft
        - convert_space

## Interpolation

::: beamax.utils.interp
    options:
      show_root_heading: false
      members:
        - make_c_function_from_grid
        - Interpolator

## Coefficient indexing

::: beamax.utils.coeff_index
    options:
      show_root_heading: false
      members:
        - batch_data
        - find_level
        - find_tensor_and_multiindex
        - compute_coeff_shapes

## Memory planning

::: beamax.utils.memory
    options:
      show_root_heading: false
      members:
        - DeviceCapabilities
        - MemoryEstimate
        - device_capabilities
        - estimate_msgb_memory
