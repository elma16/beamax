# `beamax.plotter`

Install `beamax[viz-mpl]` for the Matplotlib plotting functions. Each comparison
takes computed arrays and returns a Matplotlib figure.

## Forward recordings

Pass three pressure records in `(time, receiver)` order on the same time grid.
The dashed line marks the interior receiver with the largest RMS difference
across the two comparisons. The lower panel shows that receiver in all three records.

```python
from beamax.plotter import plot_forward_records, save_comparison_figure

fig = plot_forward_records(kwave_record, msgb_record, hybrid_record, ts)
save_comparison_figure(fig, "forward.png", fixed_canvas=True, show=True)
```

## One-face time reversal and adjoint

Pass the initial pressure, detector mask and physical grid size. Each optional
pair contains the raw k-Wave and MSGB images, in that order. This figure uses
the Chapter 5 two-packet calibration for a 128 × 128 image, including the
four-face display gain on the adjoints and the marked profile at column 87.

```python
from beamax.plotter import plot_packet_box_scanner, save_comparison_figure

fig = plot_packet_box_scanner(
    p0, face.to_mask(domain).binary_mask, domain.grid_size,
    time_reversal=(tr_kw, tr_msgb),
    adjoint=(adj_kw, adj_msgb),
)
save_comparison_figure(fig, "one_face.png", show=True)
```

::: beamax.plotter
