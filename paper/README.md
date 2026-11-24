# Paper figures

Reproducible figures for the manuscript *"SURGE: parallel kinematic wave river routing
on GPUs"*. Every figure is redrawn by one script that reads only the
compact data committed here, so no raw model output and no GPU is needed to
reproduce them.

```
paper/
├── figure_scripts/   one script per figure, and the shared style in paper_style.py
├── figure_data/      the measured values each figure draws
└── figures/          the figures as the manuscript prints them
```

## Figure, script and data

| Manuscript | Script | Reads | Produces |
|---|---|---|---|
| Fig. 1, three ways to solve one network | `fig_method_seine.py` | a routing network built by `build/` | `fig_method.pdf/png` |
| Fig. 2, the implicit difference on one river reach | `fig_river_reach_block.py` | `river_reach_shapes.json`, `river_reach_text.json` | `fig_river_reach.pdf/png` |
| Fig. 3, the iteration along one rain path | `fig_isometric_terrain.py` | the network of Fig. 1 | `fig_isometric.pdf/png` |
| Fig. 4, how the iteration closes the gap on six cells | `fig_iteration_gap.py` | nothing (solves the six cells itself) | `fig_iteration_gap.pdf/png` |
| Fig. 5, match to the serial solution at 5 arcmin | `fig_kge_convergence.py` | `metric_maps_serialC.npz`, `accuracy.json` | `fig_kge_convergence.pdf/png` |
| Fig. 6, iterations each network needs | `fig_iterations_by_resolution.py` | `iterations_by_resolution.npz` | `fig_iterations_by_resolution.pdf/png` |
| Fig. 7, wall-clock and speedup | `fig_timing_bars.py` | `resolution_scaling.json` | `fig_timing_bars.pdf/png` |
| Fig. 8, 26 basins routed on their own | `fig_basin_speedup.py` | `bench_basins_display.json`, `basin_map_01min.npz`, `routing_static_30min.npz` | `fig_basin_speedup.pdf/png` |
| Fig. 9, reservoir operation | `fig_reservoir.py` | `reservoir_timing.json`, `reservoir_convergence.json`, `reservoir_k7_scores.json` | `fig_reservoir.pdf/png` |

`fig_method_seine.py` and `fig_isometric_terrain.py` are the two scripts that need
a routing network rather than a distilled file, because they draw a real basin. Build the 5 arc-minute
network with `build/build_multiresolution.py` and point the script at the folder holding
`routing_static_05min.npz`,

```bash
SURGE_DATA=/path/to/networks python fig_method_seine.py
SURGE_DATA=/path/to/networks python fig_isometric_terrain.py
```

## What the data files hold

| File | Content |
|---|---|
| `metric_maps_serialC.npz` | KGE of SURGE against the CPU-serial solution at one, three, seven and twenty iterations, for the 106,661 cells of the 5 arcmin network whose annual mean discharge exceeds 100 m³ s⁻¹ |
| `accuracy.json` | daily discharge at the Amazon mouth through 2010, for CPU-serial and for SURGE at one, three and seven iterations |
| `iterations_by_resolution.npz` | for each of the five networks and each iteration count, the distribution of 1 − KGE across the evaluated cells, as the density this figure shades and the 5th, 50th and 95th percentiles it marks |
| `resolution_scaling.json` | wall-clock per routing time step of the four configurations at the five resolutions, with the iteration count each network needs |
| `bench_basins_display.json` | cells, levels, wall-clock and speedup of the 26 large river basins routed on their own at 1 arcmin |
| `basin_map_01min.npz` | the 26 basins painted on a coarse grid, for the map |
| `routing_static_30min.npz` | the 30 arcmin network, used for the land backdrop of the map |
| `river_reach_shapes.json`, `river_reach_text.json` | the reach outlines and labels of the schematic |
| `reservoir_timing.json` | wall-clock with and without the dams at 5, 3 and 1 arcmin, timed in the same job |
| `reservoir_convergence.json` | 1 − KGE at the 5th percentile against the iteration count, with and without the dams |
| `reservoir_k7_scores.json` | median KGE and NSE and the largest absolute bias at seven iterations, with and without the dams |

## Reproduce

```bash
cd paper/figure_scripts
python fig_river_reach_block.py
python fig_iteration_gap.py
python fig_kge_convergence.py
python fig_iterations_by_resolution.py
python fig_timing_bars.py
python fig_basin_speedup.py
python fig_reservoir.py
```

They need `numpy`, `matplotlib` and `scipy`; `fig_kge_convergence.py` also needs
`ultraplot` and `cartopy` for the map, `fig_basin_speedup.py` needs `pyproj`, and
`fig_river_reach_block.py` uses `pdftocairo` (poppler) for its PNG. The figures use the
shared house style of `paper_style.py`, which falls back to the matplotlib
defaults where the optional *Myriad Pro* font is absent. Paths are resolved
relative to the script, so the commands work from any directory.
