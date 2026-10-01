# Demo

Thirty days on the global 30 arc-minute river network, 70,648 cells. It runs on a
laptop in under a minute and needs no GPU and no download: the network is the one
in `paper/figure_data/`, and the runoff is beside this file.

```bash
python demo/run_demo.py
```

## What it does

The runoff is 2 mm/day everywhere, with 8 mm/day more over the northern hemisphere
on days 6 to 10, so there is a wave to route. SURGE routes it twice: once with the
parallel scheme at the two iterations this network needs, and once with the ordered
scheme, which takes the topological levels in turn and so returns the serial
solution. The two are then compared on the cells the thirty days fill.

## Expected output

```
network 70,648 cells, 116 levels, 20 substeps a day, on the CPU
routing 30 days twice
  parallel scheme, k = 2           3.5 s
  ordered scheme (the reference)  22.5 s

evaluated on 873 cells above 10 m3/s
  median relative difference  4.64e-03
  99th percentile             6.64e-02
  largest discharge           85 m3/s (parallel)  85 m3/s (ordered)

wrote demo/discharge_30day.npy  (30, 70648)
```

The numbers above come from four CPU threads of an AMD EPYC 7713. A GPU gives the
same figures to the last digit, the solver running in double precision, and takes
0.7 s for the parallel run. The wall-clock will differ on other hardware; the
discharges and the differences will not.

Two things in that output are the point of the paper. The parallel scheme is six
times faster here and three orders of magnitude faster on the networks of the paper,
because it has no level barriers to wait on. And its answer sits within half a per
cent of the serial solution at two iterations, which is what this network needs; a
finer network needs more, up to thirty at 1 arc-minute.

The demo evaluates on cells above 10 m3/s rather than the 100 m3/s of the paper
because thirty days from an empty network is not long enough to fill a large river.
The year-long runs in `experiments/` start from a spun-up field for that reason.

## Expected run time

| | 4 CPU threads | 1 GPU |
|---|---|---|
| parallel scheme, 30 days | 3.5 s | 0.7 s |
| ordered scheme, 30 days | 22.5 s | 43 s |

The ordered scheme is slower on the GPU than on the CPU here: it launches one kernel
group per topological level, 116 of them a substep, and at this size the launches
cost more than the arithmetic. That gap is what the paper is about, and it widens
with resolution.
