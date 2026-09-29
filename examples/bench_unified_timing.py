"""One timing protocol for every configuration of the paper (2026-09-14).

The published timings came from several scripts with different forcings, starting states
and repeat counts. This script replaces all of them with a single protocol:

  forcing     uniform runoff of RUNOFF_MMDAY, as lateral inflow per unit channel length
  state       the steady discharge field of that forcing, computed once per network by
              `steady` and shared by every configuration on every machine, so the serial
              kernel (whose Newton iteration stops on a tolerance) sees the same state
  timing      WARMUP routing time steps discarded (they include compilation), then TIMED
              time steps, one month at 20 steps per day, timed as one run with the device
              synchronised at the start, at every BLOCK steps and at the end

Usage
  steady       <net.npz> <state_out.npz>
  time         <config> <net.npz> <state.npz> <out.json> [--res res.npz] [--threads 24]
  time-basins  <config> <global_net.npz> <global_state.npz> <out.json>

config: surge (fixed point, compiled, GPU), gpu_ordered (level by level, GPU),
        cpu_parallel (fixed point, CPU threads), cpu_serial (serial C++ kernel, one core).
With --res the GPU configurations run with the reservoirs of that file active, and with
--res empty they run the reservoir code path holding no reservoir.
TIMED and WARMUP can be lowered through the environment for a smoke test only.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# the cpu_serial configuration calls the CWatM kernel, which surge.serial_ref finds
# through GPU_KWR_T5_SO; the other configurations need nothing beyond this repository
from surge.network import RoutingNetwork
from surge.kinematic_torch import TorchRouter

SWEEPS = int(os.environ.get('SURGE_SWEEPS', 3))   # the iteration count SURGE and CPU-parallel run;
NEWTON = 4                                        # each network needs its own, so it is configurable
RUNOFF_MMDAY = 2.0
WARMUP = int(os.environ.get('TIMING_WARMUP', 10))
TIMED = int(os.environ.get('TIMING_STEPS', 600))        # one month at 20 time steps per day
BLOCK = int(os.environ.get('TIMING_BLOCK', 100))
STEADY_TOL = 1e-4          # largest relative change over 100 steps, cells above 1 m3/s
STEADY_MAX_DAYS = 730


def lateral_inflow(net):
    """Uniform runoff in mm/day as inflow per unit channel length [m2/s]."""
    q = (RUNOFF_MMDAY / 1000.0 / 86400.0) * np.asarray(net.cellArea, np.float64) \
        / np.asarray(net.chanLength, np.float64)
    return np.clip(np.nan_to_num(q), 0.0, None)


def machine():
    info = dict(torch=torch.__version__, cpu_count=os.cpu_count(),
                loadavg_start=os.getloadavg())
    try:
        info['cpu'] = [l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo')
                       if l.startswith('model name')][0]
    except Exception:
        pass
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    return info


# ----------------------------------------------------------------------------- steady
def steady(net_path, out_path):
    net = RoutingNetwork(net_path)
    r = TorchRouter(net, device='cuda', dtype=torch.float32)
    qd = torch.as_tensor(lateral_inflow(net), dtype=torch.float32, device='cuda')
    q = torch.zeros(net.n_cells, device='cuda')
    nsub = int(net.noRoutingSteps)
    t0 = time.time(); step = 0; change = np.inf
    while step < STEADY_MAX_DAYS * nsub:
        prev = q.clone()
        for _ in range(100):
            q = r.substep_jacobi(q, qd, sweeps=SWEEPS, n_newton=NEWTON)
        step += 100
        wet = q > 1.0
        change = float(((q - prev).abs() / q.clamp_min(1e-3))[wet].max()) if bool(wet.any()) else np.inf
        if step % 2000 == 0:
            print('  %s step %d (%.1f days): max relative change %.2e (%.0fs)'
                  % (os.path.basename(net_path), step, step / nsub, change, time.time() - t0), flush=True)
        if step >= net.n_levels // SWEEPS + 100 and change < STEADY_TOL:
            break
    np.savez(out_path, q=q.double().cpu().numpy(), runoff_mmday=RUNOFF_MMDAY, steps=step,
             max_relative_change_per_100_steps=change, net=os.path.basename(net_path),
             n_cells=int(net.n_cells))
    print('steady %s: %d steps (%.1f days), change %.2e, mean Q %.2f m3/s -> %s'
          % (os.path.basename(net_path), step, step / nsub, change, float(q.mean()), out_path), flush=True)


# ----------------------------------------------------------------------------- timing
def make_step(config, net, qd_np, res=None, threads=24):
    """Returns (step function on the configuration's own arrays, initial-state converter, sync)."""
    if config in ('surge', 'gpu_ordered'):
        dev = 'cuda'
        if res is not None:
            from surge.reservoir import ResRouter
            r = ResRouter(net, device=dev, dtype=torch.float32)
            if res == 'empty':
                # the reservoir code path with no reservoir in it, which separates the cost of
                # the path itself from the cost of the dams
                r.set_reservoirs(np.zeros(0, np.int64), np.zeros(0), np.zeros(0))
            else:
                z = np.load(res)
                r.set_reservoirs(z['res_cell'], z['cap_m3'], z['rmean'])
            r.reset_storage()
            fn = (lambda a, b: r.jacobi_res(a, b, sweeps=SWEEPS, n_newton=NEWTON)) if config == 'surge' \
                else (lambda a, b: r.wavefront_res(a, b, n_newton=NEWTON))
        else:
            r = TorchRouter(net, device=dev, dtype=torch.float32)
            fn = (lambda a, b: r.substep_jacobi(a, b, sweeps=SWEEPS, n_newton=NEWTON)) if config == 'surge' \
                else (lambda a, b: r.substep_wavefront(a, b, n_newton=NEWTON))
        compiled = False
        if config == 'surge':
            fn = torch.compile(fn); compiled = True
        qd = torch.as_tensor(qd_np, dtype=torch.float32, device=dev)
        return (lambda q: fn(q, qd)), (lambda s: torch.as_tensor(s, dtype=torch.float32, device=dev)), \
            torch.cuda.synchronize, dict(compiled=compiled, device=dev, reservoirs=res is not None)
    if config == 'cpu_parallel':
        torch.set_num_threads(threads)
        r = TorchRouter(net, device='cpu', dtype=torch.float32)
        qd = torch.as_tensor(qd_np, dtype=torch.float32)
        return (lambda q: r.substep_jacobi(q, qd, sweeps=SWEEPS, n_newton=NEWTON)), \
            (lambda s: torch.as_tensor(s, dtype=torch.float32)), (lambda: None), \
            dict(compiled=False, device='cpu', threads=threads)
    if config == 'cpu_serial':
        from surge.serial_ref import kinematic_serial
        return (lambda q: kinematic_serial(q, qd_np, net)), (lambda s: np.array(s, np.float64)), \
            (lambda: None), dict(compiled=False, device='cpu', threads=1)
    raise SystemExit('unknown config %s' % config)


def time_config(config, net, state, res=None, threads=24):
    step, to_state, sync, meta = make_step(config, net, lateral_inflow(net), res, threads)
    q = to_state(state)
    t0 = time.time()
    try:
        for _ in range(WARMUP):
            q = step(q)
        sync()
    except Exception as e:              # a compile failure falls back to the uncompiled step
        if not meta.get('compiled'):
            raise
        print('  compile failed (%s: %s), timing uncompiled' % (type(e).__name__, str(e)[:120]), flush=True)
        torch._dynamo.reset()
        import torch as _t
        step, to_state, sync, meta = make_step('gpu_ordered' if False else config, net, lateral_inflow(net), res, threads)
        meta['compiled'] = False; meta['compile_error'] = str(e)[:300]
        q = to_state(state)
        for _ in range(WARMUP):
            q = step(q)
        sync()
    warm_s = time.time() - t0
    blocks = []
    sync(); start = time.perf_counter(); last = start
    for i in range(1, TIMED + 1):
        q = step(q)
        if i % BLOCK == 0 or i == TIMED:
            sync(); now = time.perf_counter()
            n = BLOCK if i % BLOCK == 0 else TIMED % BLOCK
            blocks.append((now - last) / n * 1e3); last = now
    total_ms = (last - start) * 1e3
    meta.update(config=config, n_cells=int(net.n_cells), levels=int(net.n_levels), warmup_steps=WARMUP,
                timed_steps=TIMED, warmup_s=warm_s, ms_per_step=total_ms / TIMED,
                block_ms_per_step=blocks, block_spread=(max(blocks) - min(blocks)) / (total_ms / TIMED),
                loadavg_end=os.getloadavg())
    return meta


def cmd_time(argv):
    config, net_path, state_path, out = argv[:4]
    res = argv[argv.index('--res') + 1] if '--res' in argv else None
    threads = int(argv[argv.index('--threads') + 1]) if '--threads' in argv else 24
    net = RoutingNetwork(net_path)
    st = np.load(state_path)
    assert int(st['n_cells']) == net.n_cells, 'state does not belong to this network'
    info = machine()
    reps = int(argv[argv.index('--repeats') + 1]) if '--repeats' in argv else 1
    runs = [time_config(config, net, st['q'], res, threads) for _ in range(reps)]
    row = runs[-1]
    if reps > 1:
        # each repeat rebuilds the router and warms up again, so the spread across repeats
        # measures what the node does to the timing, not what the configuration costs
        each = [r['ms_per_step'] for r in runs]
        ordered = sorted(each)
        row['repeats'] = reps
        row['repeats_ms'] = each
        row['ms_per_step_min'] = ordered[0]
        row['ms_per_step_spread'] = (ordered[-1] - ordered[0]) / ordered[len(ordered) // 2]
        row['ms_per_step'] = ordered[len(ordered) // 2]
    row.update(net=os.path.basename(net_path), state=os.path.basename(state_path),
               reservoir_file=(res if res == 'empty' else os.path.basename(res)) if res else None, runoff_mmday=RUNOFF_MMDAY,
               sweeps=SWEEPS, newton=NEWTON, **info)
    json.dump(row, open(out, 'w'), indent=1)
    print('%-12s %-32s %10d cells %5d lvl  %s%10.4f ms/step (blocks spread %.1f%%)  load %.1f -> %.1f  -> %s'
          % (config, os.path.basename(net_path), net.n_cells, net.n_levels,
             ('dams ' if res else '') + ('x%d ' % row['repeats'] if row.get('repeats') else ''),
             row['ms_per_step'], 100 * row.get('ms_per_step_spread', row['block_spread']),
             info['loadavg_start'][0], row['loadavg_end'][0], out),
          flush=True)


def cmd_time_basins(argv):
    from subset_basin import cell_coords, downstream_pointer, drainage_area, collect_upstream, rebuild
    from bench_basins import BASINS
    config, src, state_path, out = argv[:4]
    g = {k: v for k, v in np.load(src).items()}
    qg = np.load(state_path)['q']
    assert qg.size == int(g['mapC'])
    clat, clon = cell_coords(g['mask2d'])
    ptr = downstream_pointer(g)
    acc = drainage_area(g, ptr)
    info = machine(); rows = []
    tmp = tempfile.mkdtemp(prefix='basin_timing_')
    for name, lat, lon in BASINS:
        d = np.hypot(clat - lat, (clon - lon) * np.cos(np.radians(lat)))
        near = np.flatnonzero(d < 0.5)
        outlet = int(near[np.argmax(acc[near])]) if near.size else int(np.argmin(d))
        cells = collect_upstream(g['dirupLen'].astype(np.int64), g['dirupID'].astype(np.int64), outlet)
        path = os.path.join(tmp, name.replace('.', '') + '.npz')
        np.savez(path, **rebuild(g, cells, ptr))
        net = RoutingNetwork(path)
        # a basin drains only its own upstream cells, so the global steady field restricted
        # to them is the basin's steady field; rebuild keeps the cells in global order
        row = time_config(config, net, qg[cells])
        row.update(name=name, area_km2=float(acc[outlet] / 1e6))
        rows.append(row)
        print('%-12s %-12s %9d cells %5d lvl  %10.4f ms/step (spread %.1f%%)'
              % (config, name, net.n_cells, net.n_levels, row['ms_per_step'], 100 * row['block_spread']), flush=True)
        os.remove(path); del net
        if torch.cuda.is_available():
            torch._dynamo.reset(); torch.cuda.empty_cache()
    json.dump(dict(config=config, source=os.path.basename(src), state=os.path.basename(state_path),
                   runoff_mmday=RUNOFF_MMDAY, sweeps=SWEEPS, newton=NEWTON, warmup_steps=WARMUP,
                   timed_steps=TIMED, rows=rows, **info), open(out, 'w'), indent=1)
    print('->', out, flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd = sys.argv[1]
    if cmd == 'steady':
        steady(sys.argv[2], sys.argv[3])
    elif cmd == 'time':
        cmd_time(sys.argv[2:])
    elif cmd == 'time-basins':
        cmd_time_basins(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
