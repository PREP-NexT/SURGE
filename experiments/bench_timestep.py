"""Wall-clock of SURGE at another routing time step, under the timing protocol of
bench_unified_timing.py.

The protocol script reads the time step from the network file (4,320 s). This wrapper loads
the network with the time step given here instead, so the Newton solve runs at that step,
and leaves everything else to the protocol: compiled SURGE, the steady state of uniform
runoff, 10 warm-up and 600 timed steps, three repeats with the median kept. The wall-clock
of a simulated day is the time of a step multiplied by the steps in a day. The time step
must divide a day. Produces the 8,640 s and 21,600 s rows of Table S6.

Usage (the iteration count goes in through the environment, as for the protocol script):
  SURGE_ITERATIONS=<k> python experiments/bench_timestep.py <dt> <net.npz> <state.npz> <out.json>
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import bench_unified_timing as B
from surge.network import RoutingNetwork

dt, net_path, state_path, out = float(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
assert abs(86400.0 / dt - round(86400.0 / dt)) < 1e-9, 'the time step must divide a day'
B.RoutingNetwork = lambda p: RoutingNetwork(p, substep_seconds=dt)
B.cmd_time(['surge', net_path, state_path, out, '--repeats', '3'])
row = json.load(open(out))
row.update(routing_dt_s=dt, steps_a_day=int(round(86400.0 / dt)),
           s_per_simulated_day=row['ms_per_step'] * 86400.0 / dt / 1000.0)
json.dump(row, open(out, 'w'), indent=1)
print('dt %g s, k %d: %.3f ms a step, %.3f s a simulated day'
      % (dt, row['iterations'], row['ms_per_step'], row['s_per_simulated_day']), flush=True)
