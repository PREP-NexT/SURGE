"""surge -- GPU-parallel, differentiable kinematic-wave river routing.

The serial upstream-to-downstream routing recursion of global hydrological models
is recast as a convergent fixed-point iteration: every cell is updated at once,
and k iterations reproduce the serial solution. The iterations a network needs
follow its Courant number, two on a 30 arc-minute network and thirty on a
1 arc-minute one. Every operation is a differentiable tensor primitive, so
reverse-mode autograd gives exact gradients of any discharge functional with
respect to routing parameters, lateral inflow, or initial state.

Two GPU schemes:
  * TorchRouter.substep_parallel -- every cell at once, k iterations
  * TorchRouter.substep_ordered  -- one topological level at a time, which
                                    returns the serial solution

See the accompanying paper: "SURGE: parallel kinematic wave river routing on GPUs".
"""
from .network import RoutingNetwork
from .kinematic_torch import TorchRouter
from .reservoir import ResRouter

__all__ = ["RoutingNetwork", "TorchRouter", "ResRouter"]
__version__ = "0.1.0"
