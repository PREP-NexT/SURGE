"""surge -- GPU-parallel, differentiable kinematic-wave river routing.

The serial upstream-to-downstream routing recursion of global hydrological models
is recast as a convergent Jacobi fixed-point iteration: a fixed, small number of
fully parallel sweeps (k ~= 3) reproduces the serial solution, with a sweep count
set by the local Courant number and independent of network depth. Every operation
is a differentiable tensor primitive, so reverse-mode autograd gives exact
gradients of any discharge functional with respect to routing parameters, lateral
inflow, or initial state.

Two GPU schemes:
  * TorchRouter.substep_jacobi   -- the parallel iterative solver (k sweeps)
  * TorchRouter.substep_wavefront-- exact level-scheduled traversal (bit-level
                                    oracle; matches the serial kernel)

See the accompanying paper: "Parallelizing kinematic-wave river routing on the GPU
via a convergent fixed-point reformulation".
"""
from .network import RoutingNetwork
from .kinematic_torch import TorchRouter
from .reservoir import ResRouter

__all__ = ["RoutingNetwork", "TorchRouter", "ResRouter"]
__version__ = "0.1.0"
