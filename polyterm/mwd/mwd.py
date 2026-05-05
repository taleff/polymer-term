"""
Data class for storing molecular weight distribution fitting results

This module provides a dataclass for storing information about the
molecular weight distribution and the kinetic information used to
create it.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class MWDResult:
    """
    Container for model results.

    This immutable class stores all outputs from a kinetic model fit,
    including fitted parameters and predicted distribution.

    Attributes
    ----------
    molecular_weights : ndarray
        Molecular weights at which the fit was performed.
    intensities : ndarray
        The intensities corresponding to each molecular weight
    dead_chain_intensities : ndarray
        Predicted intensities from dead chains only.
    live_chain_intensities : ndarray
        Predicted intensities from live chains only.
    dead_chain_fraction : float
        Fraction of chains that have terminated.
    alpha : float
        Fitted ratio of termination to propagation rate constants (kt/kp).
    init : float
        Fitted or specified initial initiator concentration.
    order : float
        Termination reaction order used in the fit.
    sigma : float
        SEC line broadening parameter (fitted or fixed).
    tau : float
        SEC tailing parameter (0 if Gaussian broadening used).
    conversion : float
        Monomer conversion at the end of polymerization (0 to 1).
    r_squared : float
        Coefficient of determination for the fit.
    """
    # Molecular weight distribution information
    molecular_weights: np.ndarray
    intensities: np.ndarray
    dead_chain_intensities: np.ndarray
    live_chain_intensities: np.ndarray
    
    # Basic kinetic parameters
    dead_chain_fraction: float
    
    # Kinetic parameters
    alpha: Optional[float] = None
    init: Optional[float] = None
    order: Optional[float] = None
    sigma: Optional[float] = None
    tau: Optional[float] = None
    conversion: Optional[float] = None

    # Fit paramters
    r_squared: Optional[float] = None

    def __repr__(self) -> str:
        """Return string representation of fit results."""
        def fmt(value, spec):
            """Format value with spec, or return 'None' if value is None."""
            return f"{value:{spec}}" if value is not None else "None"

        lines = [
            "MWDResult(",
            "  =Common Parameters=",
            f"  Dead Chains: {self.dead_chain_fraction*100:.1f}%",
            "  =Kinetic Parameters=",
            f"  alpha: {fmt(self.alpha, '.6f')}",
            f"  [I]_0: {fmt(self.init, '.6f')}",
            f"  order: {fmt(self.order, '.3f')}",
            f"  sigma: {fmt(self.sigma, '.6f')}",
            f"  tau: {fmt(self.tau, '.6f')}",
            f"  conversion: {fmt(self.conversion, '.4f')}",
            "  =Fit Parameters=",
            f"  R^2 = {fmt(self.r_squared, '.6f')}",
            ")",
        ]
        return "\n".join(lines)
    
