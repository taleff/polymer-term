"""
Base classes for kinetic fitting models.

This module defines abstract base classes and result containers used by
all fitting models in the polyterm package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import minimize

from ..mwd import MolecularWeightDistribution
from ..utils import calculate_r_squared

__all__ = ['BaseModel', 'FitResult']


@dataclass(frozen=True)
class FitResult:
    """
    Container for model fitting results.

    This immutable class stores all outputs from a kinetic model fit,
    including fitted parameters, goodness-of-fit metrics, and the
    predicted distribution.

    Attributes
    ----------
    alpha : float
        Fitted ratio of termination to propagation rate constants (kt/kp).
    init : float
        Fitted or specified initial initiator concentration.
    order : float
        Fitted or specified order of termination reaction.
    sigma : float
        Fitted SEC line broadening parameter.
    conversion : float
        Monomer conversion at the end of polymerization (0 to 1).
    r_squared : float
        Coefficient of determination for the fit.
    molecular_weights : ndarray
        Molecular weights at which the fit was performed (may be
        downsampled from input).
    predicted_intensities : ndarray
        Model-predicted intensities at molecular_weights.
    dead_chain_fraction : float
        Fraction of chains that have terminated.
    hess_inv : ndarray
        Inverse of the objective function's Hessian
    fit_message : str
        Status message from the optimizer.
    additional_params : dict, optional
        Additional model-specific parameters (e.g., alpha2, alpha3 for
        multi-order models).

    Methods
    -------
    predict(molecular_weights)
        Generate predicted MWD at arbitrary molecular weights.
    """

    alpha: float
    init: float
    order: float
    sigma: float
    conversion: float
    r_squared: float
    molecular_weights: np.ndarray
    predicted_intensities: np.ndarray
    dead_chain_fraction: float
    fun: np.ndarray
    jac: np.ndarray
    hess_inv: LinearOperator
    fit_message: str
    additional_params: Optional[Dict[str, Any]] = None

    def __repr__(self) -> str:
        """Return string representation of fit results."""
        lines = [
            "FitResult(",
            f"  α = {self.alpha:.6f}",
            f"  [I]₀ = {self.init:.6f}",
            f"  order = {self.order:.3f}",
            f"  σ = {self.sigma:.6f}",
            f"  conversion = {self.conversion:.4f}",
            f"  R² = {self.r_squared:.6f}",
            f"  dead chains = {self.dead_chain_fraction:.4f}",
        ]

        print(self.fit_message)

        if self.additional_params:
            for key, value in self.additional_params.items():
                if isinstance(value, float):
                    lines.append(f"  {key} = {value:.6f}")
                else:
                    lines.append(f"  {key} = {value}")

        lines.append(")")
        return "\n".join(lines)


class BaseModel(ABC):
    """
    Abstract base class for kinetic fitting models.

    All fitting models in polyterm inherit from this class and must
    implement the fit() method.

    Parameters
    ----------
    monomer_mw : float
        Molecular weight of one monomer unit.
    init_mon : float
        Initial monomer concentration.
    combination : bool, optional
        Whether termination occurs by chain combination. Default False.
    bn : float, optional
        Inverse of propagation order in living chain. Default 1.0.
    max_fit_points : int, optional
        Maximum number of points to use in fitting (will downsample if
        needed). Default is 400.
    """

    def __init__(
        self,
        monomer_mw: float,
        init_mon: float,
        combination: bool = False,
        bn: float = 1.0,
        max_fit_points: int = 500
    ):
        """Initialize base model parameters."""
        if monomer_mw <= 0:
            raise ValueError("monomer_mw must be positive")
        if init_mon <= 0:
            raise ValueError("init_mon must be positive")
        if max_fit_points < 10:
            raise ValueError("max_fit_points must be at least 100")
        if bn <= 0:
            raise ValueError("bn must be positive")

        self.monomer_mw = monomer_mw
        self.init_mon = init_mon
        self.combination = combination
        self.bn = bn
        self.max_fit_points = max_fit_points

    @abstractmethod
    def fit(self, mwd: MolecularWeightDistribution) -> FitResult:
        """
        Fit the model to a molecular weight distribution.

        Parameters
        ----------
        mwd : MolecularWeightDistribution
            The experimental or synthetic MWD to fit.

        Returns
        -------
        FitResult
            Object containing fitted parameters and metrics.

        Notes
        -----
        This method must be implemented by subclasses.
        """
        pass

    def _prepare_data(
        self,
        mwd: MolecularWeightDistribution
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare MWD data for fitting.

        Downsamples and normalizes the distribution for efficient fitting.

        Parameters
        ----------
        mwd : MolecularWeightDistribution
            Input distribution.

        Returns
        -------
        molecular_weights : ndarray
            Processed molecular weights.
        intensities : ndarray
            Normalized intensities on log-MW scale.
        """
        # Downsample if needed
        processed_mwd = mwd.downsample(self.max_fit_points)

        # Normalize on log scale
        processed_mwd = processed_mwd.normalize_on_log_scale()

        return processed_mwd.molecular_weights, processed_mwd.intensities

    def _run_constrained_optimization(
        self,
        objective: Callable,
        init_guess: list,
        bounds: list,
        options: Optional[dict] = None
    ) -> dict:
        """Run constrained optimization using scipy.optimize.minimize.

        This is a generic wrapper around scipy.optimize.minimize that all
        models can use. Subclasses provide model-specific objective functions
        and constraints.

        Parameters
        ----------
        objective : callable
            Objective function to minimize. Should accept parameter vector
            and return scalar cost.
        init_guess : list
            Initial parameter guess.
        bounds : list of tuple
            Parameter bounds as [(lower, upper), ...].
        options : dict, optional
            Optimizer options. Default {'maxiter': 1000, 'ftol': 1e-9, 'disp': False}.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'optimizer_result': scipy OptimizeResult object
            - 'x': optimal parameters
            - 'message': optimization status message
        """
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-9, 'disp': False}

        result = minimize(
            objective,
            init_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )

        return {
            'optimizer_result': result,
            'x': result.x,
            'fun': result.fun,
            'jac': result.jac,
            'hess_inv': result.hess_inv,
            'message': f"{result.message} (success={result.success}, nit={result.nit})"
        }
