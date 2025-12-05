"""
Single termination order kinetic model.

This module implements fitting for polymerizations with a single termination
pathway.
"""

import numpy as np
from scipy.stats import poisson
from scipy.integrate import quad_vec
from scipy.special import lambertw
from scipy.optimize import NonlinearConstraint
from typing import Optional
import warnings

from .base import BaseModel, FitResult
from ..mwd import MolecularWeightDistribution
from ..core.kinetics import (
    living_chain_concentration,
    living_chain_dp,
    conversion_to_time,
    monomer_conversion,
    time_to_chain_death,
)
from ..core.distributions import (
    gaussian_broadening,
    living_distribution_integrand,
)
from ..utils import (
    calculate_number_average_dp,
    fit_right_edge,
    calculate_r_squared,
)

__all__ = ['SingleOrderModel']


class SingleOrderModel(BaseModel):
    """
    Fitting for polymerizations with a single termination pathway

    This model fits a molecular weight distribution to determine termination
    kinetics, assuming a single termination pathway with a specified
    reaction order.

    Parameters
    ----------
    monomer_mw : float
        Molecular weight of one monomer unit.
    init_mon : float
        Initial monomer concentration.
    order : float
        Order of termination reaction. Common values: 1.0 (first order),
        1.5, 2.0 (second order). This parameter is required.
    conversion : float, optional
        Monomer conversion (0 to 1). If None, will be fitted.
    init : float, optional
        Initial initiator concentration. If None, will be fitted.
    combination : bool, optional
        Whether termination occurs by chain combination. Default False.
    bn : float, optional
        Inverse of propagation order. Default 1.0.
    max_fit_points : int, optional
        Maximum points for fitting (downsamples if needed). Default 200.

    Examples
    --------
    Fit with known order, fit everything else:

    >>> model = SingleOrderModel(
    ...     monomer_mw=104.15,
    ...     init_mon=1.0,
    ...     order=1.5
    ... )
    >>> result = model.fit(experimental_mwd)
    >>> print(f"α = {result.alpha:.4f}, R² = {result.r_squared:.4f}")

    Fit with known order, conversion, and initiator:

    >>> model = SingleOrderModel(
    ...     monomer_mw=104.15,
    ...     init_mon=1.0,
    ...     order=1.5,
    ...     conversion=0.95,
    ...     init=0.02
    ... )
    >>> result = model.fit(experimental_mwd)
    """

    def __init__(
        self,
        monomer_mw: float,
        init_mon: float,
        order: float,
        conversion: Optional[float] = None,
        init: Optional[float] = None,
        combination: bool = False,
        bn: float = 1.0,
        max_fit_points: int = 500
    ):
        """Initialize single order termination model."""
        super().__init__(monomer_mw, init_mon, combination, bn, max_fit_points)

        # Validate required parameters
        if order <= 0:
            raise ValueError("order must be positive")

        # Validate optional parameters
        if conversion is not None and not (0 <= conversion <= 1):
            raise ValueError("conversion must be between 0 and 1 if specified")
        if init is not None and init <= 0:
            raise ValueError("init must be positive if specified")

        self.order = order
        self.conversion = conversion
        self.init = init

    def fit(self, mwd: MolecularWeightDistribution) -> FitResult:
        """
        Fit the model to a molecular weight distribution.

        Parameters
        ----------
        mwd : MolecularWeightDistribution
            Experimental or synthetic MWD to fit.

        Returns
        -------
        FitResult
            Fitted parameters, predicted distribution, and metrics.

        Notes
        -----
        The fitting uses nonlinear least squares optimization. Initial
        guesses are estimated from the distribution characteristics (peak
        position, number average, etc.).

        The optimization is performed on downsampled data normalized on
        a log(MW) scale for speed reasons.
        """
        # Prepare data
        fit_mws, fit_ints = self._prepare_data(mwd)

        # Calculate intelligent DP range to reduce computational cost
        # Instead of using max(fit_mws), use distribution characteristics
        nu = calculate_number_average_dp(fit_mws, fit_ints, self.monomer_mw)
        nup, _ = fit_right_edge(fit_mws, fit_ints, self.monomer_mw)

        # For living chains: Poisson(nup) has most mass within 5*sqrt(nup)
        # For dead chains: can extend to ~2*nup in worst case (high termination)
        # Conservative: cover >99.99% of distribution mass
        intelligent_max_dp = int(3 * nup + 10 * np.sqrt(nup))

        # Don't exceed original maximum, and ensure at least 2x number average
        max_mw_based_dp = int(np.max(fit_mws) / self.monomer_mw)
        max_dp = max(
            min(intelligent_max_dp, max_mw_based_dp),
            int(2 * nu)
        )

        dps = np.arange(1, max_dp, dtype=int)

        self.param_names = self._find_fitted_variables()

        # Estimate initial guesses and build parameter specifications
        init_guess, bounds = self._estimate_initial_guess(fit_mws, fit_ints, dps)

        # Build args for objective function
        args = (dps, fit_mws, fit_ints)

        # Run optimization with estimated parameter fallbacks
        result_dict = self._run_optimization(init_guess, bounds, args)

        # Calculate final metrics
        return self._build_fit_result(result_dict, fit_mws, fit_ints, dps)

    def _find_fitted_variables(self) -> list:
        """Determines the names of the fitted variables"""
        # These two variables are always fitted
        param_names = ['alpha', 'sigma']
        
        if self.init is None:
            param_names.append('init')

        if self.conversion is None:
            # Fitting time instead of conversion provides better stability
            # Time can be converted to conversion once the fitting is finished
            param_names.append('time')

        return param_names
            
    def _estimate_initial_guess(
        self,
        mws: np.ndarray,
        intensities: np.ndarray,
        dps: np.ndarray
    ) -> tuple:
        """Estimate initial parameter guesses from distribution shape.

        Returns: (initial_guess, bounds)
        """
        # Calculate distribution characteristics
        nu = calculate_number_average_dp(mws, intensities, self.monomer_mw)
        nup, sigma = fit_right_edge(mws, intensities, self.monomer_mw)
        nup = max(nup, nu * 1.001)  # Ensure nup > nu for stability

        if self.conversion is not None:
            mon_frac = 1 - self.conversion
        elif self.init is not None:
            mon_frac = max(0.001, 1 - nu * self.init / self.init_mon)
        else:
            mon_frac = 0.9  # Low conversion assumption

        # Estimate init concentration
        init = self.init if self.init is not None else (self.init_mon / nu) * (1 - mon_frac)

        # Estimate alpha
        alpha = self._estimate_alpha(self.order, mon_frac, init, nu, nup)

        # Estimate time
        conv = self.conversion if self.conversion is not None else (1 - mon_frac)
        time = conversion_to_time(alpha, init, self.order, conv, self.bn)

        # Maximum reasonable degree of polymerization and max reasonable
        # starting ratio fo termination to polymerization rate; helps reign
        # in bounds for alpha
        max_reasonable_dp = 10000
        max_starting_ratio = 0.05

        max_alpha = (max_starting_ratio * (self.init_mon**(2-self.order)) *
                     (max_reasonable_dp**(self.order-1)))
        min_alpha = max(1e-6, 0.1 * (self.init_mon**(2-self.order)) *
                        (max_reasonable_dp**(self.order-2)))
        
        init_guess = [alpha, sigma]
        bounds = [(min_alpha, max_alpha), (0.01, 0.5)]  # alpha, sigma bounds

        if 'init' in self.param_names:
            init_guess.append(init)
            bounds.append((self.init_mon / max_reasonable_dp, self.init_mon))

        if 'time' in self.param_names:
            init_guess.append(time)
            # Set upper bound to 10x the time needed for 99.99% chain death
            # Use self.init if known, otherwise use conservative estimate
            init_for_bound = self.init if self.init is not None else (self.init_mon / max_reasonable_dp)
            max_time = time_to_chain_death(0.9999, init_for_bound, self.order) * 10
            bounds.append((0, max_time))

        # Ensure initial guess respects bounds
        for i in range(len(init_guess)):
            lower, upper = bounds[i]
            init_guess[i] = np.clip(init_guess[i], lower, upper)

        return init_guess, bounds

    def _estimate_alpha(
        self,
        order: float,
        mon_frac: float,
        init: float,
        nu: float,
        nup: float
    ) -> float:
        """Estimate alpha (kt/kp) from distribution characteristics."""
        if np.isclose(mon_frac, 0):
            # High conversion case
            if order == 1:
                return 0.00001
            elif order == 2:
                ratio = self.init_mon / init / nup
                alpha = 1 - ratio
                if np.isclose(alpha, 1.0):
                    alpha = 0.999
                return alpha
            else:
                return abs(init ** (2 - order) / (order - 2))
        else:
            # General case
            if order == 1:
                return np.average([init * ((nup/nu) - 1), -init/np.log(mon_frac)])
            elif order == 2:
                ratio = self.init_mon / init / nup
                alpha = 1 - ratio + (
                    lambertw(
                        ratio * np.log(mon_frac) * (mon_frac ** ratio),
                        -1
                    ) / np.log(mon_frac)
                )
                alpha = alpha.real
                if np.isclose(alpha, 1.0):
                    alpha = 0.999
                return alpha
            else:
                return (0.5 * init ** (2 - order) / (order - 2) /
                       np.log(mon_frac))

    def _run_optimization(
        self,
        init_guess: list,
        bounds: list,
        args: tuple
    ) -> dict:
        """Run constrained optimization using scipy.optimize.minimize."""
        dps, fit_mws, fit_ints = args

        # Create objective function (returns scalar for minimize)
        objective = self._create_objective_function(dps, fit_mws, fit_ints)

        # Use base class optimization method
        return self._run_constrained_optimization(
            objective=objective,
            init_guess=init_guess,
            bounds=bounds,
        )

    def _create_objective_function(
        self,
        dps: np.ndarray,
        fit_mws: np.ndarray,
        fit_ints: np.ndarray,
    ):
        """Create objective function that unpacks parameters.

        Returns scalar sum of squares for minimize optimizer.
        """
        # Cache for broadening matrix to avoid recomputation when sigma unchanged
        cache = {'sigma': None, 'broadening': None}

        def objective(x):
            # Unpack parameters in consistent order
            params = dict(zip(self.param_names, x))

            # Fill in fixed/estimated parameters
            alpha = params['alpha']
            sigma = params['sigma']
            init = params.get('init', self.init)

            try:
                if 'time' in self.param_names:
                    pred = self._calculate_mwd(
                        alpha, init, sigma, self.order, dps, fit_mws,
                        time=params['time'],
                        broadening_cache=cache
                    )
                else:
                    pred = self._calculate_mwd(
                        alpha, init, sigma, self.order, dps, fit_mws,
                        conv=self.conversion,
                        broadening_cache=cache
                    )

                residuals = pred - fit_ints
                result = np.sum(residuals ** 2)

                # Check for NaN/Inf which don't raise exceptions
                if not np.isfinite(result):
                    return 1e10

                return result

            except (ValueError, RuntimeWarning, FloatingPointError):
                return 1e10

        return objective

    def _calculate_mwd(
        self,
        alpha: float,
        init: float,
        sigma: float,
        order: float,
        dps: np.ndarray,
        mws: np.ndarray,
        time: Optional[float] = None,
        conv: Optional[float] = None,
        broadening_cache: Optional[dict] = None
    ) -> np.ndarray:
        """Calculate MWD given either time or conversion."""
        # Determine time from conversion if needed
        if time is None:
            if np.isclose(conv, 1):
                time = time_to_chain_death(0.9999, init, self.order)
            else:
                time = conversion_to_time(alpha, init, self.order, conv, self.bn)

        # Setup broadening matrix with caching
        if broadening_cache is not None:
            # Check if we can use cached broadening matrix
            # Use exact equality check (not np.isclose) because during optimization
            # we need to recompute for every distinct sigma value to maintain
            # accurate gradient information
            if (broadening_cache['sigma'] is None or
                sigma != broadening_cache['sigma']):
                # Cache miss - compute new broadening matrix
                dps_mesh, mws_mesh = np.meshgrid(dps, mws)
                broadenings = gaussian_broadening(mws_mesh, dps_mesh * self.monomer_mw, sigma)
                broadening_cache['sigma'] = sigma
                broadening_cache['broadening'] = broadenings
            else:
                # Cache hit - reuse cached broadening matrix
                broadenings = broadening_cache['broadening']
        else:
            # No cache provided - compute directly
            dps_mesh, mws_mesh = np.meshgrid(dps, mws)
            broadenings = gaussian_broadening(mws_mesh, dps_mesh * self.monomer_mw, sigma)

        # Calculate distribution
        args = (dps, alpha, self.init_mon, init, self.order, self.combination, self.bn)
        dead_fracs, _ = quad_vec(living_distribution_integrand, 0, time, args=args)

        b = living_chain_concentration(init, self.order, time)
        nup = living_chain_dp(alpha, self.init_mon, init, self.order, time, self.bn)
        alive_fracs = b * poisson.pmf(dps, nup)
        total_fracs = alive_fracs + dead_fracs

        pred_mwd = np.matmul(broadenings, total_fracs * dps)
        return pred_mwd / np.trapezoid(pred_mwd, np.log(mws))

    def _build_fit_result(
        self,
        result_dict: dict,
        fit_mws: np.ndarray,
        fit_ints: np.ndarray,
        dps: np.ndarray
    ) -> FitResult:
        """Build FitResult object from fitted parameters."""
        # Extract optimized parameters
        x = result_dict['x']
        params = dict(zip(self.param_names, x))

        alpha = params['alpha']
        sigma = params['sigma']
        init = params.get('init', self.init)

        # Calculate conversion
        if self.conversion is not None:
            conversion = self.conversion
        else:
            time = params['time']
            conversion = monomer_conversion(time, 1/alpha, 1, 1, init, self.order, self.bn)

        # Calculate predicted intensities at fit points
        if 'time' in params:
            pred_ints = self._calculate_mwd(
                alpha, init, sigma, self.order, dps, fit_mws, time=params['time']
            )
        else:
            pred_ints = self._calculate_mwd(
                alpha, init, sigma, self.order, dps, fit_mws, conv=self.conversion
            )

        # Calculate R-squared
        r_squared = calculate_r_squared(fit_ints, pred_ints)

        # Calculate dead chain fraction
        if 'time' in params:
            time = params['time']
        else:
            time = conversion_to_time(alpha, init, self.order, conversion, self.bn)

        b = living_chain_concentration(init, self.order, time)
        dead_chain_fraction = 1 - b / init

        return FitResult(
            alpha=alpha,
            init=init,
            order=self.order,
            sigma=sigma,
            conversion=conversion,
            r_squared=r_squared,
            molecular_weights=fit_mws,
            predicted_intensities=pred_ints,
            dead_chain_fraction=dead_chain_fraction,
            fun=result_dict['fun'],
            jac=result_dict['jac'],
            hess_inv=result_dict['hess_inv'],
            fit_message=result_dict['message']
        )
    
