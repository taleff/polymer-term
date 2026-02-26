"""
Molecular weight distribution data container and manipulation.

This module provides the MolecularWeightDistribution class, the primary
interface for working with polymer molecular weight distributions.
"""

import numpy as np
import warnings
from dataclasses import dataclass

from .core.distributions import calculate_mwd
from .utils import calculate_number_average_dp

__all__ = ['MolecularWeightDistribution']


@dataclass(frozen=True)
class MolecularWeightDistribution:
    """
    Container for molecular weight distribution data.

    This immutable class represents a molecular weight distribution, either
    from experimental measurements or theoretical calculations. It provides
    properties for common metrics and methods for manipulation.

    Parameters
    ----------
    molecular_weights : ndarray
        Molecular weights at which intensities are measured/calculated.
        Must be in consistent units (e.g., g/mol, Da).
    intensities : ndarray
        Distribution intensities (weight fractions) at each molecular weight.
        Should be same length as molecular_weights.
    monomer_mw : float
        Molecular weight of one monomer unit. Same units as molecular_weights.
    is_normalized : bool, optional
        Whether the intensities are already normalized to unit area.
        Default is False.

    Attributes
    ----------
    molecular_weights : ndarray
        The molecular weight values.
    intensities : ndarray
        The intensity values.
    monomer_mw : float
        Monomer molecular weight.
    is_normalized : bool
        Normalization status.

    Notes
    -----
    IMPORTANT: All molecular weight values must use consistent units. If
    monomer_mw is in g/mol, then molecular_weights must also be in g/mol.

    This class is immutable - operations that modify the distribution
    return new instances rather than modifying in place.

    Examples
    --------
    Create from experimental data:

    >>> mwd = MolecularWeightDistribution.from_data(
    ...     molecular_weights=mw_array,
    ...     intensities=intensity_array,
    ...     monomer_mw=104.15
    ... )

    Create from kinetic parameters:

    >>> mwd = MolecularWeightDistribution.from_kinetics(
    ...     molecular_weights=np.logspace(3, 6, 500),
    ...     monomer_mw=104.15,
    ...     nu=50.0,
    ...     alpha=0.01,
    ...     init_mon=1.0,
    ...     init=0.02,
    ...     order=1.5,
    ...     sigma=0.05
    ... )

    Access properties:

    >>> print(f"Number average DP: {mwd.number_average_dp:.1f}")
    >>> print(f"Peak MW: {mwd.peak_molecular_weight:.0f}")
    """

    molecular_weights: np.ndarray
    intensities: np.ndarray
    monomer_mw: float
    is_normalized: bool = False

    def __post_init__(self):
        """Validate inputs after initialization."""
        if len(self.molecular_weights) != len(self.intensities):
            raise ValueError(
                f"Length mismatch: molecular_weights ({len(self.molecular_weights)}) "
                f"and intensities ({len(self.intensities)}) must be equal"
            )

        if self.monomer_mw <= 0:
            raise ValueError("monomer_mw must be positive")

        if np.any(self.molecular_weights <= 0):
            raise ValueError("All molecular_weights must be positive")

        if len(self.molecular_weights) < 2:
            raise ValueError("Must have at least 2 data points")

    @classmethod
    def from_data(
        cls,
        molecular_weights: np.ndarray,
        intensities: np.ndarray,
        monomer_mw: float,
        normalize: bool = True
    ) -> 'MolecularWeightDistribution':
        """
        Create MWD from experimental data.

        Parameters
        ----------
        molecular_weights : array_like
            Molecular weights from SEC/GPC measurement.
        intensities : array_like
            Measured intensities (e.g., RI detector response).
        monomer_mw : float
            Molecular weight of one monomer unit.
        normalize : bool, optional
            Whether to normalize intensities to unit area. Default is True.

        Returns
        -------
        MolecularWeightDistribution
            New MWD instance containing the data.

        Examples
        --------
        >>> import numpy as np
        >>> mws = np.array([1000, 2000, 3000, 4000])
        >>> ints = np.array([0.1, 0.5, 0.3, 0.1])
        >>> mwd = MolecularWeightDistribution.from_data(mws, ints, 100.0)
        """
        mws = np.asarray(molecular_weights, dtype=float)
        ints = np.asarray(intensities, dtype=float)

        if normalize:
            # Normalize to unit area under the curve
            area = np.trapezoid(ints, mws)
            if area > 0:
                ints = ints / area
            else:
                warnings.warn("Cannot normalize: distribution has zero or negative area")

        return cls(
            molecular_weights=mws,
            intensities=ints,
            monomer_mw=monomer_mw,
            is_normalized=normalize
        )

    @classmethod
    def from_kinetics(
        cls,
        molecular_weights: np.ndarray,
        monomer_mw: float,
        nu: float,
        alpha: float,
        init_mon: float,
        init: float,
        order: float,
        sigma: float,
        tau: float = 0,
        combination: bool = False,
        bn: float = 1.0
    ) -> 'MolecularWeightDistribution':
        """
        Generate theoretical MWD from kinetic parameters.

        Creates a molecular weight distribution based on the kinetic model
        for "living" polymerizations with termination.

        Parameters
        ----------
        molecular_weights : array_like
            Molecular weights at which to calculate the distribution.
        monomer_mw : float
            Molecular weight of one monomer unit.
        nu : float
            Kinetic chain length ([M]₀ - [M]) / [I]₀.
        alpha : float
            Ratio of termination to propagation rate constants (kt/kp).
        init_mon : float
            Initial monomer concentration.
        init : float
            Initial initiator concentration.
        order : float
            Order of the termination reaction.
        sigma : float
            SEC line broadening parameter (std dev in log MW space).
        tau : float
            SEC line broadening tailing parameter (used for exponentially
            modified Gaussians)
        combination : bool, optional
            Whether termination occurs by chain combination. Default False.
        bn : float, optional
            Inverse of propagation order in living chain. Default 1.0.

        Returns
        -------
        MolecularWeightDistribution
            New MWD instance with calculated distribution.

        Examples
        --------
        >>> mws = np.logspace(3, 6, 500)  # 1k to 1M Da
        >>> mwd = MolecularWeightDistribution.from_kinetics(
        ...     molecular_weights=mws,
        ...     monomer_mw=104.15,
        ...     nu=50.0,
        ...     alpha=0.01,
        ...     init_mon=1.0,
        ...     init=0.02,
        ...     order=1.5,
        ...     sigma=0.05
        ... )
        """
        mws = np.asarray(molecular_weights, dtype=float)

        intensities = calculate_mwd(
            molecular_weights=mws,
            monomer_mw=monomer_mw,
            nu=nu,
            alpha=alpha,
            init_mon=init_mon,
            init=init,
            order=order,
            sigma=sigma,
            tau=tau,
            combination=combination,
            bn=bn,
            live_only=False
        )

        return cls(
            molecular_weights=mws,
            intensities=intensities,
            monomer_mw=monomer_mw,
            is_normalized=True
        )

    @property
    def number_average_dp(self) -> float:
        """
        Number average degree of polymerization.

        Returns
        -------
        float
            DPn calculated from the distribution.
        """
        return calculate_number_average_dp(
            self.molecular_weights,
            self.intensities,
            self.monomer_mw
        )

    @property
    def number_average_mw(self) -> float:
        """
        Number average molecular weight (Mn).

        Returns
        -------
        float
            Mn in same units as molecular_weights.
        """
        return self.number_average_dp * self.monomer_mw

    @property
    def weight_average_mw(self) -> float:
        """
        Weight average molecular weight (Mw).

        Returns
        -------
        float
            Mw in same units as molecular_weights.

        Notes
        -----
        For a weight distribution w(M), Mw = ∫ M·w(M) dM / ∫ w(M) dM
        """
        if self.is_normalized:
            return float(np.trapezoid(
                self.molecular_weights * self.intensities,
                self.molecular_weights
            ))
        else:
            numerator = np.trapezoid(
                self.molecular_weights * self.intensities,
                self.molecular_weights
            )
            denominator = np.trapezoid(self.intensities, self.molecular_weights)
            return float(numerator / denominator if denominator > 0 else 0.0)

    @property
    def dispersity(self) -> float:
        """
        Dispersity (Đ = Mw/Mn).

        Returns
        -------
        float
            Đ (dimensionless). Lower values indicate narrower distributions.
            Đ = 1 for perfectly monodisperse polymers.
        """
        mn = self.number_average_mw
        mw = self.weight_average_mw
        return mw / mn

    @property
    def peak_molecular_weight(self) -> float:
        """
        Molecular weight at peak intensity (mode).

        Returns
        -------
        float
            MW at maximum intensity in same units as molecular_weights.
        """
        peak_idx = np.argmax(self.intensities)
        return self.molecular_weights[peak_idx]

    def normalize(self) -> 'MolecularWeightDistribution':
        """
        Return normalized copy of the distribution.

        Returns
        -------
        MolecularWeightDistribution
            New instance with intensities normalized to unit area.

        Notes
        -----
        If already normalized, returns a copy of self. This operation
        uses trapezoidal integration over molecular weight.
        """
        if self.is_normalized:
            return self

        area = np.trapezoid(self.intensities, self.molecular_weights)
        if area == 0:
            warnings.warn("Cannot normalize distribution with zero area")
            return self

        return MolecularWeightDistribution(
            molecular_weights=self.molecular_weights,
            intensities=self.intensities / area,
            monomer_mw=self.monomer_mw,
            is_normalized=True
        )

    def downsample(self, max_points: int = 500) -> 'MolecularWeightDistribution':
        """
        Return downsampled version with fewer points.

        Reduces the number of data points by uniform sampling, useful for
        speeding up fitting operations.

        Parameters
        ----------
        max_points : int, optional
            Maximum number of points to retain. Default is 200.

        Returns
        -------
        MolecularWeightDistribution
            New instance with downsampled data.

        Notes
        -----
        If the distribution has fewer points than max_points, returns self.
        Downsampling uses uniform striding (keeps every nth point).
        """
        n_points = len(self.molecular_weights)

        if n_points <= max_points:
            return self

        stride = int(n_points / max_points)

        return MolecularWeightDistribution(
            molecular_weights=self.molecular_weights[::stride],
            intensities=self.intensities[::stride],
            monomer_mw=self.monomer_mw,
            is_normalized=self.is_normalized
        )

    def normalize_on_log_scale(self) -> 'MolecularWeightDistribution':
        """
        Return distribution normalized on log(MW) scale.

        Returns
        -------
        MolecularWeightDistribution
            New instance normalized such that ∫ I d(log M) = 1.

        Notes
        -----
        Most fitting routines in this package use log-scale normalization
        internally.
        """
        area = np.trapezoid(self.intensities, np.log(self.molecular_weights))

        if area == 0:
            warnings.warn("Cannot normalize distribution with zero area")
            return self

        return MolecularWeightDistribution(
            molecular_weights=self.molecular_weights,
            intensities=self.intensities / area,
            monomer_mw=self.monomer_mw,
            is_normalized=True
        )
