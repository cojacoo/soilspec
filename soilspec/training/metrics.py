"""
Evaluation metrics for soil spectroscopy models.

Scientific Background
---------------------
Standard metrics for evaluating spectroscopy model performance,
following conventions from chemometrics and soil science literature.

References
----------
.. [1] Williams, P.C. (1987). Variables affecting near-infrared reflectance
       spectroscopy analysis. In: Near-infrared technology in the agricultural
       and food industries. AACC, St. Paul, MN.
.. [2] Chang, C.W., et al. (2001). Near-infrared reflectance spectroscopy-
       principal components regression analyses of soil properties.
       Soil Sci. Soc. Am. J. 65:480-490.
.. [3] Bellon-Maurel, V., et al. (2010). Critical review of chemometric
       indicators commonly used for assessing the quality of the prediction
       of soil attributes by NIR spectroscopy. TrAC Trends in Analytical
       Chemistry 29(9):1073-1081.
"""

import numpy as np
from typing import Optional, Union
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class RMSE:
    """
    Root Mean Squared Error - primary metric for spectroscopy models.

    Scientific Background
    ---------------------
    RMSE measures average prediction error in the same units as the target:

    .. math::

        \\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}

    **Interpretation:**

    * Lower RMSE = better predictions
    * Units: same as target variable (e.g., % for SOC)
    * Penalizes large errors more than MAE
    * Standard metric reported in soil spectroscopy literature

    **Typical RMSE values for MIR soil spectroscopy:**

    * SOC: 0.3-0.5% (excellent), 0.5-0.8% (good), >0.8% (poor)
    * Clay: 5-8% (excellent), 8-12% (good), >12% (poor)
    * pH: 0.3-0.5 (excellent), 0.5-0.8 (good), >0.8 (poor)

    Parameters
    ----------
    squared : bool, default=False
        If True, return MSE. If False, return RMSE.

    Examples
    --------
    >>> from soilspec.training.metrics import RMSE
    >>> import numpy as np
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1])
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8])
    >>>
    >>> rmse = RMSE()
    >>> error = rmse(y_true, y_pred)
    >>> print(f"RMSE: {error:.3f}")

    References
    ----------
    See [3]_ for review of chemometric metrics.
    """

    def __init__(self, squared: bool = False):
        self.squared = squared
        self.__name__ = 'mse' if squared else 'rmse'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute RMSE.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        rmse : float
            Root mean squared error (or MSE if squared=True)
        """
        mse = mean_squared_error(
            y_true, y_pred,
            sample_weight=sample_weight,
            squared=True
        )
        return mse if self.squared else np.sqrt(mse)

    def __repr__(self):
        return f"RMSE(squared={self.squared})"


class R2Score:
    """
    Coefficient of Determination (R²) - measures explained variance.

    Scientific Background
    ---------------------
    R² indicates the proportion of variance in the target explained by the model:

    .. math::

        R^2 = 1 - \\frac{\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n}(y_i - \\bar{y})^2}

    **Interpretation:**

    * R² = 1.0: Perfect predictions
    * R² = 0.0: Model performs as well as predicting the mean
    * R² < 0.0: Model worse than predicting the mean
    * Unitless, range typically [0, 1] for good models

    **Typical R² values for MIR soil spectroscopy:**

    * R² > 0.90: Excellent predictions (quantitative use)
    * R² = 0.82-0.90: Good predictions (quantitative screening)
    * R² = 0.66-0.82: Moderate (qualitative screening)
    * R² < 0.66: Poor (not recommended for prediction)

    Based on Williams (1987) classification [1]_.

    **Important:**

    * Always report R² with RMSE - R² alone can be misleading
    * Validation R² < Calibration R² indicates overfitting
    * Can be inflated by outliers or wide data range

    Parameters
    ----------
    adjusted : bool, default=False
        If True, compute adjusted R² accounting for number of features

    Examples
    --------
    >>> from soilspec.training.metrics import R2Score
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1])
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8])
    >>>
    >>> r2 = R2Score()
    >>> score = r2(y_true, y_pred)
    >>> print(f"R²: {score:.3f}")

    References
    ----------
    See [1]_ for interpretation guidelines and [3]_ for critical review.
    """

    def __init__(self, adjusted: bool = False):
        self.adjusted = adjusted
        self.__name__ = 'r2_adjusted' if adjusted else 'r2'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: Optional[int] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute R² score.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        n_features : int, optional
            Number of features (required if adjusted=True)
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        r2 : float
            R² score (or adjusted R² if adjusted=True)
        """
        r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)

        if self.adjusted:
            if n_features is None:
                raise ValueError("n_features required for adjusted R²")
            n = len(y_true)
            r2_adj = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
            return r2_adj

        return r2

    def __repr__(self):
        return f"R2Score(adjusted={self.adjusted})"


class RPD:
    """
    Ratio of Performance to Deviation - normalized performance metric.

    Scientific Background
    ---------------------
    RPD normalizes prediction error by the standard deviation of the target:

    .. math::

        \\text{RPD} = \\frac{\\text{SD}}{\\text{RMSE}} = \\frac{\\sigma_y}{\\text{RMSE}}

    where :math:`\\sigma_y` is the standard deviation of the true values.

    **Interpretation (Williams 1987 [1]_):**

    * RPD > 2.5: Excellent predictions (quantitative analysis)
    * RPD = 2.0-2.5: Good predictions (quantitative screening)
    * RPD = 1.4-2.0: Fair (qualitative screening)
    * RPD < 1.4: Poor (not recommended)

    **Advantages:**

    * Unitless - comparable across different properties
    * Accounts for data variability
    * Standard in soil spectroscopy literature

    **Disadvantages:**

    * Inflated by wide data range
    * Can be high even with poor absolute accuracy
    * Should be reported alongside RMSE

    **Modified interpretation for soil spectroscopy (Chang et al. 2001 [2]_):**

    * RPD > 2.0: Excellent
    * RPD = 1.4-2.0: Good
    * RPD < 1.4: Unreliable

    Examples
    --------
    >>> from soilspec.training.metrics import RPD
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1, 2.9, 3.7])
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8, 2.8, 3.9])
    >>>
    >>> rpd = RPD()
    >>> score = rpd(y_true, y_pred)
    >>> print(f"RPD: {score:.2f}")
    >>>
    >>> if score > 2.5:
    >>>     print("Excellent predictions")
    >>> elif score > 2.0:
    >>>     print("Good predictions")
    >>> elif score > 1.4:
    >>>     print("Fair predictions")
    >>> else:
    >>>     print("Poor predictions")

    References
    ----------
    See [1]_ for original definition and [2]_ for soil spectroscopy application.
    """

    def __init__(self):
        self.__name__ = 'rpd'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute RPD.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (note: affects RMSE but not SD calculation)

        Returns
        -------
        rpd : float
            Ratio of performance to deviation
        """
        # Standard deviation of true values
        if sample_weight is not None:
            # Weighted standard deviation
            mean = np.average(y_true, weights=sample_weight)
            variance = np.average((y_true - mean)**2, weights=sample_weight)
            sd = np.sqrt(variance)
        else:
            sd = np.std(y_true, ddof=1)  # Sample standard deviation

        # RMSE
        rmse = np.sqrt(mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight
        ))

        # RPD = SD / RMSE
        return sd / rmse if rmse > 0 else np.inf

    def __repr__(self):
        return "RPD()"


class RPIQ:
    """
    Ratio of Performance to Inter-Quartile range - robust performance metric.

    Scientific Background
    ---------------------
    RPIQ uses the inter-quartile range (IQR) instead of standard deviation,
    making it more robust to outliers:

    .. math::

        \\text{RPIQ} = \\frac{\\text{IQR}}{\\text{RMSE}} = \\frac{Q_3 - Q_1}{\\text{RMSE}}

    where :math:`Q_3` and :math:`Q_1` are the 75th and 25th percentiles.

    **Interpretation (Bellon-Maurel et al. 2010 [3]_):**

    * RPIQ > 2.5: Excellent
    * RPIQ = 2.0-2.5: Good
    * RPIQ = 1.5-2.0: Moderate
    * RPIQ < 1.5: Poor

    **Advantages over RPD:**

    * More robust to outliers (uses IQR not SD)
    * Better for skewed distributions
    * Recommended for heterogeneous datasets

    **When to use RPIQ vs RPD:**

    * Use RPIQ when:
      - Data contains outliers
      - Distribution is non-normal
      - Heterogeneous soil types
    * Use RPD when:
      - Normally distributed data
      - Comparing with literature (RPD more common)

    Examples
    --------
    >>> from soilspec.training.metrics import RPIQ
    >>>
    >>> # Data with outlier
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1, 2.9, 3.7, 8.5])  # 8.5 is outlier
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8, 2.8, 3.9, 7.2])
    >>>
    >>> rpiq = RPIQ()
    >>> score = rpiq(y_true, y_pred)
    >>> print(f"RPIQ: {score:.2f}")

    References
    ----------
    See [3]_ for critical review recommending RPIQ for soil spectroscopy.
    """

    def __init__(self):
        self.__name__ = 'rpiq'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute RPIQ.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (affects RMSE but not IQR)

        Returns
        -------
        rpiq : float
            Ratio of performance to inter-quartile range
        """
        # Inter-quartile range
        q75, q25 = np.percentile(y_true, [75, 25])
        iqr = q75 - q25

        # RMSE
        rmse = np.sqrt(mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight
        ))

        # RPIQ = IQR / RMSE
        return iqr / rmse if rmse > 0 else np.inf

    def __repr__(self):
        return "RPIQ()"


class MAE:
    """
    Mean Absolute Error - robust alternative to RMSE.

    Scientific Background
    ---------------------
    MAE measures average absolute prediction error:

    .. math::

        \\text{MAE} = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|

    **Comparison with RMSE:**

    * MAE less sensitive to outliers (absolute vs squared errors)
    * RMSE penalizes large errors more heavily
    * Both have same units as target variable
    * RMSE ≥ MAE always (equality only if all errors identical)

    **When to use:**

    * Data contains outliers → prefer MAE
    * Large errors particularly problematic → prefer RMSE
    * Report both for complete picture

    Examples
    --------
    >>> from soilspec.training.metrics import MAE
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1])
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8])
    >>>
    >>> mae = MAE()
    >>> error = mae(y_true, y_pred)
    >>> print(f"MAE: {error:.3f}")
    """

    def __init__(self):
        self.__name__ = 'mae'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute MAE.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        mae : float
            Mean absolute error
        """
        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

    def __repr__(self):
        return "MAE()"


class Bias:
    """
    Mean prediction bias - systematic over/under-prediction.

    Scientific Background
    ---------------------
    Bias measures systematic deviation from true values:

    .. math::

        \\text{Bias} = \\frac{1}{n}\\sum_{i=1}^{n}(\\hat{y}_i - y_i) = \\bar{\\hat{y}} - \\bar{y}

    **Interpretation:**

    * Bias = 0: No systematic error (unbiased)
    * Bias > 0: Systematic over-prediction
    * Bias < 0: Systematic under-prediction

    **Important:**

    * Model can have low RMSE but high bias
    * Bias can often be corrected with post-calibration
    * Should always check bias in addition to RMSE/R²

    Examples
    --------
    >>> from soilspec.training.metrics import Bias
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1])
    >>> y_pred = np.array([2.8, 3.5, 2.1, 4.4])  # Systematically high
    >>>
    >>> bias = Bias()
    >>> b = bias(y_true, y_pred)
    >>> print(f"Bias: {b:.3f}")
    >>> if b > 0:
    >>>     print("Model over-predicts on average")
    """

    def __init__(self):
        self.__name__ = 'bias'

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute bias.

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            Ground truth values
        y_pred : array-like of shape (n_samples,)
            Predicted values
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights

        Returns
        -------
        bias : float
            Mean bias (positive = over-prediction)
        """
        errors = y_pred - y_true
        if sample_weight is not None:
            return np.average(errors, weights=sample_weight)
        return np.mean(errors)

    def __repr__(self):
        return "Bias()"


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    metrics: Optional[list] = None
) -> dict:
    """
    Evaluate model with multiple metrics.

    Computes comprehensive metrics following soil spectroscopy best practices.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth values
    y_pred : array-like of shape (n_samples,)
        Predicted values
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights
    metrics : list of str, optional
        Metrics to compute. Default: ['rmse', 'r2', 'rpd', 'rpiq', 'bias']

    Returns
    -------
    results : dict
        Dictionary of metric names and values

    Examples
    --------
    >>> from soilspec.training.metrics import evaluate_model
    >>> import numpy as np
    >>>
    >>> y_true = np.array([2.5, 3.2, 1.8, 4.1, 2.9, 3.7])
    >>> y_pred = np.array([2.3, 3.5, 1.9, 3.8, 2.8, 3.9])
    >>>
    >>> results = evaluate_model(y_true, y_pred)
    >>> for metric, value in results.items():
    >>>     print(f"{metric}: {value:.3f}")
    >>>
    >>> # Interpret results
    >>> if results['r2'] > 0.90 and results['rpd'] > 2.5:
    >>>     print("Excellent predictions - suitable for quantitative use")
    >>> elif results['r2'] > 0.82 and results['rpd'] > 2.0:
    >>>     print("Good predictions - suitable for screening")
    """
    if metrics is None:
        metrics = ['rmse', 'r2', 'rpd', 'rpiq', 'bias', 'mae']

    metric_functions = {
        'rmse': RMSE(),
        'r2': R2Score(),
        'rpd': RPD(),
        'rpiq': RPIQ(),
        'bias': Bias(),
        'mae': MAE()
    }

    results = {}
    for metric_name in metrics:
        if metric_name in metric_functions:
            metric_fn = metric_functions[metric_name]
            results[metric_name] = metric_fn(y_true, y_pred, sample_weight=sample_weight)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    # Add sample size
    results['n_samples'] = len(y_true)

    return results
