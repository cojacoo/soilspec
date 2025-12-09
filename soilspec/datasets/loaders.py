"""
Data loaders for soil spectral libraries.

Provides convenient access to OSSL and other large spectral databases,
wrapping soilspecdata package with sklearn-compatible interfaces.

Scientific Background
---------------------
The Open Soil Spectral Library (OSSL) is a global compilation of soil
MIR and VISNIR spectra with associated soil properties, providing standardized
data for calibration model development.

References
----------
.. [1] Sanderman, J., et al. (2020). Mid-infrared spectroscopy for prediction
       of soil health indicators in the United States. Soil Sci. Soc. Am. J.
       84(1):251-261.
.. [2] Hengl, T., et al. (2021). African soil properties and nutrients mapped
       at 30 m spatial resolution using two-scale ensemble machine learning.
       Scientific Reports 11(1):6130.
.. [3] Viscarra Rossel, R.A., et al. (2016). A global spectral library to
       characterize the world's soil. Earth-Science Reviews 155:198-230.
.. [4] Albinet, F., et al. (2024). soilspecdata: Python package for accessing
       soil spectral libraries. https://github.com/franckalbinet/soilspecdata
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split

# Optional dependency
try:
    from soilspecdata.datasets.ossl import get_ossl
    SOILSPECDATA_AVAILABLE = True
except ImportError:
    SOILSPECDATA_AVAILABLE = False


class OSSLDataset:
    """
    Loader for Open Soil Spectral Library (OSSL) data.

    Wraps the soilspecdata package to provide convenient access to OSSL
    MIR and VISNIR spectra with associated soil properties.

    **Data Source:**

    OSSL is a global compilation of soil spectral data from multiple sources:

    * ~100,000+ MIR spectra (600-4000 cm⁻¹)
    * ~50,000+ VISNIR spectra (400-2500 nm)
    * 140+ soil properties (SOC, clay, pH, CEC, nutrients, etc.)
    * Geographic metadata (coordinates, depth, collection date)

    **Standard Usage:**

    1. Load dataset
    2. Select spectra type (MIR or VISNIR)
    3. Select target property
    4. Get aligned X, y for modeling

    **Preprocessing:**

    Data is returned as-is from OSSL (typically reflectance).
    Use soilspec.preprocessing for transformations:

    * SNV for scatter correction
    * Savitzky-Golay for derivatives
    * Trimming/resampling for standardization

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for caching downloaded data.
        Default: ~/.soilspecdata/
    auto_download : bool, default=True
        Automatically download OSSL data if not cached

    Attributes
    ----------
    ossl_ : OSSL object
        Underlying soilspecdata OSSL instance
    available_properties_ : list
        List of available soil properties

    Examples
    --------
    **Basic usage:**

    >>> from soilspec.datasets import OSSLDataset
    >>> from soilspec.preprocessing import SNVTransformer
    >>> from soilspec.models.traditional import CubistRegressor
    >>>
    >>> # Load OSSL data
    >>> ossl = OSSLDataset()
    >>>
    >>> # Get MIR spectra for SOC prediction
    >>> X, y, ids = ossl.load_mir(
    >>>     target='soc',
    >>>     wmin=600,
    >>>     wmax=4000,
    >>>     require_complete=True
    >>> )
    >>>
    >>> print(f"Loaded {X.shape[0]} samples with {X.shape[1]} wavelengths")
    >>>
    >>> # Split and train
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>>
    >>> # Train model
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    >>>     ('snv', SNVTransformer()),
    >>>     ('cubist', CubistRegressor(n_committees=20, neighbors=5))
    >>> ])
    >>> pipeline.fit(X_train, y_train)

    **Multi-property prediction:**

    >>> # Load multiple properties
    >>> X, y_dict, ids = ossl.load_mir(
    >>>     target=['soc', 'clay', 'ph'],
    >>>     wmin=600,
    >>>     wmax=4000
    >>> )
    >>>
    >>> # y_dict is DataFrame with columns: ['soc', 'clay', 'ph']

    **Geographic subsetting:**

    >>> # Get metadata for filtering
    >>> metadata = ossl.get_metadata()
    >>>
    >>> # Filter by location
    >>> usa_samples = metadata[metadata['country'] == 'USA'].index
    >>> X, y, ids = ossl.load_mir(target='soc', sample_ids=usa_samples)

    Notes
    -----
    Requires ``pip install soilspecdata`` for data access.

    See Albinet et al. [4]_ for soilspecdata package details and
    Sanderman et al. [1]_ for OSSL methodology.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        auto_download: bool = True
    ):
        if not SOILSPECDATA_AVAILABLE:
            raise ImportError(
                "soilspecdata package required for OSSL data access. "
                "Install with: pip install soilspecdata"
            )

        self.cache_dir = cache_dir
        self.auto_download = auto_download

        # Load OSSL dataset
        self.ossl_ = get_ossl()

        # Get available properties
        self.available_properties_ = self._get_available_properties()

    def load_mir(
        self,
        target: Union[str, List[str]],
        wmin: float = 600,
        wmax: float = 4000,
        require_complete: bool = True,
        sample_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        Load MIR spectra with target properties.

        Parameters
        ----------
        target : str or list of str
            Target property name(s). Examples:
            * 'soc' or 'c.tot_usda.a622_w.pct': Soil organic carbon (%)
            * 'clay.tot_usda.a334_w.pct': Clay content (%)
            * 'ph.h2o_usda.a268_index': pH
            * 'cec_usda.a723_cmolc.kg': CEC (cmol/kg)
        wmin : float, default=600
            Minimum wavenumber (cm⁻¹)
        wmax : float, default=4000
            Maximum wavenumber (cm⁻¹)
        require_complete : bool, default=True
            Only return samples with non-null target values
        sample_ids : array-like, optional
            Specific sample IDs to load

        Returns
        -------
        X : ndarray of shape (n_samples, n_wavenumbers)
            MIR spectra (reflectance or absorbance)
        y : ndarray or DataFrame
            Target values. ndarray if single target, DataFrame if multiple
        ids : ndarray
            Sample IDs

        Examples
        --------
        >>> ossl = OSSLDataset()
        >>> X, y, ids = ossl.load_mir(target='soc', wmin=600, wmax=4000)
        >>> print(f"Shape: {X.shape}, mean SOC: {y.mean():.2f}%")
        """
        # Get MIR spectra
        mir_data = self.ossl_.get_mir(wmin=wmin, wmax=wmax)

        # Get aligned data
        if isinstance(target, str):
            target = [target]

        X, y, ids = self.ossl_.get_aligned_data(
            spectra_data=mir_data,
            target_cols=target,
            require_complete=require_complete
        )

        # Filter by sample IDs if provided
        if sample_ids is not None:
            mask = np.isin(ids, sample_ids)
            X = X[mask]
            y = y[mask] if isinstance(y, np.ndarray) else y.iloc[mask]
            ids = ids[mask]

        # Convert single target DataFrame to array
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0].values

        return X, y, ids

    def load_visnir(
        self,
        target: Union[str, List[str]],
        wmin: float = 400,
        wmax: float = 2500,
        require_complete: bool = True,
        sample_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """
        Load VISNIR spectra with target properties.

        Parameters
        ----------
        target : str or list of str
            Target property name(s)
        wmin : float, default=400
            Minimum wavelength (nm)
        wmax : float, default=2500
            Maximum wavelength (nm)
        require_complete : bool, default=True
            Only return samples with non-null target values
        sample_ids : array-like, optional
            Specific sample IDs to load

        Returns
        -------
        X : ndarray of shape (n_samples, n_wavelengths)
            VISNIR spectra
        y : ndarray or DataFrame
            Target values
        ids : ndarray
            Sample IDs

        Examples
        --------
        >>> ossl = OSSLDataset()
        >>> X, y, ids = ossl.load_visnir(target='clay.tot_usda.a334_w.pct')
        """
        # Get VISNIR spectra
        visnir_data = self.ossl_.get_visnir(wmin=wmin, wmax=wmax)

        # Get aligned data
        if isinstance(target, str):
            target = [target]

        X, y, ids = self.ossl_.get_aligned_data(
            spectra_data=visnir_data,
            target_cols=target,
            require_complete=require_complete
        )

        # Filter by sample IDs if provided
        if sample_ids is not None:
            mask = np.isin(ids, sample_ids)
            X = X[mask]
            y = y[mask] if isinstance(y, np.ndarray) else y.iloc[mask]
            ids = ids[mask]

        # Convert single target DataFrame to array
        if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            y = y.iloc[:, 0].values

        return X, y, ids

    def get_metadata(self) -> pd.DataFrame:
        """
        Get sample metadata (coordinates, depth, dates, etc.).

        Returns
        -------
        metadata : DataFrame
            Sample metadata with index = sample IDs

        Examples
        --------
        >>> ossl = OSSLDataset()
        >>> meta = ossl.get_metadata()
        >>> print(meta.columns)
        >>> # Filter by country
        >>> usa_samples = meta[meta['country'] == 'USA']
        """
        return self.ossl_.get_properties(require_complete=False)

    def get_properties(
        self,
        properties: Optional[List[str]] = None,
        require_complete: bool = False
    ) -> pd.DataFrame:
        """
        Get soil properties without spectra.

        Parameters
        ----------
        properties : list of str, optional
            Property names to retrieve. If None, returns all.
        require_complete : bool, default=False
            Only return samples with non-null values in all properties

        Returns
        -------
        props : DataFrame
            Soil properties with index = sample IDs

        Examples
        --------
        >>> ossl = OSSLDataset()
        >>> props = ossl.get_properties(['soc', 'clay', 'ph'])
        >>> print(props.describe())
        """
        return self.ossl_.get_properties(
            properties=properties,
            require_complete=require_complete
        )

    def split_dataset(
        self,
        X: np.ndarray,
        y: Union[np.ndarray, pd.DataFrame],
        ids: np.ndarray,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify_bins: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split dataset into train/val/test sets.

        Parameters
        ----------
        X : ndarray
            Spectra
        y : ndarray or DataFrame
            Target values
        ids : ndarray
            Sample IDs
        test_size : float, default=0.2
            Fraction for test set
        val_size : float, optional
            Fraction for validation set. If None, no validation set.
        random_state : int, optional
            Random seed
        stratify_bins : int, optional
            Number of bins for stratified splitting by target value

        Returns
        -------
        splits : dict
            Dictionary with keys: 'X_train', 'X_test', 'y_train', 'y_test',
            'ids_train', 'ids_test' (and _val if val_size specified)

        Examples
        --------
        >>> ossl = OSSLDataset()
        >>> X, y, ids = ossl.load_mir(target='soc')
        >>>
        >>> # Train/test split
        >>> splits = ossl.split_dataset(X, y, ids, test_size=0.2, random_state=42)
        >>> X_train, y_train = splits['X_train'], splits['y_train']
        >>>
        >>> # Train/val/test split
        >>> splits = ossl.split_dataset(X, y, ids, test_size=0.2, val_size=0.1)
        """
        # Stratification
        stratify = None
        if stratify_bins is not None and isinstance(y, np.ndarray):
            # Create bins for stratification
            stratify = pd.cut(y, bins=stratify_bins, labels=False)

        # Initial split
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X, y, ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'ids_train': ids_train,
            'ids_test': ids_test
        }

        # Validation split if requested
        if val_size is not None:
            val_fraction = val_size / (1 - test_size)

            # Stratify for validation split
            stratify_val = None
            if stratify_bins is not None and isinstance(y_train, np.ndarray):
                stratify_val = pd.cut(y_train, bins=stratify_bins, labels=False)

            X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                X_train, y_train, ids_train,
                test_size=val_fraction,
                random_state=random_state,
                stratify=stratify_val
            )

            splits.update({
                'X_train': X_train,
                'X_val': X_val,
                'y_train': y_train,
                'y_val': y_val,
                'ids_train': ids_train,
                'ids_val': ids_val
            })

        return splits

    def _get_available_properties(self) -> List[str]:
        """Get list of available soil properties."""
        props = self.ossl_.get_properties(require_complete=False)
        return list(props.columns)

    def __repr__(self):
        n_props = len(self.available_properties_)
        return f"OSSLDataset(n_properties={n_props}, cache_dir='{self.cache_dir}')"


class LUCASDataset:
    """
    Loader for LUCAS Soil dataset (European topsoil).

    **Note:** Not yet implemented. LUCAS data requires separate download.

    The LUCAS (Land Use/Cover Area frame Survey) Soil dataset contains
    ~20,000 topsoil samples from across the EU with MIR/VISNIR spectra.

    References
    ----------
    .. [1] Orgiazzi, A., et al. (2018). LUCAS Soil, the largest expandable
           soil dataset for Europe: a review. European Journal of Soil Science
           69(1):140-153.
    """

    def __init__(self):
        raise NotImplementedError(
            "LUCAS dataset loader not yet implemented. "
            "For LUCAS data access, see: "
            "https://esdac.jrc.ec.europa.eu/content/lucas-2015-topsoil-data"
        )
