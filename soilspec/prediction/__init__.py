"""
Prediction and uncertainty quantification module.

**STATUS:** Module placeholder for future features (v0.2.0+)

Planned Features
----------------
* **SpectralPredictor**: Unified prediction interface for all model types
* **Ensemble Uncertainty**: Variance-based uncertainty from model ensembles
* **Conformal Prediction**: Distribution-free prediction intervals
* **Dropout Uncertainty**: Bayesian approximation via MC dropout (deep learning)

Current Workaround
------------------
For now, use model-specific prediction methods:

**Basic prediction (all models):**

>>> from soilspec.models.traditional import MBLRegressor, CubistRegressor
>>> from soilspec.models.deep_learning import SimpleCNN1D
>>>
>>> # Traditional models
>>> mbl = MBLRegressor(k=50)
>>> mbl.fit(X_train, y_train)
>>> y_pred = mbl.predict(X_test)
>>>
>>> # Deep learning models
>>> from soilspec.training import DeepLearningTrainer
>>> cnn = SimpleCNN1D(input_size=1801)
>>> trainer = DeepLearningTrainer(model=cnn)
>>> trainer.fit(X_train, y_train, X_val, y_val)
>>> y_pred = trainer.predict(X_test)

**Uncertainty quantification (MBL only):**

>>> # MBL provides local variance estimates
>>> mbl = MBLRegressor(k=50, return_variance=True)
>>> mbl.fit(X_train, y_train)
>>> y_pred, y_var = mbl.predict_with_uncertainty(X_test)
>>>
>>> # Compute 95% prediction intervals
>>> y_std = np.sqrt(y_var)
>>> lower = y_pred - 1.96 * y_std
>>> upper = y_pred + 1.96 * y_std

**Ensemble uncertainty (manual):**

>>> from sklearn.ensemble import BaggingRegressor
>>> from soilspec.preprocessing import SNVTransformer
>>> from sklearn.pipeline import Pipeline
>>>
>>> # Create ensemble of MBL models
>>> base_model = Pipeline([
...     ('snv', SNVTransformer()),
...     ('mbl', MBLRegressor(k=50))
... ])
>>> ensemble = BaggingRegressor(
...     base_estimator=base_model,
...     n_estimators=10,
...     random_state=42
... )
>>> ensemble.fit(X_train, y_train)
>>>
>>> # Get predictions from all ensemble members
>>> predictions = np.array([
...     estimator.predict(X_test)
...     for estimator in ensemble.estimators_
... ])
>>>
>>> # Compute mean and variance
>>> y_pred = predictions.mean(axis=0)
>>> y_std = predictions.std(axis=0)

**Conformal prediction (manual):**

>>> from sklearn.model_selection import train_test_split
>>>
>>> # Split data: train / calibration / test
>>> X_train_val, X_test, y_train_val, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=42
... )
>>> X_train, X_cal, y_train, y_cal = train_test_split(
...     X_train_val, y_train_val, test_size=0.2, random_state=42
... )
>>>
>>> # Train model
>>> model = MBLRegressor(k=50)
>>> model.fit(X_train, y_train)
>>>
>>> # Compute calibration residuals
>>> y_cal_pred = model.predict(X_cal)
>>> residuals = np.abs(y_cal - y_cal_pred)
>>>
>>> # Get prediction interval at 90% coverage
>>> alpha = 0.10
>>> quantile = np.quantile(residuals, 1 - alpha)
>>>
>>> # Make predictions with intervals
>>> y_test_pred = model.predict(X_test)
>>> lower = y_test_pred - quantile
>>> upper = y_test_pred + quantile

Future Development
------------------
We plan to implement these methods in v0.2.0:

1. **Conformal prediction** (Vovk et al. 2005)
   - Distribution-free prediction intervals
   - Guaranteed coverage under exchangeability
   - Efficient implementation for large datasets

2. **Quantile regression forests**
   - Full predictive distribution
   - No distributional assumptions
   - Compatible with ensemble models

3. **Bayesian deep learning**
   - MC dropout (Gal & Ghahramani 2016)
   - Variational inference
   - Predictive uncertainty for CNNs

4. **Ensemble methods**
   - Stacking different model types (MBL + Cubist + CNN)
   - Weighted averaging based on calibration
   - Out-of-bag predictions

References
----------
.. [1] Vovk, V., Gammerman, A., Shafer, G. (2005). Algorithmic Learning
       in a Random World. Springer.
.. [2] Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian approximation.
       ICML 2016.
.. [3] Ramirez-Lopez et al. (2013). MBL provides local uncertainty estimates.
       Geoderma 195:268-279.

Notes
-----
**Why not implemented yet?**

1. Core modeling functionality was priority for v0.1.0
2. Uncertainty quantification requires careful validation
3. Different methods suit different use cases (no one-size-fits-all)
4. MBL already provides local variance (covers many use cases)

**Timeline:** Planned for v0.2.0 (Q2-Q3 2025)
"""

# No imports yet - module is placeholder
__all__ = []
