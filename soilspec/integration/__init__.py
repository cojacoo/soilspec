"""
Integration with external tools and model repositories.

**STATUS:** Module placeholder for future features (v0.2.0+)

Planned Features
----------------
* **Model Zoo**: Download pre-trained soil spectroscopy models
* **OSSL Model Loader**: Load pre-trained OSSL Cubist models
* **Export/Import**: Save/load models in standard formats (ONNX, PMML)
* **Integration**: Compatibility with other spectroscopy packages

Current Workaround
------------------
For now, use standard model persistence:

**Scikit-learn models (MBL, Cubist):**

>>> from soilspec.models.traditional import MBLRegressor
>>> import joblib
>>>
>>> # Train and save
>>> model = MBLRegressor(k=50)
>>> model.fit(X_train, y_train)
>>> joblib.dump(model, 'mbl_model.pkl')
>>>
>>> # Load and predict
>>> loaded_model = joblib.load('mbl_model.pkl')
>>> y_pred = loaded_model.predict(X_test)

**PyTorch models (1D CNNs):**

>>> from soilspec.models.deep_learning import SimpleCNN1D
>>> from soilspec.training import DeepLearningTrainer
>>> import torch
>>>
>>> # Train and save
>>> model = SimpleCNN1D(input_size=1801)
>>> trainer = DeepLearningTrainer(model=model)
>>> trainer.fit(X_train, y_train, X_val, y_val)
>>> torch.save(model.state_dict(), 'cnn_model.pth')
>>>
>>> # Load and predict
>>> loaded_model = SimpleCNN1D(input_size=1801)
>>> loaded_model.load_state_dict(torch.load('cnn_model.pth'))
>>> loaded_model.eval()

**Full pipeline with preprocessing:**

>>> from sklearn.pipeline import Pipeline
>>> from soilspec.preprocessing import SNVTransformer
>>> import joblib
>>>
>>> pipeline = Pipeline([
...     ('snv', SNVTransformer()),
...     ('model', MBLRegressor(k=50))
... ])
>>> pipeline.fit(X_train, y_train)
>>> joblib.dump(pipeline, 'full_pipeline.pkl')

Future Development
------------------
If you're interested in contributing model zoo functionality or have
pre-trained models to share, please open an issue at:
https://github.com/[username]/soilspec/issues

We're particularly interested in:
* Pre-trained OSSL models for common soil properties
* Multi-property prediction models
* Transfer learning benchmarks
* Model format standardization (ONNX export)

Notes
-----
**Why not implemented yet?**

1. No public repository of pre-trained soil spectroscopy models exists
2. OSSL models are typically trained from scratch using their data
3. Model sharing requires infrastructure (hosting, versioning, validation)
4. Focus on v0.1.0 was core modeling functionality

**Timeline:** Planned for v0.2.0 (Q2 2025)
"""

# No imports yet - module is placeholder
__all__ = []
