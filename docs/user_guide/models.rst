Models Guide
============

Choosing and using models for soil spectroscopy. See :doc:`../api/models` for detailed API documentation.

Model Selection
---------------

**Strong Models (Recommended)**:

* **Cubist**: OSSL standard, interpretable rules, excellent performance
* **MBL**: Best for transfer learning across instruments/regions

**Baseline Models** (for comparison only):

* PLS: Simple, fast, but weaker than Cubist/MBL
* Random Forest: Often worse than Cubist for spectroscopy

**Deep Learning** (optional):

* 1D CNN: Can match or exceed traditional models with large datasets (>10k samples)

When to Use Each Model
----------------------

**Use Cubist when:**

* You want OSSL-compatible methodology
* Interpretability is important (view rules)
* You have moderate to large training sets (>500 samples)

**Use MBL when:**

* Applying models across different instruments
* Transfer learning to new geographic regions
* You need uncertainty estimates
* Large calibration libraries available (>1000 samples)

**Use 1D CNN when:**

* Very large training sets available (>10k samples)
* Maximum performance is critical
* Computational resources available (GPU)

See :doc:`../api/models` for complete documentation with scientific references.
