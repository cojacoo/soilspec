Installation
============

Requirements
------------

**Python Version**: 3.9 or higher

**Core Dependencies**:

* numpy >= 1.24.0
* scipy >= 1.10.0
* pandas >= 2.0.0
* scikit-learn >= 1.3.0
* pywavelets >= 1.4.0
* cubist >= 0.3.0 (pjaselin/Cubist package)
* matplotlib >= 3.7.0
* seaborn >= 0.12.0
* brukeropusreader >= 1.3.0 (for reading Bruker OPUS files)

Installation Methods
--------------------

PyPI Installation
~~~~~~~~~~~~~~~~~

**Basic installation** (recommended for most users):

.. code-block:: bash

   pip install soilspec

**With deep learning support** (requires PyTorch):

.. code-block:: bash

   pip install soilspec[deep-learning]

**With MLOps tools** (MLflow, Weights & Biases):

.. code-block:: bash

   pip install soilspec[mlops]

**Complete installation** (all optional dependencies):

.. code-block:: bash

   pip install soilspec[all]

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

**For contributors** or to access latest features:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/soilspec.git
   cd soilspec

   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate

   # Install in editable mode with dev dependencies
   pip install -e ".[dev]"

This installs:

* All core dependencies
* Development tools (pytest, black, ruff, mypy)
* Jupyter for notebooks

Conda Installation
~~~~~~~~~~~~~~~~~~

**Coming soon** - conda-forge package in development.

For now, use pip within conda environment:

.. code-block:: bash

   conda create -n soilspec python=3.11
   conda activate soilspec
   pip install soilspec

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import soilspec
   print(soilspec.__version__)

   # Test core modules
   from soilspec.preprocessing import SNVTransformer
   from soilspec.features import PeakIntegrator
   from soilspec.models.traditional import MBLRegressor, CubistRegressor
   from soilspec.knowledge import SpectralBandDatabase

   # Check Cubist availability
   from soilspec.models.traditional.cubist_wrapper import CUBIST_AVAILABLE
   print(f"Cubist available: {CUBIST_AVAILABLE}")

   # Load spectral band database
   bands = SpectralBandDatabase()
   print(f"Loaded {len(bands.bands)} spectral bands")

If you see version number and no errors, installation was successful!

Optional Dependencies
---------------------

Deep Learning
~~~~~~~~~~~~~

For 1D CNN and physics-guided neural networks:

.. code-block:: bash

   pip install torch>=2.0.0 lightning>=2.0.0

Or use the shortcut:

.. code-block:: bash

   pip install soilspec[deep-learning]

MLOps and Experiment Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For experiment tracking and model deployment:

.. code-block:: bash

   pip install mlflow>=2.8.0 wandb>=0.15.0

Or use:

.. code-block:: bash

   pip install soilspec[mlops]

GPU Support
~~~~~~~~~~~

For GPU-accelerated deep learning:

.. code-block:: bash

   # CUDA 11.8 (check your CUDA version)
   pip install torch --index-url https://download.pytorch.org/whl/cu118

See `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ for details.

Troubleshooting
---------------

Cubist Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If Cubist installation fails:

.. code-block:: bash

   # Ensure C compiler is available
   # Ubuntu/Debian:
   sudo apt-get install build-essential

   # macOS:
   xcode-select --install

   # Windows: Install Visual Studio Build Tools

   # Then retry:
   pip install cubist

NumPy/SciPy Issues on Apple Silicon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For M1/M2 Macs, use Homebrew Python or Miniforge:

.. code-block:: bash

   # Using Miniforge (recommended)
   brew install miniforge
   conda init zsh  # or bash
   conda create -n soilspec python=3.11
   conda activate soilspec
   pip install soilspec

Import Errors
~~~~~~~~~~~~~

If you see ``ModuleNotFoundError``:

1. Verify you're in correct environment:

   .. code-block:: bash

      which python
      python -c "import sys; print(sys.executable)"

2. Reinstall soilspec:

   .. code-block:: bash

      pip uninstall soilspec
      pip install soilspec

3. Check for package conflicts:

   .. code-block:: bash

      pip list | grep -E "numpy|scipy|sklearn"

Next Steps
----------

After installation:

* :doc:`quickstart` - Get started with basic usage
* :doc:`preprocessing` - Learn spectral preprocessing methods
* :doc:`models` - Train MBL and Cubist models
* :doc:`examples` - See complete examples

Getting Help
------------

If you encounter issues:

1. Check `GitHub Issues <https://github.com/yourusername/soilspec/issues>`_
2. Search `Documentation <https://soilspec.readthedocs.io>`_
3. Open a new issue with:

   * Python version (``python --version``)
   * OS and architecture
   * Full error message
   * Minimal reproducible example
