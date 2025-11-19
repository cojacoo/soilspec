Contributing
============

We welcome contributions to soilspec! This guide explains how to contribute.

Development Setup
-----------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/soilspec.git
   cd soilspec

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install in development mode
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=soilspec --cov-report=html

   # Run specific test file
   pytest tests/test_models/test_mbl.py

Code Style
----------

We use:

* **black** for code formatting
* **ruff** for linting
* **mypy** for type checking

Run before committing:

.. code-block:: bash

   # Format code
   black soilspec tests

   # Lint
   ruff check soilspec tests

   # Type check
   mypy soilspec

Or use pre-commit:

.. code-block:: bash

   pre-commit run --all-files

Contributing Guidelines
-----------------------

**Design Principles**:

1. **Use proven methods**: Prioritize literature-validated approaches
2. **Wrap, don't reimplement**: Use scipy/sklearn/pywavelets when possible
3. **Sklearn compatibility**: All transformers should follow sklearn API
4. **Scientific rigor**: Include references for all methods
5. **Clear documentation**: Every function needs docstring + scientific background

**Pull Request Process**:

1. Create feature branch: ``git checkout -b feature/my-feature``
2. Write tests for new code (aim for >80% coverage)
3. Add docstrings with scientific background and references
4. Update documentation in ``docs/``
5. Run tests and linting
6. Submit PR with clear description

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   pip install -r requirements-docs.txt
   make html
   open _build/html/index.html

**Writing Documentation**:

* API docs: Use numpy-style docstrings
* User guides: Use reStructuredText (.rst)
* Include scientific background and references for all methods
* Add usage examples

Adding Scientific Methods
--------------------------

When adding a new method:

1. **Literature search**: Find peer-reviewed references
2. **Implementation**: Wrap existing libraries when possible
3. **Tests**: Unit tests + validation against paper results
4. **Documentation**:

   * Docstring with scientific background
   * Mathematical formulation
   * When to use / not use
   * Example usage
   * References in BibTeX format

**Example template**:

.. code-block:: python

   class NewMethod(BaseEstimator, TransformerMixin):
       \"\"\"
       One-line description.

       Scientific Background
       ---------------------
       Detailed explanation of the method, its physical basis,
       and why it's appropriate for soil spectroscopy.

       Mathematical Formulation
       ------------------------
       .. math::

          y = f(x)

       Parameters
       ----------
       param1 : type
           Description

       References
       ----------
       .. [1] Author et al. (2020). Title. Journal.

       Examples
       --------
       >>> from soilspec.module import NewMethod
       >>> method = NewMethod()
       >>> result = method.fit_transform(X)
       \"\"\"

Reporting Issues
----------------

When reporting bugs:

1. Check `existing issues <https://github.com/yourusername/soilspec/issues>`_
2. Include:

   * Python version
   * soilspec version
   * Operating system
   * Minimal reproducible example
   * Full error traceback

Code of Conduct
---------------

Be respectful and constructive. We're building scientific software together!
