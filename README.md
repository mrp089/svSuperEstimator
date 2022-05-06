# SimVascular's SuperEstimator

A framework for multi-fidelity estimation of boundary condition
parameters for cardiovascular fluid dynamics simulations.

## Installation

After cloning this repository, svSuperEstimator and all its dependencies can be
installed easily via pip. Just navigate to the root folder of the repository and enter:

```bash
pip install .
```

### For Contributers

If you are contributing to svSuperEstimator, it is highly recommended to use a virtual
environment like [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
After installing Miniconda you can create a new environment and enter it using:

```bash
conda create -n superestimator python=3.9
conda activate superestimator
```

After that you can install the svSuperEstimator **with development related dependencies**
using:

```bash
pip install -e .[dev]
```

*If you are using the `zsh` shell, enter: `pip install -e ".[dev]"`*
