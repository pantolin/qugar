# qugar

## About qugar
Quadratures for Unfitted GeometRies

# Installation
First clone the repository and go inside
```bash
git clone https://github.com/pantolin/qugar.git
cd qugar
```
## C++ installation
To build and install the C++ library, from the `qugar` directory run:
```bash
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -B build cpp/
ninja install -C build
```
In the [section below](#additional-cmake-options) additional CMake configuration options are provided.

## Python interface installation

QUGaR's Python interface requires the C++ library. So before installing the Python interface, make sure you have the C++ library installed.

> [!NOTE]  
> In the future, it will be possible to install the Python interface independently of the C++ library.
This will allow to use DOLFINx with custom quadratures, but without providing tools to compute these quadratures.
For generating quadrature points and weights, other libraries, as, e.g., 
[CutCells](https://github.com/sclaus2/CutCells), could be potentially used.

For building the Python interface, go to the `python/` directory under the root `qugar/` directory and run
```bash
python3 -m pip -v install -r ./python/build-requirements.txt
python3 -m pip -v install --no-build-isolation ./python -U
```
Check the section [Optional Python dependencies](#optional-python-dependencies) for optional features.


### Additional CMake options

 Some algorithms can benefit from the use of [LAPACKE](https://www.netlib.org/lapack/lapacke.html) for solving eigenvalue problems. For instance, when finding roots of polynomials.
If LAPACKE is not found during the configuration the message `LAPACKE not found. Algoim's eigenvalue solvers will be used.` is displayed.
However, if present in your computer, you can help CMake in finding LAPACKE by providing its path by appending `-DLAPACKE_DIR=PATH_TO_LAPACKE` to the previous command.

Other options that can be provided to CMake are `qugar_BUILD_DOC` (for building C++ documentation), `BUILD_TESTING` (for enabling testing), `qugar_WITH_DEMOS` (for building some basic demos), and `qugar_DEVELOPER_MODE` (for activating multiple compiler options, this option is automatically activated when compiling in debug mode).

In addition, the preferred installation path can be set through `CMAKE_INSTALL_PREFIX`.

If not C++ compiler is found, or you want to use a specific one, set the variable `CMAKE_CXX_COMPILER`.
For the case of the Python interface building, append
`--config-settings=cmake.define.CMAKE_CXX_COMPILER=<PATH>` to the pip command.

### Optional Python dependencies
QUGaR's Python interface is designed to interact nicely with [FEniCSx](https://fenicsproject.org). So, if you want to solve your PDEs using unfitted domains through FEniCSx, make sure to install a compatible version of [DOLFINx](https://github.com/FEniCS/dolfinx) (check the file [pyproject.toml](./python/pyproject.toml) for required version compatibility).

However, it is also possible to use QUGaR's Python interface without DOLFINx. Check [demo_no_fenicsx.py](./python/demo/demo_no_fenicsx.py)

QUGaR's Python interface provides some extra visualization features through the [VTK](https://vtk.org) library. To enable such features, install `vtk` (for instance, from [PyPI](https://pypi.org/project/vtk/) or [conda-forge](https://anaconda.org/conda-forge/vtk)).

> [!WARNING]  
> In the mid-future, [VTK](https://vtk.org) dependency will be replaced by [pyvista](https://pyvista.org).