# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
# ---

# # $L^2$ projection
#
# This demo is implemented in {download}`demo_L2_projection.py`. It
# illustrates:
#
# - How to perform a $L^2$ projection using QUGaR and FEniCSx.
# - How to export the generated result to VTK files.
#
# ## Problem definition
#
# Let us consider the domain $\Omega \subset \mathbb{R}^n$,
# immersed in a mesh $\mathcal{T}(\Omega)$ (see
# [Divergence theorem demo](demo_div_thm.md) first for further details
# about the immersed setting), and the function
# $f:\Omega\to\mathbb{R}$ defined as $f(x,y,z) = \sin(x)\cos(y)\sin(z)$.
# Let us also consider a finite element space $V$ defined over the mesh
# $\mathcal{T}(\Omega)$, then, computing the $L^2$ projection
# of $f$ onto $V$ is equivalent to solving the following variational
# problem:
# $
# \begin{align}
#   \text{Find } u\in V \text{ such that } \forall v\in V:
#   \int_{\Omega} u v \text{d} x = \int_{\Omega} f v \text{d} x
# \end{align}
# $
#
# ## Implementation
#
# ### Modules import
# First we add the needed modules and functions to be used:

# +
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import numpy as np
import ufl

import qugar
import qugar.impl
from qugar.dolfinx import form_custom
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

# -

# We define the floating point type to be used in the computations.
# So far, QUGaR supports 32 and 64 bits real floating point types.
# In the near future support for 64 and 128 bits complex types will be
# also available.

# +
dtype = np.float64
# -

# ### Geometry and mesh

# We define the geometry to be considered: in this case a
# {py:class}`Schoen gyroid<qugar.impl.create_Schoen>` in 3D defined
# through an implicit function.
# The domain can be easily changed by just choosing a different implicit
# function among the available ones (see
# [Implicit functions demo](demo_impl_funcs.md) for some examples).

# +
impl_func = qugar.impl.create_Schoen([1.0, 1.0, 1.0])
# -

# We create an {py:class}`unfitted Cartesian mesh<qugar.mesh.UnfittedCartMesh>`
# (corresponding to $\mathcal{T}$) in which we embed the domain $\Omega$.
# This is a Cartesian mesh corresponding to the domain $[0,1]^3$ and
# with `n_cells` cells per direction.

# +
n_cells = 4

unf_mesh = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, impl_func, n_cells, exclude_empty_cells=True, dtype=dtype
)
# -

# The option `exclude_empty_cells` (set to `True`) prevents the
# generation of empty cells in the mesh (those not intersecting $\Omega$).
# This is useful to eliminate inactive degrees of freedom in the
# problem.

# ### Spaces and functions

# We create the function $\mathbf{f}$.

# +
x = ufl.SpatialCoordinate(unf_mesh)
f = ufl.sin(x[0]) * ufl.cos(x[1]) * ufl.sin(x[2])
# -

# and the finite element space $V$ over the unfitted mesh
# (corresponding to the mesh $\mathcal{T}$), and the test and trial
# functions $u$ and $v$.
# The finite element space is defined as a Lagrange space of the given
# `degree` (2 in this case).

# +
degree = 2
V = dolfinx.fem.functionspace(unf_mesh, ("Lagrange", degree))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
# -

# ### Linear forms

# We define the variational problem to be solved, which is
# equivalent to the $L^2$ projection of $f$ onto $V$, and generate
# the corresponding bilinear and linear forms `a` and `L` using
# QUGaR's custom form functions.

# +
F = u * v * ufl.dx - f * v * ufl.dx

a = form_custom(ufl.lhs(F), unf_mesh, dtype=dtype)
L = form_custom(ufl.rhs(F), unf_mesh, dtype=dtype)
# -

# Using those forms we can assemble the corresponding
# stiffness {py:class}`PETSc matrix<dolfinx.fem.petsc.assemble_matrix>`
# `A` and right-hand-side
# {py:class}`PETSc vector<dolfinx.fem.petsc.assemble_vector>` `b`
# array for each coefficient.
#

# +
A = dolfinx.fem.petsc.assemble_matrix(a, coeffs=a.pack_coefficients())
A.assemble()

b = dolfinx.fem.petsc.assemble_vector(L, coeffs=L.pack_coefficients())
# -

# ### Linear system solution

# We solve the associated linear system
# $A\mathbf{u} = \mathbf{b}$, where $\mathbf{u}$ is the
# solution of the problem. The solution is stored in a
# a finite element function `uh` defined over the same finite
# element space $V$ as the trial functions.
#
# In this case we use a direct solver (LU) to solve the
# linear system.

# +
uh = dolfinx.fem.Function(V)

solver = PETSc.KSP().create(unf_mesh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

solver.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()
# -

# Due to the matrix being ill-conditioned, the solver above
# may fail to provide a stable solution.
# As a (brutal) temporary workaround we solve the system
# by transforming the matrix to a dense matrix and using
# a direct solver (LU) provided by NumPy.
# Suitable preconditioner (e.g., Jacobi) will be implemented
# in the future.

# +
A_full = A.convert("dense").getDenseArray()
uh.x.array[:] = np.linalg.solve(A_full, b.array)
print(f"Matrix conditioning: {np.linalg.cond(A_full)}")
print(f"Diagonal minimum value: {np.min(np.diag(A_full))}")
# -

# ### Error calculation

# The $L^2$ error of the projection can be computed as

# +
error_form: qugar.dolfinx.CustomForm = form_custom((uh - f) ** 2 * ufl.dx, unf_mesh, dtype=dtype)
error_L2 = np.sqrt(
    unf_mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(error_form, coeffs=error_form.pack_coefficients()),
        op=MPI.SUM,
    )
)
if unf_mesh.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")
# -

# ### Visualization

# Finally, we visualize the obtained solution by generating
# an auxiliar mesh that reparameterizes the unfitted domain.
# This mesh approximates the domain using an arbitrary order fitted
# tessellation made of discontinuous (and possibly degenerated)
# Lagrange elements.
#
# An extra mesh corresponding to the wirebasket
# (the line intersections of the domain $\Omega$ with the Cartesian mesh
# $\mathcal{T}$ iso-planes) is also generated to improve the solution's visualization.

# +
reparam_degree = 3
reparam = qugar.reparam.create_reparam_mesh(unf_mesh, degree=reparam_degree, levelset=False)
reparam_mesh = reparam.create_mesh()
reparam_mesh_wb = reparam.create_mesh(wirebasket=True)
# -

# In order to visualize the computed solution, we interpolate it into the
# reparameterized mesh. For doing so we create a new
# space using the reparameterized mesh and interpolate
# the solution into it (using the `interpolate_nonmatching` DOLFINx's
# function). Note that even if the element type of `V_reparam` is
# continuous (`CG`), the underlying mesh is discontinuous, so it will be
# the generated function.

# +
V_reparam = dolfinx.fem.functionspace(reparam_mesh, ("CG", reparam_degree))
uh_reparam = dolfinx.fem.Function(V_reparam, dtype=dtype)

cmap = reparam_mesh.topology.index_map(reparam_mesh.topology.dim)
num_cells = cmap.size_local + cmap.num_ghosts
cells = np.arange(num_cells, dtype=np.int32)
interpolation_data = dolfinx.fem.create_interpolation_data(V_reparam, V, cells, padding=1.0e-14)

uh_reparam.interpolate_nonmatching(uh, cells, interpolation_data=interpolation_data)
# -


# Both meshes are then exported to VTK files and can be visualized using
# [ParaView](https://www.paraview.org/). In the case in which
# `reparam_degree > 1`, the parameter `Nonlinear Subdivision Level` value
# (under the advanced properties menu in ParaView) can be adjusted to
# generate higher-quality visualizations.

# +
results_folder = Path("results")
results_folder.mkdir(exist_ok=True, parents=True)
filename = results_folder / "L2_projection"

with dolfinx.io.VTKFile(reparam_mesh.comm, filename.with_suffix(".pvd"), "w") as vtk:
    vtk.write_function(uh_reparam)
    vtk.write_mesh(reparam_mesh_wb)
# -


# Other DOLFINx file writers (as
# [VTXWriter](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.VTXWriter)
# or [XDMFFile](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.XDMFFile))
# should be also usable, however, be aware that between `VTKFFile` and
# `XDMFFile`, the former is the recommended option for high-degrees
# (as hinted in [DOLFINx documentation](https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.io.html#dolfinx.io.VTKFile)).
