from mpi4py import MPI

import numpy as np

import qugar
import qugar.cpp
from qugar.impl.impl_function import (
    # create_Fischer_Koch_S,
    # create_Schoen,
    # create_Schoen_FRD,
    # create_Schoen_IWP,
    # create_Schwarz_Diamond,
    # create_Schwarz_Primitive,
    create_sphere,
)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    func = create_sphere(radius=1.0, use_bzr=False)
    # func = create_Schoen(periods=[2, 2, 2])
    # func = create_Schoen_FRD(periods=[2, 2, 2])
    # func = create_Schoen_IWP(periods=[2, 2, 2])
    # func = create_Fischer_Koch_S(periods=[2, 2, 2])
    # func = create_Schwarz_Diamond(periods=[2, 2, 2])
    # func = create_Schwarz_Primitive(periods=[2, 2, 2])

    mesh = qugar.mesh.create_Cartesian_mesh(comm, [4, 4, 4], np.zeros(3), np.ones(3))
    domain = qugar.impl.create_unfitted_impl_domain(func, mesh)
    # quad_gen = qugar.quad.create_quadrature_generator(domain)
    # quad_vtk = qugar.vtk.quadrature_to_VTK(domain, n_pts_dir=4)
    # qugar_vtk = qugar.vtk.write_VTK_to_file(quad_vtk, "quad")
    domain.quadrature_to_VTK_file("quad")

    reparam = qugar.reparam.create_reparam_mesh(domain, n_pts_dir=4, levelset=False)
    reparam.to_VTK_file("test")
