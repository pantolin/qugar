from mpi4py import MPI

from dolfinx import mesh

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)
topology = domain.topology
topology.create_connectivity(2, 1)
topology.create_connectivity(1, 2)
conn = topology.connectivity(2, 1)
exterior_facets = mesh.exterior_facet_indices(topology)
index_map = domain.topology.index_map(2)
print(type(index_map.size_local))
print(type(index_map.size_global))
print(type(topology.original_cell_index.size))
kk = index_map.local_to_global(0)
print(domain)
