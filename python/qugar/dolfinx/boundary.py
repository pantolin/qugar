# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"Data structures for computing quantities on custom boundaries."

from qugar.utils import has_FEniCSx

if not has_FEniCSx:
    raise ValueError("FEniCSx installation not found is required.")

from typing import Optional

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import numpy.typing as npt
import ufl
from ufl.geometry import Jacobian
from ufl.measure import integral_type_to_measure_name
from ufl.operators import inv, transpose
from ufl.protocols import id_or_none


class ParamNormal(dolfinx.fem.Constant):
    """Constant type for defining the normal of a custom unfitted
    boundary in the parametric domain of the cell.

    It is a constant vector set to zero, with the same number of
    coordinates as the normal, whose only purpose is to represent the
    normal. Its value will be determined at runtime at
    every quadrature point of the boundary.
    """

    def __init__(self, domain: ufl.AbstractDomain) -> None:
        """Initializes the normals.

        Args:
            domain (ufl.AbstractDomain): An AbstractDomain object (most
                often a Mesh) to which the normal is associated to.
        """

        assert isinstance(domain, dolfinx.mesh.Mesh)

        dtype = domain.geometry.x.dtype
        param_dim = domain.topology.dim

        zeros = np.zeros(param_dim, dtype=dtype)
        super().__init__(domain, zeros)


def _compute_vector_norm(vec):
    """Computes the norm of a vector.

    Args:
        vec: Vector whose norm is computed.

    Returns:
        Computed norm.
    """
    return ufl.sqrt(ufl.inner(vec, vec))


def mapped_normal(domain: ufl.AbstractDomain | dolfinx.mesh.Mesh, normalize: bool = True):
    """
    Returns a normal vector of a custom unfitted boundary mapped with
    the domain's geometry. I.e., the normal in the physical space.

    Args:
        domain (ufl.AbstractDomain | dolfinx.mesh.Mesh): An AbstractDomain object (most
            often a Mesh) to which the normal is associated to.
        normalize (bool): If ``True``, the returned normal is
            normalized, otherwise (if ``False``) it is not normalized
            and its norm is the ratio of the deformed to reference
            area cells.

    Returns:
        Mapped normal.
    """

    N = ParamNormal(domain)

    DF = Jacobian(domain)

    if DF.ufl_shape[0] != DF.ufl_shape[1]:
        metric = transpose(DF) * DF  # type: ignore
        n = DF * inv(metric) * N  # type: ignore
    else:
        n = transpose(inv(DF)) * N  # type: ignore

    if normalize:
        return n / _compute_vector_norm(n)  # type: ignore
    else:
        return n


class dx_bdry_unf(ufl.Measure):
    """This is a new ufl Measure class for an unfitted custom boundary.

    It has the same functionalities as ufl.dx with the only difference
    that when multiplied by an integrand, it introduces the necessary
    correction for accounting for the boundary orientation by using
    its normal.

    In order to differentiate the generated measure from others,
    sets the option `unfitted_custom_boundary` equal to ``True`` in
    the quadrature's metadata, and uses a custom quadrature with two
    (fake) points.
    """

    # Static definition of (fake) custom quadrature points and weights.
    _weights = np.array([-1.0, -1.0], dtype=np.float64)
    _points_2D = np.array([[0.1, 0.1], [0.2, 0.2]], dtype=np.float64)
    _points_3D = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]], dtype=np.float64)

    def __init__(
        self,
        domain: ufl.AbstractDomain | dolfinx.mesh.Mesh,
        subdomain_id: str | int | tuple[int] = "everywhere",
        metadata: dict | None = None,
        subdomain_data: dolfinx.mesh.MeshTags
        | list[tuple[int, npt.NDArray[np.int32]]]
        | None = None,
        degree: Optional[int] = None,
    ):
        """Initialize.

        Args:
            domain (ufl.AbstractDomain | dolfinx.mesh.Mesh): An AbstractDomain object (most
                often a Mesh).
            subdomain_id (str | int | tuple[int], optional): either
                string "everywhere", a single subdomain id int, or tuple
                of ints. Defaults to "everywhere".
            metadata (dict | None, optional): Dictionary with additional
                compiler-specific parameters for optimization or
                debugging of generated code. Defaults to None.
            subdomain_data (dolfinx.mesh.MeshTags | list[tuple[int, npt.NDArray[np.int32]]] | None, optional):
                Object representing data to interpret subdomain_id with. Defaults to None.
            degree (int, optional): The degree of the quadrature rule.
        """  # noqa: E501

        assert domain is not None

        metadata = {} if metadata is None else metadata.copy()
        if degree is not None:
            metadata["quadrature_degree"] = degree

        metadata.update(self._create_custom_metadata(domain))

        super().__init__(
            integral_type="cell",
            domain=domain,
            subdomain_id=subdomain_id,  # type: ignore
            metadata=metadata,
            subdomain_data=subdomain_data,
        )

        n = mapped_normal(domain, normalize=False)
        # This is the missing term in Nanson's formula (the Jacobian
        # determinant should by already included in dx).
        self._measure_complement = _compute_vector_norm(n)

        self._integral_type_mod = "unfitted_custom_boundary"
        self._measure_name = "dx_bdry_unf"

    def _create_custom_metadata(self, domain: ufl.AbstractDomain) -> dict:
        """Creates the custom measure metadata. It has an associated
        fake quadrature with only two points in the reference domain,
        and includes the flag ``custom_unfitted_boundary`` set to
        ``True``.

        Args:
            domain (ufl.AbstractDomain): An AbstractDomain object (most
                often a Mesh).

        Returns:
            dict: Generated metadata.
        """

        assert isinstance(domain, dolfinx.mesh.Mesh)

        tdim = domain.topology.dim
        points = self._points_2D if tdim == 2 else self._points_3D

        metadata = {}
        metadata["quadrature_points"] = points
        metadata["quadrature_weights"] = self._weights
        metadata["quadrature_rule"] = "custom"
        metadata["custom_unfitted_boundary"] = True

        return metadata

    def __rmul__(self, integrand):
        """Multiply a scalar expression with measure to construct a form
        with a single integral.

        This is to implement the notation
        ``form = integrand * measure_complement * self`` where
        ``measure_complement`` is the norm of a mapped normal to the
        boundary, according to Nanson's formula.

        Integration properties are taken from this Measure object.
        """
        return super().__rmul__(integrand * self._measure_complement)

    def __str__(self) -> str:
        """Formats the class instance as a string.

        Returns:
            str: Generated string.
        """
        parent_name = integral_type_to_measure_name[self._integral_type]
        parent_str = super().__str__()
        print(parent_str)
        print()
        return parent_str.replace(parent_name, self._measure_name, 1)

    def __repr__(self) -> str:
        """Return a repr string for this Measure.

        Returns:
            str: Generated representation.
        """
        parent_repr = super().__repr__()
        new_repr = parent_repr.replace(type(self).__name__, "Measure", 1)
        new_repr = new_repr.replace(repr(self._integral_type), repr(self._integral_type_mod), 1)
        return new_repr

    def __hash__(self) -> int:
        """Return a hash value for this Measure.

        Returns:
            int: Generated hash.
        """
        metadata_hashdata = tuple(sorted((k, id(v)) for k, v in list(self._metadata.items())))
        hashdata = (
            self._integral_type_mod,
            self._subdomain_id,
            hash(self._domain),
            metadata_hashdata,
            id_or_none(self._subdomain_data),
        )
        return hash(hashdata)

    def __eq__(self, other) -> bool:
        """Checks if two `dx_bdry_unf` measures are equal.

        Returns:
            bool: ``True`` if both measures are equal, ``False``
            otherwise.
        """
        return isinstance(other, dx_bdry_unf) and super().__eq__(other)
