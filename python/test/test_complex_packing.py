# --------------------------------------------------------------------------
#
# Copyright (C) 2025-present by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

"""Byte-level validation of the on-the-fly complex-coefficient packing layout.

The on-the-fly packer writes per-cell smuggled data (n_pts header, points,
weights, ...) into a numpy array typed as the form's coefficients dtype
(real or complex). For complex coeffs the smuggled region is written
through a real view (``coeffs.view(real_dtype)``) so each complex cell
holds 2 real slots. The kernel reads it back via
``((const T*)w) + w_custom_offset`` so the offset is in real units.

This test exercises that layout directly -- no full form assembly, which
would require PETSc compiled with complex scalars. It verifies:

- The offset stored in the trailing cell decodes to the expected
  real-unit position.
- The smuggled data at that position byte-matches what was packed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _emulate_kernel_read(coeffs, entity_row, col_in_row, real_dtype,
                          want_real_slots):
    """Mimic the kernel: w := row-start pointer; read ptrdiff_t at
    w[col_in_row] (within-row column of the offset cell); then read
    ``want_real_slots`` real_dtype values starting at ``(T*)w + offset``.
    """
    raw = np.ascontiguousarray(coeffs).view(np.uint8).ravel()
    bytes_per_row = coeffs.shape[1] * coeffs.dtype.itemsize
    byte_start = entity_row * bytes_per_row + col_in_row * coeffs.dtype.itemsize
    intp_bytes = raw[byte_start : byte_start + np.intp().itemsize]
    offset = int(np.frombuffer(intp_bytes.tobytes(), dtype=np.intp)[0])
    ratio = coeffs.dtype.itemsize // real_dtype.itemsize
    real_row_start = entity_row * coeffs.shape[1] * ratio
    real = coeffs.view(real_dtype).ravel()
    return offset, real[real_row_start + offset
                        : real_row_start + offset + want_real_slots]


def _pack_byte_layout(coeffs_dtype, real_dtype, n_cols_complex, n_extra_cols,
                     entities, payloads):
    """Build a fake new_coeffs the same way custom_coefficients.py does:
    one row per entity, plus extra rows for the smuggled payload region.
    """
    ratio = coeffs_dtype.itemsize // real_dtype.itemsize  # 1 or 2
    n_cols_real = n_cols_complex * ratio
    n_entities = len(entities)
    sizes = np.array([p.size for p in payloads], dtype=np.intp)
    cum = np.concatenate(([0], np.cumsum(sizes)))
    first_extra = n_entities * n_cols_real
    abs_offsets = cum[:-1] + first_extra  # real units
    rel_offsets = abs_offsets - np.arange(n_entities) * n_cols_real

    n_extra_rows = int(np.ceil(sizes.sum() / n_cols_real))
    coeffs = np.zeros((n_entities + n_extra_rows, n_cols_complex),
                      dtype=coeffs_dtype)

    if coeffs_dtype.itemsize <= np.intp().itemsize:
        view = rel_offsets.astype(np.intp).view(coeffs_dtype).reshape(
            -1, n_extra_cols)
    else:
        intps_per_cell = coeffs_dtype.itemsize // np.intp().itemsize
        padded = np.zeros(n_entities * intps_per_cell, dtype=np.intp)
        padded[::intps_per_cell] = rel_offsets
        view = padded.view(coeffs_dtype).reshape(-1, n_extra_cols)
    coeffs[:n_entities, -n_extra_cols:] = view

    real_view = coeffs.view(real_dtype).ravel()
    pos = first_extra
    for p in payloads:
        real_view[pos : pos + p.size] = p
        pos += p.size

    return coeffs, abs_offsets


@pytest.mark.parametrize(
    ("coeffs_dtype", "real_dtype"),
    [
        (np.dtype(np.float32), np.dtype(np.float32)),
        (np.dtype(np.float64), np.dtype(np.float64)),
        (np.dtype(np.complex64), np.dtype(np.float32)),
        (np.dtype(np.complex128), np.dtype(np.float64)),
    ],
    ids=["float32", "float64", "complex64", "complex128"],
)
def test_kernel_read_matches_pack(coeffs_dtype, real_dtype):
    """For each coefficient dtype, what the kernel reads back must
    byte-match what the packer wrote."""
    n_cols_complex = 5
    n_extra_cols = 1 if coeffs_dtype.itemsize >= np.intp().itemsize else 2
    n_cols_complex_total = n_cols_complex + n_extra_cols
    rng = np.random.default_rng(0)
    payloads = [rng.random(7).astype(real_dtype),
                rng.random(11).astype(real_dtype)]
    coeffs, expected_abs = _pack_byte_layout(
        coeffs_dtype, real_dtype, n_cols_complex_total, n_extra_cols,
        entities=[0, 1], payloads=payloads,
    )

    col_in_row = n_cols_complex_total - n_extra_cols
    for i, p in enumerate(payloads):
        rel, got = _emulate_kernel_read(coeffs, i, col_in_row, real_dtype,
                                         p.size)
        ratio = coeffs_dtype.itemsize // real_dtype.itemsize
        abs_pos = rel + i * n_cols_complex_total * ratio
        assert abs_pos == expected_abs[i], (
            f"{coeffs_dtype.name}: offset mismatch at entity {i}, "
            f"got abs={abs_pos} expected {expected_abs[i]}"
        )
        np.testing.assert_allclose(got, p, err_msg=(
            f"{coeffs_dtype.name}: payload bytes mismatch at entity {i}"))
