# -*- coding: utf-8 -*-
"""
GStools subpackage providing export routines.

.. currentmodule:: gstools.tools.export

The following functions are provided

.. autosummary::
   to_vtk_structured
   vtk_export_structured
   to_vtk_unstructured
   vtk_export_unstructured
   to_vtk
   vtk_export
"""
# pylint: disable=C0103, E1101
from __future__ import print_function, division, absolute_import

import numpy as np
from pyevtk.hl import gridToVTK, pointsToVTK

from gstools.tools.geometric import pos2xyz

try:
    import pyvista as pv
except ImportError:
    pv = None

__all__ = [
    "to_vtk_structured",
    "vtk_export_structured",
    "to_vtk_unstructured",
    "vtk_export_unstructured",
    "to_vtk",
    "vtk_export",
]


# export routines #############################################################


def _vtk_structured_helper(pos, fields):
    """An internal helper to extract what is needed for the vtk rectilinear grid
    """
    if not isinstance(fields, dict):
        fields = {"field": fields}
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.array([0])
    if z is None:
        z = np.array([0])
    # need fortran order in VTK
    for field in fields:
        fields[field] = fields[field].reshape(-1, order="F")
        if len(fields[field]) != len(x) * len(y) * len(z):
            raise ValueError(
                "gstools.vtk_export_structured: "
                "field shape doesn't match the given mesh"
            )
    return x, y, z, fields


def to_vtk_structured(pos, fields):  # pragma: no cover
    """Create a vtk structured rectilinear grid from a field.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.

    Returns
    -------
    :class:`pyvista.RectilinearGrid`
        A PyVista rectilinear grid of the structured field data. Data arrays
        live on the point data of this PyVista dataset.
    """
    x, y, z, fields = _vtk_structured_helper(pos=pos, fields=fields)
    try:
        import pyvista as pv

        grid = pv.RectilinearGrid(x, y, z)
        grid.point_arrays.update(fields)
    except ImportError:
        raise ImportError("Please install PyVista to create VTK datasets.")
    return grid


def vtk_export_structured(filename, pos, fields):  # pragma: no cover
    """Export a field to vtk structured rectilinear grid file.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtr) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    """
    x, y, z, fields = _vtk_structured_helper(pos=pos, fields=fields)
    return gridToVTK(filename, x, y, z, pointData=fields)


def _vtk_unstructured_helper(pos, fields):
    if not isinstance(fields, dict):
        fields = {"field": fields}
    x, y, z = pos2xyz(pos)
    if y is None:
        y = np.zeros_like(x)
    if z is None:
        z = np.zeros_like(x)
    for field in fields:
        fields[field] = fields[field].reshape(-1)
        if (
            len(fields[field]) != len(x)
            or len(fields[field]) != len(y)
            or len(fields[field]) != len(z)
        ):
            raise ValueError(
                "gstools.vtk_export_unstructured: "
                "field shape doesn't match the given mesh"
            )
    return x, y, z, fields


def to_vtk_unstructured(pos, fields):  # pragma: no cover
    """Export a field to vtk structured rectilinear grid file.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Unstructured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.

    Returns
    -------
    :class:`pyvista.UnstructuredGrid`
        A PyVista unstructured grid of the unstructured field data. Data arrays
        live on the point data of this PyVista dataset. This is essentially
        a point cloud with no topology.
    """
    x, y, z, fields = _vtk_unstructured_helper(pos=pos, fields=fields)
    try:
        import pyvista as pv

        grid = pv.PolyData(np.c_[x, y, z]).cast_to_unstructured_grid()
        grid.point_arrays.update(fields)
    except ImportError:
        raise ImportError("Please install PyVista to create VTK datasets.")
    return grid


def vtk_export_unstructured(filename, pos, fields):  # pragma: no cover
    """Export a field to vtk unstructured grid file.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        Unstructured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    """
    x, y, z, fields = _vtk_unstructured_helper(pos=pos, fields=fields)
    return pointsToVTK(filename, x, y, z, data=fields)


def to_vtk(pos, fields, mesh_type="unstructured"):  # pragma: no cover
    """Create a VTK/PyVista grid.

    Parameters
    ----------
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        [Un]structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: structured

    Returns
    -------
    :class:`pyvista.RectilinearGrid` or :class:`pyvista.UnstructuredGrid`
        This will return a PyVista object for the given field data in its
        appropriate type. Structured meshes will return a
        :class:`pyvista.RectilinearGrid` and unstructured meshes will return
        an :class:`pyvista.UnstructuredGrid` object.
    """
    if mesh_type == "structured":
        grid = to_vtk_structured(pos=pos, fields=fields)
    else:
        grid = to_vtk_unstructured(pos=pos, fields=fields)
    return grid


def vtk_export(
    filename, pos, fields, mesh_type="unstructured"
):  # pragma: no cover
    """Export a field to vtk.

    Parameters
    ----------
    filename : :class:`str`
        Filename of the file to be saved, including the path. Note that an
        ending (.vtr or .vtu) will be added to the name.
    pos : :class:`list`
        the position tuple, containing main direction and transversal
        directions
    fields : :class:`dict` or :class:`numpy.ndarray`
        [Un]structured fields to be saved.
        Either a single numpy array as returned by SRF,
        or a dictionary of fields with theirs names as keys.
    mesh_type : :class:`str`, optional
        'structured' / 'unstructured'. Default: structured
    """
    if mesh_type == "structured":
        return vtk_export_structured(filename=filename, pos=pos, fields=fields)
    else:
        return vtk_export_unstructured(
            filename=filename, pos=pos, fields=fields
        )
