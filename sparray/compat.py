import numpy as np
import scipy.sparse as ss

__all__ = [
    'broadcast_to', 'broadcast_shapes', 'ufuncs_with_fixed_point_at_zero',
    'intersect1d_sorted'
]

if hasattr(np, 'broadcast_to'):
  broadcast_to = np.broadcast_to
else:
  def broadcast_to(array, shape, subok=False):
    '''copied in reduced form from numpy 1.10'''
    shape = tuple(shape)
    array = np.array(array, copy=False, subok=subok)
    broadcast = np.nditer((array,), flags=['multi_index', 'zerosize_ok'],
                          op_flags=['readonly'], itershape=shape, order='C'
                          ).itviews[0]
    if type(array) is not type(broadcast):
      broadcast = broadcast.view(type=type(array))
      if broadcast.__array_finalize__:
        broadcast.__array_finalize__(array)
    return broadcast


# Re-create np.broadcast rules, but for shapes instead of array-likes
def broadcast_shapes(*shapes):
  # this uses a tricky hack to make fake ndarrays
  x = np.array(0)
  fake_arrays = [broadcast_to(x, s) for s in shapes]
  return np.broadcast(*fake_arrays).shape


if hasattr(ss.base, '_ufuncs_with_fixed_point_at_zero'):
  ufuncs_with_fixed_point_at_zero = ss.base._ufuncs_with_fixed_point_at_zero
else:
  ufuncs_with_fixed_point_at_zero = frozenset((
      np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
      np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad, np.rad2deg,
      np.floor, np.ceil, np.trunc, np.sqrt))


try:
  import pyximport
  pyximport.install()
  from _bench import intersect1d_sorted
except ImportError:

  def intersect1d_sorted(a, b, return_inds=False):
    # current technique adapted from http://stackoverflow.com/a/12427633/10601
    c = np.concatenate((a, b))
    c.sort(kind='mergesort')
    mask = np.zeros(len(c), dtype=bool)
    np.equal(c[1:], c[:-1], out=mask[:-1])
    c = c[mask]
    if not return_inds:
      return c
    a_inds = np.searchsorted(a, c)
    b_inds = np.searchsorted(b, c)
    return c, a_inds, b_inds
