import numpy as np
import scipy.sparse as ss

__all__ = [
    'broadcast_to', 'broadcast_shapes', 'ufuncs_with_fixed_point_at_zero',
    'intersect1d_sorted', 'union1d_sorted', 'combine_ranges', 'len_range'
]


def _broadcast_to(array, shape, subok=False):
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


def _intersect1d_sorted(a, b, return_inds=False):
  # technique adapted from http://stackoverflow.com/a/12427633/10601
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


def _union1d_sorted(a, b, return_masks=False):
  a = np.asanyarray(a)
  b = np.asanyarray(b)
  common_mask = np.in1d(a, b, assume_unique=True)
  common = a[common_mask]
  b_mask = np.in1d(b, common, assume_unique=True, invert=True)
  b_only = b[b_mask]
  c = np.concatenate((a, b_only))
  c.sort(kind='mergesort')
  mask = np.ones(len(c), dtype=bool)
  np.not_equal(c[1:], c[:-1], out=mask[1:])
  c = c[mask]
  if not return_masks:
    return c
  lut = np.in1d(c, b_only) + 2 * np.in1d(c, common)
  a_mask = ~common_mask
  return c, lut, a_mask, b_mask


def _combine_ranges(ranges, shape, result_size, inner=False):
  if inner:
    return np.ravel_multi_index([np.arange(*row) for row in ranges], shape)
  strides = np.ones(len(shape), dtype=ranges.dtype)
  np.cumprod(shape[:0:-1], out=strides[1:])
  flat_ranges = ranges * strides[::-1, None]
  flat_idxs = (np.arange(*row) for row in flat_ranges)
  result = next(flat_idxs)
  for idx in flat_idxs:
    result = np.add.outer(result, idx).ravel()
  return result[:result_size]


def _len_range(start, stop, step):
  if step > 0:
    if start >= stop:
      return 0
    return (stop - start - 1) // step + 1
  # negative step case
  if stop >= start:
    return 0
  return (start - stop - 1) // -step + 1


# Apply the shims where necessary

if hasattr(np, 'broadcast_to'):  # pragma: no cover
  broadcast_to = np.broadcast_to
else:  # pragma: no cover
  broadcast_to = _broadcast_to

if hasattr(ss.base, '_ufuncs_with_fixed_point_at_zero'):  # pragma: no cover
  ufuncs_with_fixed_point_at_zero = ss.base._ufuncs_with_fixed_point_at_zero
else:  # pragma: no cover
  ufuncs_with_fixed_point_at_zero = frozenset((
      np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
      np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad, np.rad2deg,
      np.floor, np.ceil, np.trunc, np.sqrt))

try:  # pragma: no cover
  # use pre-compiled _merge.so library
  from _merge import (
      intersect1d_sorted, union1d_sorted, combine_ranges, len_range)
except ImportError:
  # try compiling it ourselves on the fly
  try:
    import pyximport
    pyximport.install(setup_args={'include_dirs': np.get_include()})
    from _merge import (
        intersect1d_sorted, union1d_sorted, combine_ranges, len_range)
  except ImportError:
    # fall back to pure-Python versions
    intersect1d_sorted = _intersect1d_sorted
    union1d_sorted = _union1d_sorted
    combine_ranges = _combine_ranges
    len_range = _len_range
