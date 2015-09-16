import numpy as np
import scipy.sparse as ss

__all__ = ['broadcast_to', 'ufuncs_with_fixed_point_at_zero']

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

if hasattr(ss.base, '_ufuncs_with_fixed_point_at_zero'):
  ufuncs_with_fixed_point_at_zero = ss.base._ufuncs_with_fixed_point_at_zero
else:
  ufuncs_with_fixed_point_at_zero = frozenset((
    np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
    np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad, np.rad2deg,
    np.floor, np.ceil, np.trunc, np.sqrt))