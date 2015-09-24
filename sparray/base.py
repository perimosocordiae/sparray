from __future__ import absolute_import
import numbers
import numpy as np
import scipy.sparse as ss

from .compat import (
    broadcast_to, broadcast_shapes, ufuncs_with_fixed_point_at_zero,
    intersect1d_sorted, union1d_sorted
)


class SpArray(object):
  '''Simple sparse ndarray-like, similar to scipy.sparse matrices.
  Defined by three member variables:
    self.data : array of nonzero values (may include zeros)
    self.indices : sorted int64 array of nonzero flat indices
    self.shape : tuple of integers, ala ndarray shape
  '''
  __array_priority__ = 999

  def __init__(self, indices, data, shape=None, is_canonical=False):
    indices = np.array(indices, dtype=int, copy=False).ravel()
    data = np.array(data, copy=False).ravel()
    assert len(indices) == len(data), '# inds (%d) != # data (%d)' % (
        len(indices), len(data))
    if not is_canonical:
      # sort and sum duplicates, but allow explicit zeros
      indices, inv_ind = np.unique(indices, return_inverse=True)
      data = np.bincount(inv_ind, weights=data).astype(data.dtype, copy=False)
    if shape is None:
      self.shape = (indices[-1]+1,)
    else:
      self.shape = shape
      assert np.prod(shape) >= len(data)
    self.indices = indices
    self.data = data

  @staticmethod
  def from_ndarray(arr):
    '''Converts an array-like to a SpArray object.'''
    arr = np.array(arr, copy=False)
    mask = arr.flat != 0
    idx, = np.nonzero(mask)
    return SpArray(idx, arr.flat[mask], shape=arr.shape, is_canonical=True)

  @staticmethod
  def from_spmatrix(mat):
    '''Converts a scipy.sparse matrix to a SpArray object'''
    mat = mat.tocoo()
    inds = np.ravel_multi_index((mat.row, mat.col), mat.shape)
    return SpArray(inds, mat.data, shape=mat.shape)

  def toarray(self):
    a = np.zeros(self.shape, dtype=self.data.dtype)
    a.flat[self.indices] = self.data
    return a

  def tocoo(self):
    assert len(self.shape) == 2
    row, col = np.unravel_index(self.indices, self.shape)
    return ss.coo_matrix((self.data, (row, col)), shape=self.shape)

  def getnnz(self):
    '''Get the count of explicitly-stored values'''
    return len(self.indices)

  nnz = property(fget=getnnz, doc=getnnz.__doc__)

  def nonzero(self):
    '''Returns a tuple of arrays containing indices of non-zero elements.
    Note: Does not include explicitly-stored zeros.
    '''
    nz_inds = self.indices[self.data!=0]
    return np.unravel_index(nz_inds, self.shape)

  def __len__(self):
    # Mimic ndarray here, instead of spmatrix
    return self.shape[0]

  def __bool__(self):
    if np.prod(self.shape) <= 1:
      return bool(len(self.data))
    raise ValueError("The truth value of an array with more than one "
                     "element is ambiguous. Use a.any() or a.all().")

  __nonzero__ = __bool__

  def transpose(self, *axes):
    if self.ndim < 2:
      return self
    # axes control dimension order, defaults to reverse
    if not axes:
      axes = range(self.ndim-1, -1, -1)
    elif len(axes) == 1 and self.ndim > 1:
      axes = axes[0]
    new_shape = tuple(self.shape[i] for i in axes)
    if self.shape == new_shape:
      return self
    # Hack: convert our flat indices into the new shape's flat indices.
    old_multi_index = np.unravel_index(self.indices, self.shape)
    new_multi_index = tuple(old_multi_index[i] for i in axes)
    new_inds = np.ravel_multi_index(new_multi_index, new_shape)
    return SpArray(new_inds, self.data, new_shape)

  def diagonal(self, offset=0, axis1=0, axis2=1):
    if axis1 == axis2:
      raise ValueError('axis1 and axis2 cannot be the same')
    if self.ndim < 2:
      raise ValueError('diagonal requires at least two dimensions')
    # TODO: support offsets, different axes, ndim > 2, etc
    if self.ndim > 2:
      raise NotImplementedError('diagonal() is NYI for ndim > 2')
    if offset != 0 or axis1 != 0 or axis2 != 1:
      raise NotImplementedError('diagonal() is NYI for non-default parameters')
    idx = np.arange(min(self.shape))
    return self._slice_multi((idx, idx), inner=True)

  def __repr__(self):
    return '<%s-SpArray of type %s\n\twith %d stored elements>' % (
        self.shape, self.data.dtype, self.getnnz())

  def __str__(self):
    lines = []
    multi_inds = np.unravel_index(self.indices, self.shape)
    for x in zip(self.data, *multi_inds):
      lines.append('  %s\t%s' % (x[1:], x[0]))
    return '\n'.join(lines)

  def reshape(self, new_shape):
    try:
      idx = new_shape.index(-1)
    except ValueError:
      assert np.prod(new_shape) >= len(self.data)
    else:
      assert sum(d == -1 for d in new_shape) == 1, 'Only one -1 allowed'
      new_shape = list(new_shape)
      new_shape[idx] = np.prod(self.shape) // -np.prod(new_shape)
    return SpArray(self.indices, self.data, shape=new_shape, is_canonical=True)

  def resize(self, new_shape):
    assert np.prod(new_shape) >= len(self.data)
    self.shape = new_shape

  def ravel(self):
    n = int(np.prod(self.shape))
    return SpArray(self.indices, self.data, shape=(n,), is_canonical=True)

  def __iter__(self):
    for i in range(self.shape[0]):
      yield self[i]

  def _prepare_indices(self, index):
    # avoid dealing with non-tuple cases
    if not isinstance(index, tuple):
      mut_indices = [index]
    else:
      mut_indices = list(index)
    missing_dims = len(self.shape) - len(mut_indices)

    # check for Ellipsis
    ell_inds = [i for i, idx in enumerate(mut_indices) if idx is Ellipsis]
    if ell_inds:
      # according to http://sourceforge.net/p/numpy/mailman/message/12594675/,
      # only the first Ellipsis is "real", and the rest are just slice(None)
      for i in ell_inds[1:]:
        mut_indices[i] = slice(None)
      # insert as many colons as we need at the first Ellipsis position
      ell_pos = ell_inds[0]
      mut_indices[ell_pos:ell_pos+1] = [slice(None)] * (missing_dims+1)
    elif missing_dims > 0:
      mut_indices.extend([slice(None)] * missing_dims)

    # check for array-like indices
    # TODO: handle indices with ndim > 1
    for i, idx in enumerate(mut_indices):
      if isinstance(idx, (slice, numbers.Integral)):
        continue
      if not hasattr(idx, 'ndim'):
        idx = np.array(idx, copy=False, subok=True, order='A')
      if idx.ndim > 1:
        raise NotImplementedError('Multi-dimensional indexing is NYI')
      if idx.dtype in (bool, np.bool_):
        idx, = idx.nonzero()
      mut_indices[i] = idx

    if len(mut_indices) > len(self.shape):
      raise IndexError('too many indices for SpArray')
    # indices now match our shape, and each index is int|slice|array
    assert len(mut_indices) == len(self.shape)

    # do some simple checking / fixup
    for axis, (idx, dim) in enumerate(zip(mut_indices, self.shape)):
      if isinstance(idx, numbers.Integral):
        if not (-dim <= idx < dim):
          raise IndexError('index %d is out of bounds '
                           'for axis %d with size %d' % (idx, axis, dim))
        if idx < 0:
          mut_indices[axis] += dim
    return tuple(mut_indices)

  def _slice_multi(self, indices, inner=True):
    '''Helper for making a new SpArray using (int,array-like) indices.
    dense[ii,jj]         -> sparse._slice_multi((ii, jj), inner=True)
    dense[ii[:,None],jj] -> sparse._slice_multi((ii, jj), inner=False)
    '''
    shape = tuple(len(idx) for idx in indices
                  if not isinstance(idx, numbers.Integral))
    if inner:
      assert len(set(shape)) == 1
      shape = (shape[0],)
    flat_idx = np.ravel_multi_index(indices, self.shape)
    _, data_inds, new_indices = intersect1d_sorted(self.indices, flat_idx,
                                                   return_inds=True)
    new_data = self.data[data_inds]
    return SpArray(new_indices, new_data, shape, is_canonical=True)

  def __getitem__(self, indices):
    indices = self._prepare_indices(indices)

    # trivial case: all slices are colons
    if all(isinstance(idx, slice) and idx == slice(None) for idx in indices):
      return self

    # simple case: all indices are simple int indexes
    if all(isinstance(idx, numbers.Integral) for idx in indices):
      flat_idx = np.ravel_multi_index(indices, self.shape)
      i = np.searchsorted(self.indices, flat_idx)
      if i >= len(self.indices) or self.indices[i] != flat_idx:
        return 0
      return self.data[i]

    # non-fancy case: all indices are slices or integers
    if not any(hasattr(idx, 'shape') for idx in indices):
      # convert slices to ranges
      # TODO: find a way to avoid this, combining into a "flat range"?
      indices = [np.arange(*idx.indices(dim)) if isinstance(idx, slice) else idx
                 for idx, dim in zip(indices, self.shape)]
      return self._slice_multi(indices, inner=False)

    # TODO: implement the harder cases
    raise NotImplementedError('Fancy slicing is still NYI')

  def _pairwise_sparray(self, other, ufunc):
    '''Helper function for the pattern: ufunc(sparse, sparse) -> sparse
    other : SpArray with the same shape
    ufunc : vectorized binary function
    '''
    idx, lut, lhs_only, rhs_only = union1d_sorted(self.indices, other.indices,
                                                  return_masks=True)
    data = np.empty_like(idx, dtype=np.promote_types(self.dtype, other.dtype))
    data[lut==0] = ufunc(self.data[lhs_only], 0)
    data[lut==1] = ufunc(0, other.data[rhs_only])
    data[lut==2] = ufunc(self.data[~lhs_only], other.data[~rhs_only])
    return SpArray(idx, data, self.shape, is_canonical=True)

  def _pairwise_sparray_fixed_zero(self, other, ufunc):
    '''Helper function for the pattern: ufunc(sparse, sparse) -> sparse
    other : SpArray with the same shape
    ufunc : vectorized binary function, where ufunc(x, 0) -> 0
    '''
    idx, lhs_inds, rhs_inds = intersect1d_sorted(self.indices, other.indices,
                                                 return_inds=True)
    lhs = self.data[lhs_inds]
    rhs = other.data[rhs_inds]
    data = ufunc(lhs, rhs)
    return SpArray(idx, data, self.shape, is_canonical=True)

  def _pairwise_dense2dense(self, other, ufunc):
    '''Helper function for the pattern: ufunc(dense, sparse) -> dense
    other : ndarray
    '''
    result = other.copy(order='C')
    result.flat[self.indices] = ufunc(result.flat[self.indices], self.data)
    return result

  def _pairwise_dense2sparse(self, other, ufunc):
    '''Helper function for the pattern: ufunc(dense, sparse) -> sparse
    other : array_like
    '''
    other = np.asanyarray(other)
    return self._with_data(ufunc(self.data, other.flat[self.indices]))

  def _handle_broadcasting(self, other):
    if other.shape == self.shape:
      return self, other
    # Find a shape that we can broadcast to
    bshape = broadcast_shapes(self.shape, other.shape)
    # Do broadcasting for the lhs
    if self.shape == bshape:
      lhs = self
    else:
      lhs = self._broadcast(bshape)
    # Do broadcasting for the rhs
    if other.shape == bshape:
      rhs = other
    elif isinstance(other, SpArray):
      rhs = other._broadcast(bshape)
    else:
      rhs = broadcast_to(other, bshape, subok=True)
    return lhs, rhs

  def _broadcast(self, shape):
    # TODO: fix this hack! Need to avoid densifying here.
    return SpArray.from_ndarray(broadcast_to(self.toarray(), shape))

  def __add__(self, other):
    if np.isscalar(other):
      if other == 0:
        return self.copy()
      raise NotImplementedError('adding a nonzero scalar to a sparse array '
                                'is not supported')
    if ss.issparse(other):
      # np.matrix + np.array always returns np.matrix, so for now we punt
      return self.tocoo() + other
    lhs, rhs = self._handle_broadcasting(other)
    assert isinstance(lhs, SpArray)
    if isinstance(rhs, SpArray):
      return lhs._pairwise_sparray(rhs, np.add)
    # dense addition
    return lhs._pairwise_dense2dense(rhs, np.add)

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self.__add__(-other)

  def __rsub__(self, other):
    return (-self).__add__(other)

  def __mul__(self, other):
    if np.isscalar(other):
      return self._with_data(self.data * other)
    if ss.issparse(other):
      # np.matrix * np.array always returns np.matrix, so for now we punt
      return self.tocoo().multiply(other)
    lhs, rhs = self._handle_broadcasting(other)
    assert isinstance(lhs, SpArray)
    if isinstance(rhs, SpArray):
      return lhs._pairwise_sparray_fixed_zero(rhs, np.multiply)
    # dense * sparse -> sparse
    return lhs._pairwise_dense2sparse(rhs, np.multiply)

  def __rmul__(self, other):
    return self.__mul__(other)

  def multiply(self, other):
    '''Element-wise multiplication. Alias for self * other'''
    return self.__mul__(other)

  def __div__(self, other):
    return self._divide(other)

  def __rdiv__(self, other):
    return self._divide(other, rdivide=True)

  def __truediv__(self, other):
    return self._divide(other, div_func=np.true_divide)

  def __rtruediv__(self, other):
    return self._divide(other, div_func=np.true_divide, rdivide=True)

  def __floordiv__(self, other):
    return self._divide(other, div_func=np.floor_divide)

  def __rfloordiv__(self, other):
    return self._divide(other, div_func=np.floor_divide, rdivide=True)

  def _divide(self, other, div_func=np.divide, rdivide=False):
    # Don't bother keeping sparsity if rhs is sparse
    if ss.issparse(other) or isinstance(other, SpArray):
      other = other.toarray()
      if not rdivide:
        return div_func(self.toarray(), other)
    if rdivide:
      return div_func(other, self.toarray())
    # Punt truediv to __mul__
    if div_func is np.true_divide:
      return self.__mul__(1. / other)
    # Non-truediv cases
    if np.isscalar(other):
      return self._with_data(div_func(self.data, other))
    lhs, rhs = self._handle_broadcasting(other)
    # dense / sparse -> sparse
    return lhs._pairwise_dense2sparse(rhs, div_func)

  def __matmul__(self, other):
    return self.dot(other)

  def dot(self, other):
    ax1 = len(self.shape) - 1
    ax2 = max(0, len(other.shape) - 2)
    if self.shape[ax1] != other.shape[ax2]:
      raise ValueError('shapes %s and %s not aligned' % (self.shape,
                                                         other.shape))
    # if other is sparse, use spmatrix dot
    if ss.issparse(other) or isinstance(other, SpArray):
      out_shape = self.shape[:-1] + other.shape[:ax2] + other.shape[ax2+1:]
      lhs_shape = (int(np.product(self.shape[:-1])), self.shape[ax1])
      lhs = self.reshape(lhs_shape).tocoo()
      if isinstance(other, SpArray):
        # transpose so ax2 comes first
        axes = (ax2,) + tuple(range(ax2)) + tuple(range(ax2+1,len(other.shape)))
        other = other.transpose(*axes)
        # reshape to 2d for spmatrix
        rhs_shape = (other.shape[0], int(np.product(other.shape[1:])))
        other = other.reshape(rhs_shape).tocoo()
      result = lhs.dot(other)
      # convert back to a SpArray with the correct shape
      if not out_shape:  # scalar case, return a scalar
        return result[0,0]
      return SpArray.from_spmatrix(result).reshape(out_shape)

    # other is dense
    if self.ndim == 1 and other.ndim == 1:
      # TODO: allow other shapes for self here
      return other[self.indices].dot(self.data)
    # dense rhs always returns dense result
    return self.toarray().dot(other)

  def _rdot(self, other):
    # This only gets called for dense other,
    # because spmatrix.dot(x) calls np.asarray(x)
    return other.dot(self.toarray())

  def __pow__(self, exponent):
    # TODO: Should probably warn when we're losing sparsity
    if exponent == 0:
      return np.ones(self.shape, dtype=self.dtype)
    elif exponent < 0:
      return self.toarray() ** exponent
    return self._with_data(self.data ** exponent)

  def _with_data(self, data):
    return SpArray(self.indices.copy(), data, self.shape, is_canonical=True)

  def minimum(self, other):
    if np.isscalar(other) and other >= 0:
      return self._with_data(np.minimum(self.data, other))
    if isinstance(other, SpArray):
      return self._pairwise_sparray(other, np.minimum)
    if ss.issparse(other):
      # For now, convert to SpArray first and then do the operation
      return self._pairwise_sparray(SpArray.from_spmatrix(other), np.minimum)
    # Probably won't get a sparse result
    return np.minimum(self.toarray(), other)

  def maximum(self, other):
    if np.isscalar(other) and other <= 0:
      return self._with_data(np.maximum(self.data, other))
    if isinstance(other, SpArray):
      return self._pairwise_sparray(other, np.maximum)
    if ss.issparse(other):
      # For now, convert to SpArray first and then do the operation
      return self._pairwise_sparray(SpArray.from_spmatrix(other), np.maximum)
    # Probably won't get a sparse result
    return np.maximum(self.toarray(), other)

  def sum(self, axis=None, dtype=None):
    if dtype is None:
      dtype = self.dtype
    if axis is None:
      return self.data.sum(dtype=dtype)
    # XXX: we don't support tuples of axes, yet
    axis = int(axis)
    new_shape = self.shape[:axis] + self.shape[axis+1:]
    if not new_shape:
      return self.data.sum(dtype=dtype)
    axis_inds = np.unravel_index(self.indices, self.shape)
    axis_inds = axis_inds[:axis] + axis_inds[axis+1:]
    flat_inds = np.ravel_multi_index(axis_inds, new_shape)
    new_idx, data_idx = np.unique(flat_inds, return_inverse=True)
    # Note: we can't use:
    #    new_data = np.zeros(new_idx.shape, dtype=dtype)
    #    new_data[data_idx] += self.data
    # here, because the fancy index doesn't return a proper view.
    new_data = np.bincount(data_idx, self.data.astype(dtype, copy=False))
    return SpArray(new_idx, new_data, shape=new_shape, is_canonical=True)

  def mean(self, axis=None, dtype=None):
    if dtype is None:
      dtype = self.dtype
    # Mimic numpy upcasting
    if np.can_cast(dtype, np.float_):
      dtype = np.float_
    elif np.can_cast(dtype, np.complex_):
      dtype = np.complex_
    s = self.sum(axis=axis, dtype=dtype)
    if axis is None:
      num_elts = np.prod(self.shape)
    else:
      # XXX: we don't support tuples of axes, yet
      num_elts = self.shape[axis]
    if num_elts != 1:
      s /= num_elts
    return s

  def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
    '''ufunc dispatcher. Mostly copied from scipy.sparse.spmatrix'''
    out = kwargs.pop('out', None)

    without_self = list(inputs)
    del without_self[pos]
    without_self = tuple(without_self)

    if func is np.multiply:
      result = self.__mul__(*without_self)
    elif func is np.add:
      result = self.__add__(*without_self)
    elif func is np.dot:
      if pos == 0:
        result = self.dot(inputs[1])
      else:
        result = self._rdot(inputs[0])
    elif func is np.subtract:
      if pos == 0:
        result = self.__sub__(inputs[1])
      else:
        result = self.__rsub__(inputs[0])
    elif func in (np.divide, np.true_divide, np.floor_divide):
      result = self._divide(*without_self, div_func=func, rdivide=(pos==1))
    elif func in (np.minimum, np.maximum):
      result = getattr(self, func.__name__)(*without_self)
    elif func is np.absolute:
      result = abs(self)
    elif func in (np.conj, np.conjugate):
      result = self.conj()
    elif func in ufuncs_with_fixed_point_at_zero:
      result = getattr(self, func.__name__)()
    else:
      return NotImplemented

    if out is not None:
      if not isinstance(out, SpArray) and isinstance(result, SpArray):
        out[...] = result.toarray()
      else:
        out.data = result.data
        out.indices = result.indices
        out.shape = result.shape
      result = out

    return result

  def __getattr__(self, attr):
    if attr == 'dtype':
      return self.data.dtype
    if attr == 'A':
      return self.toarray()
    if attr == 'T':
      return self.transpose()
    if attr == 'size':
      return self.getnnz()
    if attr == 'ndim':
      return len(self.shape)
    raise AttributeError(attr + " not found")

  # The following code is completely ripped from scipy.sparse.data._data_matrix.
  # I'm including it here because I don't want to inherit from spmatrix.

  def __abs__(self):
    return self._with_data(abs(self.data))

  @property
  def real(self):
    return self._with_data(self.data.real)

  @property
  def imag(self):
    return self._with_data(self.data.imag)

  def __neg__(self):
    return self._with_data(-self.data)

  def __imul__(self, other):  # self *= other
    if np.isscalar(other):
      self.data *= other
      return self
    else:
      return NotImplemented

  def __itruediv__(self, other):  # self /= other
    if np.isscalar(other):
      recip = 1.0 / other
      self.data *= recip
      return self
    else:
      return NotImplemented

  def astype(self, t):
    return self._with_data(self.data.astype(t))

  def conj(self):
    return self._with_data(self.data.conj())

  def conjugate(self):
    return self.conj()

  def copy(self):
    return self._with_data(self.data.copy())

  def min(self):
    # TODO: axis kwarg
    return self.data.min()

  def max(self):
    # TODO: axis kwarg
    return self.data.max()


# Add the numpy unary ufuncs for which func(0) = 0
for npfunc in ufuncs_with_fixed_point_at_zero:
  name = npfunc.__name__

  def _create_method(op):
    def method(self):
      result = op(self.data)
      x = self._with_data(result)
      return x

    method.__doc__ = ("Element-wise %s.\n\n"
                      "See numpy.%s for more information." % (name, name))
    method.__name__ = name
    return method

  setattr(SpArray, name, _create_method(npfunc))
