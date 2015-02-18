import numpy as np
import scipy.sparse as ss


class SpArray(object):
  '''Simple sparse ndarray-like, similar to scipy.sparse matrices.
  Defined by three member variables:
    self.data : array of nonzero values (may include zeros)
    self.indices : int64 array of nonzero flat indices
    self.shape : tuple of integers, ala ndarray shape
  '''
  def __init__(self, indices, data, shape=None):
    indices = np.array(indices, dtype=int).ravel()
    data = np.array(data).ravel()
    assert len(indices) == len(data), '# inds (%d) != # data (%d)' % (
        len(indices), len(data))
    if shape is None:
      self.shape = (indices.max()+1,)
    else:
      self.shape = shape
      assert np.prod(shape) >= len(data)
    self.indices = indices
    self.data = data

  @staticmethod
  def from_ndarray(arr):
    mask = arr.flat != 0
    return SpArray(np.nonzero(mask)[0], arr.flat[mask], shape=arr.shape)

  @staticmethod
  def from_spmatrix(mat):
    mat = mat.tocoo()
    inds = np.ravel_multi_index((mat.row, mat.col), mat.shape)
    return SpArray(inds, mat.data, shape=mat.shape)

  def toarray(self):
    a = np.zeros(self.shape, dtype=self.data.dtype)
    a.flat[self.indices] = self.data
    return a

  def tocoo(self):
    assert len(self.shape) == 2
    row,col = np.unravel_index(self.indices, self.shape)
    return ss.coo_matrix((self.data, (row, col)), shape=self.shape)

  def getnnz(self):
    return len(self.indices)

  def transpose(self, *axes):
    if self.ndim < 2:
      return self
    # axes control dimension order, defaults to reverse
    if not axes:
      axes = range(self.ndim-1, -1, -1)
    elif len(axes) == 1 and self.ndim > 1:
      axes = axes[0]
    new_shape = tuple(self.shape[i] for i in axes)
    # Hack: convert our flat indices into the new shape's flat indices.
    old_multi_index = np.unravel_index(self.indices, self.shape)
    new_multi_index = tuple(old_multi_index[i] for i in axes)
    new_inds = np.ravel_multi_index(new_multi_index, new_shape)
    return SpArray(new_inds, self.data, new_shape)

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
    return SpArray(self.indices, self.data, shape=new_shape)

  def resize(self, new_shape):
    assert np.prod(new_shape) >= len(self.data)
    self.shape = new_shape

  def ravel(self):
    return self.reshape((np.prod(self.shape),))

  def __add__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse addition
      return NotImplemented
    # dense addition
    # TODO: optimize
    return self.toarray() + other

  def __sub__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    # TODO: optimize
    return self.toarray() - other

  def __rsub__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    # TODO: optimize
    return other - self.toarray()

  def dot(self, other):
    if self.shape[-1] != other.shape[0]:
      raise ValueError('Dimension mismatch: %s dot %s' % (
          self.shape, other.shape))
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    # TODO: optimize
    return self.toarray().dot(other)

  def _rdot(self, other):
    if other.shape[-1] != self.shape[0]:
      raise ValueError('Dimension mismatch: %s dot %s' % (
          other.shape, self.shape))
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    # TODO: optimize
    return other.dot(self.toarray())

  def _with_data(self, data):
    return SpArray(self.indices.copy(), data, self.shape)

  def __mul__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    if not np.isscalar(other):
      other = np.asanyarray(other)
      if other.ndim == 0 and other.dtype == np.object_:
        return NotImplemented  # Not interpretable as an array
      other = other.flat[self.indices]
    return self._with_data(self.data * other)

  def __rmul__(self, other):
    return self.__mul__(other)

  def _divide(self, other, true_divide=False, rdivide=False):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    if rdivide:
      # Div by zero means we won't get a sparse result, so punt.
      # TODO: true_divide
      return other / self.toarray()
    if not np.isscalar(other):
      other = np.asanyarray(other)
      if other.ndim == 0 and other.dtype == np.object_:
        # Not interpretable as an array
        return NotImplemented
      other = other.flat[self.indices]
    # TODO: true_divide
    return self._with_data(self.data / other)

  def __div__(self, other):
    return self._divide(other)

  def __rdiv__(self, other):
    return self._divide(other, rdivide=True)

  def minimum(self, other):
    if np.isscalar(other) and other >= 0:
      return self._with_data(np.minimum(self.data, other))
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # Probably won't get a sparse result
    return np.minimum(self.toarray(), other)

  def maximum(self, other):
    if np.isscalar(other) and other <= 0:
      return self._with_data(np.maximum(self.data, other))
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # Probably won't get a sparse result
    return np.maximum(self.toarray(), other)

  def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
    '''ufunc dispatcher. Mostly copied from scipy.sparse.spmatrix'''
    out = kwargs.pop('out', None)
    if method != '__call__' or kwargs:
      return NotImplemented

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
    elif func is np.divide:
      true_divide = (4/3) > 1
      result = self._divide(*without_self, true_divide=true_divide,
                            rdivide=(pos == 1))
    elif func is np.true_divide:
      result = self._divide(*without_self, true_divide=True, rdivide=(pos == 1))
    elif func in (np.minimum, np.maximum):
      result = getattr(self, func.__name__)(*without_self)
    elif func is np.absolute:
      result = abs(self)
    elif func in (np.conj, np.conjugate):
      result = self.conj()
    elif func in ss.base._ufuncs_with_fixed_point_at_zero:
      result = getattr(self, func.__name__)()
    else:
      return NotImplemented

    if out is not None:
      if not isinstance(out, SpArray) and isinstance(result, SpArray):
        out[...] = result.toarray()
      else:
        out[...] = result
      result = out

    return result

  def __getattr__(self, attr):
    if attr == 'dtype':
      return self.data.dtype
    if attr == 'A':
      return self.toarray()
    if attr == 'T':
      return self.transpose()
    if attr == 'real':
      return self._real()
    if attr == 'imag':
      return self._imag()
    if attr == 'size':
      return self.getnnz()
    if attr == 'ndim':
      return len(self.shape)
    raise AttributeError(attr + " not found")

  # The following code is completely ripped from scipy.sparse.data._data_matrix.
  # I'm including it here because I don't want to inherit from spmatrix.

  def __abs__(self):
    return self._with_data(abs(self.data))

  def _real(self):
    return self._with_data(self.data.real)

  def _imag(self):
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

  def copy(self):
    return self._with_data(self.data.copy())

  def min(self):
    # TODO: axis kwarg
    return self.data.min()

  def max(self):
    # TODO: axis kwarg
    return self.data.max()

# Add the numpy unary ufuncs for which func(0) = 0
for npfunc in ss.base._ufuncs_with_fixed_point_at_zero:
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
