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
    indices = np.array(indices).ravel()
    data = np.array(data).ravel()
    assert len(indices) == len(data), '# inds (%d) != # data (%d)' % (
        len(indices), len(data))
    if shape is None:
      self.shape = data.shape
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
  def from_sparse(mat):
    inds = np.ravel_multi_index(mat.nonzero(), mat.shape)
    return SpArray(inds, mat.data, shape=mat.shape)

  def toarray(self):
    a = np.zeros(self.shape, dtype=self.data.dtype)
    a.flat[self.indices] = self.data
    return a

  def tocoo(self):
    assert len(self.shape) == 2
    row,col = np.unravel_index(self.indices, self.shape)
    return ss.coo_matrix((self.data, (row, col)), shape=self.shape)

  def __repr__(self):
    return '<%s-SpArray of type %s\n\twith %d stored elements>' % (
        self.shape, self.data.dtype, len(self.indices))

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
    return self.toarray() + other

  def __sub__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    return self.toarray() - other

  def __rsub__(self, other):
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    # dense version
    return other - self.toarray()

  def _with_data(self, data, copy=True):
    if copy:
      return SpArray(self.indices.copy(), data, self.shape)
    return SpArray(self.indices, data, self.shape)

  def __mul__(self, other):
    if np.isscalar(other):
      return self._mul_scalar(other)
    if isinstance(other, SpArray) or ss.issparse(other):
      # TODO: sparse version
      return NotImplemented
    other = np.asanyarray(other)
    if other.ndim == 0 and other.dtype == np.object_:
      # Not interpretable as an array
      return NotImplemented
    # TODO: dense version
    return NotImplemented

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

  # The following code is completely ripped from scipy.sparse.data._data_matrix.
  # I'm including it here because I don't want to inherit from spmatrix.

  def _get_dtype(self):
      return self.data.dtype

  def _set_dtype(self,newtype):
      self.data.dtype = newtype
  dtype = property(fget=_get_dtype,fset=_set_dtype)

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
      return self._with_data(self.data.copy(), copy=True)

  def _mul_scalar(self, other):
      return self._with_data(self.data * other)


# Add the numpy unary ufuncs for which func(0) = 0
for npfunc in ss.base._ufuncs_with_fixed_point_at_zero:
  name = npfunc.__name__

  def _create_method(op):
    def method(self):
      result = op(self.data)
      x = self._with_data(result, copy=True)
      return x

    method.__doc__ = ("Element-wise %s.\n\n"
                      "See numpy.%s for more information." % (name, name))
    method.__name__ = name
    return method

  setattr(SpArray, name, _create_method(npfunc))
