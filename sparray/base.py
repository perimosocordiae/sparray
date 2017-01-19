from __future__ import absolute_import
import numpy as np

__all__ = ['is_sparray']


class _BaseSparray(object):
  '''Base-class for Sparray types.'''
  __array_priority__ = 999

  def __len__(self):
    # Mimic ndarray here, instead of spmatrix
    return self.shape[0]

  def __bool__(self):
    if np.prod(self.shape) <= 1:
      return bool(self.nnz)
    raise ValueError("The truth value of an array with more than one "
                     "element is ambiguous. Use a.any() or a.all().")

  __nonzero__ = __bool__

  def __iter__(self):
    for i in range(self.shape[0]):
      yield self[i]

  def __eq__(self, other):
    return self._comparison(other, '__eq__', np.equal, '==')

  def __ne__(self, other):
    return self._comparison(other, '__ne__', np.not_equal, '!=')

  def __lt__(self, other):
    return self._comparison(other, '__lt__', np.less, '<')

  def __le__(self, other):
    return self._comparison(other, '__le__', np.less_equal, '<=')

  def __gt__(self, other):
    return self._comparison(other, '__gt__', np.greater, '>')

  def __ge__(self, other):
    return self._comparison(other, '__ge__', np.greater_equal, '>=')

  def __radd__(self, other):
    return self.__add__(other)

  def __sub__(self, other):
    return self.__add__(-other)

  def __rsub__(self, other):
    return (-self).__add__(other)

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

  def __matmul__(self, other):
    return self.dot(other)

  def _rdot(self, other):
    # This only gets called for dense other,
    # because spmatrix.dot(x) calls np.asarray(x)
    return other.dot(self.toarray())

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

  def __getattr__(self, attr):
    if attr == 'A':
      return self.toarray()
    if attr == 'T':
      return self.transpose()
    if attr == 'size':
      return self.getnnz()
    if attr == 'ndim':
      return len(self.shape)
    raise AttributeError(attr + " not found")

  def __iadd__(self, other):
    raise NotImplementedError('in-place add is not supported')

  def __isub__(self, other):
    raise NotImplementedError('in-place subtract is not supported')

  def conjugate(self):
    return self.conj()


def is_sparray(x):
  return isinstance(x, _BaseSparray)
