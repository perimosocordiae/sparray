import numpy as np
import scipy.sparse as ss


class SpArray(object):
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
