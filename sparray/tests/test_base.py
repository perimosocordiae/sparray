import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal
from sparray import SpArray

dense2d = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]], dtype=float) / 2.
dense2d_indices = [1,3,4,5,6,7,9,10,11]
dense2d_data = [0,2,2.5,3.5,3,1,0.5,1.5,4]

sparse2d = ss.csr_matrix(dense2d)
with warnings.catch_warnings():
  warnings.simplefilter('ignore')  # Ignore efficiency warning
  sparse2d[0,1] = 0  # Add the explicit zero to match indices,data

dense1d = np.arange(5) - 2
dense1d_indices = [0,1,3,4]
dense1d_data = [-2,-1,1,2]


def assert_sparse_equal(a, b):
  if hasattr(a, 'A'):
    a = a.A
  if hasattr(b, 'A'):
    b = b.A
  return assert_array_equal(a, b)


class BaseSpArrayTest(unittest.TestCase):
  '''Base class that other tests can inherit from'''
  def setUp(self):
    self.sp1d = SpArray(dense1d_indices, dense1d_data, shape=dense1d.shape)
    self.sp2d = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    self.pairs = [
        (dense1d, self.sp1d),
        (dense2d, self.sp2d),
        (np.array([]), SpArray([],[],shape=(0,))),
        (np.zeros((1,2,3)), SpArray([],[],shape=(1,2,3))),
    ]

  def _same_op(self, op, assertFn):
    for d, s in self.pairs:
      assertFn(op(s), op(d))


class TestCreation(unittest.TestCase):
  def test_init(self):
    a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    assert_array_equal(a.toarray(), dense2d)
    b = SpArray(dense1d_indices, dense1d_data, shape=dense1d.shape)
    assert_array_equal(b.toarray(), dense1d)
    b = SpArray(dense1d_indices, dense1d_data)
    assert_array_equal(b.toarray(), dense1d)

  def test_from_ndarray(self):
    for arr in (dense2d, dense1d):
      a = SpArray.from_ndarray(arr)
      assert_array_equal(a.toarray(), arr)

  def test_from_spmatrix(self):
    for fmt in ('csr', 'csc', 'coo', 'dok', 'lil', 'dia'):
      a = SpArray.from_spmatrix(sparse2d.asformat(fmt))
      assert_array_equal(a.toarray(), dense2d,
                         'Failed to convert from %s' % fmt.upper())


if __name__ == '__main__':
  unittest.main()
