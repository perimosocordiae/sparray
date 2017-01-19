import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparray import FlatSparray

dense2d = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]], dtype=float) / 2.
dense2d_indices = [1,3,4,5,6,7,9,10,11]
dense2d_data = [0,2,2.5,3.5,3,1,0.5,1.5,4]

# Ignore efficiency warnings
warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)
sparse2d = ss.csr_matrix(dense2d)
sparse2d[0,1] = 0  # Add the explicit zero to match indices,data

dense1d = np.arange(5) - 2
dense1d_indices = [0,1,3,4]
dense1d_data = [-2,-1,1,2]

dense3d = np.arange(24).reshape((3,2,4))[::-1]
dense3d[[0,2],:,2:] = 0
dense3d[1,0,:] = 0


def assert_sparse_equal(a, b, err_msg=''):
  if hasattr(a, 'A'):
    a = a.A
  if hasattr(b, 'A'):
    b = b.A
  return assert_array_equal(a, b, err_msg=err_msg)


def assert_sparse_almost_equal(a, b, err_msg=''):
  if hasattr(a, 'A'):
    a = a.A
  if hasattr(b, 'A'):
    b = b.A
  return assert_array_almost_equal(a, b, err_msg=err_msg)


class BaseSparrayTest(unittest.TestCase):
  '''Base class that other tests can inherit from'''
  def setUp(self):
    self.sp1d = FlatSparray(dense1d_indices, dense1d_data, shape=dense1d.shape)
    self.sp2d = FlatSparray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    self.sp3d = FlatSparray.from_ndarray(dense3d)
    self.pairs = [
        (dense1d, self.sp1d),
        (dense2d, self.sp2d),
        (np.array([]), FlatSparray([],[],shape=(0,))),
        (np.zeros((1,2,3)), FlatSparray([],[],shape=(1,2,3))),
        (dense3d, self.sp3d),
    ]

  def _same_op(self, op, assertFn):
    for d, s in self.pairs:
      assertFn(op(s), op(d))


class TestCreation(unittest.TestCase):
  def test_init(self):
    a = FlatSparray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    assert_array_equal(a.toarray(), dense2d)
    b = FlatSparray(dense1d_indices, dense1d_data, shape=dense1d.shape)
    assert_array_equal(b.toarray(), dense1d)
    b = FlatSparray(dense1d_indices, dense1d_data)
    assert_array_equal(b.toarray(), dense1d)

  def test_from_ndarray(self):
    for arr in (dense2d, dense1d, dense3d):
      a = FlatSparray.from_ndarray(arr)
      assert_array_equal(a.toarray(), arr)

  def test_from_spmatrix(self):
    for fmt in ('csr', 'csc', 'coo', 'dok', 'lil', 'dia'):
      a = FlatSparray.from_spmatrix(sparse2d.asformat(fmt))
      assert_array_equal(a.toarray(), dense2d,
                         'Failed to convert from %s' % fmt.upper())


if __name__ == '__main__':
  unittest.main()
