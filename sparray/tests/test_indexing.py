import numpy as np
import unittest
from numpy.testing import assert_array_equal

from .test_base import (
    BaseSparrayTest, dense1d, dense2d, sparse2d, dense3d, assert_sparse_equal)


class TestIndexing(BaseSparrayTest):

  def test_simple_indexing(self):
    for i in [0, 1, len(dense1d) - 1, -1]:
      self.assertEqual(dense1d[i], self.sp1d[i])
    for i in [0, 1, len(dense2d) - 1, -1]:
      for j in [0, 1, dense2d.shape[1] - 1, -1]:
        self.assertEqual(dense2d[i,j], self.sp2d[i,j])
    # check out of bounds indexes
    self.assertRaises(IndexError, lambda: self.sp1d[len(dense1d)])

  def test_ellipses(self):
    assert_array_equal(dense1d[...], self.sp1d[...].toarray())
    assert_array_equal(dense2d[...], self.sp2d[...].toarray())
    # two ellipses is an error in recent numpy
    self.assertRaises(IndexError, lambda: self.sp1d[...,...])
    # three ellipses is too many for any numpy
    self.assertRaises(IndexError, lambda: self.sp1d[...,...,...])

  def test_partial_indexing(self):
    for i in [0, 1, len(dense2d) - 1, -1]:
      assert_array_equal(dense2d[i], self.sp2d[i].toarray())
    for j in [0, 1, dense2d.shape[1] - 1, -1]:
      assert_array_equal(dense2d[:,j], self.sp2d[:,j].toarray())

  def test_iter(self):
    assert_array_equal(dense1d, list(self.sp1d))
    for dense_row, sparse_row in zip(dense2d, self.sp2d):
      assert_array_equal(dense_row, sparse_row.toarray())

  def test_diagonal(self):
    assert_array_equal(dense2d.diagonal(), self.sp2d.diagonal().toarray())
    self.assertRaises(ValueError, lambda: self.sp1d.diagonal())
    self.assertRaises(ValueError, lambda: self.sp2d.diagonal(0,1,1))

  def test_offset_diagonal(self):
    for k in [1, -1, 2, -2, 3, -3, 4, -4]:
      assert_sparse_equal(dense2d.diagonal(offset=k),
                          self.sp2d.diagonal(offset=k),
                          err_msg='Mismatch for k=%d' % k)

  def test_slicing(self):
    assert_array_equal(dense1d[1:], self.sp1d[1:].toarray())
    assert_array_equal(dense2d[1:,1:], self.sp2d[1:,1:].toarray())

  def test_mixed_fancy_indexing(self):
    idx = [0,2]
    assert_array_equal(dense2d[:,idx], self.sp2d[:,idx].toarray())
    assert_array_equal(dense2d[idx,:], self.sp2d[idx,:].toarray())

    assert_array_equal(dense3d[idx,:,idx], self.sp3d[idx,:,idx].toarray())
    assert_array_equal(dense3d[[1],:,idx], self.sp3d[[1],:,idx].toarray())
    assert_array_equal(dense3d[:,[1],idx], self.sp3d[:,[1],idx].toarray())
    assert_array_equal(dense3d[idx,[1],:], self.sp3d[idx,[1],:].toarray())

    assert_array_equal(dense3d[2,:,idx], self.sp3d[2,:,idx].toarray())
    assert_array_equal(dense3d[:,1,idx], self.sp3d[:,1,idx].toarray())
    assert_array_equal(dense3d[idx,1,:], self.sp3d[idx,1,:].toarray())

  def test_inner_indexing(self):
    idx = [0,2]
    assert_array_equal(dense1d[idx], self.sp1d[idx].toarray())
    assert_array_equal(dense2d[idx,idx], self.sp2d[idx,idx].toarray())

  @unittest.expectedFailure
  def test_outer_indexing(self):
    ii = np.array([1,3])[:,None]
    jj = np.array([0,2])
    assert_array_equal(dense2d[ii,jj], self.sp2d[ii,jj].toarray())

  def test_1d_boolean(self):
    for idx in ([0,0,0,0,0], [1,0,0,0,0], [0,1,1,0,0], [1,1,1,1,1]):
      idx = np.array(idx, dtype=bool)
      assert_array_equal(dense1d[idx], self.sp1d[idx].toarray())
    for idx in ([0,0,0,0], [1,0,0,0], [0,1,1,0], [1,1,1,1]):
      idx = np.array(idx, dtype=bool)
      assert_array_equal(dense2d[idx], self.sp2d[idx].toarray())
    for idx in ([0,0,0], [1,0,0], [0,1,1], [1,1,1]):
      idx = np.array(idx, dtype=bool)
      assert_array_equal(dense2d[:,idx], self.sp2d[:,idx].toarray())

  def test_nd_boolean(self):
    idx = ((np.arange(12) % 3) == 0).reshape(dense2d.shape)
    assert_array_equal(dense2d[idx], self.sp2d[idx])
    idx[:,:] = False
    assert_array_equal(dense2d[idx], self.sp2d[idx])
    idx[:,:] = True
    assert_array_equal(dense2d[idx], self.sp2d[idx])


class TestAssignment(BaseSparrayTest):
  def test_scalar_assignment_in_structure(self):
    a = self.sp1d.copy()
    a_dense = dense1d.copy()
    a[3] = 99
    a_dense[3] = 99
    assert_array_equal(a_dense, a.toarray())

    a = self.sp2d.copy()
    a_dense = dense2d.copy()
    a[3,1] = 99
    a_dense[3,1] = 99
    assert_array_equal(a_dense, a.toarray())

  def test_scalar_assignment_not_in_structure(self):
    a = self.sp1d.copy()
    a_dense = dense1d.copy()
    a[2] = 99
    a_dense[2] = 99
    assert_array_equal(a_dense, a.toarray())

    a = self.sp2d.copy()
    a_dense = dense2d.copy()
    a[0,2] = 99
    a_dense[0,2] = 99
    assert_array_equal(a_dense, a.toarray())

  def test_subarray_assignment(self):
    a = self.sp1d.copy()
    a_dense = dense1d.copy()
    a[:3] = 99
    a_dense[:3] = 99
    assert_array_equal(a_dense, a.toarray())

    a = self.sp2d.copy()
    a_dense = dense2d.copy()
    a[:2,:2] = 99
    a_dense[:2,:2] = 99
    assert_array_equal(a_dense, a.toarray())

  def test_setdiag(self):
    for k in [0, 1, -1, 2, -2]:
      a = self.sp2d.copy()
      a_sparse = sparse2d.copy()
      a.setdiag(99, offset=k)
      a_sparse.setdiag(99, k=k)
      assert_sparse_equal(a_sparse, a, err_msg='Mismatch for k=%d' % k)


if __name__ == '__main__':
  unittest.main()
