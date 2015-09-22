from __future__ import absolute_import
import numpy as np
import unittest
from numpy.testing import assert_array_equal

from .test_base import BaseSpArrayTest, dense2d, dense1d


class TestIndexing(BaseSpArrayTest):

  def test_simple_indexing(self):
    for i in [0, 1, len(dense1d)-1, -1]:
      self.assertEqual(dense1d[i], self.sp1d[i])
    for i in [0, 1, len(dense2d)-1, -1]:
      for j in [0, 1, dense2d.shape[1]-1, -1]:
        self.assertEqual(dense2d[i,j], self.sp2d[i,j])
    # check out of bounds indexes
    self.assertRaises(IndexError, lambda: self.sp1d[len(dense1d)])

  def test_ellipses(self):
    assert_array_equal(dense1d[...], self.sp1d[...].toarray())
    assert_array_equal(dense2d[...], self.sp2d[...].toarray())
    # two ellipses
    assert_array_equal(dense1d[...,...], self.sp1d[...,...].toarray())
    # three ellipses is too many
    self.assertRaises(IndexError, lambda: self.sp1d[...,...,...])

  def test_partial_indexing(self):
    for i in [0, 1, len(dense2d)-1, -1]:
      assert_array_equal(dense2d[i], self.sp2d[i].toarray())
    for j in [0, 1, dense2d.shape[1]-1, -1]:
      assert_array_equal(dense2d[:,j], self.sp2d[:,j].toarray())

  def test_iter(self):
    assert_array_equal(dense1d, list(self.sp1d))
    for dense_row, sparse_row in zip(dense2d, self.sp2d):
      assert_array_equal(dense_row, sparse_row.toarray())

  def test_diagonal(self):
    assert_array_equal(dense2d.diagonal(), self.sp2d.diagonal().toarray())
    self.assertRaises(ValueError, lambda: self.sp1d.diagonal())
    self.assertRaises(ValueError, lambda: self.sp2d.diagonal(0,1,1))

  @unittest.expectedFailure
  def test_offset_diagonal(self):
    for k in [1, -1, 2, -2]:
      assert_array_equal(dense2d.diagonal(offset=k),
                         self.sp2d.diagonal(offset=k).toarray())

  @unittest.expectedFailure
  def test_slicing(self):
    assert_array_equal(dense1d[1:], self.sp1d[1:].toarray())
    assert_array_equal(dense2d[1:,1:], self.sp2d[1:,1:].toarray())

  @unittest.expectedFailure
  def test_inner_indexing(self):
    idx = [0,2]
    assert_array_equal(dense1d[idx], self.sp1d[idx].toarray())
    assert_array_equal(dense2d[idx,idx], self.sp1d[idx,idx].toarray())

  @unittest.expectedFailure
  def test_outer_indexing(self):
    ii = np.array([1,3])[:,None]
    jj = np.array([0,2])
    assert_array_equal(dense2d[ii,jj], self.sp2d[ii,jj].toarray())

  @unittest.expectedFailure
  def test_1d_boolean(self):
    idx = np.random.randint(2, size=dense1d.shape).astype(bool)
    assert_array_equal(dense1d[idx], self.sp1d[idx])
    idx = np.random.randint(2, size=dense2d.shape[0]).astype(bool)
    assert_array_equal(dense2d[idx], self.sp2d[idx])
    idx = np.random.randint(2, size=dense2d.shape[1]).astype(bool)
    assert_array_equal(dense2d[:,idx], self.sp2d[:,idx])


if __name__ == '__main__':
  unittest.main()
