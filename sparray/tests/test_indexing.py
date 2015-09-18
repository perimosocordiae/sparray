from __future__ import absolute_import
import numpy as np
import unittest
from numpy.testing import assert_array_equal

from .test_ufuncs import TestUfuncsBase, dense2d, dense1d


class TestIndexing(TestUfuncsBase):

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


if __name__ == '__main__':
  unittest.main()
