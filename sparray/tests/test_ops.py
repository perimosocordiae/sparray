from __future__ import absolute_import
import unittest
import numpy as np
from numpy.testing import assert_array_equal

from .test_base import (
    assert_sparse_equal, BaseSparrayTest, dense2d,
    dense1d_indices, dense1d_data
)


class TestOps(BaseSparrayTest):
  def test_tocoo(self):
    assert_array_equal(self.sp2d.tocoo().A, dense2d)

  def test_repr(self):
    for _, s in self.pairs:
      self.assertRegexpMatches(repr(s), r'<\(.*?\)-FlatSparray')

  def test_str(self):
    expected = '\n'.join('  (%d,)\t%d' % x
                         for x in zip(dense1d_indices, dense1d_data))
    self.assertEqual(str(self.sp1d), expected)

  def test_resize(self):
    self.sp2d.resize((5,3))
    assert_array_equal(self.sp2d.A, np.vstack((dense2d,np.zeros((1,3)))))
    self.sp2d.resize((12,))
    assert_array_equal(self.sp2d.A, dense2d.ravel())

  def test_reshape(self):
    b = self.sp2d.reshape((6,2))
    self.assertIsNot(self.sp2d, b)
    assert_array_equal(self.sp2d.A, dense2d)
    assert_array_equal(b.toarray(), dense2d.reshape((6,2)))
    b = self.sp2d.reshape((2,-1))
    assert_array_equal(b.toarray(), dense2d.reshape((2,-1)))

  def test_ravel(self):
    self._same_op(lambda x: x.ravel(), assert_sparse_equal)
    # Make sure we're not modifying anything in-place
    b = self.sp2d.ravel()
    self.assertIsNot(self.sp2d, b)
    assert_array_equal(self.sp2d.A, dense2d)

  def test_astype(self):
    self._same_op(lambda x: x.astype(np.float32), assert_sparse_equal)
    # Make sure we're not modifying anything in-place
    b = self.sp2d.astype(np.float32)
    self.assertIsNot(self.sp2d, b)
    self.assertIs(self.sp2d.dtype, dense2d.dtype)

  def test_copy(self):
    b = self.sp2d.copy()
    self.assertIsNot(self.sp2d, b)
    assert_array_equal(self.sp2d.A, b.A)
    b.data[2] *= 3  # modify b's members
    assert_array_equal(self.sp2d.A, dense2d)

  def test_transpose(self):
    self._same_op(lambda x: x.transpose(), assert_sparse_equal)
    # test non-default axes
    assert_array_equal(self.sp2d.transpose(0,1).A,
                       dense2d.transpose(0,1))
    assert_array_equal(self.sp2d.transpose((0,1)).A,
                       dense2d.transpose((0,1)))

  def test_nonzero(self):
    self._same_op(lambda x: x.nonzero(), assert_array_equal)

  def test_len(self):
    self._same_op(len, self.assertEqual)

  def test_bool(self):
    for d, s in self.pairs:
      try:
         res = bool(d)
      except ValueError:
         self.assertRaises(ValueError, bool, s)
      else:
         self.assertEqual(bool(s), res)

  def test_properties(self):
    self._same_op(lambda x: x.dtype, self.assertEqual)
    self._same_op(lambda x: x.shape, self.assertEqual)
    self._same_op(lambda x: x.ndim, self.assertEqual)
    # size means something different for sparse objects
    self._same_op(lambda x: x.size, self.assertLessEqual)

  def test_transform_attrs(self):
    for attr in ('T', 'real', 'imag'):
      self._same_op(lambda x: getattr(x, attr), assert_sparse_equal)

  def test_nonexistent_attr(self):
    self.assertRaises(AttributeError, lambda: getattr(self.sp2d, 'xxxx'))


if __name__ == '__main__':
  unittest.main()
