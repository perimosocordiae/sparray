from __future__ import division, absolute_import
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from .test_ufuncs import TestUfuncsBase, dense2d


class TestTrueDivision(TestUfuncsBase):

  def test_truediv(self):
    c = 3
    assert_array_almost_equal(dense2d / c, (self.sp2d / c).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(c / dense2d, c / self.sp2d)

  def test_itruediv(self):
    self.sp2d /= 1
    assert_array_almost_equal(dense2d, self.sp2d.toarray())
    b = np.random.random(dense2d.shape)
    self.sp2d /= b
    assert_array_almost_equal(dense2d / b, self.sp2d.toarray())


if __name__ == '__main__':
  unittest.main()
