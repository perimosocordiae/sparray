# This file should only be imported in versions 3.5+
import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal

from .test_base import BaseSparrayTest, dense1d, dense2d

# Check for numpy 1.10+
HAS_NUMPY_MATMUL = True
try:
  np.arange(3) @ np.arange(3)
except TypeError:
  HAS_NUMPY_MATMUL = False


class TestMatmulOperator(BaseSparrayTest):

  @unittest.skipUnless(HAS_NUMPY_MATMUL, 'Requires numpy with @ support')
  def test_matmul(self):
    b = np.random.random((dense2d.shape[1], dense2d.shape[0]))
    assert_array_almost_equal(dense2d @ b, self.sp2d @ b)

    b = np.random.random(dense1d.shape[0])
    self.assertAlmostEqual(dense1d @ b, self.sp1d @ b)

    # Test bad alignment for dot
    b = np.random.random(dense1d.shape[0] + 1)
    self.assertRaises(ValueError, lambda: self.sp1d @ b)
