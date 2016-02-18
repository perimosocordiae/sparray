import unittest
import numpy as np
from numpy.testing import assert_array_equal

import sparray.compat as sc


class CompatibilityTest(unittest.TestCase):

  def test_intersect1d_sorted(self):
    a, b = [0, 1, 4, 6, 8, 9], [2, 4, 5, 6, 7]
    assert_array_equal(sc._intersect1d_sorted(a, b), [4, 6])

    x, a_inds, b_inds = sc._intersect1d_sorted(a, b, return_inds=True)
    assert_array_equal(x, [4, 6])
    assert_array_equal(a_inds, [2, 3])
    assert_array_equal(b_inds, [1, 3])

  def test_union1d_sorted(self):
    a, b = [0, 2, 4, 8], [1, 4, 6, 8]
    expected = [0, 1, 2, 4, 6, 8]
    assert_array_equal(sc._union1d_sorted(a, b), expected)

    x, lut, a_only, b_only = sc._union1d_sorted(a, b, return_masks=True)
    assert_array_equal(x, expected)
    assert_array_equal(lut, [0, 1, 0, 2, 1, 2])
    assert_array_equal(a_only, [True, True, False, False])
    assert_array_equal(b_only, [True, False, True, False])

  def test_broadcast_to(self):
    for x in (np.array(0), np.ma.array(0)):
      for shape in [(5,6), (1200,), (2,3,4,5)]:
        res = sc._broadcast_to(x, shape, subok=True)
        self.assertEqual(res.shape, shape)
        self.assertIs(type(res), type(x))


if __name__ == '__main__':
  unittest.main()
