import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparray import FlatSparray
from sparray.compat import ufuncs_with_fixed_point_at_zero

from .test_base import (
    assert_sparse_equal, BaseSparrayTest, dense2d, sparse2d, dense1d
)


# Check for __numpy_ufunc__
class _UFuncCheck:
  def __array__(self):
    return np.array([1])

  def __numpy_ufunc__(self, *a, **kwargs):
    global HAS_NUMPY_UFUNC
    HAS_NUMPY_UFUNC = True


HAS_NUMPY_UFUNC = False
np.add(_UFuncCheck(), np.array([1]))


class TestUfuncs(BaseSparrayTest):

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_add_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.add(dense2d, b), np.add(self.sp2d, b))
    assert_array_equal(np.add(b, dense2d), np.add(b, self.sp2d))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_sub_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.subtract(dense2d, b), np.subtract(self.sp2d, b))
    assert_array_equal(np.subtract(b, dense2d), np.subtract(b, self.sp2d))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.multiply(dense2d, b),
                       np.multiply(self.sp2d, b).toarray())
    assert_array_equal(np.multiply(b, dense2d),
                       np.multiply(b, self.sp2d).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_spmatrix_ufunc(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(b.multiply(sparse2d), b.multiply(self.sp2d))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_sparray_ufunc(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_sparse_equal(np.multiply(dense2d, s), self.sp2d * b)
    # Test broadcasting
    for shape in [(sparse2d.shape[0], 1), (1, sparse2d.shape[1])]:
      s = ss.rand(*shape, density=0.5)
      b = FlatSparray.from_spmatrix(s)
      assert_sparse_equal(np.multiply(dense2d, s), self.sp2d * b)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_div_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_almost_equal(np.divide(dense2d, b),
                              np.divide(self.sp2d, b).toarray())
    assert_array_almost_equal(np.true_divide(dense2d, b),
                              np.true_divide(self.sp2d, b).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(np.divide(b, dense2d), np.divide(b, self.sp2d))
      assert_array_almost_equal(np.true_divide(b, dense2d),
                                np.true_divide(b, self.sp2d))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_div_sparray_ufunc(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    # spmatrix / spmatrix is broken in scipy, so we compare against ndarrays
    c = s.toarray()
    with np.errstate(divide='ignore', invalid='ignore'):
      e1 = np.true_divide(dense2d, c)
      e2 = np.true_divide(c, dense2d)
      e3 = np.floor_divide(dense2d, c)
      e4 = np.floor_divide(c, dense2d)
    with warnings.catch_warnings(record=True) as ws:
      warnings.simplefilter("always")
      assert_array_equal(e1, np.true_divide(self.sp2d, b))
      assert_array_equal(e2, np.true_divide(b, self.sp2d))
      assert_array_equal(e3, np.floor_divide(self.sp2d, b))
      assert_array_equal(e4, np.floor_divide(b, self.sp2d))
      # each operation may raise div by zero and/or invalid value warnings
      for w in ws:
        self.assertIn(str(w.message).split()[0], ('divide','invalid'))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_conjugate_ufunc(self):
    assert_array_equal(np.conjugate(dense2d), np.conjugate(self.sp2d).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_dot_ufunc(self):
    b = np.random.random((dense2d.shape[1], dense2d.shape[0]))
    # XXX: in older numpy, ndarray.dot(FlatSparray) wraps us in an object array.
    assert_array_equal(b.dot(dense2d), b.dot(self.sp2d))
    assert_array_equal(np.dot(dense2d, b), np.dot(self.sp2d, b))
    assert_array_equal(np.dot(b, dense2d), np.dot(b, self.sp2d))

    b = np.random.random(dense1d.shape[0])
    # XXX: in older numpy, ndarray.dot(FlatSparray) wraps us in an object array.
    self.assertEqual(b.dot(dense1d), b.dot(self.sp1d))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_sparray_ufunc(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_array_equal(np.minimum(dense2d, s),
                       np.minimum(self.sp2d, b).toarray())
    assert_array_equal(np.maximum(dense2d, s),
                       np.maximum(self.sp2d, b).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_spmatrix_ufunc(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_array_equal(np.minimum(dense2d, b),
                         np.minimum(self.sp2d, b).toarray())
      assert_array_equal(np.maximum(dense2d, b),
                         np.maximum(self.sp2d, b).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_scalar_ufunc(self):
    b = 3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b).A)
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b))
    b = -3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b).A)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_abs_ufunc(self):
    assert_array_equal(np.abs(dense2d), np.abs(self.sp2d).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_fixed_point_at_zero_ufuncs(self):
    with np.errstate(invalid='ignore', divide='ignore'):
      for ufunc in ufuncs_with_fixed_point_at_zero:
        assert_array_equal(ufunc(dense2d), ufunc(self.sp2d).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_not_implemented_ufunc(self):
    self.assertRaises(TypeError, np.log, self.sp2d)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_dense_out_kwarg(self):
    b = 3
    out1 = np.zeros_like(dense2d)
    out2 = np.zeros_like(dense2d)
    np.multiply(dense2d, b, out=out1)
    res = np.multiply(self.sp2d, b, out=out2)
    self.assertIs(res, out2)
    assert_array_equal(out1, out2)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_sparray_out_kwarg(self):
    res = np.add(self.sp2d, 0, out=self.sp2d)
    self.assertIs(res, self.sp2d)
    assert_array_equal(dense2d, self.sp2d.toarray())


if __name__ == '__main__':
  unittest.main()
