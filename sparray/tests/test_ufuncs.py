import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparray import SpArray
from sparray.compat import ufuncs_with_fixed_point_at_zero

dense2d = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]], dtype=float) / 2.
dense1d = np.array([1,2,0,1,0])

sparse2d = ss.csr_matrix(dense2d)
with warnings.catch_warnings():
  warnings.simplefilter('ignore')  # Ignore efficiency warning
  sparse2d[0,1] = 0  # Add the explicit zero to match indices,data


# Check for __numpy_ufunc__
class _UFuncCheck(object):
    def __array__(self):
        return np.array([1])

    def __numpy_ufunc__(self, *a, **kwargs):
        global HAS_NUMPY_UFUNC
        HAS_NUMPY_UFUNC = True

HAS_NUMPY_UFUNC = False
np.add(_UFuncCheck(), np.array([1]))


def assert_sparse_equal(a, b):
  if hasattr(a, 'A'):
    a = a.A
  if hasattr(b, 'A'):
    b = b.A
  return assert_array_equal(a, b)


class TestUfuncsBase(unittest.TestCase):
  def setUp(self):
    dense2d_indices = [1,3,4,5,6,7,9,10,11]
    dense2d_data = [0,2,2.5,3.5,3,1,0.5,1.5,4]
    self.sp2d = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    self.sp1d = SpArray([0,1,3], [1,2,1], shape=dense1d.shape)


class TestUfuncs(TestUfuncsBase):

  def test_add_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d + b, self.sp2d + b)
    assert_array_equal(b + dense2d, b + self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d + b, self.sp2d + b)
    assert_array_equal(b + dense2d, b + self.sp2d)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_add_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.add(dense2d, b), np.add(self.sp2d, b))
    assert_array_equal(np.add(b, dense2d), np.add(b, self.sp2d))

  def test_add_spmatrix(self):
    for fmt in ('coo', 'csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d + b, self.sp2d + b)
      assert_sparse_equal(b + sparse2d, b + self.sp2d)

  def test_add_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_array_equal(dense2d + s, (self.sp2d + b).toarray())
    assert_array_equal(s + dense2d, (b + self.sp2d).toarray())

  def test_add_scalar(self):
    b = 0
    assert_array_equal(dense2d + b, (self.sp2d + b).toarray())
    assert_array_equal(b + dense2d, (b + self.sp2d).toarray())
    b = 1
    self.assertRaises(NotImplementedError, lambda: self.sp2d + b)
    self.assertRaises(NotImplementedError, lambda: b + self.sp2d)

  def test_add_inplace(self):
    self.sp2d += 0
    assert_array_equal(dense2d, self.sp2d.toarray())
    # sparray += sparray
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    self.sp2d += b
    assert_array_equal(dense2d + s, self.sp2d.toarray())

  def test_sub_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d - b, self.sp2d - b)
    assert_array_equal(b - dense2d, b - self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d - b, self.sp2d - b)
    assert_array_equal(b - dense2d, b - self.sp2d)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_sub_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.subtract(dense2d, b), np.subtract(self.sp2d, b))
    assert_array_equal(np.subtract(b, dense2d), np.subtract(b, self.sp2d))

  def test_mul_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d * b, (self.sp2d * b).toarray())
    assert_array_equal(b * dense2d, (b * self.sp2d).toarray())
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d * b, (self.sp2d * b).toarray())
    assert_array_equal(b * dense2d, (b * self.sp2d).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_ndarray_ufunc(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.multiply(dense2d, b),
                       np.multiply(self.sp2d, b).toarray())
    assert_array_equal(np.multiply(b, dense2d),
                       np.multiply(b, self.sp2d).toarray())

  def test_mul_scalar(self):
    for b in (3, -3.5, 0):
      assert_array_equal(dense2d * b, (self.sp2d * b).toarray())
      assert_array_equal(b * dense2d, (b * self.sp2d).toarray())

  def test_mul_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.multiply(b), self.sp2d * b)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_spmatrix_ufunc(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(b.multiply(sparse2d), b.multiply(self.sp2d))

  def test_mul_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_sparse_equal(s.multiply(dense2d), b * self.sp2d)
    # Test broadcasting
    for shape in [(sparse2d.shape[0], 1), (1, sparse2d.shape[1])]:
      s = ss.rand(*shape, density=0.5)
      b = SpArray.from_spmatrix(s)
      assert_sparse_equal(s.multiply(dense2d), b * self.sp2d)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_mul_sparray_ufunc(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_sparse_equal(np.multiply(dense2d, s), self.sp2d * b)
    # Test broadcasting
    for shape in [(sparse2d.shape[0], 1), (1, sparse2d.shape[1])]:
      s = ss.rand(*shape, density=0.5)
      b = SpArray.from_spmatrix(s)
      assert_sparse_equal(np.multiply(dense2d, s), self.sp2d * b)

  def test_imul(self):
    b = np.random.random(dense2d.shape)
    a = self.sp2d.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())
    b = 3
    a = self.sp2d.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())

  def test_div_ndarray(self):
    b = np.random.random(dense2d.shape)
    c = np.random.random((dense2d.shape[0], 1))  # Test broadcasting
    assert_array_almost_equal(dense2d / b, (self.sp2d / b).toarray())
    assert_array_almost_equal(dense2d / c, (self.sp2d / c).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(b / dense2d, b / self.sp2d)

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

  def test_div_scalar(self):
    c = 3
    assert_array_almost_equal(dense2d / c, (self.sp2d / c).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(c / dense2d, c / self.sp2d)

  def test_div_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      # spmatrix / spmatrix is broken in scipy, so we compare against ndarrays
      # also, np.true_divide(spmatrix, x) wraps the spmatrix in an object array
      c = b.toarray()
      with np.errstate(divide='ignore', invalid='ignore'):
        e1 = dense2d / c
        e2 = c / dense2d
        e3 = dense2d // c
        e4 = c // dense2d
      with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        assert_array_equal(e1, self.sp2d / b)
        assert_array_equal(e2, b / self.sp2d)
        assert_array_equal(e3, self.sp2d // b)
        assert_array_equal(e4, b // self.sp2d)
        # each operation may raise div by zero and/or invalid value warnings
        for w in ws:
          self.assertIn(str(w.message).split()[0], ('divide','invalid'))

  def test_div_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
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

  def test_idiv(self):
    self.sp2d /= 1
    assert_array_almost_equal(dense2d, self.sp2d.toarray())
    b = np.random.random(dense2d.shape)
    self.sp2d /= b
    assert_array_almost_equal(dense2d / b, self.sp2d.toarray())

  def test_neg(self):
    assert_array_equal(-dense2d, (-self.sp2d).toarray())

  def test_conj(self):
    assert_array_equal(dense2d.conj(), self.sp2d.conj().toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_conjugate_ufunc(self):
    assert_array_equal(np.conjugate(dense2d), np.conjugate(self.sp2d).toarray())

  def test_dot(self):
    b = np.random.random((dense2d.shape[1], dense2d.shape[0]))
    assert_array_equal(dense2d.dot(b), self.sp2d.dot(b))

    b = np.random.random(dense1d.shape[0])
    self.assertAlmostEqual(dense1d.dot(b), self.sp1d.dot(b))

    # Test bad alignment for dot
    b = np.random.random(dense1d.shape[0] + 1)
    self.assertRaises(ValueError, lambda: self.sp1d.dot(b))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_dot_ufunc(self):
    b = np.random.random((dense2d.shape[1], dense2d.shape[0]))
    # XXX: for older numpys, ndarray.dot(SpArray) wraps us in an object array.
    assert_array_equal(b.dot(dense2d), b.dot(self.sp2d))
    assert_array_equal(np.dot(dense2d, b), np.dot(self.sp2d, b))
    assert_array_equal(np.dot(b, dense2d), np.dot(b, self.sp2d))

    b = np.random.random(dense1d.shape[0])
    # XXX: for older numpys, ndarray.dot(SpArray) wraps us in an object array.
    self.assertEqual(b.dot(dense1d), b.dot(self.sp1d))

  def test_dot_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(dense2d.shape[1], dense2d.shape[0], density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.dot(b), self.sp2d.dot(b))
      # XXX: spmatrix.dot(SpArray) calls np.asarray on us,
      #  which just wraps us in an object array.
      # assert_sparse_equal(b.dot(sparse2d), b.dot(self.sp2d))

  def test_dot_sparray(self):
    m,n = dense2d.shape
    shapes = ((n,), (n,m), (2,n,m))
    for shape in shapes:
      d = np.random.random(shape)
      d.flat[np.random.randint(2, size=d.size)] = 0
      e = dense2d.dot(d)
      b = SpArray.from_ndarray(d)
      assert_array_almost_equal(e, self.sp2d.dot(b).toarray())

    d = np.random.random(dense1d.shape[0])
    d.flat[np.random.randint(2, size=d.size)] = 0
    b = SpArray.from_ndarray(d)
    self.assertEqual(dense1d.dot(d), self.sp1d.dot(b))
    self.assertEqual(d.dot(dense1d), b.dot(self.sp1d))

  def test_minmax(self):
    self.assertEqual(dense2d.min(), self.sp2d.min())
    self.assertEqual(dense2d.max(), self.sp2d.max())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b))

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_array_equal(np.minimum(dense2d, s),
                       np.minimum(self.sp2d, b).toarray())
    assert_array_equal(np.maximum(dense2d, s),
                       np.maximum(self.sp2d, b).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_array_equal(np.minimum(dense2d, b),
                         np.minimum(self.sp2d, b).toarray())
      assert_array_equal(np.maximum(dense2d, b),
                         np.maximum(self.sp2d, b).toarray())

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_minmax_imum_scalar(self):
    b = 3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b).A)
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b))
    b = -3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.sp2d, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.sp2d, b).A)

  @unittest.skipUnless(HAS_NUMPY_UFUNC, 'Requires __numpy_ufunc__ support')
  def test_abs_ufunc(self):
    assert_array_equal(np.abs(dense2d), np.abs(self.sp2d).toarray())

  def test_abs(self):
    assert_array_equal(abs(dense2d), abs(self.sp2d).toarray())

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
