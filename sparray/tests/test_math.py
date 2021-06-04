import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparray import FlatSparray
from sparray.compat import ufuncs_with_fixed_point_at_zero

from .test_base import (
    assert_sparse_equal, assert_sparse_almost_equal,
    BaseSparrayTest, dense2d, sparse2d, dense1d
)


class TestMath(BaseSparrayTest):
  def setUp(self):
    BaseSparrayTest.setUp(self)
    # add complex test data
    d = dense2d * 1j
    self.pairs.append((d, FlatSparray.from_ndarray(d)))

  def test_add_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d + b, self.sp2d + b)
    assert_array_equal(b + dense2d, b + self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d + b, self.sp2d + b)
    assert_array_equal(b + dense2d, b + self.sp2d)

  def test_add_spmatrix(self):
    for fmt in ('coo', 'csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d + b, self.sp2d + b)
      assert_sparse_equal(b + sparse2d, b + self.sp2d)

  def test_add_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_array_equal(dense2d + s, (self.sp2d + b).toarray())
    assert_array_equal(s + dense2d, (b + self.sp2d).toarray())
    # Test broadcasting
    s = ss.rand(sparse2d.shape[0], 1, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    # XXX using .A to avoid a scipy.sparse bug
    assert_sparse_equal(dense2d + s.A, self.sp2d + b)
    assert_sparse_equal(s.A + dense2d, b + self.sp2d)

  def test_add_scalar(self):
    b = 0
    self._same_op(lambda x: x + b, assert_sparse_equal)
    self._same_op(lambda x: b + x, assert_sparse_equal)
    b = 1
    self._same_op(lambda x: x + b, assert_array_equal)
    self._same_op(lambda x: b + x, assert_array_equal)

  def test_sub_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d - b, self.sp2d - b)
    assert_array_equal(b - dense2d, b - self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d - b, self.sp2d - b)
    assert_array_equal(b - dense2d, b - self.sp2d)

  def test_mul_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d * b, (self.sp2d * b).toarray())
    assert_array_equal(b * dense2d, (b * self.sp2d).toarray())
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d * b, (self.sp2d * b).toarray())
    assert_array_equal(b * dense2d, (b * self.sp2d).toarray())

  def test_mul_scalar(self):
    for b in (3, -3.5, 0):
      self._same_op(lambda x: x * b, assert_sparse_equal)
      self._same_op(lambda x: b * x, assert_sparse_equal)

  def test_mul_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.multiply(b), self.sp2d * b)
      assert_sparse_equal(sparse2d.multiply(b), self.sp2d.multiply(b))

  def test_mul_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_sparse_equal(s.multiply(dense2d), b * self.sp2d)
    # Test broadcasting
    for shape in [(sparse2d.shape[0], 1), (1, sparse2d.shape[1])]:
      s = ss.rand(*shape, density=0.5)
      b = FlatSparray.from_spmatrix(s)
      assert_sparse_equal(s.multiply(dense2d), b * self.sp2d)

  def test_imul(self):
    b = np.random.random(dense2d.shape)
    a = self.sp2d.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())
    b = 3
    a = self.sp2d.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())

  def test_div_scalar(self):
    self._same_op(lambda x: x / 3, assert_sparse_almost_equal)
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(3 / dense2d, 3 / self.sp2d)

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
    b = FlatSparray.from_spmatrix(s)
    # spmatrix / spmatrix is broken in scipy, so we compare against ndarrays
    c = s.toarray()
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

  def test_div_ndarray(self):
    b = np.random.random(dense2d.shape)
    c = np.random.random((dense2d.shape[0], 1))  # Test broadcasting
    assert_array_almost_equal(dense2d / b, (self.sp2d / b).toarray())
    assert_array_almost_equal(dense2d / c, (self.sp2d / c).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(b / dense2d, b / self.sp2d)

  def test_idiv(self):
    self.sp2d /= 1
    assert_array_almost_equal(dense2d, self.sp2d.toarray())
    b = np.random.random(dense2d.shape)
    self.sp2d /= b
    assert_array_almost_equal(dense2d / b, self.sp2d.toarray())

  def test_neg(self):
    self._same_op(lambda x: -x, assert_sparse_equal)

  def test_conj(self):
    self._same_op(lambda x: x.conj(), assert_sparse_equal)
    self._same_op(lambda x: x.conjugate(), assert_sparse_equal)

  def test_pow(self):
    self._same_op(lambda x: x**2, assert_sparse_equal)
    with np.errstate(divide='ignore', invalid='ignore'):
      self._same_op(lambda x: x**-1.5, assert_array_almost_equal)
      self._same_op(lambda x: x**0, assert_array_almost_equal)

  def test_dot_ndarray(self):
    b = np.random.random(dense2d.shape[::-1])
    assert_array_equal(dense2d.dot(b), self.sp2d.dot(b))

    b = np.random.random(dense1d.shape[0])
    self.assertAlmostEqual(dense1d.dot(b), self.sp1d.dot(b))

    # Test bad alignment for dot
    b = np.random.random(dense1d.shape[0] + 1)
    self.assertRaises(ValueError, lambda: self.sp1d.dot(b))

  def test_dot_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(dense2d.shape[1], dense2d.shape[0], density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.dot(b), self.sp2d.dot(b))
      # XXX: spmatrix.dot(FlatSparray) calls np.asarray on us,
      #  which just wraps us in an object array.
      # assert_sparse_equal(b.dot(sparse2d), b.dot(self.sp2d))

  def test_dot_sparray(self):
    m,n = dense2d.shape
    shapes = ((n,), (n,m), (2,n,m))
    for shape in shapes:
      d = np.random.random(shape)
      d.flat[np.random.randint(2, size=d.size)] = 0
      e = dense2d.dot(d)
      b = FlatSparray.from_ndarray(d)
      assert_array_almost_equal(e, self.sp2d.dot(b).toarray())

    d = np.random.random(dense1d.shape[0])
    d.flat[np.random.randint(2, size=d.size)] = 0
    b = FlatSparray.from_ndarray(d)
    self.assertEqual(dense1d.dot(d), self.sp1d.dot(b))
    self.assertEqual(d.dot(dense1d), b.dot(self.sp1d))

  def test_minmax(self):
    self.assertEqual(dense2d.min(), self.sp2d.min())
    self.assertEqual(dense2d.max(), self.sp2d.max())

  def test_minmax_imum_ndarray(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.minimum(dense2d, b), self.sp2d.minimum(b))
    assert_array_equal(np.maximum(dense2d, b), self.sp2d.maximum(b))

  def test_minmax_imum_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_sparse_equal(s.minimum(dense2d), b.minimum(self.sp2d))
    assert_sparse_equal(s.maximum(dense2d), b.maximum(self.sp2d))

  def test_minmax_imum_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(b.minimum(dense2d), self.sp2d.minimum(b))
      assert_sparse_equal(b.maximum(dense2d), self.sp2d.maximum(b))

  def test_minmax_imum_scalar(self):
    b = 3
    assert_array_equal(np.minimum(dense2d, b), self.sp2d.minimum(b).A)
    assert_array_equal(np.maximum(dense2d, b), self.sp2d.maximum(b))
    b = -3
    assert_array_equal(np.minimum(dense2d, b), self.sp2d.minimum(b))
    assert_array_equal(np.maximum(dense2d, b), self.sp2d.maximum(b).A)

  def test_abs(self):
    self._same_op(abs, assert_sparse_equal)

  def test_sum(self):
    # axis=None
    self._same_op(lambda x: x.sum(), self.assertEqual)
    # axis=0
    self.assertEqual(dense1d.sum(axis=0), self.sp1d.sum(axis=0))
    assert_sparse_equal(dense2d.sum(axis=0), self.sp2d.sum(axis=0))
    # axis=1
    assert_sparse_equal(dense2d.sum(axis=1), self.sp2d.sum(axis=1))

  def test_mean(self):
    # axis=None, uses assert_array_almost_equal to handle NaN values
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', category=RuntimeWarning)
      self._same_op(lambda x: x.mean(), assert_array_almost_equal)
    # axis=0
    self.assertEqual(dense1d.mean(axis=0), self.sp1d.mean(axis=0))
    assert_sparse_equal(dense2d.mean(axis=0), self.sp2d.mean(axis=0))
    # axis=1
    assert_sparse_equal(dense2d.mean(axis=1), self.sp2d.mean(axis=1))

  def test_fixed_point_at_zero_methods(self):
    with np.errstate(invalid='ignore', divide='ignore'):
      for ufunc in ufuncs_with_fixed_point_at_zero:
        method = getattr(self.sp2d, ufunc.__name__)
        assert_array_equal(ufunc(dense2d), method().toarray())

  def test_comparison_scalar(self):
    # equal
    self._same_op(lambda x: x == 1, assert_sparse_equal)
    self._same_op(lambda x: x == 0, assert_array_equal)
    # not equal
    self._same_op(lambda x: x != 0, assert_sparse_equal)
    self._same_op(lambda x: x != 1, assert_array_equal)
    # less than
    self._same_op(lambda x: x < -1, assert_sparse_equal)
    self._same_op(lambda x: x < 1, assert_array_equal)
    # less equal
    self._same_op(lambda x: x <= -1, assert_sparse_equal)
    self._same_op(lambda x: x <= 0, assert_array_equal)
    # greater than
    self._same_op(lambda x: x > 1, assert_sparse_equal)
    self._same_op(lambda x: x > -1, assert_array_equal)
    # greater equal
    self._same_op(lambda x: x >= 1, assert_sparse_equal)
    self._same_op(lambda x: x >= 0, assert_array_equal)

  def test_eq_nonscalar(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d == b, self.sp2d == b)
    assert_array_equal(b == dense2d, b == self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d == b, self.sp2d == b)
    assert_array_equal(b == dense2d, b == self.sp2d)
    # Test spmatrix
    for fmt in ('coo', 'csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d == b, self.sp2d == b)
      # spmatrix doesn't know how to handle us
      # assert_sparse_equal(b == sparse2d, b == self.sp2d)

  def test_comparison_nonscalar(self):
    # Only testing < here, because the other ops use the same code.
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d < b, self.sp2d < b)
    assert_array_equal(b < dense2d, b < self.sp2d)
    # Test broadcasting
    b = np.random.random((dense2d.shape[0], 1))
    assert_array_equal(dense2d < b, self.sp2d < b)
    assert_array_equal(b < dense2d, b < self.sp2d)
    # Test sparray
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = FlatSparray.from_spmatrix(s)
    assert_sparse_equal(sparse2d < s, self.sp2d < b)
    assert_sparse_equal(s < sparse2d, b < self.sp2d)
    # Test spmatrix
    for fmt in ('coo', 'csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d < b, self.sp2d < b)
      # spmatrix doesn't know how to handle us
      # assert_sparse_equal(b < sparse2d, b < self.sp2d)


if __name__ == '__main__':
  unittest.main()
