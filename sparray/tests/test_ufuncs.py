import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sparray import SpArray

dense2d = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]], dtype=float) / 2.

sparse2d = ss.csr_matrix(dense2d)
with warnings.catch_warnings():
  warnings.simplefilter('ignore')  # Ignore efficiency warning
  sparse2d[0,1] = 0  # Add the explicit zero to match indices,data


def assert_sparse_equal(a, b):
  return assert_array_equal(a.toarray(), b.toarray())


class TestUfuncs(unittest.TestCase):
  def setUp(self):
    dense2d_indices = [1,3,4,5,6,7,9,10,11]
    dense2d_data = [0,2,2.5,3.5,3,1,0.5,1.5,4]
    self.a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)

  def test_add_dense(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d + b, self.a + b)
    assert_array_equal(b + dense2d, b + self.a)
    assert_array_equal(np.add(dense2d, b), np.add(self.a, b))
    assert_array_equal(np.add(b, dense2d), np.add(b, self.a))

  def test_add_spmatrix(self):
    for fmt in ('coo', 'csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d + b, self.a + b)
      assert_sparse_equal(b + sparse2d, b + self.a)

  def test_add_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_array_equal(dense2d + s, (self.a + b).toarray())
    assert_array_equal(s + dense2d, (b + self.a).toarray())

  def test_add_scalar(self):
    b = 0
    assert_array_equal(dense2d + b, (self.a + b).toarray())
    assert_array_equal(b + dense2d, (b + self.a).toarray())
    b = 1
    self.assertRaises(NotImplementedError, lambda: self.a + b)
    self.assertRaises(NotImplementedError, lambda: b + self.a)

  def test_add_inplace(self):
    self.a += 0
    assert_array_equal(dense2d, self.a.toarray())
    # np.add with out kwarg
    res = np.add(self.a, 0, out=self.a)
    self.assertIs(res, self.a)
    assert_array_equal(dense2d, self.a.toarray())
    # sparray += sparray
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    self.a += b
    assert_array_equal(dense2d + s, self.a.toarray())

  def test_sub(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d - b, self.a - b)
    assert_array_equal(b - dense2d, b - self.a)
    assert_array_equal(np.subtract(dense2d, b), np.subtract(self.a, b))
    assert_array_equal(np.subtract(b, dense2d), np.subtract(b, self.a))

  def test_mul(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(dense2d * b, (self.a * b).toarray())
    assert_array_equal(b * dense2d, (b * self.a).toarray())
    assert_array_equal(np.multiply(dense2d, b),
                       np.multiply(self.a, b).toarray())
    assert_array_equal(np.multiply(b, dense2d),
                       np.multiply(b, self.a).toarray())
    b = 3  # scalar case
    assert_array_equal(dense2d * b, (self.a * b).toarray())
    assert_array_equal(b * dense2d, (b * self.a).toarray())

  def test_mul_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(*sparse2d.shape, density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.multiply(b), self.a * b)
      assert_sparse_equal(b.multiply(sparse2d), b.multiply(self.a))

  def test_mul_sparray(self):
    s = ss.rand(*sparse2d.shape, density=0.5)
    b = SpArray.from_spmatrix(s)
    assert_array_equal(np.multiply(dense2d, s), (self.a * b).toarray())
    assert_array_equal(s.multiply(dense2d), (b * self.a).toarray())

  def test_imul(self):
    b = np.random.random(dense2d.shape)
    a = self.a.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())
    b = 3
    a = self.a.copy()
    a *= b
    assert_array_equal(dense2d * b, a.toarray())

  def test_div(self):
    b = np.random.random(dense2d.shape)
    c = 3  # scalar case
    assert_array_almost_equal(dense2d / b, (self.a / b).toarray())
    assert_array_almost_equal(np.divide(dense2d, b),
                              np.divide(self.a, b).toarray())
    assert_array_almost_equal(np.true_divide(dense2d, b),
                              np.true_divide(self.a, b).toarray())
    assert_array_almost_equal(dense2d / c, (self.a / c).toarray())
    with np.errstate(divide='ignore'):
      assert_array_almost_equal(b / dense2d, b / self.a)
      assert_array_almost_equal(np.divide(b, dense2d), np.divide(b, self.a))
      assert_array_almost_equal(np.true_divide(b, dense2d),
                                np.true_divide(b, self.a))
      assert_array_almost_equal(c / dense2d, c / self.a)

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
        assert_array_equal(e1, self.a / b)
        assert_array_equal(e2, b / self.a)
        assert_array_equal(e3, self.a // b)
        assert_array_equal(e4, b // self.a)
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
      assert_array_equal(e1, np.true_divide(self.a, b))
      assert_array_equal(e2, np.true_divide(b, self.a))
      assert_array_equal(e3, np.floor_divide(self.a, b))
      assert_array_equal(e4, np.floor_divide(b, self.a))
      # each operation may raise div by zero and/or invalid value warnings
      for w in ws:
        self.assertIn(str(w.message).split()[0], ('divide','invalid'))

  def test_idiv(self):
    self.a /= 1
    assert_array_almost_equal(dense2d, self.a.toarray())
    b = np.random.random(dense2d.shape)
    self.a /= b
    assert_array_almost_equal(dense2d / b, self.a.toarray())

  def test_neg(self):
    assert_array_equal(-dense2d, (-self.a).toarray())

  def test_conj(self):
    assert_array_equal(dense2d.conj(), self.a.conj().toarray())
    assert_array_equal(np.conjugate(dense2d), np.conjugate(self.a).toarray())

  def test_dot(self):
    b = np.random.random((dense2d.shape[1], dense2d.shape[0]))
    assert_array_equal(dense2d.dot(b), self.a.dot(b))
    assert_array_equal(b.dot(dense2d), b.dot(self.a))
    assert_array_equal(np.dot(dense2d, b), np.dot(self.a, b))
    assert_array_equal(np.dot(b, dense2d), np.dot(b, self.a))

  def test_dot_spmatrix(self):
    for fmt in ('csr', 'csc'):
      b = ss.rand(dense2d.shape[1], dense2d.shape[0], density=0.5, format=fmt)
      assert_sparse_equal(sparse2d.dot(b), self.a.dot(b))
      # XXX: spmatrix.dot(SpArray) calls np.asarray on us,
      #  which just wraps us in an object array.
      # assert_sparse_equal(b.dot(sparse2d), b.dot(self.a))

  def test_dot_sparray(self):
    m,n = dense2d.shape
    shapes = ((n,), (n,m), (2,n,m))
    for shape in shapes:
      d = np.random.random(shape)
      d.flat[np.random.randint(2, size=d.size)] = 0
      e = dense2d.dot(d)
      b = SpArray.from_ndarray(d)
      assert_array_equal(e, self.a.dot(b).toarray())

  def test_minmax(self):
    self.assertEqual(dense2d.min(), self.a.min())
    self.assertEqual(dense2d.max(), self.a.max())

  def test_minmax_imum(self):
    b = np.random.random(dense2d.shape)
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.a, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.a, b))
    b = 3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.a, b).toarray())
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.a, b))
    b = -3
    assert_array_equal(np.minimum(dense2d, b), np.minimum(self.a, b))
    assert_array_equal(np.maximum(dense2d, b), np.maximum(self.a, b).toarray())

  def test_abs(self):
    assert_array_equal(np.abs(dense2d), np.abs(self.a).toarray())
    assert_array_equal(abs(dense2d), abs(self.a).toarray())

  def test_fixed_point_at_zero_ufuncs(self):
    with np.errstate(invalid='ignore', divide='ignore'):
      for ufunc in ss.base._ufuncs_with_fixed_point_at_zero:
        assert_array_equal(ufunc(dense2d), ufunc(self.a).toarray())


if __name__ == '__main__':
  unittest.main()
