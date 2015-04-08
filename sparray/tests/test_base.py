import unittest
import numpy as np
import scipy.sparse as ss
import warnings
from numpy.testing import assert_array_equal
from sparray import SpArray

dense2d = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]], dtype=float) / 2.
dense2d_indices = [1,3,4,5,6,7,9,10,11]
dense2d_data = [0,2,2.5,3.5,3,1,0.5,1.5,4]

sparse2d = ss.csr_matrix(dense2d)
with warnings.catch_warnings():
  warnings.simplefilter('ignore')  # Ignore efficiency warning
  sparse2d[0,1] = 0  # Add the explicit zero to match indices,data

dense1d = np.arange(5) - 2
dense1d_indices = [0,1,3,4]
dense1d_data = [-2,-1,1,2]


class TestCreation(unittest.TestCase):
  def test_init(self):
    a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    assert_array_equal(a.toarray(), dense2d)
    b = SpArray(dense1d_indices, dense1d_data, shape=dense1d.shape)
    assert_array_equal(b.toarray(), dense1d)
    b = SpArray(dense1d_indices, dense1d_data)
    assert_array_equal(b.toarray(), dense1d)

  def test_from_ndarray(self):
    for arr in (dense2d, dense1d):
      a = SpArray.from_ndarray(arr)
      assert_array_equal(a.toarray(), arr)

  def test_from_spmatrix(self):
    for fmt in ('csr', 'csc', 'coo', 'dok', 'lil', 'dia'):
      a = SpArray.from_spmatrix(sparse2d.asformat(fmt))
      assert_array_equal(a.toarray(), dense2d,
                         'Failed to convert from %s' % fmt.upper())


class TestConversion(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    self.b = SpArray(dense1d_indices, dense1d_data)

  def test_tocoo(self):
    assert_array_equal(self.a.tocoo().A, dense2d)

  def test_repr(self):
    ra = '<(4, 3)-SpArray of type float64\n\twith 9 stored elements>'
    self.assertEqual(repr(self.a), ra)
    self.assertEqual(repr(self.b),
                     '<(5,)-SpArray of type int64\n\twith 4 stored elements>')

  def test_str(self):
    expected = '\n'.join('  (%d,)\t%d' % x
                         for x in zip(dense1d_indices, dense1d_data))
    self.assertEqual(str(self.b), expected)


class TestOps(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)
    self.b = SpArray(dense1d_indices, dense1d_data, shape=dense1d.shape)

  def test_resize(self):
    self.a.resize((5,3))
    assert_array_equal(self.a.toarray(), np.vstack((dense2d,np.zeros((1,3)))))
    self.a.resize((12,))
    assert_array_equal(self.a.toarray(), dense2d.ravel())

  def test_reshape(self):
    b = self.a.reshape((6,2))
    self.assertIsNot(self.a, b)
    assert_array_equal(self.a.toarray(), dense2d)
    assert_array_equal(b.toarray(), dense2d.reshape((6,2)))
    b = self.a.reshape((2,-1))
    assert_array_equal(b.toarray(), dense2d.reshape((2,-1)))

  def test_ravel(self):
    b = self.a.ravel()
    self.assertIsNot(self.a, b)
    assert_array_equal(self.a.toarray(), dense2d)
    assert_array_equal(b.toarray(), dense2d.ravel())

  def test_astype(self):
    self.assertIs(self.a.dtype, dense2d.dtype)
    b = self.a.astype(np.float32)
    self.assertIsNot(self.a, b)
    assert_array_equal(b.toarray(), dense2d.astype(np.float32))

  def test_copy(self):
    b = self.a.copy()
    self.assertIsNot(self.a, b)
    assert_array_equal(self.a.toarray(), b.toarray())
    b.data[2] *= 3  # modify b's members
    assert_array_equal(self.a.toarray(), dense2d)

  def test_transpose(self):
    assert_array_equal(dense2d.transpose(), self.a.transpose().toarray())
    assert_array_equal(dense2d.transpose(0,1), self.a.transpose(0,1).toarray())
    assert_array_equal(dense2d.transpose((0,1)),
                       self.a.transpose((0,1)).toarray())
    assert_array_equal(dense1d.transpose(), self.b.transpose().toarray())


class TestAttrs(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(dense2d_indices, dense2d_data, shape=dense2d.shape)

  def test_prop_attrs(self):
    for attr in ('dtype', 'size', 'shape', 'ndim'):
      exp = getattr(sparse2d, attr)
      act = getattr(self.a, attr)
      self.assertEqual(getattr(sparse2d, attr), getattr(self.a, attr),
                       'attr "%s" mismatch: %s != %s' % (attr, exp, act))

  def test_transform_attrs(self):
    assert_array_equal(sparse2d.A, self.a.A)
    for attr in ('T', 'real', 'imag'):
      assert_array_equal(getattr(sparse2d, attr).A,
                         getattr(self.a, attr).toarray())

  def test_nonexistent_attr(self):
    self.assertRaises(AttributeError, lambda: getattr(self.a, 'xxxx'))


if __name__ == '__main__':
  unittest.main()
