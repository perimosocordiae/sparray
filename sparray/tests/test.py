import unittest
import numpy as np
import scipy.sparse as ss
from numpy.testing import assert_array_equal
from sparray import SpArray

foo = np.array([[0,0,0],[4,5,7],[6,2,0],[1,3,8]])
foo_indices = [1,3,4,5,6,7,9,10,11]
foo_data = [0,4,5,7,6,2,1,3,8]

bar = np.arange(5)


class TestCreation(unittest.TestCase):
  def test_init(self):
    a = SpArray(foo_indices, foo_data, shape=foo.shape)
    assert_array_equal(a.toarray(), foo)
    b = SpArray(bar, bar)
    assert_array_equal(b.toarray(), bar)

  def test_from_ndarray(self):
    a = SpArray.from_ndarray(foo)
    assert_array_equal(a.toarray(), foo)
    b = SpArray.from_ndarray(bar)
    assert_array_equal(b.toarray(), bar)

  def test_from_sparse(self):
    for cls in (ss.csr_matrix, ss.coo_matrix):
      a = SpArray.from_sparse(cls(foo))
      assert_array_equal(a.toarray(), foo, 'Failed to convert from %s' % cls)


class TestConversion(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(foo_indices, foo_data, shape=foo.shape)
    self.b = SpArray(bar, bar)

  def test_tocoo(self):
    assert_array_equal(self.a.tocoo().A, foo)

  def test_repr(self):
    ra = '<(4, 3)-SpArray of type int64\n\twith 9 stored elements>'
    self.assertEqual(repr(self.a), ra)
    self.assertEqual(repr(self.b),
                     '<(5,)-SpArray of type int64\n\twith 5 stored elements>')

  def test_str(self):
    self.assertEqual(str(self.b), '\n'.join('  (%d,)\t%d' % (i,i) for i in bar))


class TestOps(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(foo_indices, foo_data, shape=foo.shape)

  def test_resize(self):
    self.a.resize((5,3))
    assert_array_equal(self.a.toarray(), np.vstack((foo,np.zeros((1,3)))))
    self.a.resize((12,))
    assert_array_equal(self.a.toarray(), foo.ravel())

  def test_reshape(self):
    b = self.a.reshape((6,2))
    self.assertIsNot(self.a, b)
    assert_array_equal(self.a.toarray(), foo)
    assert_array_equal(b.toarray(), foo.reshape((6,2)))
    b = self.a.reshape((2,-1))
    assert_array_equal(b.toarray(), foo.reshape((2,-1)))

  def test_ravel(self):
    b = self.a.ravel()
    self.assertIsNot(self.a, b)
    assert_array_equal(self.a.toarray(), foo)
    assert_array_equal(b.toarray(), foo.ravel())


class TestUfuncs(unittest.TestCase):
  def setUp(self):
    self.a = SpArray(foo_indices, foo_data, shape=foo.shape)

  def test_add(self):
    b = np.random.random(foo.shape)
    assert_array_equal(foo + b, self.a + b)
    assert_array_equal(b + foo, b + self.a)

  def test_sub(self):
    b = np.random.random(foo.shape)
    assert_array_equal(foo - b, self.a - b)
    assert_array_equal(b - foo, b - self.a)

  def test_mul(self):
    b = np.random.random(foo.shape)
    assert_array_equal(foo * b, (self.a * b).toarray())
    assert_array_equal(b * foo, (b * self.a).toarray())
    b = 3  # scalar case
    assert_array_equal(foo * b, (self.a * b).toarray())
    assert_array_equal(b * foo, (b * self.a).toarray())

  def test_dot(self):
    b = np.random.random((foo.shape[1], foo.shape[0]))
    assert_array_equal(foo.dot(b), self.a.dot(b))
    assert_array_equal(b.dot(foo), b.dot(self.a))

if __name__ == '__main__':
  unittest.main()
