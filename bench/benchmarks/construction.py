import numpy as np
import scipy.sparse as ss

from sparray import FlatSparray


class Construction2D(object):
  def setup(self):
    num_rows, num_cols = 3000, 4000
    self.spm = ss.rand(num_rows, num_cols, density=0.1, format='coo')
    self.arr = self.spm.A
    self.data = self.spm.data
    self.indices = self.spm.row * num_cols + self.spm.col
    self.spm_csr = self.spm.tocsr()

  def time_init(self):
    FlatSparray(self.indices, self.data, shape=self.arr.shape)

  def time_from_ndarray(self):
    FlatSparray.from_ndarray(self.arr)

  def time_from_spmatrix_coo(self):
    FlatSparray.from_spmatrix(self.spm)

  def time_from_spmatrix_csr(self):
    FlatSparray.from_spmatrix(self.spm_csr)


class ConstructionND(object):
  params = [[(1200000,), (1200,1000), (120,100,100), (20,30,40,50)]]
  param_names = ['shape']

  def setup(self, shape):
    nnz = 10000
    size = np.prod(shape)
    self.indices = np.random.choice(size, nnz, replace=False)
    self.sorted_indices = np.sort(self.indices)
    self.data = np.ones(nnz, dtype=float)
    arr = np.zeros(size, dtype=float)
    arr[self.sorted_indices] = 1
    self.arr = arr.reshape(shape)

  def time_init(self, shape):
    FlatSparray(self.indices, self.data, shape=shape)

  def time_canonical_init(self, shape):
    FlatSparray(self.sorted_indices, self.data, shape=shape, is_canonical=True)

  def time_from_ndarray(self, shape):
    FlatSparray.from_ndarray(self.arr)
