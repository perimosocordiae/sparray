import scipy.sparse as ss
import warnings
warnings.simplefilter('ignore', ss.SparseEfficiencyWarning)

from sparray import FlatSparray


class Operations(object):
  params = [['FlatSparray', 'csr_matrix']]
  param_names = ['arr_type']

  def setup(self, arr_type):
    mat = ss.rand(3000, 4000, density=0.1, format='csr')
    if arr_type == 'FlatSparray':
      self.arr = FlatSparray.from_spmatrix(mat)
    else:
      self.arr = mat

  def time_scalar_multiplication(self, arr_type):
    self.arr * 3

  def time_sum(self, arr_type):
    self.arr.sum()

  def time_getitem_scalar(self, arr_type):
    self.arr[154, 145]

  def time_getitem_subarray(self, arr_type):
    self.arr[:5, :5]

  def time_getitem_row(self, arr_type):
    self.arr[876]

  def time_getitem_col(self, arr_type):
    self.arr[:,273]

  def time_diagonal(self, arr_type):
    self.arr.diagonal()


class ImpureOperations(object):
  params = [['FlatSparray', 'csr_matrix']]
  param_names = ['arr_type']
  number = 1  # make sure we re-run setup() before each timing

  def setup(self, arr_type):
    mat = ss.rand(3000, 4000, density=0.1, format='csr')
    if arr_type == 'FlatSparray':
      self.arr = FlatSparray.from_spmatrix(mat)
    else:
      self.arr = mat

  def time_setdiag(self, arr_type):
    self.arr.setdiag(99)
