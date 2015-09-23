cimport cython
import numpy as np


def intersect1d_sorted(a, b, return_inds=False):
  c = np.empty(len(a) + len(b), dtype=a.dtype)
  end = merge_unique(a, b, c)
  c = c[:end]
  if not return_inds:
    return c
  a_inds = np.searchsorted(a, c)
  b_inds = np.searchsorted(b, c)
  return c, a_inds, b_inds


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t merge(int[::1] a, int[::1] b, int[::1] result) nogil:
  cdef Py_ssize_t idx = 0, i = 0, j = 0, na = a.shape[0], nb = b.shape[0]
  while i < na and j < nb:
    if a[i] < b[j]:
      result[idx] = a[i]
      i += 1
    else:
      result[idx] = b[j]
      j += 1
    idx += 1
  while i < na:
    result[idx] = a[i]
    i += 1
    idx += 1
  while j < nb:
    result[idx] = b[j]
    j += 1
    idx += 1
  return idx


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t merge_unique(int[::1] a, int[::1] b, int[::1] result) nogil:
  cdef Py_ssize_t idx = 0, i = 0, j = 0, na = a.shape[0], nb = b.shape[0]
  while i < na and j < nb:
    if a[i] < b[j]:
      result[idx] = a[i]
      i += 1
    elif b[j] < a[i]:
      result[idx] = b[j]
      j += 1
    else:
      result[idx] = a[i]
      i += 1
      j += 1
    idx += 1
  while i < na:
    result[idx] = a[i]
    i += 1
    idx += 1
  while j < nb:
    result[idx] = b[j]
    j += 1
    idx += 1
  return idx
