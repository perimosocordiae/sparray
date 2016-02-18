cimport cython
cimport numpy as np
import numpy as np
np.import_array()


def intersect1d_sorted(long[::1] a, long[::1] b, bint return_inds=False):
  c = np.empty(min(len(a), len(b)), dtype=np.int64)
  cdef Py_ssize_t end = merge_intersect(a, b, c)
  c = c[:end]
  if not return_inds:
    return c
  a_inds = np.searchsorted(a, c)
  b_inds = np.searchsorted(b, c)
  return c, a_inds, b_inds


def union1d_sorted(long[::1] a, long[::1] b, bint return_masks=False):
  c = np.empty(len(a) + len(b), dtype=np.int64)
  lut = np.empty(len(a) + len(b), dtype=np.uint8)
  a_mask = np.zeros_like(a, dtype=np.uint8)
  b_mask = np.zeros_like(b, dtype=np.uint8)
  cdef Py_ssize_t end = merge_unique(a, b, c, lut, a_mask, b_mask)
  c = c[:end]
  if not return_masks:
    return c
  lut = lut[:end]
  return c, lut, a_mask.view(dtype=bool), b_mask.view(dtype=bool)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t merge_unique(long[::1] a, long[::1] b, long[::1] result,
                             np.uint8_t[::1] lut, np.uint8_t[::1] a_mask,
                             np.uint8_t[::1] b_mask) nogil:
  cdef Py_ssize_t idx = 0, i = 0, j = 0, na = a.shape[0], nb = b.shape[0]
  while i < na and j < nb:
    if a[i] < b[j]:
      result[idx] = a[i]
      a_mask[i] = 1
      lut[idx] = 0
      i += 1
    elif b[j] < a[i]:
      result[idx] = b[j]
      b_mask[j] = 1
      lut[idx] = 1
      j += 1
    else:
      result[idx] = a[i]
      lut[idx] = 2
      i += 1
      j += 1
    idx += 1
  while i < na:
    result[idx] = a[i]
    a_mask[i] = 1
    lut[idx] = 0
    i += 1
    idx += 1
  while j < nb:
    result[idx] = b[j]
    b_mask[j] = 1
    lut[idx] = 1
    j += 1
    idx += 1
  return idx


@cython.boundscheck(False)
@cython.wraparound(False)
cdef Py_ssize_t merge_intersect(long[::1] a, long[::1] b, long[::1] result) nogil:
  cdef Py_ssize_t idx = 0, i = 0, j = 0, na = a.shape[0], nb = b.shape[0]
  while i < na and j < nb:
    if a[i] < b[j]:
      i += 1
    elif b[j] < a[i]:
      j += 1
    else:
      result[idx] = a[i]
      i += 1
      j += 1
      idx += 1
  return idx
