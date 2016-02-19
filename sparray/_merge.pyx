cimport cython
cimport numpy as np
import numpy as np
np.import_array()


def combine_ranges(long[:,::1] ranges, shape, long result_size,
                   bint inner=False):
  cdef long[::1] result = np.zeros(result_size, dtype=np.int64)
  cdef long[::1] shape_arr = np.asarray(shape, dtype=np.int64)
  combine_ranges_fast(ranges, shape_arr, result, result_size, inner)
  return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void combine_ranges_fast(long[:,::1] ranges, long[::1] shape,
                              long[::1] result, long result_size,
                              bint inner) nogil:
  # convert shape to strides in-place
  cdef Py_ssize_t ndim = shape.shape[0], cp = 1, tmp, i
  for i in range(ndim - 1, 0, -1):
    tmp = shape[i]
    shape[i] = cp
    cp *= tmp
  shape[0] = cp

  cdef Py_ssize_t start, stop, step, idx, c, range_len
  # easier case: inner indexing
  if inner:
    for i in range(ndim):
      start = ranges[i,0] * shape[i]
      stop = ranges[i,1] * shape[i]
      step = ranges[i,2] * shape[i]
      range_len = len_range(start, stop, step)
      if range_len == 1:
        # one item, broadcast over the whole result
        for c in range(result_size):
          result[c] += start
      else:
        # range_len should be equal to result_size
        idx = start
        for c in range(result_size):
          result[c] += idx
          idx += step
    return

  # harder case: outer indexing
  # add each range to the result, repeated/tiled as appropriate
  cdef Py_ssize_t t, r, num_tiles, num_repeats = result_size
  for i in range(ndim):
    start = ranges[i,0] * shape[i]
    stop = ranges[i,1] * shape[i]
    step = ranges[i,2] * shape[i]
    range_len = len_range(start, stop, step)
    num_tiles = result_size / num_repeats
    num_repeats /= range_len
    c = 0
    for t in range(num_tiles):
      idx = start
      while idx < stop:
        for r in range(num_repeats):
          result[c] += idx
          c += 1
        idx += step


@cython.cdivision(True)
cpdef inline long len_range(long start, long stop, long step) nogil:
  if step > 0:
    if start >= stop:
      return 0
    return (stop - start - 1) / step + 1
  # negative step case
  if stop >= start:
    return 0
  return (start - stop - 1) / -step + 1


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
