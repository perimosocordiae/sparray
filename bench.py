#!/usr/bin/env python
from __future__ import print_function
import timeit
import numpy as np
import scipy.sparse as ss

from sparray import SpArray


def run_bench(fn, arrays, number=100, repeats=3):
  for a in arrays:
    try:
      res = timeit.repeat(lambda: fn(a), repeat=repeats, number=number)
    except TypeError:
      res = [np.inf]
    yield min(res) * 1e6 / number


def format_time(usec):
  if np.isinf(usec):
    return "N/A"
  if usec < 1000:
    return "%.3g us" % usec
  msec = usec / 1000.
  if msec < 1000:
    return "%.3g ms" % msec
  sec = msec / 1000.
  return "%.3g s" % sec


def main():
  arr = ss.rand(1000, 500, density=0.1)
  fmts = ['csr', 'coo', 'csc', 'dense', 'SpArray']
  arrays = [arr.tocsr(), arr.tocoo(), arr.tocsc(), arr.toarray(),
            SpArray.from_spmatrix(arr)]
  benches = [
      ('arr * 3', lambda a: a * 3),
      ('arr[876]', lambda a: a[876]),
      ('arr.sum()', lambda a: a.sum()),
      ('arr[:,273]', lambda a: a[:,273]),
      ('arr[:5,:5]', lambda a: a[:5,:5]),
      ('arr[154,145]', lambda a: a[154,145]),
  ]

  label_size = max(len(b[0]) for b in benches)
  fmt_size = 8
  print(' ' * label_size, *[f.ljust(fmt_size) for f in fmts])
  for label, fn in benches:
    result = list(run_bench(fn, arrays))
    ratio = np.array(result[:-1]) / result[-1]
    ratio = ['N/A' if np.isinf(r) else ('%.2fx' % r) for r in ratio]
    print(label.ljust(label_size), *[r.ljust(fmt_size) for r in ratio], end=' ')
    print(format_time(result[-1]).ljust(fmt_size))

if __name__ == '__main__':
  main()
