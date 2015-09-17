#!/usr/bin/env python
from __future__ import print_function
import timeit
import scipy.sparse as ss

from sparray import SpArray


def run_bench(fn, formats, number=100, repeats=3):
  results = []
  for fmt, a in formats.iteritems():
    res = timeit.repeat(lambda: fn(a), repeat=repeats, number=number)
    results.append((min(res), fmt))
  for tot_time, fmt in sorted(results):
    print('', fmt, format_time(tot_time, number), sep='\t')


def format_time(tot_time, number):
  usec = tot_time * 1e6 / number
  if usec < 1000:
    return "%.3g usec" % usec
  msec = usec / 1000.
  if msec < 1000:
    return "%.3g msec" % msec
  sec = msec / 1000.
  return "%.3g sec" % sec


def main():
  arr = ss.rand(1000, 500, density=0.1)
  fmts = dict(csr=arr.tocsr(),
              coo=arr.tocoo(),
              csc=arr.tocsc(),
              dense=arr.toarray(),
              SpArray=SpArray.from_spmatrix(arr))
  print('Benchmark: `arr.sum()`')
  run_bench(lambda a: a.sum(), fmts)

if __name__ == '__main__':
  main()
