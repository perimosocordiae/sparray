from __future__ import absolute_import

# Check for python 3.5+
try:
  from ._matmul import TestMatmulOperator
except SyntaxError:
  pass

