# Check for python 3.5+
try:
  from ._matmul import TestMatmulOperator
except SyntaxError:
  pass
