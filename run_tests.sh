#!/bin/sh

nosetests --with-cov --cov-report html --cov=sparray/ \
  sparray/tests/ && coverage report
