#!/bin/sh

nosetests --with-cov --cov-report html && coverage report
