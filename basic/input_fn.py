#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES= []

def main():
    training_set = pd.read_csv('boston_train.csv', skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv('boston_test.csv', skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv('boston_predict.csv', skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

if __name__ == '__main__':
    tf.app.run()

