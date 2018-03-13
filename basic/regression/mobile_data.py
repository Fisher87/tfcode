#!/usr/bin/env python
# coding=utf-8

"""utility functions for loading automobile data set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
import collections

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

float=np.float32

COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
    ])

def raw_dataframe():
    path = tf.keras.utils.get_file(URL.split('/')[-1], URL)

    df = pd.read_csv(path, names=COLUMN_TYPES.keys(), 
                           dtype=COLUMN_TYPES, na_values="?")

    return df


def load_data(y_name='price', train_fraction=0.7, seed=None):
    """
    Args:
        y_name: the column to return as the label.
        train_fraction: the fraction of the data set to use for training.
        seed: the random seed to use when shuffling the data.
    """
    data = raw_dataframe()

    # delete rows with NAN
    data = data.dropna()

    np.random.seed(seed)
    x_train = data.sample(frac=train_fraction, random_state=seed)
    x_test = data.drop(x_train.index)

    y_train = x_train.pop(y_name)
    y_test  = x_test.pop(y_name)

    return (x_train, y_train), (x_test, y_test)
