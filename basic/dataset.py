#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
import tensorflow as tf

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

defaults = collections.OrderedDict([
    ("symboling", [0]),
    ("normalized-losses", [0.0]),
    ("make", [""]),
    ("fuel-type", [""]),
    ("aspiration", [""]),
    ("num-of-doors", [""]),
    ("body-style", [""]),
    ("drive-wheels", [""]),
    ("engine-location", [""]),
    ("wheel-base", [0.0]),
    ("length", [0.0]),
    ("width", [0.0]),
    ("height", [0.0]),
    ("curb-weight", [0.0]),
    ("engine-type", [""]),
    ("num-of-cylinders", [""]),
    ("engine-size", [0.0]),
    ("fuel-system", [""]),
    ("bore", [0.0]),
    ("stroke", [0.0]),
    ("compression-ratio", [0.0]),
    ("horsepower", [0.0]),
    ("peak-rpm", [0.0]),
    ("city-mpg", [0.0]),
    ("highway-mpg", [0.0]),
    ("price", [0.0])
])  # pyformat: disable


types = collections.OrderedDict((key,type(value[0]))
                                for key,value in defaults.items())

# print(types)
def _get_imports85():
    path = tf.contrib.keras.utils.get_file(URL.split('/')[-1], URL)
    return path

def dataset(y_name='price', train_fraction=0.7):
    path = _get_imports85()

    def decode_line(line):
        # decode the line to a tuple of items based on the types of csv_header.values()
        items = tf.decode_csv(line, list(defaults.values()))

        # convert the keys and items to a dict
        paris = zip(defaults.keys(), items)
        features_dict = dict(paris)

        # remove the label from the features_dict
        label = features_dict.pop(y_name)

        return  features_dict, label

    def has_no_question_marks(line):
        chars = tf.string_split(line[tf.newaxis], "").values
        is_question = tf.equal(chars, "?")
        any_question = tf.reduce_any(is_question)
        no_question = ~any_question

        return no_question

    def in_training_set(line):
        num_buckets = 1000000
        bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
        return bucket_id < int(train_fraction * num_buckets)

    def in_test_set(line):
        return ~in_training_set(line)

    base_dataset = (tf.contrib.data
                    .TextLineDataset(path)
                    .filter(has_no_question_marks))

    train = (base_dataset
            .filter(in_training_set)
            .map(decode_line)
            .cache())
    test = (base_dataset
            .filter(in_test_set)
            .map(decode_line)
            .cache())

    return train, test 

# with tf.Session() as sess:
#     dataset = dataset()
#     print(dir(dataset))
# import inspect
# print(inspect.getsource(tf.string_to_hash_bucket_fast))
