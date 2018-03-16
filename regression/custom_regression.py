#!/usr/bin/env python
# coding=utf-8

"""Regression using the DNNRegression Estimator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import mobile_data

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', default=100, type=int, help="batch size")
parse.add_argument('--train_steps', default=1000, type=int, 
                   help='number of training steps.')

data = mobile_data.load_data()

def my_dnn_regressor_fn(features, labels, mode, params):
    # extract the input into a dense layer, accoring to the feature_columns.
    top = tf.feature_column.input_layer(features, params['feature_columns'])

    # iterate over the 'hidden units' list of layer sizes, default is [20]
    for units in params.get('hidden_units', [20]):
        top = tf.layers.dense(inputs=top, units=units, activation=tf.nn.relu)

    # connect a linear output layer on top.
    output_layer = tf.layers.dense(inputs=top, units=1)

    # reshape the output layer to 1-dim Tensor 
    predictions = tf.squeeze(output_layer, 1)

    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode, predictions={'price':predictions})

    average_loss = tf.losses.mean_squared_error(labels, predictions)
    batch_size = tf.shape(labels)[0]
    total_loss = tf.to_float(batch_size)*average_loss

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get('optimizer', tf.train.AdamOptimizer)
        optimizer = optimizer(params.get('learning_rate', None))
        train_op  = optimizer.minimize(
                loss = average_loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL
    print(labels)
    print(predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)

    eval_metrics = {'rmse':rmse}

    return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metrics)


def from_dataset(ds):
    return lambda:ds.make_one_shot_iterator().get_next()

def make_dataset(features, labels=None):
    # features = dict(features)
    # for key in features:
    #     features[key] = np.array(features[key])
    # items=[features]
    # if labels is not None:
    #     items.append(np.array(labels, dtype=np.float32))
    # return tf.data.Dataset.from_tensor_slices(tuple(items))
    data = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return data

def main(argv):
    args = parse.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = mobile_data.load_data()
    train_y /= 10000
    test_y /= 10000

    # print(train_x['body-style'])
    # print(train_x['make'])
    # print(pd.unique(train_x['body-style']))

    # build train dataset.
    train = (make_dataset(train_x, train_y)
            .shuffle(1000).batch(args.batch_size)
            .repeat())
    # build validation dataset.
    test  = (make_dataset(test_x, test_y)
            .batch(args.batch_size))

    body_style_vocab = ['hardtop', 'wagon', 'sedan', 'hatchback', 'convertible']
    body_style = tf.feature_column.categorical_column_with_vocabulary_list(
            key="body-style", vocabulary_list=body_style_vocab)
    make = tf.feature_column.categorical_column_with_hash_bucket(key='make',
            hash_bucket_size=50)

    feature_columns = [
            tf.feature_column.numeric_column(key='curb-weight'),
            tf.feature_column.numeric_column(key='highway-mpg'),
            tf.feature_column.indicator_column(body_style),
            tf.feature_column.embedding_column(make, dimension=3)
            ]
    # print(feature_columns)

    model = tf.estimator.Estimator(
            model_fn = my_dnn_regressor_fn,
            params={
                'feature_columns':feature_columns,
                'learning_rate':0.001,
                'optimizer':tf.train.AdamOptimizer,
                'hidden_units':[20,20]
                }
            )
    model.train(
            input_fn=from_dataset(train),
            steps=args.train_steps)

    eval_result = model.evaluate(input_fn=from_dataset(test))

    print('\n'+80*'*')
    print('\nRMSE error for the test set:{:.0f}'
            .format(10000*eval_result['rmse']))
    print()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
