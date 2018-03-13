#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
import numpy as np
import tensorflow as tf

STEPS = 1000
PRICE_NORM_FACTOR = 1000

def main(argv):
    assert len(argv)==1
    (train, test) = dataset.dataset()

    def to_thousands(features, labels):
        return features, labels/PRICE_NORM_FACTOR

    train = train.map(to_thousands)
    test  = test.map(to_thousands)

    # build the training input_fn
    def input_train():
        return (
                # shuffling with a buffer larger than the data set ensures that
                # the examples are well mixed.
                train.shuffle(1000).batch(128)
                .repeat().make_one_shot_iterator().get_next())

    def input_test():
        return (
                test.shuffle(1000).batch(128)
                .make_one_shot_iterator().get_next())

    # configure feature columns
    feature_columns = [
        tf.feature_column.numeric_column(key='curb-weight'),
        tf.feature_column.numeric_column(key='highway-mpg'),
            ]

    # build the Estimator
    model = tf.estimator.LinearRegressor(feature_columns=feature_columns)

    # train model
    model.train(input_fn=input_train, steps=STEPS)

    # evaluate how the model performs on data it has not yet seen
    eval_result = model.evaluate(input_fn=input_test)

    average_loss = eval_result['average_loss']
    
    # convert MSE to Root Mean Square Error
    print('\n' + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}"
            .format(PRICE_NORM_FACTOR * average_loss**0.5))

    # run the model in prediction mode.
    input_dict = {
            'curb-weight':np.array([2000,3000]),
            'highway-mpg':np.array([30,40])
            }

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            input_dict, shuffle=False
            )

    predict_results = model.predict(input_fn=predict_input_fn)

    # print the prediction results
    print('\nPrediction results:')
    for i, prediction in enumerate(predict_results):
        msg = ("Curb weight: {: 4d}lbs, "
           "Highway: {: 0d}mpg, "
           "Prediction: ${: 9.2f}")
	msg = msg.format(input_dict["curb-weight"][i], input_dict["highway-mpg"][i],
			 PRICE_NORM_FACTOR * prediction["predictions"][0])
    
        print("   "+msg)
    print()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

