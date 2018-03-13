#!/usr/bin/env python
# coding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import argparse
import tensorflow as tf

import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size.')
parser.add_argument('--train_steps', default=1000, type=int, 
                    help='number of training steps.')


def main(argv):
    args = parser.parse_args(argv[1:])
    batch_size = args.batch_size
    train_steps= args.train_steps

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    # print(train_y)

    # configure feature cloumns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # print(my_feature_columns)

    # build estimator
    classifer = tf.estimator.DNNClassifier(
            feature_columns = my_feature_columns,
            hidden_units=[10, 10],
            n_classes = 3
            )

    classifer.train(
            input_fn = lambda:iris_data.train_input_fn(train_x, train_y, batch_size),
            steps=train_steps
            )

    eval_result = classifer.evaluate(
            input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, batch_size),
            )
    print('\nTest SET ACCURACY:{accuracy:0.3f}\n'.format(**eval_result))

    # generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
            'SepalLength': [5.1, 5.9, 6.9],
	    'SepalWidth': [3.3, 3.0, 3.1],
	    'PetalLength': [1.7, 4.2, 5.4],
	    'PetalWidth': [0.5, 1.5, 2.1],}
    predictions = classifer.predict(
            input_fn=lambda:iris_data.eval_input_fn(predict_x, labels=None,
                                                    batch_size=batch_size)
            )
    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ("{:.1f}%"), excepted "{}"')
        class_id = pred_dict['class_ids'][0]
        prob = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id], 100*prob, expec))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


