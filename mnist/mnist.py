#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import dataset

class Model(object):
    def __init__(self, data_format):
        # channel first => 'batch, channel, height, weight'.
        if data_format == 'channels_first':
            self._input_shape = [-1, 1, 28, 28]
        # channel last  => 'batch, height, weight, channel'.
        else:
            assert data_format=='channels_last'
            self._input_shape = [-1, 28, 28, 1]

        '''构建卷积层'''
        # Conv2D:args(filter, kernel_size, strides, padding, data_format, 
        #             dilation_rate, activation,...)
        self.conv1 = tf.layers.Conv2D(
                32, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(
                64, 5, padding='same', data_format=data_format, activation=tf.nn.relu)
        '''构建全连接层'''
        # Dense:args(units, activation, use_bias, kernel_initializer,...)
        self.fc1 = tf.layers.Dense(1024, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(10, activation=tf.nn.relu)
        
        # Dropout:args(rate, noise_shape, seed, name)
        self.dropout = tf.layers.Dropout(0.4)

        # MaxPooling2D:args(pool_size, strides, padding, data_format, name)
        # pool_size:an integer or tuple/list of 2 integers:(pool_height, pool_width)
        # strides:an integer or tuple/list of 2 integers:(stride_height, stride_width)
        self.max_pool2d = tf.layers.MaxPooling2D(
                (2,2), (2,2), padding='same', data_format=data_format)

    def __call__(self, inputs, training):
        y = tf.reshape(inputs, self._input_shape)
        y = self.conv1(y)
        y = self.max_pool2d(y)
        y = self.conv2(y)
        y = self.max_pool2d(y)
        y = tf.layers.flatten(y)
        y = self.fc1(y)
        y = self.dropout(y, training=training)
        return self.fc2(y)

def model_fn(features, labels, mode, params):
    '''overide tf.estimator.Estimator model_fn function.'''
    model = Model(params['data_format'])
    image = features
    if isinstance(image, dict):
        image = features['image']

    if mode==tf.estimator.ModeKeys.PREDICT:
        # predict 过程直接将新的数据feed到模型输出结果
        logits = model(image, training=False)
        predictions = {
                'classes':tf.argmax(logits, axis=1),
                'probabilities':tf.nn.softmax(logits)
                }
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                prdictions=predictions,
                export_outputs={
                    'classify':tf.estimator.export.PredictOutput(predictions)}
                )

    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        logits = model(image, training=True)
        loss   = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        accuracy = tf.metrics.accuracy(
                labels=labels, predictions=tf.argmax(logits, axis=1))

        tf.identity(accuracy[1], name='train_accuracy')

        # 使用summary和tensorboard实现tensorflow的可视化.
        tf.summary.scalar('train_accuracy', accuracy[1])
        return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.TRAIN,
                loss=loss,
                train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))

    if mode==tf.estimator.ModeKeys.EVAL:
        logits = model(image, training=False)
        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits=logits)
        return tf.estimato.EstimatorSpec(
                mode=tf.estimator.ModeKeys.EVAL,
                loss=loss,
                eval_metric_ops={
                    'accuracy':tf.metrics.accuracy(
                        labels=labels, predictions=tf.argmax(logits, axis=1))
                    }
                )

def validate_batch_size_for_multi_gpu(batch_size):
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')
    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)

def main(argv):
    # print(argv)
    model_function = model_fn

    if FLAGS.multi_gpu:
        validate_batch_size_for_multi_gpu(FLAGS.batch_size)
        
        model_function = tf.contrib.estimator.replicate_model_fn(
                model_fn, loss_reduction=tf.loss.Reduction.MEAN)

    data_format = FLAGS.data_format
    if data_format is None:
        data_format = ('channels_first'
                       if tf.test.is_built_with_cuda() else 'channels_last')

    mnist_classifier = tf.estimator.Estimator(
            model_fn = model_function,
            model_dir= FLAGS.model_dir,
            params   = {
                'data_format':data_format,
                'multi_gpu':FLAGS.multi_gpu,
                }
            )

    def train_input_fn():
        ds = dataset.train(FLAGS.data_dir)
        ds = ds.cache().shuffle(buffer_size=50000).batch(FLAGS.batch_size).repeat(
                FLAGS.train_epochs)
        (images, labels) = ds.make_one_shot_iterator().get_next()
        return images, labels

    # set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {'train_accuracy':'train_accuracy'}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)
    mnist_classifier.train(
        input_fn=train_input_fn,
        hooks = [logging_hook])

    # evaluate the model and print results
    def eval_input_fn():
        return dataset.test(FLAGS.data_dir).batch(
                FLAGS.batch_size).make_one_shot_iterator().get_next()

    eval_results = mnist_classifier.evaluate(
                        input_fn=eval_input_fn)
    print()
    print('Evaluation results:\n\t%s' % eval_results)

    # export the model
    if FLAGS.export_dir is not None:
        image = tf.placeholder(tf.float32, [None, 28, 28])
        input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                {'image':image})
        mnist_classifier.export_savedmodel(FLAGS.export_dir, input_fn)


class MNISTArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(MNISTArgParser, self).__init__()
        self.add_argument(
            '--multi_gpu', 
            action='store_true',
            help='If set, run across all available GPUs.')
        self.add_argument(
            '--batch_size', 
            type=int,
            default=100,
            help='Number of images to process in a batch.')
        self.add_argument(
            '--data_dir',
            type=str,
            default='/tmp/mnist_data',
            help='Path to directory containing the mnist datasets.')
        self.add_argument(
            '--model_dir',
            type=str,
            default='/tmp/mnist_model',
            help='Path to stor model.')
        self.add_argument(
            '--train_epochs',
            type=int,
            default=40,
            help='Number of epochs to train.')
        self.add_argument(
            '--data_format',
            type=str,
            default='channels_last',
            choices=['channels_first', 'channels_last'],
	    help='A flag to override the data format used in the model. '
		 'channels_first provides a performance boost on GPU but is not always '
		 'compatible with CPU. If left unspecified, the data format will be '
		 'chosen automatically based on whether TensorFlow was built for CPU or '
		 'GPU.')
	self.add_argument(
	    '--export_dir',
	    type=str,
	    help='The directory where the exported SavedModel will be stored.')


if __name__ == '__main__':
    parser = MNISTArgParser()
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main, argv=[sys.argv[0]] + unparsed)
