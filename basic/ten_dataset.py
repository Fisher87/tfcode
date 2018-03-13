#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

dataset1  = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4,10]))
# dataset1 = tf.contrib.data.Dataset.range(100)
# iterator = dataset1.make_one_shot_iterator()
# next_element = iterator.get_next()

# with tf.Session() as sess:
#     # print dataset1.output_types, dataset1.output_shapes
#     # print dataset1.map(lambda x:x + 1)
#     for i in range(10):
#         value = sess.run(next_element)
#         print value

iterator = dataset1.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    next = iterator.get_next()
    # print dir(next)
    sess.run(next.eval())


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()
