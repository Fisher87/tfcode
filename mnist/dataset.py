#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import gzip
import numpy as np
import tensorflow as tf
from six.moves import urllib

def download(directory, filename):
    '''download file is not already exist.'''
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
    zipped_filepath = filepath+'.gz'
    print('Downloading %s to %s' %(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, 'rb') as f_in, open(filepath, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath

def read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def check_image_file_header(filename):
    '''validate the filename corresponds to images for the MNIST dataset.'''
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        num_images = read32(f)
        rows = read32(f)
        cols = read32(f)
        print(magic, num_images, rows, cols)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' %(magic, f.name))
        if rows!=28 or cols!=28:
            raise ValueError(
                    'Invalid MNIST file %s:Excepted 28x28 images, found %dx%s'%
                    (f.name, rows, cols))

def check_label_file_header(filename):
    with tf.gfile.Open(filename, 'rb') as f:
        magic = read32(f)
        num_items = read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' %(magic, f.name))

def dataset(directory, images_file, labels_file):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_label_file_header(labels_file)

    def decode_image(image):
        image = tf.decode_raw(image, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [784])
        return image/255.0
    def decode_label(label):
        label = tf.decode_raw(label, tf.uint8)
        return tf.to_int32(label)

    images = tf.data.FixedLengthRecordDataset(
            images_file, 28*28, header_bytes=16).map(decode_image)
    labels = tf.data.FixedLengthRecordDataset(
            labels_file, 1, header_bytes=8).map(decode_label)
    return tf.data.Dataset.zip((images, labels))

def train(directory):
    return dataset(directory, 'train-images-idx3-ubyte', 
                            'train-labels-idx1-ubyte')

def test(directory):
    return dataset(directory, 'train-images-idx3-ubyte', 
                            'train-labels-idx1-ubyte')

# if __name__ == '__main__':
#     directory=''
#     train = dataset(directory, 'train-images-idx3-ubyte', 
#                     'train-labels-idx1-ubyte')
#     print(train)



