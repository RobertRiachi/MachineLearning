# pylint: disable-messing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

# from six.moves import urllib

import tensorflow as tf

#images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
#labels = tf.placeholder(tf.int32, shape=[None])


def weight_variable(shape):
    normal = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(normal)


def bias_variable(shape):
    const = tf.constant(0.1, shape=shape)
    return tf.Variable(const)


def convolution2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#The input is 32x32 pixels and 3 input channels RGB
#Cifar image is defined as: [height, width, 3]
#Dimensions [batch, height, width, 3]
def inference(images):

    #weighting for the first convolution
    w_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    #Apply Convolution and ReLU of output size 64
    conv1 = convolution2d(images, w_conv1) + b_conv1
    conv1_out = tf.nn.relu(conv1)

    #Pool 2x2 it
    conv1_pooled = max_pool_2x2(conv1_out)

    #Apply local response normalization
    normalized1 = tf.nn.local_response_normalization(input=conv1_pooled)

    w_conv2 = weight_variable([5, 5, 64, 128])
    b_conv2 = bias_variable([128])

    conv2 = convolution2d(normalized1, w_conv2) + b_conv2
    conv2_out = tf.nn.relu(conv2)

    #Pool 2x2
    conv2_pooled = max_pool_2x2(conv2_out)

    normalized2 = tf.nn.local_response_normalization(input=conv2_pooled)

    #FIRST fully connected layer (Image size is 8*8 after 2 max poolings
    #Connecting all the feature maps of the 2nd conv layer to 1024 neurons
    w_fc1 = weight_variable([8 * 8 * 128, 1024])
    b_fc1 = bias_variable([1024])

    #flatten the output of the layers to a vector of 8*8*128 with a batch dimension
    flattened = tf.reshape(normalized2, [-1, 8 * 8 * 128])
    #result is a tensor of [batch, 1024]

    #left multiply flattened to the weights so that the batch dimension is left untouched...
    fc1_output = tf.matmul(flattened, w_fc1) + b_fc1
    fc1_activated = tf.nn.relu(fc1_output)

    #Final layer goes from 1024 down to 10 to match the number of labels
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    fc2_output = tf.matmul(fc1_activated, w_fc2) + b_fc2
    prediction = tf.nn.softmax(fc2_output)

    return prediction


def loss(prediction, labels):
    cross_entropy_loss = -tf.reduce_sum(labels * tf.log(prediction))
    return cross_entropy_loss


