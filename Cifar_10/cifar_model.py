
import tensorflow as tf

#The input is 32x32 pixels and 3 input channels RGB
#Cifar image is defined as: [height, width, 3]
images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
labels = tf.placeholder(tf.int32, shape=[None])


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


#weighting for the first convolution
W_conv1 = weight_variable([5, 5, 3, 64])
B_conv1 = bias_variable([64])

#Apply Convolution and ReLU of output size 64
conv1 = convolution2d(images, W_conv1) + B_conv1
conv1_out = tf.nn.relu(conv1)

#Pool 2x2 it
conv1_pooled = max_pool_2x2(conv1_out)

#Apply local response normalization



