
import tensorflow as tf

# images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
# labels = tf.placeholder(tf.int32, shape=[None])

import os
import sys
import tarfile

import urllib

import cifar_input

# SOURCE: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10.py
# define flags for use in the model
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(flag_name='batch_size', default_value=128,
                            docstring="""number of images to process in a batch""")

tf.app.flags.DEFINE_string(flag_name='data_dir', default_value='/tmp/cifar10_data/',
                           docstring="""Path to CIFAR-10 data directory.""")

# Global constants
IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_CLASSES = cifar_input.NUM_CLASSES

# How many images per training cycle do we use for training
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

# How many images per training cycle do we use for evaluating our model
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999           # The Decay for the moving average
NUM_EPOCHS_PER_DECAY = 350.0            # Epochs after which learning rate decays
LEARNING_RATE_DECAY_FACTOR = 0.1        # Learning rate decay factor
INITIAL_LEARNING_RATE = 0.1             # Initial learning rate


def weight_variable_with_decay(name, shape, stddev, wd):
    normal = tf.truncated_normal(shape=shape, stddev=stddev)

    var = tf.Variable(initial_value=normal, name=name)

    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def distorted_inputs():
    """Construct distorted input for CIFAR training using the Reader OPS.
        SOURCE: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10.py"""

    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    return cifar_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def bias_variable(name, shape):
    const = tf.constant(0.1, shape=shape)
    return tf.Variable(const, name=name)


def convolution2d(input, filter):
    return tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def activation_summary(x):
    """HHelper to create summaries for activations.
        Creates a summary that provides a histogram of activations for model training visualization

        Source: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10.py

    """


# The input is 32x32 pixels and 3 input channels RGB
# Cifar image is defined as: [height, width, 3]
# Dimensions [batch, height, width, 3]
def inference(images):
    """Builds the cifar10 model"""

    with tf.variable_scope('conv1') as scope:
        # weighting for the first convolution
        w_conv1 = weight_variable_with_decay(name='weights', shape=[5, 5, 3, 64], stddev=1e-4, wd=0.0)
        b_conv1 = bias_variable('biases', [64])

        # Apply Convolution and ReLU of output size 64
        conv = tf.nn.conv2d(images, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = conv + b_conv1
        conv1_out = tf.nn.relu(conv1, name=scope.name)
        activation_summary(conv1_out)


    # Pool 2x2 it
    conv1_pooled = max_pool_2x2(conv1_out)

    # Apply local response normalization
    normalized1 = tf.nn.local_response_normalization(input=conv1_pooled)

    with tf.variable_scope('conv2') as scope:
        # weighting for the second convolution
        w_conv2 = weight_variable_with_decay('weights', shape=[5, 5, 64, 64], stddev=1e-4, wd=0.0)
        b_conv2 = bias_variable('biases', [64])

        conv = tf.nn.conv2d(normalized1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = conv + b_conv2
        conv2_out = tf.nn.relu(conv2)
        activation_summary(conv2_out)

    # Pool 2x2
    conv2_pooled = max_pool_2x2(conv2_out)

    normalized2 = tf.nn.local_response_normalization(input=conv2_pooled)

    with tf.variable_scope('fully_connected_1') as scope:

        # FIRST fully connected layer (Image size is 6*6 after 2 max poolings
        # Connecting all the feature maps of the 2nd conv layer to 512 neurons
        w_fc1 = weight_variable_with_decay('weights', shape=[6 * 6 * 64, 512], stddev=0.04, wd=0.004)
        b_fc1 = bias_variable('biases', [512])

        # flatten the output of the layers to a vector of 6*6*128 with a batch dimension
        flattened = tf.reshape(normalized2, [FLAGS.batch_size, 6 * 6 * 64])
        # result is a tensor of [batch, 1024]

        # left multiply flattened to the weights so that the batch dimension is left untouched...
        fc1_output = tf.matmul(flattened, w_fc1) + b_fc1
        fc1_activated = tf.nn.relu(fc1_output, name=scope.name)
        activation_summary(fc1_activated)

    with tf.variable_scope('fully_connected_2') as scope:

        # This layer goes from 512 down to 256
        w_fc2 = weight_variable_with_decay('weights', shape=[512, 256], stddev=0.04, wd=0.004)
        b_fc2 = bias_variable('biases', [256])

        fc2_output = tf.matmul(fc1_activated, w_fc2) + b_fc2
        fc2_activated = tf.nn.relu(fc2_output, name=scope.name)
        activation_summary(fc2_activated)

    with tf.variable_scope('fully_connected_3') as scope:

        # This layer goes from 256 down to 128
        w_fc3 = weight_variable_with_decay('weights', shape=[256, 128], stddev=0.04, wd=0.004)
        b_fc3 = bias_variable('biases', [128])

        fc3_output = tf.matmul(fc2_activated, w_fc3) + b_fc3
        fc3_activated = tf.nn.relu(fc3_output, name=scope.name)
        activation_summary(fc3_activated)

    with tf.variable_scope('softmax_linear') as scope:

        # Final layer takes us from 128 to #Of classes

        w_final = weight_variable_with_decay('weights', shape=[128, NUM_CLASSES], stddev=1/128.0, wd=0.0) # Best stddev is 1/NUM_INPUT_NEURONS according to karpathy
        biases = bias_variable('biases', shape=[NUM_CLASSES])

        softmax_linear = tf.nn.softmax(tf.matmul(fc3_activated, w_final) + biases, name=scope.name)
        activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):

    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_loss)

    # The total loss is defined as cross entropy +
    # L2 regularization given by adding weight decay to the variables in the fully connected layers
    # This function will add all the losses (cross entropy and weight decay terms)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summaries(total_loss):
    """Adds summaries for the loss in the CIFAR-10 model
        Generates moving average for all losses and associated summaries for visualization of the neural network

        SOURCE: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10.py
    """

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the lass
        # as the original loss nae
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


# FROM https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10.py
# functions to help extract the data
def train(total_loss, global_step):
    """DEFINE TRAIN EPOCH FOR CIFAR MODEL."""
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Calculate the learning rate for this step
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning rate', lr)

    # Generate moving averages of all losses and associated summamries
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients through back propagation
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Create training summaries
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    for grad, var in grads:
        tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # return the train operation
    return train_op


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def maybe_download_and_extract():
    """Downloads and extracts the data"""
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)

        print()
        statinfo = os.stat(filepath)
        print('successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


