import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

#shape=[None, 28*28] specifies a matrix with a variable size in width, and 28*28=784 in height
x = tf.placeholder(tf.float32, shape=[None, 28*28])

y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#Filter parameters for convolution
W_conv1 = weight_variable([5, 5, 1, 32])
#Filter is 5x5, 1 is the number of input channels (black and white so just 1),
#  and 32 is the number of output channels (ReLU neurons)

#Our bias variable has the same number of entries as the output channels (32)
b_conv1 = bias_variable([32])

#Flatten the 28*28 image to a tensor of [-1,28,28,1] => [batch, height, width, input_channels]
#-1 specifies that the tensor will preserve its original size by resizing that dimension
x_image = tf.reshape(x, [-1,28,28,1])

#We run a convolution layer on the x_image using the W, b parameters
convolution1 = conv2d(x_image, W_conv1) + b_conv1
#We pass the convolution through a ReLU layer with input size [32]
h_conv1 = tf.nn.relu(convolution1)
#ReLU neurons take their inputs and perform f(x)=max(0,x) on them

#We max pool the result 2x2
h_pool1 = max_pool_2x2(h_conv1)

#<THE UUGE>
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#</THE UUGE>

#We define our weights as such. The input volume is 7*7(image size after 2 max pools) * 64 (output size of last layer)
#1024 is our selected number of neurons, so we use it as the height of W and height of B
W_fc1 = weight_variable([7 * 7 * 64, 1024])
B_fc1 = bias_variable([1024])

#the flattened matrix becomes [-1, 7*7*64] => [batch, pixels]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + B_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#good ole softmax and cross entropy
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
