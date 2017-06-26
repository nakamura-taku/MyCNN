# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime

NUM_CLASSES = 5
IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3
# train.txt & test.txt are like following...
# train.txt
# /home/nakamura/CNN/nakamura-smile.jpg 0
# /home/nakamura/CNN/nakamura-angry.jpg 1
# /home/nakamura/CNN/nakamura-sad.jpg 2
# ...

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
flags.DEFINE_string('train', 'train_mylab.txt', 'File name of train data')
flags.DEFINE_string('test', 'test_mylab.txt', 'File name of train data')
flags.DEFINE_string('train_dir', './FE_train',
                           """Directory where to write event logs. """)
flags.DEFINE_string('checkpoint_path', './checkpoint',
                    """Directory where to write checkpoint file and meta. """)
flags.DEFINE_integer('max_steps', 100,
                            """Number of batches to run.""")
flags.DEFINE_integer('batch_size', 10,
                           """batch size"""
                           """Must divide evenly into the dataset sizes.""")
flags.DEFINE_float('learning_rate', 1e-4, """Initial learning rate""")

def inference(images_placeholder, keep_prob):
    """ A function for making a model of prediction
    
    Argument:
        image_placeholder: placeholder of images.
        keep_prob: placeholder of rate of dropout.
    
    Return:
        y_conv: like probabilities of each classes.
    """
    # initialize weights by normal distribution( SD=0.1 )
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)

    # initialize bias by normal distribution( SD=0.1 )
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        print("initial:{}".format(initial))
        return tf.Variable(initial)

    # make convolutional layers
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    # make pooling layers
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1],
                              strides=[1,2,2,1], padding='SAME')

    # reshape a input data (IMAGE_SIZExIMAGE_SIZEx1)
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    
    # make first convolutional layer (output:IMAGE_SIZExIMAGE_SIZE,32)
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # make first pooling layer (output: (IMAGE_SIZE/2)x(IMAGE_SIZE/2),32)
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    # make second convolutional layer (output:(IMAGE_SIZE/2)x(IMAGE_SIZE/2),64)
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # make second pooling layer (output:(IMAGE_SIZE/4)x(IMAGE_SIZE/4),64)
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # make third convolutional layer (output:(IMAGE_SIZE/4)x(IMAGE_SIZE/4),128)
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([5, 5, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # make third pooling layer (output:(IMAGE_SIZE/8)x(IMAGE_SIZE/8),128)
    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)

    # make fourth convolutional layer (output:(IMAGE_SIZE/8)x(IMAGE_SIZE/8),256)
    with tf.name_scope('conv4') as scope:
        W_conv4 = weight_variable([5, 5, 128, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    # make fourth pooling layer (output:(IMAGE_SIZE/16)x(IMAGE_SIZE/16),256)
    with tf.name_scope('pool4') as scope:
        h_pool4 = max_pool_2x2(h_conv4)

    # make first fully connected layer
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([(IMAGE_SIZE/8)*(IMAGE_SIZE/8)*256, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, (IMAGE_SIZE/8)*(IMAGE_SIZE/8)*256])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
        # setting dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # make second fully connected layer
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, 512])
        b_fc2 = bias_variable([512])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # make third fully connected layer
    with tf.name_scope('fc3') as scope:
        W_fc3 = weight_variable([512, NUM_CLASSES])
        b_fc3 = bias_variable([NUM_CLASSES])

    # Normalize by softmax function
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    # return possibilities of each labels
    return y_conv

# we get prediction by inference(), so let's define loss for back propagation
def loss(logits, labels):
    """ A function for calculating loss
        
    Argument:
        logits: tensor of logit, float - [batch_size, NUM_CLASSES]
        labels: tensor of label, int32 - [batch_size, NUM_CLASSES]
        
    Return:
        cross_entropy: tensor of cross entropy, float
    """

    # calculating cross entropy
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    # drowing in TensorBoard
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

# we get loss function, so let's train the model with back propagation method
def training(loss, learning_rate):
    """A function of defining training operation
        
    Argument:
        loss: tensor of loss, The result of loss().
        learning_rate: learning rate
    
    Return:
        train_step: operation of training
    """

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

# we need to check accuracy
def accuracy(logits, labels):
    """A function of calculating accuracy
    
    Argument:
        logits: the result of inference()
        labels: tensor of label, int32 - [batch_size, NUM_CLASSES]
    Return:
        accuracy: rate of accuracy(float)
    """
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    # get the date
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    # open file for training
    f = open(FLAGS.train, 'r')
    # array to input data
    train_image = []
    train_label = []
    for line in f:
        # separate space apart from newlines
        line = line.rstrip()
        l = line.split()
        # read the data & reshape to IMAGE_SIZExIMAGE_SIZE
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        # after line up, converse to float(0-1)
        train_image.append(img.flatten().astype(np.float32)/255.0)
        # prepare the label by 1-of-k style
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)
    # converse numpy style
    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    f.close()

    f = open(FLAGS.test, 'r')
    test_image = []
    test_label = []
    for line in f:
        line = line.rstrip()
        l = line.split()
        img = cv2.imread(l[0])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32)/255.0)
        tmp = np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        test_label.append(tmp)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    f.close()

    # start training
    with tf.Graph().as_default():
        # temporary tensor for inputting images
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # temporary tensor for inputting labels
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # temporary tensor for inputting rate of dropout
        keep_prob = tf.placeholder("float")

        # call inference() to make a model
        print("start_logits")
        logits = inference(images_placeholder, keep_prob)
        # call loss() to calculate loss
        loss_value = loss(logits, labels_placeholder)
        # call training() to train
        train_op = training(loss_value, FLAGS.learning_rate)
        # calculate accuracy
        acc = accuracy(logits, labels_placeholder)

        # prepare saving
        saver = tf.train.Saver()
        # make a session
        sess = tf.Session()
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # set values discribed on TensorBoard
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)

        # execute training
        for step in range(FLAGS.max_steps):
            for i in range(len(train_image)/FLAGS.batch_size):
                # execute training by batch_size imgages
                batch = FLAGS.batch_size*i
                # select data to be input to placeholder with feed_dict
                sess.run(train_op, feed_dict={
                         images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                         labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                         keep_prob: 0.5})

            # calculate accuracy per 1 step
            train_accuracy = sess.run(acc, feed_dict={
                                    images_placeholder: train_image,
                                    labels_placeholder: train_label,
                                    keep_prob: 1.0})
            print "step %d, training accuracy %g"%(step, train_accuracy)
            summary_str = sess.run(summary_op, feed_dict={
                                   images_placeholder: train_image,
                                   labels_placeholder: train_label,
                                   keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

        # after finishing training, show the accuracy towards test data
        test_accuracy = sess.run(acc, feed_dict={
                                 images_placeholder: test_image,
                                 labels_placeholder: test_label,
                                 keep_prob: 1.0})
        print("test_accuracy:{}".format(test_accuracy))

        #print "test accuracy %g"%sess.run(acc, feed_dict={
        #                                  images_placeholder: test_image,
        #                                  labels_placeholder: test_label,
        #                                  keep_prob: 1.0})

        # save the final model
        checkpoint_path = os.path.join(os.path.expanduser(FLAGS.checkpoint_path), subdir)
        os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(os.path.expanduser(checkpoint_path), subdir)
        save_path = saver.save(sess, checkpoint_path)













