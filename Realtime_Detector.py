# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import dlib

NUM_CLASSES = 5
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

def inference(images_placeholder, keep_prob):
    """ A function for making a model of prediction
        
        Argument:
        image_placeholder: placeholder of images.
        keep_prob: placeholder of rate of dropout.
        
        Return:
        cross_entropy: the result of calculating
        """
    def weight_variable(shape):
        initial = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(x,W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    x_image = tf.reshape(images_placeholder, [-1,28,28,3])
    
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
    
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    return y_conv

if __name__ == '__main__':
    test_image = []

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")
    
    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()
    
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "(The pass of your directory including checkpoint file)/(checkpoint file name)")
    
    labels = ["neutral", "happiness", "surprise", "anger", "sadness"]
    # prepare D-lib
    detector = dlib.get_frontal_face_detector()
    font = cv2.FONT_HERSHEY_PLAIN
    # start the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception, 'video not found'
    # get a frame
    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Histogram flattening
            RGB = cv2.split(frame)
            Blue = RGB[0]
            Green = RGB[1]
            Red = RGB[2]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            Blue_c = clahe.apply(Blue)
            Green_c = clahe.apply(Green)
            Red_c = clahe.apply(Red)
            frame = np.dstack((Blue_c,Green_c))
            frame = np.dstack((frame,Red_c))
            # cutout your face from the frame
            dets = detector(frame, 1)
            for d in dets:
                cropped = frame[d.top():d.bottom(),d.left():d.right(),:]
                cv2.imshow("cropped_face", cropped)
                face = cv2.resize(cropped, (28, 28))
                face_flatten = face.flatten().astype(np.float32)/255.0
                # predict
                pred = np.argmax(logits.eval(feed_dict={
                                     images_placeholder: [face_flatten],
                                     keep_prob: 1.0 })[0])
                label = labels[pred]
                cv2.putText(cropped,label,(10,20),font, 2.0,(255,255,0))
                print(pred)
                cv2.imshow("cropped_face", cropped)
        cv2.imshow("capture", frame)

        # wait for input from keyboard
        key = cv2.waitKey(1) & 0xFF
        # stop the roop by click "q"
        if key == ord('q'):
            break
            
    # stop the camera
    cap.release()
    cv2.destroyAllWindows()




























