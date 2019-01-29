#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import svm
from sklearn.metrics import accuracy_score


mnist = input_data.read_data_sets("mnist/", one_hot=True)


# In[2]:


# clears up names and variables
tf.reset_default_graph()

inputs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input')
compare = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='compare')


# In[3]:


#encoder


#convolution: input, number of filters, kenerl size, activation function, padding, name
#maxpool: input, kernel size, stride length, name

#output => 28x28x32
convolution_1 = tf.layers.conv2d(inputs, 32, [3,3], activation=tf.nn.relu, padding='same', name='convolution_1')
#output => 14x14x32
maxpool_1 = tf.layers.max_pooling2d(convolution_1, [2,2], strides=2, name='maxpool_1')

#output => 14x14x64
convolution_2 = tf.layers.conv2d(maxpool_1, 64, [3,3], activation=tf.nn.relu, padding='same', name='convolution_2')

#output => 14x14x32
convolution_3 = tf.layers.conv2d(convolution_2, 32, [3,3], activation=tf.nn.relu, padding='same', name='convolution_3')
#output => 7x7x32
maxpool_2 = tf.layers.max_pooling2d(convolution_3, [2,2], strides=2, name='maxpool_2')


# In[4]:


# flatten image and feature extraction

# 1568
flatten = tf.layers.flatten(maxpool_2)
# 128
dense = tf.layers.dense(flatten, 128, activation=tf.nn.relu)
# 1568
unflatten = tf.layers.dense(dense, 1568, activation=tf.nn.relu)

#output => 7x7x32
reshaping = tf.reshape(unflatten, [-1, 7, 7, 32])


# In[5]:


#decoder


#output => 7x7x32
convolution_4 = tf.layers.conv2d(reshaping, 32, [3,3], activation=tf.nn.relu, padding='same', name='convolution_4')
#output => 14x14x32
upsampling_1 = tf.image.resize_nearest_neighbor(convolution_4, [14,14], name='upsampling_1')

#output => 14x14x64
convolution_5 = tf.layers.conv2d(upsampling_1, 64, [3,3], activation=tf.nn.relu, padding='same', name='convolution_5')

#output => 14x14x32
convolution_6 = tf.layers.conv2d(convolution_5, 32, [3,3], activation=tf.nn.relu, padding='same', name='convolution_6')
#output => 28x28x32
upsampling_2 = tf.image.resize_nearest_neighbor(convolution_6 , [28,28], name='upsampling_2')

#output => 28x28x1
logits = tf.layers.conv2d(upsampling_2, 1, [3,3], activation=None, padding='same')

#reconstructs the image
decoder = tf.nn.sigmoid(logits, name='decoder')


# In[6]:


#calculates cross entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=compare, logits=logits)
loss = tf.reduce_mean(loss)

#define optimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


# In[7]:


#train the autoencoder

def train_autoencoder():
    epochs = 1
    batches = 20
    start1 = time.time()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # start training
    with tf.Session() as session:
        session.run(init)

        for i in range(epochs):
            for j in range(mnist.train.num_examples // batches):
                batch = mnist.train.next_batch(batches)
                imgs = batch[0].reshape((-1, 28, 28, 1))
                loss_val, _ = session.run([loss, optimizer], feed_dict={inputs: imgs, compare: imgs})

            if i == (epochs - 1):
                saver.save(session, 'model/final_model.ckpt')

        end1 = time.time()
        time1 = end1 - start1

        return time1


# In[8]:


#train the svm

def train_svm():   
    new_saver = tf.train.Saver()
    start2 = time.time()

    with tf.Session() as session: 
        new_saver.restore(session, tf.train.latest_checkpoint('model'))

        training_data = []
        for i in range(mnist.train.num_examples):
            img = mnist.train.images[i]
            reconstructed = session.run(dense, feed_dict={inputs: img.reshape((1, 28, 28, 1))})
            training_data.append(reconstructed[0])

        classifier = svm.SVC(gamma='scale', decision_function_shape='ovo')
        classifier.fit(training_data, np.argmax(mnist.train.labels, axis=1))

        end2 = time.time()
        time2 = end2 - start2
        
        return time2, classifier


# In[9]:


# run the full test of the model

def run_model(classifier):
    new_saver = tf.train.Saver()

    with tf.Session() as session:
        new_saver.restore(session, tf.train.latest_checkpoint('model'))

        prediction = []
        for i in range(mnist.test.num_examples): 
            img = mnist.test.images[i]
            reconstructed = session.run(dense, feed_dict={inputs: img.reshape((1, 28, 28, 1))})
            test = classifier.predict([reconstructed[0]])
            prediction.append(test)

        accuracy = accuracy_score(np.argmax(mnist.test.labels, axis=1), prediction)
        
        return accuracy 


# In[10]:


# run training and testing

sum_time = 0
sum_accuracy = 0
count = 1
iterations = 1

for i in range(iterations):
    time_1 = train_autoencoder()
    time_2, classifier = train_svm()
    accuracy = run_model(classifier)
    
    sum_time += time_1 + time_2
    sum_accuracy += accuracy
    print("Time: {}".format(time_1 + time_2))
    print("Accuracy: {}".format(accuracy))
    print("Test Run {} Complete...\n".format(count))
    count += 1
    
print("\nTotal Time: {}".format(sum_time))
print("Average Time: {}".format(sum_time/iterations))
print("Average Accuracy: {}".format(sum_accuracy/iterations))

