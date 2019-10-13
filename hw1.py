import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels[:, -1, :, :], logits=preds[:, -1, :, :]))
    #############################


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        N = self.num_classes
        K = self.samples_per_class - 1
        B = FLAGS.meta_batch_size 


        #Update final N examples in index K+1 to have 0 labels, not their ground truth
        zeros_labels = tf.zeros_like(tf.expand_dims(input_labels[:, 0, :, :], 1))
        labels_w_zeros = tf.concat([input_labels[:,:K,:,:], zeros_labels], axis=1)
        #print(labels_w_zeros.get_shape())

        #Concatenate inputs [B, K + 1, N, 784] and outputs [B, K + 1, N, N]
        #into [B, K + 1, N, 784+N] tensor to pass into LSTM (adding in labels as our classification technique)
        concatenated = tf.concat([input_images, labels_w_zeros], axis = 3)

        #Reshape inputs [B, K + 1, N, N] and outputs [B, K + 1, N, N] 
        #into [B, (K+1)*N, 784+N] to pass into LSTM 
        concatenated_shaped = tf.reshape(concatenated,(-1, (K + 1)*N, 784+N))

        #Pass into LSTM 
        h1 = tf.nn.relu(self.layer1(concatenated_shaped))
        out_shaped = self.layer2(h1)

        #Reshape back into labels [B, K + 1, N, N]!
        out = tf.reshape(out_shaped, (-1, K + 1, N, N))
        #############################
        return out

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))
labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.0005)
#Even greater grad step! *
#For original learning rate: optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)


test_accuracy = []
iteration_number = []

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}
        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)    
            print("Test Accuracy", (1.0 * (pred == l)).mean())
            test_accuracy.append((1.0 * (pred == l)).mean())
            iteration_number.append(step)

print(test_accuracy)
plt.plot(iteration_number, test_accuracy)
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.title("K=" + str(FLAGS.num_samples) + ", N=" + str(FLAGS.num_classes) + ", B=" + str(FLAGS.meta_batch_size))
plt.savefig("K_" + str(FLAGS.num_samples) + "N_" + str(FLAGS.num_classes) + "B_" + str(FLAGS.meta_batch_size))
plt.show()

