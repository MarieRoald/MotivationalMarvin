from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

import collections

# quick test inspired by
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537



def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary




def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    #print(content)
    content = [content[i].split() for i in range(len(content))]
    #print(content)
    content = np.concatenate(np.array(content),0)
    #print(content)
    content = np.reshape(content, [-1, ])
    return content






def simple_RNN(x, W, b,n_input,n_hidden):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    print (x)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    print ("outputs[-1] {}".format(outputs[-1]))

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], W) + b

