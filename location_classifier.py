from read_location_names import read_location_names
from embedding import *
import tensorflow as tf


class location_classifier():
    def __init__(self):
        # shape: (batch_size, num_words, embedding_dim)
        self.x = tf.placeholder(tf.float32, [None, None, embedding_dim])
        self.y = tf.placeholder(tf.int32, [None])

        self.hidden_dim = 4
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
        cell_outputs, _ = tf.nn.dynamic_rnn(self.lstm_cell, self.x, time_major=False)
        W = tf.get_variable('W', [self.hidden_dim, 2], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [2])
        



