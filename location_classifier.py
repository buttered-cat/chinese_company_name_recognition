import random
from read_location_names import read_location_names
from embedding import *
import gensim
import numpy as np
import tensorflow as tf


class location_classifier():
    def __init__(self):
        self.hidden_dim = 4
        self.learning_rate = 1e-3
        self.beta1 = 0.5
        self.test_data_portion = 1/5
        self.epoch = 200
        self.sess = tf.Session()

        # shape: (batch_size, num_words, embedding_dim)
        self.x = tf.placeholder(tf.float32, [None, None, embedding_dim])
        self.y = tf.placeholder(tf.int32, [None])

        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
        # assuming the inputs have the same sequence length
        # seq_len = tf.shape(self.x)[1]
        cell_outputs, _ = tf.nn.dynamic_rnn(self.lstm_cell, self.x, time_major=False, dtype=tf.float32)
        W = tf.get_variable('W', [self.hidden_dim, 2], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [2])
        cell_outputs = tf.squeeze(cell_outputs[:, -1, :])
        logits = tf.matmul(cell_outputs, W) + b
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits, name='softmax_entropy')
        predictions = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        self.pred_acc = tf.reduce_mean(tf.cast(tf.equal(predictions, self.y), tf.float32))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1)
        grads = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
        print('graph construction finished.')

    def get_data(self):
        embedding_model = gensim.models.Word2Vec.load(embedding_model_path)
        locations, locations_with_suffix = read_location_names()
        locations += locations_with_suffix

        # shape: (num_locations, location_word_len)
        location_word_vec = []
        max_word_seq_len = 0
        # segment all locations
        for location in locations:
            segmented_loc = jieba.cut(location, cut_all=False)
            word_seq = []
            for word in segmented_loc:
                word_seq.append(embedding_model[word])

            # try:
            #     for word in segmented_loc:
            #             word_seq.append(embedding_model[word])
            # except KeyError:
            #     word_seq = []
            #     generated_sentences = replace_locations_in_sentence(' '.join(segmented_loc), keep_sentence_structure=True)
            #     embedding_model.train(generated_sentences)
            #     embedding_model.save(embedding_model_path)
            #     for word in segmented_loc:
            #         word_seq.append(embedding_model[word])

            location_word_vec.append(word_seq)
            word_seq_len = len(word_seq)
            if word_seq_len > max_word_seq_len:
                max_word_seq_len = word_seq_len

        num_pos_samples = len(location_word_vec)
        # [0]: is location; [1]: is not location
        # location_labels = np.zeros([num_pos_samples])

        # locations = set(locations)
        non_locations = generate_non_locations(set(locations))

        train_bucket = dict()
        test_bucket = dict()
        bucket_temp = dict()
        for seq_len in range(1, max_word_seq_len):
            indices = random.sample(range(len(non_locations) - seq_len), num_pos_samples//max_word_seq_len)
            num_samples = len(indices)
            index_list = range(num_samples)
            test_data_index_indices = random.sample(index_list, int(num_samples * self.test_data_portion))
            test_data_indices = [indices[i] for i in test_data_index_indices]
            train_data_indices = [indices[i] for i in list(set(index_list) - set(test_data_index_indices))]

            # shape: ((num_samples, seq_len), (num_samples))
            train_bucket[seq_len] = [
                [ [embedding_model[non_locations[i+offset]] for offset in range(seq_len)] for i in train_data_indices ],
                np.ones([len(indices)])
            ]
            test_bucket[seq_len] = [
                [ [embedding_model[non_locations[i+offset]] for offset in range(seq_len)] for i in test_data_indices ],
                np.ones([len(indices)])
            ]
            bucket_temp[seq_len] = []

        for location in location_word_vec:
            # shape: (num_samples, seq_len)
            bucket_temp[len(location)].append(location)

        for seq_len in range(1, max_word_seq_len):
            num_samples = len(bucket_temp[seq_len])
            index_list = range(num_samples)
            test_data_index_indices = random.sample(index_list, int(num_samples * self.test_data_portion))
            test_data_indices = [index_list[i] for i in test_data_index_indices]
            train_data_indices = [index_list[i] for i in list(set(index_list) - set(test_data_index_indices))]

            train_bucket[seq_len][1] = np.concatenate((train_bucket[seq_len][1], np.zeros([len(train_data_indices)])), axis=0)
            train_bucket[seq_len][0] += [bucket_temp[seq_len][i] for i in train_data_indices]
            test_bucket[seq_len][1] = np.concatenate((test_bucket[seq_len][1], np.zeros([len(test_data_indices)])), axis=0)
            test_bucket[seq_len][0] += [bucket_temp[seq_len][i] for i in test_data_indices]

        return train_bucket, test_bucket

    def run_model(self, bucket, ops_to_run):
        max_len = 0
        loss = 0
        acc = 0
        for seq_len in bucket:
            if seq_len > max_len:
                max_len = seq_len
            iter_acc, iter_loss, _ = self.sess.run(ops_to_run,
                                                   feed_dict={self.x: bucket[seq_len][0], self.y: bucket[seq_len][1]})
            print('\tsequence length %i: acc = %f, loss = %f' % (seq_len, iter_acc, iter_loss))
            loss += iter_loss
            acc += iter_acc

        return acc/max_len, loss/max_len

    def train_and_test(self):
        train_bucket, test_bucket = self.get_data()

        print('Training:')
        for e in range(self.epoch):
            train_acc, train_loss = self.run_model(train_bucket, [self.pred_acc, self.loss, self.train_op])
            if e % 10 == 0:
                print('\tepoch %i: acc = %f, loss = %f' % (e, train_acc, train_loss))

        print('\nTesting:')
        test_acc, test_loss = self.run_model(test_bucket, [self.pred_acc, self.loss])
        print('\tacc = %f, loss = %f' % (test_acc, test_loss))

model = location_classifier()
model.train_and_test()
