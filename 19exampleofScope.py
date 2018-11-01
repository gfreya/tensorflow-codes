from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(1)

class TrainConfig:
    batch_size = 20
    time_steps = 20
    input_size = 10
    output_size = 2
    cell_size = 11
    learning_rate = 0.01

class TestConfig(TrainConfig):
    time_step = 1

class RNN(object):
    pass

if __name__ =='__main__':
    train_config = TrainConfig()
    test_config = TestConfig()
    with tf.variable_scope('train_rnn'):
        train_rnn1 = RNN(train_config)
    with tf.variable_scope('test_run'):
        test_rnn1 = RNN(test_config)

    with tf.variable_scope('run') as scope:
        sess = tf.Session()
        train_rnn2 = RNN(train_config)
        scope.reuse_variables()
        test_rnn2 = RNN(test_config)
        sess.run(tf.initialize_all_variables())