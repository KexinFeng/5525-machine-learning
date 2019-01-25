import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time, os
import matplotlib.pyplot as plt
import csv, codecs

## improvement:
# drop
# softmax on last_output
# dynamic_rnn
# multilayer lstm

# input matrix->vector


# buid lstm cell

# cell = MultiRNNCell(drops)
#
# lstm = BasicLSTMCell(64)

mnist = input_data.read_data_sets("./MNIST-data/", one_hot=True)


n_input = 28   # row length of 28 x 28 image
n_steps = 28   # 28 time steps
n_classes = 10 # output classes

n_hidden = 128 # hidden state size = lstm_size

learning_rate = 0.001



x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
init_state = tf.placeholder(tf.float32, [None, 2*n_hidden])



weights = {
    'hidden' : tf.Variable(tf.truncated_normal([n_input, n_hidden], stddev=0.1)),
    'out' : tf.Variable(tf.truncated_normal([n_hidden, n_classes], stddev=0.1))
}
biases = {
    'hidden' : tf.Variable(tf.constant(0.1, shape=[n_hidden])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}



def lstm_layer(X, init_state, weights, biases ):
    X = tf.transpose(X,[1, 0, 2])
    X = tf.reshape(X, [-1, n_input])
    # X.shape >> (n_steps * batch_size, n_input)

    lstm = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=False)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.3)

    X = tf.matmul(X, weights['hidden']) + biases['hidden']
    # apply weight(rotation) on each of the sequence element, which is vector.
    X = tf.split(X, n_steps, 0)
    # X.shape >> n_steps batch_size * n_input
    outputs, final_state = tf.contrib.rnn.static_rnn(cell=drop, inputs=X, initial_state= init_state)
    last_output = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])
    return last_output

with tf.variable_scope("Build_lstm_layer") as scope:
    last_output = lstm_layer(x, init_state, weights, biases)

with tf.name_scope("train_and_loss"):
    # train
    # def build_lost_pred_and_opt(lstm_outputs, labels, learning_rate):
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)

with tf.name_scope("accuracy"):
    prediciton = tf.equal(tf.argmax(last_output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediciton, tf.float32))


NUM_THREADS = 2
stat_step = 10
batch_size = 64
test_size = 256
n_epochs = 2


loss_t = []
acc_t = []
time_t = []

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS, inter_op_parallelism_threads=NUM_THREADS, log_device_placement=False)) as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./lstm_graph', sess.graph)
    writer.close()

    n_batches = mnist.train.num_examples // batch_size
    for idx in range(1, n_batches * n_epochs + 1):
        x_batch, y_batch = mnist.train.next_batch(batch_size)
        x_batch = x_batch.reshape((-1, 28, 28))
        # x_batch = x_batch.reshape((batch_size, n_steps, n_input))

        start_time = time.time()

        train_init_state = np.zeros((batch_size, 2 * n_hidden))
        train_feed = {x: x_batch, y: y_batch, init_state: train_init_state}

        loss, _ = sess.run([loss_function, optimizer], feed_dict=train_feed)
        if idx % stat_step ==0:
            acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch, init_state: train_init_state})
            print('iteration: {}/{}'.format(idx, n_batches),
                  'epoch: {}/{}'.format(idx//n_batches, n_epochs),
                  " current_loss: {:.6f}".format(loss),
                  " Training accuracy: {:.5f}".format(acc))
            # print("step : " + str(idx*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(accuracy))

            # output plot
            loss_t.append(loss)
            acc_t.append(acc)
            end_time = time.time()
            time_t.append((end_time - start_time)/stat_step)
            start_time = time.time()


    # test
    test_images = mnist.test.images[:test_size]
    test_images = test_images.reshape((-1, 28, 28))
    test_labels = mnist.test.labels[:test_size]

    test_init_state = np.zeros((test_size, 2 * n_hidden))
    test_feed = {x: test_images,
                y: test_labels,
                init_state: test_init_state}
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)

    print("test accuracy: ", test_accuracy)

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name,'w+','utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("save complete")

try:
    os.mkdir('files_to_plot')
except:
    pass

output_file_name = './files_to_plot/lstm.csv'

data_write_csv(output_file_name, [loss_t])
data_write_csv(output_file_name, [acc_t])
data_write_csv(output_file_name, [time_t])

plt.figure()
plt.plot(loss_t)
plt.title('loss_t')

plt.figure()
plt.plot(acc_t)
plt.title('training_accuracy')

plt.figure()
plt.plot(time_t)
plt.title('time/display_step')

plt.show()