import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from tqdm import *
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output= tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

prediction = recurrent_network(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

    
def rgb_to_greyscale(pixels):
    r, g, b, a = pixels
    return .2126 * (r / 255) + .7152 * (g / 255) + 0.0722 * (b/255)

def train(x):
    iv = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(iv)
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in trange(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))


                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',
                  accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))
4
        saver.save(sess, './checkpoint/rnn.ckpt')

def test():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/rnn.ckpt')
        for i in range(10):
            img = Image.open("test/{}.png".format(i))
            features = np.array(list(map(rgb_to_greyscale, img.getdata()))).reshape(28, 28)

            result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
            print(i, result[0])
            if i != result[0]:
               img.show()

train(x)
test()
