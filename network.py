import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

from PIL import Image

n_nodes = [500, 500, 500]
in_classes = 784
out_classes = 10
batch_size = 100
n_epochs = 10
x = tf.placeholder('float', [None, in_classes])
y = tf.placeholder('float')
r = lambda *d: tf.Variable(tf.random_normal(d))

def neural(data, n_nodes, f=tf.nn.relu):
    layers = [{'weights': r(a, b), 'biases': r(b), 'f': c} for a, b, c in
              zip([in_classes]+n_nodes,
                  n_nodes+[out_classes],
                  [f for _ in n_nodes]+[lambda _:_])]
    calc = data
    for l in layers:
        calc = l['f'](tf.add(tf.matmul(calc, l['weights']), l['biases']))

    return calc

prediction = neural(x, n_nodes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init_vars = tf.global_variables_initializer()
saver = tf.train.Saver()

def rgb_to_greyscale(pixels):
    r, g, b, a = pixels
    return .2126 * (r / 255) + .7152 * (g / 255) + 0.0722 * (b/255)

def train(x):
    with tf.Session() as sess:
        sess.run(init_vars)
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss:',epoch_loss)
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        saver.save(sess, './checkpoint/mnist.ckpt')

def test():
    with tf.Session() as sess:
        sess.run(init_vars)
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/mnist.ckpt')
        for i in range(10):
            img = Image.open("test/{}.png".format(i))
            features = np.array(list(map(rgb_to_greyscale, img.getdata())))

            result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
            print(i, result[0])

train(x)
test()
