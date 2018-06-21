import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

##from PIL import Image

n_classes=10
batch_size=128
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    #                        size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_net(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}
    
    biases = {'B_conv1': tf.Variable(tf.random_normal([32])),
              'B_conv2': tf.Variable(tf.random_normal([64])),
              'B_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    conv1 = conv2d(x, weights['W_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(x, weights['W_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['B_fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']
    
    return output

prediction = conv_net(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
hm_epochs = 10
    
##def rgb_to_greyscale(pixels):
##    r, g, b, a = pixels
##    return .2126 * (r / 255) + .7152 * (g / 255) + 0.0722 * (b/255)

def train(x):
    iv = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(iv)
        saver = tf.train.Saver()
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        saver.save(sess, './checkpoint/convnet.ckpt')

##def test():
##    with tf.Session() as sess:
##        sess.run(tf.global_variables_initializer())
##        saver = tf.train.Saver()
##        saver.restore(sess, './checkpoint/mnist.ckpt')
##        for i in range(10):
##            img = Image.open("test/{}.png".format(i))
##            features = np.array(list(map(rgb_to_greyscale, img.getdata())))
##
##            result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
##            print(i, result[0])
##            if i != result[0]:
##               img.show()

train(x)
##test()
