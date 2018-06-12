import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist= input_data.read_data_sets("/tmp/data/", one_hot=True)

from PIL import Image

#10 clasees, 0-9
n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10
batch_size=100
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1=tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    li= tf.nn.relu(l1)
    l2=tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2= tf.nn.relu(l2)
    l3=tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3= tf.nn.relu(l3)
    output= tf.matmul(l3, output_layer['weights'])+ output_layer['biases']
    return output

prediction = neural(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
hm_epochs = 10
init_vars = tf.global_variables_initializer()
saver = tf.train.Saver()

def rgb_to_greyscale(pixels):
    r, g, b, a = pixels
    return .2126 * (r / 255) + .7152 * (g / 255) + 0.0722 * (b/255)

def train(x):
    
    with tf.Session() as sess:
        sess.run(init_vars)
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
            if i != result[0]:
               img.show()

train(x)
test()
