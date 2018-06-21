import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

hm_epochs = 10
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def rnn(x):
    layer = {'weights':t f.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)
    
    output= tf.matmul(??????, layer['weights']) + layer['biases']
    return output

prediction = rnn(x)
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
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))
        saver.save(sess, './checkpoint/rnn.ckpt')

def test():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './checkpoint/rnn.ckpt')
        for i in range(10):
            img = Image.open("test/{}.png".format(i))
            features = np.array(list(map(rgb_to_greyscale, img.getdata())))

            result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [features]}), 1))
            print(i, result[0])
            if i != result[0]:
               img.show()

train(x)
test()
