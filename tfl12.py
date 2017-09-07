import tensorflow as tf
import numpy as np

#restore from the  file
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

#not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"savedir/save_net.ckpt")
    print("weights:",sess.run(W))
    print("bises:",sess.run(b))
