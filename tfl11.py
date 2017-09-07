import tensorflow as tf

#Save to file, only save the variable
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32, name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,'savedir/save_net.ckpt')
    print('Save to:', save_path)
