from __future__ import print_function
import tensorflow as tf
tf.set_random_seed(1) #reproducible

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=initializer) #使用get_variable方法构建name_scope是无效的
    var2 = tf.get_variable(name='var2', shape=[1], dtype=tf.float32)
    var21 = tf.get_variable(name='var2', shape=[2,1], dtype=tf.float32)
    var22 = tf.get_variable(name='var2', shape=[2,2], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(var1.name)
    print(sess.run(var1))
    print(var2.name)
    print(sess.run(var2))
    print(var21.name)
    print(sess.run(var21))
    print(var22.name)
    print(sess.run(var22))