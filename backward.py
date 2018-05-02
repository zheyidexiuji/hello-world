# -*- coding: utf-8 -*-
import tensorflow as tf
import forward
from tensorflow.examples.tutorials.mnist import input_data
BATC_SIZE = 200
LEARING_RATE_BASE = 0.001
LEARING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS =50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "minist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
    y = forward.forward(x,REGULARIZER)
    global_step = tf.Variable(0,trainable=False)
    
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer(LEARING_RATE_BASE).minimize(loss,global_step=global_step)   
    #saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            xs,ys = mnist.train.next_batch(BATC_SIZE)
            _,loss_value,step=sess.run([train_step,loss,global_step],feed_dict={x:xs,y_:ys})
            if i%1000 == 0 :
                print("%d    %g",step,loss_value)
def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    backward(mnist)
if __name__ =="__main__":
    main()
                
            