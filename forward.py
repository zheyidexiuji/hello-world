# -*- coding: utf-8 -*-
import tensorflow as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weigth(shape,regularizer):
    ww1 = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!= None:
        tf.add_to_collection("loss",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return ww1
    
def get_bais(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
def forward(x,regularizer):
    w1 = get_weigth([INPUT_NODE,LAYER1_NODE],regularizer)
    b1 = get_bais([LAYER1_NODE])
    l1 = tf.nn.relu(tf.matmul(x,w1)+b1)
    
    w2 = get_weigth([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2 = get_bais([OUTPUT_NODE])
    y = tf.matmul(l1,w2)+b2
    return y
