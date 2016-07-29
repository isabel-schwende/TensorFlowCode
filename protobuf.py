############################################################################################
#Code by Isabel Schwende 
#Minimal code for saving a CNN in tensorflow with constant weights
#
#Inspired by code in this thread -> https://github.com/tensorflow/tensorflow/issues/616
############################################################################################

import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

LOG_DIR = '/tmp/alex_log' #adapt if necessary
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# original graph construction with weights as variables
vars = {}
with tf.Graph().as_default() as g_1:
    # Constrction of the complete graph
    # PUT YOUR CODE HERE
    for v in tf.trainable_variables():
        vars[v.value().name] = sess.run(v)
    graph1 = g_1.as_graph_def()
    # Optional code if you want to view your graph with tensorboard
    #writer = tf.train.SummaryWriter('/Users/isabel/Desktop/SandboxProjects', graph1)
    #writer.close()

# convert variables to constants
consts = {}
with tf.Graph().as_default() as g_2:
    # create one graph as combination of g1 and g2
    # mapping string to tensor objects in import_graph_def
    for k in vars.keys():
                consts[k] = tf.constant(vars[k])
    tf.import_graph_def(graph1,input_map={name:consts[name] for name in consts.keys()})
    #print consts #print the constants if neccessary
    
    tf.train.write_graph(sess.graph_def,'.','graph.pb')
    print os.path.join('.','graph.pb')
    # Optional code if you want to view your graph with tensorboard
    #graph2 = g_2.as_graph_def()
    #writer = tf.train.SummaryWriter('.', graph2)
    #writer.close()
