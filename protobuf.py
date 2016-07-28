import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

LOG_DIR = '/tmp/alex_log'#adapt 

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# original graph such as for training or loading existing graph
vars = {}
with tf.Graph().as_default() as g_1:
    # All the blah trainging stuff       
    # isa addition
    for v in tf.trainable_variables():
        vars[v.value().name] = sess.run(v)
        
    graph1 = g_1.as_graph_def()
    #writer = tf.train.SummaryWriter('/Users/isabel/Desktop/SandboxProjects', graph1)
    #writer.close()

# convert numpy array to constant
consts = {}
with tf.Graph().as_default() as g_2:
    # create one graph as combination of g1 and g2
    # mapping string to tensor objects 
    for k in vars.keys():
                consts[k] = tf.constant(vars[k])
    tf.import_graph_def(graph1,input_map={name:consts[name] for name in consts.keys()})
    #print consts
    
    tf.train.write_graph(sess.graph_def,'.','graph.pb')
    print os.path.join('.','graph.pb')
    #graph2 = g_2.as_graph_def()
    #writer = tf.train.SummaryWriter('.', graph2)
    #writer.close()
