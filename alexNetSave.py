################################################################################
#Michael Guerzhoy, 2016
#AlexNet implementation in TensorFlow, with weights
#with changes by Isabel Schwende
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
################################################################################

from numpy import *
import numpy as np
import os
import os.path
from scipy.misc import imread

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib

from caffe_classes import class_names
from graph_util import mygraph
from freeze_graph import freeze_graph

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

LOG_DIR = '/tmp/alex_log'#adapt 
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

#Read Image

x_dummy = (np.random.random((1,)+ xdim)/255.).astype(float32)
i = x_dummy.copy()
i[0,:,:,:] = (imread("poodle.png")[:,:,:3]).astype(float32)
i = i-mean(i)
# loading trained weights
net_data = load("bvlc_alexnet.npy").item()
	
with tf.Graph().as_default() as g_1:	
	#### Initialize network and finalize graph1
	g_1,prob = mygraph(g_1,i,net_data)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		#Test output:
		output = sess.run(prob)
		inds = argsort(output)[0,:]
		for i in range(5):
			print class_names[inds[-1-i]], output[0, inds[-1-i]]
		#### End network test ####
	
		#### Saving network data ####
		checkpoint_prefix = os.path.join(LOG_DIR, "saved_checkpoint")
    		checkpoint_state_name = "checkpoint_state"
    		
    		output_graph_name = "output_graph.pb"
	
		saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
		checkpoint_path = saver.save(
		  sess,
		  checkpoint_prefix,
		  global_step=0,
		  latest_filename=checkpoint_state_name)

	input_graph_name = "input_graph.pb"

#	tf.train.write_graph(g_1.as_graph_def(), LOG_DIR, input_graph_name,False)
	
